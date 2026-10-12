"""Shared camera/trial runtime for desktop and mobile browser frontends."""
from __future__ import annotations

import copy
import math
import queue
import shutil
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd

from scripts.camera_io import LANDMARK_COUNT, build_capture_row, make_trial_id, receive_frame
from scripts.device_control import DeviceControl
from scripts.history import History, now_iso
from scripts.recording import Recording, find_ffmpeg
from scripts.realtime_pipeline import append_jsonl, process_trial
from scripts.realtime_predictors import load_predictor


class StateConflict(ValueError):
    pass


class RealtimeEngine:
    def __init__(self, args, *, predictor=None, recorder_factory=Recording, ffmpeg=None):
        self.args = args
        if not math.isfinite(args.duration) or args.duration <= 0 or args.minimum_valid_frames < 2 or not 0 <= args.minimum_success_rate <= 1:
            raise ValueError("収集時間・品質条件の設定が不正です")
        self.predictor = predictor or load_predictor(args.predictor, model_path=args.model,
            metadata_path=args.metadata, n_jobs=args.model_n_jobs)
        self.recording_enabled = getattr(args, "record_video", True)
        self.ffmpeg = (ffmpeg or find_ffmpeg(getattr(args, "ffmpeg", None))) if self.recording_enabled else None
        self.history = History(args.log, getattr(args, "journal", Path("results/realtime_journal.jsonl")),
                               getattr(args, "video_dir", Path("video_data/realtime")))
        self.recorder_factory = recorder_factory
        self.lock = threading.RLock()
        self.control = DeviceControl(args.control_port, args.disable_control)
        self.stop_event = threading.Event()
        self.camera_connected = False
        self.camera_error = None
        self.storage_error = None
        self.last_frame = None
        self.last_frame_at = 0.0
        self.face_detected = False
        self.show_mesh = args.show_facemesh
        self.preview_jpeg = None
        self.preview_frame = None
        self.preview_sequence = 0
        self.last_preview = 0.0
        self.frames = queue.Queue(maxsize=2)
        self.active_trial = None
        self.last_result = None
        self.manual_recording = None
        self.recordings = {}
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trial-inference")
        self.future = None
        self.server = None
        self.connection = None
        self.landmarker = None
        self.threads = []

    def start(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp = mp
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            self.server.bind((self.args.host, self.args.port))
            self.server.listen(1)
            self.server.settimeout(0.2)
            self.landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(self.args.face_model)),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3))
            self.history.recover()
            self.control.start()
            for target, name in ((self._receive_loop, "camera-receive"), (self._analysis_loop, "camera-analysis")):
                thread = threading.Thread(target=target, name=name, daemon=True)
                self.threads.append(thread)
                thread.start()
        except Exception:
            self.close()
            raise

    def _receive_loop(self):
        while not self.stop_event.is_set():
            try:
                connection, address = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.connection = connection
            self.control.set_target(self.args.raspi_ip or address[0])
            connection.settimeout(2)
            buffered = b""
            print(f"[CAMERA] Piから映像接続: {address[0]}:{address[1]}", flush=True)
            try:
                while not self.stop_event.is_set():
                    frame, buffered = receive_frame(connection, buffered)
                    stamp = time.monotonic()
                    self.accept_frame(frame, stamp)
                    # Drop only preview/analysis backlog, never accumulate stale live video.
                    try:
                        self.frames.put_nowait((frame, stamp))
                    except queue.Full:
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.frames.put_nowait((frame, stamp))
                        except queue.Full:
                            pass
            except Exception as exc:
                print(f"[CAMERA ERROR] 映像受信失敗: {type(exc).__name__}: {exc}", flush=True)
                self.disconnect(str(exc))
            finally:
                connection.close()
                self.connection = None
                reason = "PCサーバー終了" if self.stop_event.is_set() else (self.camera_error or "Pi側から映像接続が閉じられました")
                print(f"[CAMERA] 映像切断: {reason}", flush=True)
                self.disconnect(reason)

    def accept_frame(self, frame, stamp):
        with self.lock:
            self.last_frame, self.last_frame_at = frame, stamp
            self.camera_connected, self.camera_error = True, None
            trial = self.active_trial
            if trial and trial["status"] == "collecting" and stamp >= trial["started"] + self.args.duration:
                self._finish_recording(trial["video_id"], trial["started"] + self.args.duration)
            for recording in list(self.recordings.values()):
                recording.push(frame, stamp)

    def _analysis_loop(self):
        timestamp = -1
        while not self.stop_event.is_set():
            try:
                try:
                    frame, stamp = self.frames.get(timeout=0.05)
                except queue.Empty:
                    self.tick(time.monotonic())
                    continue
                timestamp = max(timestamp + 1, int(stamp * 1000))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detection = self.landmarker.detect_for_video(
                    self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp)
                landmarks = detection.face_landmarks[0] if detection.face_landmarks else None
                self.analyze_frame(frame, stamp, timestamp, landmarks)
                self.tick(time.monotonic())
            except Exception as exc:
                print(f"[CAMERA ERROR] FaceMesh／プレビュー処理失敗: {type(exc).__name__}: {exc}", flush=True)
                self.disconnect("映像処理エラー: " + str(exc))

    def analyze_frame(self, frame, stamp, timestamp, landmarks):
        valid = landmarks is not None and len(landmarks) >= LANDMARK_COUNT
        height, width = frame.shape[:2]
        with self.lock:
            if not self.camera_connected or time.monotonic() - stamp > 2:
                return
            self.face_detected = valid
            self.control.set(face=valid)
            trial = self.active_trial
            if trial and trial["status"] == "collecting" and 0 <= stamp - trial["started"] <= self.args.duration:
                trial["rows"].append(build_capture_row(landmarks, frame_index=len(trial["rows"]) + 1,
                    timestamp_ms=timestamp, elapsed_sec=stamp - trial["started"],
                    frame_width=width, frame_height=height))
            if stamp - self.last_preview < 0.1:
                return
            show_mesh = self.show_mesh
        preview = cv2.resize(frame, (640, 400))
        if show_mesh and valid:
            for point in landmarks:
                cv2.circle(preview, (int(point.x * 640), int(point.y * 400)), 1, (0, 255, 0), -1)
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.preview_jpeg, self.preview_frame = encoded.tobytes(), preview
                self.preview_sequence += 1
                self.last_preview = stamp

    def _ensure_ready(self):
        if self.stop_event.is_set():
            raise StateConflict("サーバーは終了処理中です")
        if not self.camera_connected or self.last_frame is None or time.monotonic() - self.last_frame_at > 2:
            raise StateConflict("カメラの接続を待ってください")
        if self.storage_error:
            raise StateConflict("保存先を確認して再起動してください: " + self.storage_error)

    def _new_recording(self, kind, started, trial_id=None):
        video_id = "video_" + uuid.uuid4().hex
        self.history.update(video_id, kind="recording", source=kind, trial_id=trial_id, status="recording")
        try:
            if shutil.disk_usage(self.history.video_dir).free < 100 * 1024 * 1024:
                raise OSError("録画用の空き容量が100MB未満です")
            recording = self.recorder_factory(video_id, self.history.video_dir / (video_id + ".mp4"),
                self.last_frame, started, self.ffmpeg, self._recording_completed)
            self.recordings[video_id] = recording
            self.workers = [worker for worker in self.workers if worker.thread.is_alive()]
            self.workers.append(recording)
        except Exception as exc:
            self.history.update(video_id, status="failed", error_message=str(exc))
        return video_id

    def _finish_recording(self, video_id, stamp, reason=None):
        recording = self.recordings.pop(video_id, None)
        if recording:
            try:
                self.history.update(video_id, status="finalizing")
            except OSError as exc:
                self.storage_error = str(exc)
            finally:
                recording.finish(max(recording.started, stamp), reason)

    def _recording_completed(self, video_id, status, error, duration):
        with self.lock:
            self.recordings.pop(video_id, None)
            if self.manual_recording == video_id:
                self.manual_recording = None
            try:
                self.history.update(video_id, status=status, error_message=error, duration_seconds=duration)
            except OSError as exc:
                self.storage_error = str(exc)

    def start_trial(self):
        with self.lock:
            self._ensure_ready()
            if self.active_trial is not None:
                raise StateConflict("すでに収集・判定中です")
            started = time.monotonic()
            trial_id = make_trial_id()
            video_id = self._new_recording("trial", started, trial_id) if self.recording_enabled else None
            try:
                self.history.update(trial_id, kind="trial", status="collecting", video_id=video_id)
            except OSError:
                self._finish_recording(video_id, started, "試行開始ログを保存できませんでした")
                raise
            self.active_trial = {"id": trial_id, "video_id": video_id, "started": started,
                                 "status": "collecting", "rows": []}
            self.last_result = None
            return {"trial_id": trial_id, "video_id": video_id}

    def start_recording(self):
        with self.lock:
            self._ensure_ready()
            if not self.recording_enabled:
                raise StateConflict("録画が無効です")
            if self.manual_recording:
                raise StateConflict("すでに任意録画中です")
            video_id = self._new_recording("manual", time.monotonic())
            if video_id not in self.recordings:
                raise StateConflict("録画を開始できません。履歴のエラーを確認してください")
            self.manual_recording = video_id
            return {"video_id": video_id}

    def stop_recording(self):
        with self.lock:
            if not self.manual_recording:
                raise StateConflict("任意録画は開始されていません")
            video_id, self.manual_recording = self.manual_recording, None
            self._finish_recording(video_id, time.monotonic())
            return {"video_id": video_id}

    def tick(self, stamp):
        with self.lock:
            trial = self.active_trial
            if trial and trial["status"] == "collecting" and stamp - trial["started"] >= self.args.duration:
                self._finish_recording(trial["video_id"], trial["started"] + self.args.duration)
                try:
                    self.history.update(trial["id"], status="processing")
                except OSError as exc:
                    self.storage_error = str(exc)
                trial["status"] = "processing"
                self.control.set(phase="processing")
                self.future = self.executor.submit(process_trial, pd.DataFrame(trial["rows"]),
                    trial_id=trial["id"], predictor=self.predictor,
                    minimum_valid_frames=self.args.minimum_valid_frames,
                    minimum_success_rate=self.args.minimum_success_rate, log_path=self.args.log,
                    save_trials_dir=self.args.save_trials_dir, video_id=trial["video_id"])
            if self.future is not None and self.future.done():
                try:
                    result = self.future.result()
                    self.last_result = {"status": "success", **asdict(result)}
                    self.control.result(result.prediction.predicted_name)
                except Exception as exc:
                    self.last_result = {"status": "rejected", "trial_id": trial["id"], "error_message": str(exc)}
                    if isinstance(exc, OSError):
                        self.storage_error = str(exc)
                    self.control.set(phase="face")
                try:
                    self.history.update(trial["id"], status="finished")
                except OSError as exc:
                    self.storage_error = str(exc)
                self.future = None
                self.active_trial = None

    def disconnect(self, reason):
        with self.lock:
            self.camera_connected, self.camera_error = False, reason
            self.face_detected = False
            self.preview_jpeg, self.preview_frame = None, None
            self.last_frame = None
            self.control.set(face=False)
            for video_id in list(self.recordings):
                self._finish_recording(video_id, self.last_frame_at, reason or "映像切断")
            self.manual_recording = None
            trial = self.active_trial
            if trial and trial["status"] == "collecting":
                row = {"recorded_at": now_iso(), "trial_id": trial["id"], "status": "interrupted",
                       "video_id": trial["video_id"], "error_message": reason or "映像切断",
                       "total_frames": len(trial["rows"]),
                       "valid_frames": sum(row["facemesh_ok"] for row in trial["rows"])}
                self.last_result = row
                self.active_trial = None
                self.control.set(phase="face")
                try:
                    append_jsonl(self.args.log, row)
                except OSError as exc:
                    self.storage_error = str(exc)
                try:
                    self.history.update(trial["id"], status="interrupted", error_message=reason)
                except OSError as exc:
                    self.storage_error = str(exc)

    def snapshot(self):
        with self.lock:
            trial = self.active_trial
            elapsed = time.monotonic() - trial["started"] if trial else 0
            return {"camera_connected": self.camera_connected, "camera_error": self.camera_error,
                "storage_error": self.storage_error, "face_detected": self.face_detected,
                "show_mesh": self.show_mesh, "phase": trial["status"] if trial else "idle",
                "trial_id": trial["id"] if trial else None,
                "remaining_seconds": max(0, self.args.duration - elapsed) if trial else 0,
                "total_frames": len(trial["rows"]) if trial else 0,
                "valid_frames": sum(row["facemesh_ok"] for row in trial["rows"]) if trial else 0,
                "result": copy.deepcopy(self.last_result), "manual_recording": self.manual_recording,
                "control": self.control.snapshot(), "recording_enabled": self.recording_enabled}

    def close(self):
        self.stop_event.set()
        for sock in (self.connection, self.server):
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass
        for thread in self.threads:
            thread.join(timeout=4)
        self.disconnect("サーバーを終了しました")
        self.executor.shutdown(wait=True)
        self.tick(time.monotonic())
        for worker in self.workers:
            worker.join()
        self.control.close()
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
