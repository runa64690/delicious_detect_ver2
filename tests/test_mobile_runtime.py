from __future__ import annotations

import json
import mimetypes
import pickle
import socket
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from raspberrypi.led_control import LedRingController, handle_message
from raspberrypi.pi_send_camera import control_server
from scripts.device_control import DeviceControl
from scripts.history import History, read_jsonl
from scripts.recording import Recording, find_ffmpeg
from scripts.realtime_engine import RealtimeEngine, StateConflict
from scripts.realtime_pipeline import append_jsonl
from web_app import create_app, parse_args
from test_realtime_inference import FakePredictor, raw_trial_with_one_failure


def wait_until(condition, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for condition")


class StubRecording:
    def __init__(self, video_id, path, frame, started, ffmpeg, completed):
        self.video_id, self.path, self.started = video_id, path, started
        self.completed = completed
        self.frames = []
        self.thread = SimpleNamespace(is_alive=lambda: False)

    def push(self, frame, stamp):
        self.frames.append(stamp)

    def finish(self, stamp, reason=None):
        self.path.write_bytes(b"fake-mp4-data")
        self.completed(self.video_id, "interrupted" if reason else "completed", reason, stamp-self.started)

    def join(self):
        pass


class FakePixels:
    def __init__(self):
        self.colors = [(0, 0, 0)] * 12
        self.brightness = 0.15
        self.shown = 0

    def __len__(self):
        return len(self.colors)

    def __setitem__(self, index, color):
        self.colors[index] = color

    def fill(self, color):
        self.colors[:] = [color] * len(self.colors)

    def show(self):
        self.shown += 1


class MobileRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.args = parse_args([])
        self.args.log = self.root / "predictions.jsonl"
        self.args.journal = self.root / "journal.jsonl"
        self.args.video_dir = self.root / "videos"
        self.args.duration = 0.3
        self.args.minimum_valid_frames = 2
        self.args.minimum_success_rate = 0.5
        self.args.disable_control = True
        self.engine = RealtimeEngine(self.args, predictor=FakePredictor(),
                                     recorder_factory=StubRecording, ffmpeg="unused")
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.engine.accept_frame(self.frame, time.monotonic())

    def tearDown(self):
        self.engine.close()
        self.directory.cleanup()

    def complete_trial(self, valid=True):
        trial = self.engine.active_trial
        sample = raw_trial_with_one_failure().iloc[0]
        landmarks = [SimpleNamespace(x=sample[f"lm{i}_x_px"] / 64,
                                    y=sample[f"lm{i}_y_px"] / 48) for i in range(478)] if valid else None
        for index in range(3):
            stamp = trial["started"] + 0.03 + index * 0.08
            self.engine.accept_frame(self.frame, stamp)
            self.engine.analyze_frame(self.frame, stamp, int(stamp * 1000), landmarks)
        self.engine.tick(trial["started"] + self.args.duration)
        wait_until(lambda: self.engine.future.done())
        self.engine.tick(trial["started"] + self.args.duration + 0.01)

    def test_parallel_recording_and_trial_with_durable_links(self):
        manual = self.engine.start_recording()["video_id"]
        started = self.engine.start_trial()
        with self.assertRaises(StateConflict):
            self.engine.start_trial()
        with self.assertRaises(StateConflict):
            self.engine.start_recording()
        self.complete_trial()
        self.assertEqual(self.engine.snapshot()["result"]["status"], "success")
        self.assertEqual(self.engine.manual_recording, manual)
        self.engine.stop_recording()
        history = self.engine.history.items()
        trial = next(row for row in history if row["id"] == started["trial_id"])
        self.assertEqual(trial["video_id"], started["video_id"])
        self.assertEqual(trial["video_status"], "completed")
        self.assertNotEqual(manual, trial["video_id"])
        restored = History(self.args.log, self.args.journal, self.args.video_dir)
        self.assertEqual(len(restored.items()), 3)

    def test_quality_rejection_keeps_video_and_reason(self):
        started = self.engine.start_trial()
        self.complete_trial(valid=False)
        row = next(row for row in self.engine.history.items() if row["id"] == started["trial_id"])
        self.assertEqual(row["status"], "rejected")
        self.assertEqual(row["video_status"], "completed")
        self.assertEqual(row["error_type"], "TrialQualityError")

    def test_camera_disconnect_aborts_trial_but_accepts_new_session(self):
        self.engine.start_recording()
        self.engine.start_trial()
        self.engine.disconnect("test disconnect")
        self.assertIsNone(self.engine.active_trial)
        self.assertIsNone(self.engine.manual_recording)
        self.assertEqual(read_jsonl(self.args.log)[0]["status"], "interrupted")
        self.assertTrue(all(row["status"] == "interrupted" for row in self.engine.history.items()))
        with self.assertRaises(StateConflict):
            self.engine.start_trial()
        self.engine.accept_frame(self.frame, time.monotonic())
        self.engine.start_trial()
        self.assertEqual(self.engine.snapshot()["phase"], "collecting")

    def test_recording_failure_does_not_destroy_prediction(self):
        def fail(*args):
            raise OSError("disk full")
        self.engine.recorder_factory = fail
        started = self.engine.start_trial()
        self.complete_trial()
        row = next(row for row in self.engine.history.items() if row["id"] == started["trial_id"])
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["video_status"], "failed")
        self.assertEqual(row["video_error"], "disk full")

    def test_trial_recording_stops_at_deadline_even_if_analysis_lags(self):
        started = self.engine.start_trial()
        deadline = self.engine.active_trial["started"] + self.args.duration
        self.engine.accept_frame(self.frame, deadline + 0.5)
        row = self.engine.history.records[started["video_id"]]
        self.assertAlmostEqual(row["duration_seconds"], self.args.duration)

    def test_recovery_handles_legacy_logs_and_unfinished_sessions(self):
        append_jsonl(self.args.log, {"trial_id": "old_trial", "status": "success", "recorded_at": "2020-01-01"})
        started = self.engine.start_trial()
        restored = History(self.args.log, self.args.journal, self.args.video_dir)
        restored.recover()
        items = restored.items()
        old = next(row for row in items if row["id"] == "old_trial")
        self.assertIsNone(old["video_status"])
        trial = next(row for row in items if row["id"] == started["trial_id"])
        self.assertEqual(trial["status"], "interrupted")

    def test_api_serializes_start_and_downloads_by_id(self):
        with TestClient(create_app(engine=self.engine, manage_runtime=False)) as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/static/app.js").status_code, 200)
            self.assertEqual(client.post("/api/trials", json={}).status_code, 202)
            self.assertEqual(client.post("/api/trials", json={}).status_code, 409)
            # New browser/client sees the same server-owned capture.
            self.assertEqual(client.get("/api/state").json()["phase"], "collecting")
            self.complete_trial()
            video_id = next(row["video_id"] for row in client.get("/api/history").json()["items"] if row["kind"] == "trial")
            response = client.get(f"/api/videos/{video_id}", headers={"Range": "bytes=0-3"})
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, b"fake")
            self.assertIn("attachment", client.get(f"/api/videos/{video_id}?download=true").headers["content-disposition"])
            self.assertEqual(client.get("/api/videos/unknown").status_code, 404)
            self.assertTrue(client.get("/api/logs/csv").content.startswith(b"\xef\xbb\xbf"))
            self.assertIn("predicted_name", client.get("/api/logs/jsonl").text)
            self.assertEqual(client.get("/api/history?offset=-1").status_code, 422)

    def test_api_validates_led_and_blocks_cross_origin_mutations(self):
        with TestClient(create_app(engine=self.engine, manage_runtime=False)) as client:
            payload = {"mode":"manual", "rgb":[10, 20, 30], "brightness":0.1}
            self.assertEqual(client.put("/api/led", json=payload).status_code, 202)
            self.assertEqual(self.engine.control.snapshot()["rgb"], [10,20,30])
            self.assertEqual(client.put("/api/led", json={**payload, "brightness":1}).status_code, 422)
            self.assertEqual(client.put("/api/led", json={**payload, "rgb":[True,0,0]}).status_code, 422)
            self.assertEqual(client.put("/api/ir", json={"enabled":True}).status_code, 202)
            self.assertTrue(self.engine.control.snapshot()["ir"])
            self.assertEqual(client.post("/api/trials", json={}, headers={"Origin":"https://unrelated.example"}).status_code, 403)
            self.assertEqual(client.post("/api/trials", content="").status_code, 415)

    def test_web_assets_keep_browser_mime_types_despite_host_file_associations(self):
        # Windows registry associations can make .js text/plain. Combined with
        # nosniff, browsers refuse to execute it and never request /api/state.
        mimetypes.init()
        with patch.dict(mimetypes.types_map, {extension: "text/plain" for extension in (".js", ".css", ".html")}), \
             TestClient(create_app(engine=self.engine, manage_runtime=False)) as client:
            for path, expected in (("/", "text/html"),
                                   ("/static/app.js", "text/javascript"),
                                   ("/static/style.css", "text/css")):
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["content-type"].split(";")[0], expected)
                    self.assertEqual(response.headers["x-content-type-options"], "nosniff")

    def test_truncated_log_does_not_swallow_next_record(self):
        self.args.log.write_text('{"broken":', encoding="utf-8")
        append_jsonl(self.args.log, {"trial_id":"after_crash", "status":"interrupted"})
        self.assertEqual(read_jsonl(self.args.log)[0]["trial_id"], "after_crash")

    def test_simultaneous_trial_requests_start_only_one_trial(self):
        barrier = threading.Barrier(2)
        results = []
        def start():
            barrier.wait()
            try:
                results.append(self.engine.start_trial())
            except StateConflict:
                results.append("conflict")
        threads = [threading.Thread(target=start) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        self.assertEqual(results.count("conflict"), 1)
        self.assertEqual(len(self.engine.recordings), 1)

    def test_disk_failure_during_disconnect_still_stops_all_work(self):
        self.engine.start_trial()
        self.engine.start_recording()
        with patch.object(self.engine.history, "update", side_effect=OSError("full")), \
             patch("scripts.realtime_engine.append_jsonl", side_effect=OSError("full")):
            self.engine.disconnect("lost camera")
        self.assertFalse(self.engine.recordings)
        self.assertIsNone(self.engine.active_trial)
        self.assertIsNone(self.engine.manual_recording)
        self.assertEqual(self.engine.snapshot()["storage_error"], "full")
        self.assertEqual(self.engine.snapshot()["result"]["status"], "interrupted")

    def test_journal_failure_during_processing_does_not_leave_trial_busy(self):
        self.engine.start_trial()
        with patch.object(self.engine.history, "update", side_effect=OSError("full")):
            self.complete_trial()
        self.assertEqual(self.engine.snapshot()["phase"], "idle")
        self.assertEqual(self.engine.snapshot()["result"]["status"], "success")
        self.assertEqual(self.engine.snapshot()["storage_error"], "full")
        with self.assertRaises(StateConflict):
            self.engine.start_trial()

    def test_desktop_keys_use_shared_engine(self):
        import realtime_classify
        with patch.object(realtime_classify, "parse_args", return_value=self.args), \
             patch.object(realtime_classify, "RealtimeEngine", return_value=self.engine), \
             patch.object(self.engine, "start"), patch.object(self.engine, "close"), \
             patch.object(realtime_classify.cv2, "imshow"), \
             patch.object(realtime_classify.cv2, "destroyAllWindows"), \
             patch.object(realtime_classify.cv2, "waitKey", side_effect=[13, ord("f"), ord(" "), ord("q")]):
            self.assertEqual(realtime_classify.main(), 0)
        self.assertIsNotNone(self.engine.active_trial)
        self.assertTrue(self.engine.show_mesh)
        self.assertTrue(self.engine.control.snapshot()["ir"])

    def test_busy_camera_port_fails_before_history_recovery(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            self.args.host = "127.0.0.1"
            self.args.port = occupied.getsockname()[1]
            with patch.object(self.engine.history, "recover") as recovery:
                with self.assertRaises(OSError):
                    self.engine.start()
                recovery.assert_not_called()

    def test_camera_socket_reconnect_does_not_restart_aborted_trial(self):
        self.args.host, self.args.port, self.args.duration = "127.0.0.1", 0, 10
        detector = SimpleNamespace(detect_for_video=lambda *_: SimpleNamespace(face_landmarks=[]), close=lambda: None)
        from mediapipe.tasks.python import vision
        with patch.object(vision.FaceLandmarker, "create_from_options", return_value=detector):
            self.engine.start()
        self.engine.camera_connected = False
        port = self.engine.server.getsockname()[1]
        ok, jpeg = cv2.imencode(".jpg", self.frame)
        payload = pickle.dumps(jpeg)
        message = struct.pack("Q", len(payload)) + payload
        with socket.create_connection(("127.0.0.1", port)) as camera:
            camera.sendall(message)
            wait_until(lambda: self.engine.snapshot()["camera_connected"])
            self.engine.start_trial()
        wait_until(lambda: not self.engine.snapshot()["camera_connected"])
        self.assertIsNone(self.engine.active_trial)
        with socket.create_connection(("127.0.0.1", port)) as camera:
            camera.sendall(message)
            wait_until(lambda: self.engine.snapshot()["camera_connected"])
            self.assertIsNone(self.engine.active_trial)
            self.engine.start_trial()
            self.assertIsNotNone(self.engine.active_trial)


class RecordingTests(unittest.TestCase):
    def test_failed_encoder_is_reported_without_publishing_a_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.mp4"
            outcomes = []
            frame = np.zeros((48,64,3), dtype=np.uint8)
            recording = Recording("test", path, frame, 100, str(Path(directory) / "missing-ffmpeg"),
                                  lambda *row: outcomes.append(row))
            recording.finish(100.1)
            recording.join()
            self.assertEqual(outcomes[0][1], "failed")
            self.assertFalse(path.exists())

    def test_real_ffmpeg_duration_and_decoding(self):
        ffmpeg = find_ffmpeg()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.mp4"
            outcomes = []
            frame = np.zeros((48,64,3), dtype=np.uint8)
            recording = Recording("test", path, frame, 100.0, ffmpeg, lambda *row: outcomes.append(row))
            recording.push(np.full_like(frame, 100), 100.14)
            recording.push(np.full_like(frame, 200), 100.41)
            recording.finish(100.6)
            recording.join()
            self.assertEqual(outcomes[0][1], "completed", outcomes)
            video = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(video.isOpened())
                self.assertEqual(video.get(cv2.CAP_PROP_FRAME_COUNT), 18)
                self.assertEqual(video.get(cv2.CAP_PROP_FPS), 30)
                self.assertTrue(video.read()[0])
            finally:
                video.release()
            data = path.read_bytes()
            self.assertLess(data.index(b"moov"), data.index(b"mdat"))


class LedTests(unittest.TestCase):
    def setUp(self):
        self.pixels = FakePixels()
        self.ir_values = []
        self.led = LedRingController(self.pixels, self.ir_values.append)

    def state(self, **fields):
        return {"id":1, "command":"SET_STATE", "mode":"manual", "rgb":[12,34,56],
                "brightness":0.1, "ir":True, "face":True, "phase":"face", "result":"water", **fields}

    def test_manual_color_survives_auto_events_and_restores_auto(self):
        response = handle_message(json.dumps(self.state()), self.led)
        self.assertTrue(response["ok"])
        self.assertGreater(self.pixels.shown, 0)
        for command in ["LED_PROCESSING", "LED_RESULT_SPARKLING", "LED_FACE_OFF"]:
            self.led.legacy(command)
            self.assertEqual(self.pixels.colors[0], (12,34,56))
        self.led.apply(self.state(mode="auto", phase="result", result="sparkling_water"))
        self.assertEqual(self.pixels.colors[0], (255,0,0))
        self.led.result_until = 0
        self.led.render()
        self.assertEqual(self.pixels.colors[0], (0,255,0))
        self.led.off()
        self.assertEqual(self.pixels.colors[0], (0,0,0))
        self.assertFalse(self.ir_values[-1])

    def test_invalid_state_has_negative_ack_and_does_not_apply(self):
        response = handle_message(json.dumps(self.state(brightness=1)), self.led)
        self.assertFalse(response["ok"])
        self.assertEqual(self.pixels.shown, 0)

    def test_control_roundtrip_and_reconnect(self):
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            port = reserve.getsockname()[1]
        stop, ready, errors = threading.Event(), threading.Event(), []
        server = threading.Thread(target=control_server, args=(self.led, port, stop, ready, errors), daemon=True)
        server.start()
        self.assertTrue(ready.wait(2))
        self.assertFalse(errors)
        client = DeviceControl(port)
        client.set_target("127.0.0.1")
        client.set(mode="manual", rgb=[60,70,80], ir=True)
        client.start()
        try:
            wait_until(lambda: client.snapshot()["status"] == "applied")
            self.assertEqual(self.pixels.colors[0], (60,70,80))
            self.assertTrue(self.ir_values[-1])
            client.sock.shutdown(socket.SHUT_RDWR)
            wait_until(lambda: client.snapshot()["status"] == "disconnected")
            wait_until(lambda: client.snapshot()["status"] == "applied")
            self.assertEqual(self.pixels.colors[0], (60,70,80))
        finally:
            client.close()
            stop.set()
            server.join(3)
        self.assertFalse(self.ir_values[-1])


if __name__ == "__main__":
    unittest.main()
