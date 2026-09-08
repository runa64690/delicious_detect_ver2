#!/usr/bin/env python3
"""Stream Raspberry Pi camera frames and classify one trial after Enter."""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from scripts.realtime_pipeline import ProcessedTrial, process_trial
from scripts.realtime_predictors import load_predictor


LANDMARK_COUNT = 478
DEFAULT_MODEL_DIR = Path("results/models/random_forest_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi映像を表示し、Enter後の1試行を水・炭酸水分類します。"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument(
        "--raspi-ip",
        default=None,
        help="IR制御先IP。省略時は映像接続元のRaspberry Pi IPを自動使用する",
    )
    parser.add_argument("--control-port", type=int, default=10000)
    parser.add_argument("--disable-control", action="store_true")
    parser.add_argument("--face-model", type=Path, default=Path("face_landmarker.task"))
    parser.add_argument("--predictor", default="handcrafted_rf")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR / "model.joblib")
    parser.add_argument(
        "--metadata", type=Path, default=DEFAULT_MODEL_DIR / "metadata.json"
    )
    parser.add_argument("--model-n-jobs", type=int, default=-1)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--minimum-valid-frames", type=int, default=180)
    parser.add_argument("--minimum-success-rate", type=float, default=0.70)
    parser.add_argument(
        "--log", type=Path, default=Path("results/realtime_predictions.jsonl")
    )
    parser.add_argument(
        "--save-trials-dir", type=Path, default=None,
        help="指定時は収集したraw/normalized CSVも保存する",
    )
    parser.add_argument("--show-facemesh", action="store_true")
    return parser.parse_args()


def make_trial_id() -> str:
    return "realtime_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def build_capture_row(
    landmarks: Any | None,
    *,
    frame_index: int,
    timestamp_ms: int,
    elapsed_sec: float,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    valid = landmarks is not None and len(landmarks) >= LANDMARK_COUNT
    row: dict[str, Any] = {
        "video_name": "realtime",
        "rec_frame": frame_index,
        "global_frame": frame_index,
        "timestamp_ms": timestamp_ms,
        "elapsed_sec": elapsed_sec,
        "facemesh_ok": int(valid),
    }
    for landmark_id in range(LANDMARK_COUNT):
        if valid:
            landmark = landmarks[landmark_id]
            # Match video_record.py and the training data's integer pixel inputs.
            row[f"lm{landmark_id}_x_px"] = int(landmark.x * frame_width)
            row[f"lm{landmark_id}_y_px"] = int(landmark.y * frame_height)
        else:
            row[f"lm{landmark_id}_x_px"] = None
            row[f"lm{landmark_id}_y_px"] = None
    return row


def receive_frame(connection: socket.socket, buffered: bytes) -> tuple[Any, bytes]:
    payload_size = struct.calcsize("Q")
    while len(buffered) < payload_size:
        packet = connection.recv(4096)
        if not packet:
            raise ConnectionError("Raspberry Piとの接続が切断されました")
        buffered += packet
    message_size = struct.unpack("Q", buffered[:payload_size])[0]
    buffered = buffered[payload_size:]
    while len(buffered) < message_size:
        packet = connection.recv(4096)
        if not packet:
            raise ConnectionError("Raspberry Piとの接続が切断されました")
        buffered += packet
    frame_payload = buffered[:message_size]
    buffered = buffered[message_size:]
    encoded = pickle.loads(frame_payload)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("受信したJPEGフレームを復号できません")
    return frame, buffered


def draw_lines(frame: Any, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    for index, (message, color) in enumerate(lines):
        cv2.putText(
            frame,
            message,
            (20, 35 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )


def connect_control(
    args: argparse.Namespace, *, stream_peer_ip: str
) -> socket.socket | None:
    if args.disable_control:
        return None
    target_ip = args.raspi_ip or stream_peer_ip
    if args.raspi_ip is not None and args.raspi_ip != stream_peer_ip:
        print(
            "[WARN] 指定したIR制御先IPが映像接続元IPと異なります: "
            f"control={args.raspi_ip}, stream={stream_peer_ip}"
        )
    deadline = time.monotonic() + 3.0
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        control.settimeout(max(0.2, deadline - time.monotonic()))
        try:
            control.connect((target_ip, args.control_port))
            control.settimeout(None)
            print(f"IR・LED制御接続: {target_ip}:{args.control_port}")
            return control
        except OSError as exc:
            last_error = exc
            control.close()
            time.sleep(0.2)
    print(
        f"[WARN] IR・LED制御へ接続できません ({target_ip}:{args.control_port})。"
        f"映像処理は継続します: {last_error}"
    )
    return None


def send_control_command(
    control: socket.socket | None, command: str
) -> socket.socket | None:
    """Send one newline-delimited command without stopping video on failure."""
    if control is None:
        return None
    try:
        control.sendall((command + "\n").encode("utf-8"))
        return control
    except OSError as exc:
        print(f"[WARN] 制御命令を送信できません ({command}): {exc}")
        control.close()
        return None


def main() -> int:
    args = parse_args()
    if args.duration <= 0:
        print("[FATAL] --durationは0より大きくしてください", file=sys.stderr)
        return 1

    try:
        predictor = load_predictor(
            args.predictor,
            model_path=args.model,
            metadata_path=args.metadata,
            n_jobs=args.model_n_jobs,
        )
    except Exception as exc:
        print(f"[FATAL] モデルを読み込めません: {exc}", file=sys.stderr)
        return 1
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.face_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection: socket.socket | None = None
    control: socket.socket | None = None
    landmarker = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trial-inference")
    processing: Future[ProcessedTrial] | None = None
    try:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"映像接続待機中: {args.host}:{args.port}")
        connection, address = server.accept()
        print(f"映像接続: {address}")
        control = connect_control(args, stream_peer_ip=str(address[0]))
        control = send_control_command(control, "LED_OFF")
        landmarker = vision.FaceLandmarker.create_from_options(face_options)

        buffered = b""
        show_facemesh = args.show_facemesh
        collecting = False
        collection_started = 0.0
        trial_id = ""
        collected_rows: list[dict[str, Any]] = []
        global_frame = 0
        last_timestamp_ms = -1
        last_face_led_state: bool | None = None
        status_message = f"Press ENTER to start a {args.duration:g}s trial"
        status_color = (255, 255, 255)
        result_lines: list[tuple[str, tuple[int, int, int]]] = []

        while True:
            frame, buffered = receive_frame(connection, buffered)
            global_frame += 1
            height, width = frame.shape[:2]
            display = frame.copy()

            timestamp_ms = max(last_timestamp_ms + 1, int(time.perf_counter() * 1000))
            last_timestamp_ms = timestamp_ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = landmarker.detect_for_video(mp_image, timestamp_ms)
            landmarks = detection.face_landmarks[0] if detection.face_landmarks else None
            valid_landmarks = landmarks is not None and len(landmarks) >= LANDMARK_COUNT
            instruction_lines: list[tuple[str, tuple[int, int, int]]] = []

            if valid_landmarks != last_face_led_state:
                face_command = "LED_FACE_ON" if valid_landmarks else "LED_FACE_OFF"
                control = send_control_command(control, face_command)
                last_face_led_state = valid_landmarks

            if show_facemesh and valid_landmarks:
                for landmark in landmarks:
                    cv2.circle(
                        display,
                        (int(landmark.x * width), int(landmark.y * height)),
                        1,
                        (0, 255, 0),
                        -1,
                    )

            if collecting:
                elapsed = time.perf_counter() - collection_started
                collected_rows.append(
                    build_capture_row(
                        landmarks,
                        frame_index=len(collected_rows) + 1,
                        timestamp_ms=timestamp_ms,
                        elapsed_sec=elapsed,
                        frame_width=width,
                        frame_height=height,
                    )
                )
                remaining = max(args.duration - elapsed, 0.0)
                valid_count = sum(row["facemesh_ok"] for row in collected_rows)
                status_message = (
                    f"RECORDING {remaining:.1f}s  FaceMesh "
                    f"{valid_count}/{len(collected_rows)}"
                )
                status_color = (0, 0, 255)
                if remaining <= 3.0:
                    instruction_lines = [
                        ("Release cup from mouth", (0, 0, 255)),
                        ("Cup off", (0, 0, 255)),
                    ]
                if elapsed >= args.duration:
                    collecting = False
                    raw_trial = pd.DataFrame(collected_rows)
                    processing = executor.submit(
                        process_trial,
                        raw_trial,
                        trial_id=trial_id,
                        predictor=predictor,
                        minimum_valid_frames=args.minimum_valid_frames,
                        minimum_success_rate=args.minimum_success_rate,
                        log_path=args.log,
                        save_trials_dir=args.save_trials_dir,
                    )
                    control = send_control_command(control, "LED_PROCESSING")
                    status_message = "PROCESSING..."
                    status_color = (0, 255, 255)

            if processing is not None and processing.done():
                try:
                    outcome = processing.result()
                    water = outcome.prediction.probabilities["water"]
                    sparkling = outcome.prediction.probabilities["sparkling_water"]
                    status_message = "RESULT: " + outcome.prediction.predicted_name.upper()
                    status_color = (0, 255, 0)
                    result_command = (
                        "LED_RESULT_SPARKLING"
                        if outcome.prediction.predicted_label == 1
                        else "LED_RESULT_WATER"
                    )
                    control = send_control_command(control, result_command)
                    result_lines = [
                        (f"Water score: {water:.3f}", (255, 255, 0)),
                        (f"Sparkling score: {sparkling:.3f}", (255, 255, 0)),
                        (
                            "Quality: "
                            f"{outcome.quality.valid_frames}/"
                            f"{outcome.quality.total_frames} "
                            f"({outcome.quality.facemesh_success_rate:.1%})",
                            (255, 255, 0),
                        ),
                    ]
                    print(
                        f"[{outcome.trial_id}] {outcome.prediction.predicted_name} "
                        f"water={water:.3f} sparkling={sparkling:.3f} "
                        f"FaceMesh={outcome.quality.facemesh_success_rate:.1%}"
                    )
                except Exception as exc:
                    control = send_control_command(control, "LED_IDLE")
                    status_message = "NO RESULT - CHECK CONSOLE"
                    status_color = (0, 0, 255)
                    result_lines = [(str(exc)[:80], (0, 0, 255))]
                    print(f"[TRIAL ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
                processing = None

            face_status = "FaceMesh: OK" if valid_landmarks else "FaceMesh: LOST"
            face_color = (0, 255, 0) if valid_landmarks else (0, 0, 255)
            lines = [
                (status_message, status_color),
                (face_status, face_color),
                *instruction_lines,
                *result_lines,
                ("ENTER: start  SPACE: IR  F: mesh  Q: quit", (255, 255, 255)),
            ]
            draw_lines(display, lines)
            cv2.imshow("Real-time Water / Sparkling Classifier", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (10, 13):
                if collecting:
                    print("[INFO] すでに収集中です")
                elif processing is not None:
                    print("[INFO] 前の試行を処理中です")
                else:
                    collecting = True
                    collection_started = time.perf_counter()
                    trial_id = make_trial_id()
                    collected_rows = []
                    result_lines = []
                    print(f"[{trial_id}] {args.duration:.1f}秒の収集を開始")
            elif key == ord("f"):
                show_facemesh = not show_facemesh
            elif key == ord(" "):
                control = send_control_command(control, "TOGGLE_IR")
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        if landmarker is not None:
            landmarker.close()
        if connection is not None:
            connection.close()
        if control is not None:
            send_control_command(control, "LED_OFF")
            control.close()
        server.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
