#!/usr/bin/env python3
"""Shared CLI options, camera transport, capture rows and desktop drawing."""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2



LANDMARK_COUNT = 478
DEFAULT_MODEL_DIR = Path("results/models/random_forest_v1")


def create_parser() -> argparse.ArgumentParser:
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
    return parser


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
    if not 0 < message_size <= 16 * 1024 * 1024:
        raise ValueError("不正な映像フレーム長です")
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
