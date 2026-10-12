#!/usr/bin/env python3
"""Desktop frontend for the shared camera/trial runtime."""
from __future__ import annotations

import sys

import cv2
import numpy as np

# Re-export the original helpers for existing callers and tests.
from scripts.camera_io import (
    LANDMARK_COUNT, DEFAULT_MODEL_DIR, build_capture_row, connect_control,
    create_parser, draw_lines, make_trial_id, receive_frame, send_control_command,
)
from scripts.realtime_engine import RealtimeEngine, StateConflict


def parse_args():
    parser = create_parser()
    parser.add_argument("--record-video", action="store_true", help="判定区間のMP4も保存する（FFmpegが必要）")
    return parser.parse_args()


def main() -> int:
    engine = None
    try:
        engine = RealtimeEngine(parse_args())
        engine.start()
        print("映像接続待機中。ENTER: 判定 / F: FaceMesh / SPACE: IR / Q: 終了")
        while True:
            state = engine.snapshot()
            with engine.lock:
                display = engine.preview_frame.copy() if engine.preview_frame is not None else np.zeros((400, 640, 3), dtype=np.uint8)
            lines = [("ENTER: trial  SPACE: IR  F: mesh  Q: quit", (255, 255, 255)),
                     ("FaceMesh: " + ("OK" if state["face_detected"] else "LOST"), (0, 255, 0))]
            if not state["camera_connected"]:
                lines.append(("Waiting for camera...", (0, 255, 255)))
            elif state["phase"] == "collecting":
                lines.append((f"RECORDING {state['remaining_seconds']:.1f}s", (0, 0, 255)))
                if state["remaining_seconds"] <= 3:
                    lines.append(("Release cup from mouth", (0, 0, 255)))
            elif state["phase"] == "processing":
                lines.append(("PROCESSING...", (0, 255, 255)))
            result = state["result"]
            if result:
                if result["status"] == "success":
                    prediction = result["prediction"]
                    lines.append(("RESULT: " + prediction["predicted_name"].upper(), (0, 255, 0)))
                    for name, score in prediction["probabilities"].items():
                        lines.append((f"{name}: {score:.3f}", (255, 255, 0)))
                else:
                    lines.append(("NO RESULT - see prediction log", (0, 0, 255)))
            draw_lines(display, lines)
            cv2.imshow("Real-time Water / Sparkling Classifier", display)
            key = cv2.waitKey(20) & 0xFF
            try:
                if key == ord("q"):
                    break
                if key in (10, 13):
                    print(engine.start_trial())
                elif key == ord("f"):
                    with engine.lock:
                        engine.show_mesh = not engine.show_mesh
                elif key == ord(" "):
                    engine.control.set(ir=not state["control"]["ir"])
            except StateConflict as exc:
                print("[INFO]", exc)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine:
            engine.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
