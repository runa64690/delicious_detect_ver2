"""One reconnecting control connection, with correlated Pi acknowledgements."""
from __future__ import annotations

import json
import socket
import threading
import time


class DeviceControl:
    def __init__(self, port=10000, disabled=False):
        self.port, self.disabled = port, disabled
        self.lock = threading.RLock()
        self.target = None
        self.desired = {"mode": "auto", "rgb": [0, 0, 0], "brightness": 0.15,
                        "ir": False, "face": False, "phase": "face", "result": "water"}
        self.result_until = 0.0
        self.version = 0
        self.applied_version = -1
        self.status = "disabled" if disabled else "disconnected"
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="pi-control", daemon=True)
        self.sock = None

    def start(self):
        if not self.disabled:
            self.thread.start()

    def set_target(self, target):
        with self.lock:
            self.target = target

    def set(self, **values):
        with self.lock:
            if any(self.desired.get(key) != value for key, value in values.items()):
                self.desired.update(values)
                self.version += 1
                if self.status == "applied":
                    self.status = "sending"

    def result(self, name):
        with self.lock:
            self.result_until = time.monotonic() + 3
            self.set(phase="result", result=name)

    def snapshot(self):
        with self.lock:
            return {**self.desired, "status": self.status, "error": self.error}

    def _run(self):
        sequence = 0
        while not self.stop_event.is_set():
            with self.lock:
                target = self.target
            if not target:
                self.stop_event.wait(0.2)
                continue
            try:
                with socket.create_connection((target, self.port), timeout=2) as sock:
                    self.sock = sock
                    sock.settimeout(2)
                    buffered = b""
                    last_sent = 0.0
                    self.applied_version = -1
                    while not self.stop_event.is_set():
                        with self.lock:
                            if target != self.target:
                                break
                            if self.desired["phase"] == "result" and time.monotonic() >= self.result_until:
                                self.set(phase="face")
                            version, desired = self.version, dict(self.desired)
                        if version != self.applied_version or time.monotonic() - last_sent >= 1:
                            sequence += 1
                            with self.lock:
                                self.status = "sending" if version != self.applied_version else "applied"
                            message = {"id": sequence, "command": "SET_STATE", **desired}
                            sock.sendall((json.dumps(message) + "\n").encode())
                            while b"\n" not in buffered:
                                packet = sock.recv(4096)
                                if not packet:
                                    raise ConnectionError("LED制御接続が切れました")
                                buffered += packet
                                if len(buffered) > 16384:
                                    raise ValueError("不正な制御応答です")
                            line, buffered = buffered.split(b"\n", 1)
                            response = json.loads(line)
                            if response.get("id") != sequence or not response.get("ok"):
                                raise ValueError(response.get("error", "LED制御応答が一致しません"))
                            with self.lock:
                                self.applied_version = version
                                self.status = "applied" if self.version == version else "sending"
                                self.error = None
                            last_sent = time.monotonic()
                        self.stop_event.wait(0.05)
            except (OSError, ValueError) as exc:
                with self.lock:
                    self.status, self.error = "disconnected", str(exc)
                self.stop_event.wait(1)
            finally:
                self.sock = None

    def close(self):
        self.stop_event.set()
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.thread.is_alive():
            self.thread.join(timeout=4)
