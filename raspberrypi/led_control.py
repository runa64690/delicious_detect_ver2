"""Hardware-independent LED state machine, also used by fake-device tests."""
import colorsys
import math
import threading
import time


class LedRingController:
    def __init__(self, pixels, set_ir, *, max_brightness=0.15):
        self.pixels, self.set_ir = pixels, set_ir
        self.max_brightness = max_brightness
        self.lock = threading.RLock()
        self.mode, self.phase = "auto", "face"
        self.rgb = (0, 0, 0)
        self.brightness = max_brightness
        self.face = self.ir = False
        self.result = "water"
        self.result_until = 0
        self.offset = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def apply(self, state):
        mode, rgb, brightness = state.get("mode"), state.get("rgb"), state.get("brightness")
        if mode not in {"auto", "manual"}:
            raise ValueError("mode must be auto or manual")
        if not isinstance(rgb, list) or len(rgb) != 3 or any(type(x) is not int or not 0 <= x <= 255 for x in rgb):
            raise ValueError("rgb must contain three integers from 0 to 255")
        if type(brightness) not in (int, float) or not math.isfinite(brightness) or not 0 <= brightness <= self.max_brightness:
            raise ValueError("brightness exceeds configured limit")
        if type(state.get("ir")) is not bool or type(state.get("face")) is not bool:
            raise ValueError("ir and face must be booleans")
        if state.get("phase") not in {"face", "processing", "result"} or state.get("result") not in {"water", "sparkling_water"}:
            raise ValueError("invalid automatic LED state")
        with self.lock:
            self.mode, self.rgb, self.brightness = mode, tuple(rgb), brightness
            self.face, self.phase, self.result = state["face"], state["phase"], state["result"]
            self.result_until = time.monotonic() + 3
            self.ir = state["ir"]
            self.set_ir(self.ir)
            self.render()

    def legacy(self, command):
        with self.lock:
            if command == "TOGGLE_IR":
                self.ir = not self.ir
                self.set_ir(self.ir)
            elif command in {"LED_FACE_ON", "LED_FACE_OFF"}:
                self.face = command == "LED_FACE_ON"
            elif command == "LED_PROCESSING":
                self.phase = "processing"
            elif command in {"LED_RESULT_WATER", "LED_RESULT_SPARKLING"}:
                self.phase = "result"
                self.result = "water" if command == "LED_RESULT_WATER" else "sparkling_water"
                self.result_until = time.monotonic() + 3
            elif command == "LED_IDLE":
                self.phase = "face"
            elif command == "LED_OFF":
                self.mode, self.phase, self.face = "auto", "face", False
            else:
                raise ValueError("Unknown command: " + command)
            self.render()

    def render(self):
        with self.lock:
            if self.phase == "result" and time.monotonic() >= self.result_until:
                self.phase = "face"
            self.pixels.brightness = self.brightness
            if self.mode == "manual":
                self.pixels.fill(self.rgb)
            elif self.phase == "processing":
                for index in range(len(self.pixels)):
                    hue = ((self.offset + index * 256 / len(self.pixels)) % 256) / 256
                    self.pixels[index] = tuple(int(value * 255) for value in colorsys.hsv_to_rgb(hue, 1, 1))
                self.offset = (self.offset + 5) % 256
            else:
                color = ((0, 0, 255) if self.result == "water" else (255, 0, 0)) if self.phase == "result" else ((0, 255, 0) if self.face else (0, 0, 0))
                self.pixels.fill(color)
            self.pixels.show()

    def off(self):
        with self.lock:
            self.mode, self.phase, self.face, self.ir = "auto", "face", False, False
            self.set_ir(False)
            self.pixels.fill((0, 0, 0))
            self.pixels.show()

    def _run(self):
        while not self.stop_event.wait(0.05):
            try:
                self.render()
            except Exception:
                # The next SET_STATE request reports hardware errors to the PC.
                continue

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        self.off()


KNOWN_COMMANDS = {"TOGGLE_IR", "LED_FACE_ON", "LED_FACE_OFF", "LED_PROCESSING",
                  "LED_RESULT_WATER", "LED_RESULT_SPARKLING", "LED_IDLE", "LED_OFF"}


def handle_message(line, controller):
    import json
    if not line.startswith("{"):
        controller.legacy(line)
        return None  # Legacy PC programs never read responses.
    request_id = None
    try:
        message = json.loads(line)
        request_id = message.get("id")
        if message.get("command") != "SET_STATE":
            raise ValueError("Unknown JSON command")
        controller.apply(message)
        return {"id": request_id, "ok": True}
    except Exception as exc:
        return {"id": request_id, "ok": False, "error": str(exc)}
