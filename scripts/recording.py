"""Bounded, asynchronous H.264 recording; timestamps never affect inference."""
from __future__ import annotations

import math
import queue
import shutil
import subprocess
import threading
from pathlib import Path


def find_ffmpeg(executable: str | None = None) -> str:
    if not executable:
        executable = shutil.which("ffmpeg")
    if not executable:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    # Check the actual encoder, not merely the presence of an executable.
    result = subprocess.run(
        [executable, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=size=16x16:rate=1", "-frames:v", "1", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-f", "null", "-"],
        capture_output=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError("FFmpeg H.264エンコーダーを利用できません: " + result.stderr.decode(errors="replace"))
    return executable


class Recording:
    FPS = 30

    def __init__(self, video_id, path: Path, frame, started: float, ffmpeg: str, completed):
        self.video_id, self.path, self.started = video_id, path, started
        self.ffmpeg, self.completed = ffmpeg, completed
        self.height, self.width = frame.shape[:2]
        self.frames = queue.Queue(maxsize=90)
        self.frames.put((started, frame))
        self.ended = threading.Event()
        self.stopped_at = started
        self.reason = None
        self.error = None
        self.process = None
        self.thread = threading.Thread(target=self._run, name=f"record-{video_id}", daemon=True)
        self.thread.start()

    def push(self, frame, timestamp):
        if self.ended.is_set():
            return
        if frame.shape[:2] != (self.height, self.width):
            self.error = "録画中に解像度が変更されました"
            self.finish(timestamp)
            return
        try:
            self.frames.put_nowait((timestamp, frame))
        except queue.Full:
            self.error = "録画処理が追いつかず、バッファが満杯になりました"
            self.finish(timestamp)

    def finish(self, timestamp, reason=None):
        if not self.ended.is_set():
            self.stopped_at, self.reason = timestamp, reason
            self.ended.set()

    def _run(self):
        temporary = self.path.with_suffix(".partial.mp4")
        status = "failed"
        emitted = 0
        try:
            # A file, rather than PIPE, prevents stderr from blocking the encoder.
            with self.path.with_suffix(".ffmpeg.log").open("wb") as errors:
                self.process = subprocess.Popen(
                    [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                     "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}",
                     "-r", str(self.FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
                     "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
                     "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(temporary)],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                previous = None
                while not self.ended.is_set() or not self.frames.empty():
                    if self.error:
                        raise RuntimeError(self.error)
                    try:
                        stamp, frame = self.frames.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    target = max(0, int((stamp - self.started) * self.FPS))
                    if previous is not None:
                        payload = previous.tobytes()
                        while emitted < target:
                            self.process.stdin.write(payload)
                            emitted += 1
                    previous = frame
                if self.error:
                    raise RuntimeError(self.error)
                target = max(1, math.ceil((self.stopped_at - self.started) * self.FPS))
                if previous is not None:
                    payload = previous.tobytes()
                    while emitted < target:
                        self.process.stdin.write(payload)
                        emitted += 1
                self.process.stdin.close()
                if self.process.wait(timeout=20) != 0:
                    raise RuntimeError("動画のエンコードに失敗しました。FFmpegログを確認してください")
            temporary.replace(self.path)
            status = "interrupted" if self.reason else "completed"
        except Exception as exc:
            self.error = str(exc)
        finally:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait(timeout=5)
                if self.process.stdin and not self.process.stdin.closed:
                    try:
                        self.process.stdin.close()
                    except OSError:
                        pass
            while not self.frames.empty():
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    break
            self.completed(self.video_id, status, self.error or self.reason, emitted / self.FPS)

    def join(self):
        self.thread.join(timeout=25)
        if self.thread.is_alive() and self.process is not None:
            self.error = "録画の終了がタイムアウトしました"
            self.process.kill()
            self.thread.join(timeout=5)
