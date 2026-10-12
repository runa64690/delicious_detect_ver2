"""LAN-only mobile controller. Run: python web_app.py"""
from __future__ import annotations

import asyncio
import csv
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from scripts.camera_io import create_parser
from scripts.realtime_engine import RealtimeEngine, StateConflict

ROOT = Path(__file__).resolve().parent


class LedSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["auto", "manual"]
    rgb: tuple[Annotated[int, Field(strict=True, ge=0, le=255)],
               Annotated[int, Field(strict=True, ge=0, le=255)],
               Annotated[int, Field(strict=True, ge=0, le=255)]] = (0, 0, 0)
    brightness: float = Field(default=0.15, ge=0, le=0.15, allow_inf_nan=False)


class Switch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool


def parse_args(argv=None):
    parser = create_parser()
    parser.description = "Androidブラウザ用の映像・判定・録画・LED操作サーバー"
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8000)
    parser.add_argument("--video-dir", type=Path, default=Path("video_data/realtime"))
    parser.add_argument("--journal", type=Path, default=Path("results/realtime_journal.jsonl"))
    parser.add_argument("--ffmpeg", default=None, help="FFmpeg実行ファイル。省略時はPATHまたは同梱版")
    parser.set_defaults(record_video=True)
    return parser.parse_args(argv)


def create_app(args=None, *, engine=None, manage_runtime=True):
    @asynccontextmanager
    async def lifespan(app):
        runtime = engine if engine is not None else RealtimeEngine(args or parse_args([]))
        app.state.engine = runtime
        try:
            if manage_runtime:
                runtime.start()
            yield
        finally:
            if manage_runtime:
                runtime.close()

    app = FastAPI(title="Delicious Detector", lifespan=lifespan)

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.url.netloc:
                return JSONResponse({"detail": "別のサイトからの操作は許可されていません"}, status_code=403)
            if not request.headers.get("content-type", "").startswith("application/json"):
                return JSONResponse({"detail": "JSONで送信してください"}, status_code=415)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(StateConflict)
    async def conflict(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(OSError)
    async def storage_failure(request, exc):
        request.app.state.engine.storage_error = str(exc)
        return JSONResponse({"detail": "保存処理に失敗しました。PCの保存先を確認してください"}, status_code=503)

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web/index.html", media_type="text/html")

    # Windows can register .js as text/plain. Never infer executable asset MIME
    # types from host associations: nosniff correctly blocks that response.
    @app.get("/static/app.js", include_in_schema=False)
    def javascript():
        return FileResponse(ROOT / "web/app.js", media_type="text/javascript")

    @app.get("/static/style.css", include_in_schema=False)
    def stylesheet():
        return FileResponse(ROOT / "web/style.css", media_type="text/css")

    @app.get("/api/state")
    def state(request: Request):
        return request.app.state.engine.snapshot()

    @app.post("/api/trials", status_code=202)
    def start_trial(request: Request):
        return request.app.state.engine.start_trial()

    @app.post("/api/recordings", status_code=202)
    def start_recording(request: Request):
        return request.app.state.engine.start_recording()

    @app.post("/api/recordings/stop", status_code=202)
    def stop_recording(request: Request):
        return request.app.state.engine.stop_recording()

    @app.put("/api/led", status_code=202)
    def led(settings: LedSettings, request: Request):
        control = request.app.state.engine.control
        control.set(mode=settings.mode, rgb=list(settings.rgb), brightness=settings.brightness)
        return control.snapshot()

    @app.put("/api/ir", status_code=202)
    def ir(settings: Switch, request: Request):
        control = request.app.state.engine.control
        control.set(ir=settings.enabled)
        return control.snapshot()

    @app.put("/api/view")
    def view(settings: Switch, request: Request):
        runtime = request.app.state.engine
        with runtime.lock:
            runtime.show_mesh = settings.enabled
        return {"show_mesh": settings.enabled}

    @app.get("/api/history")
    def history(request: Request, offset: int = 0, limit: int = 50):
        if offset < 0 or not 1 <= limit <= 200:
            raise HTTPException(422, "履歴の取得範囲が不正です")
        items = request.app.state.engine.history.items()
        return {"items": items[offset:offset + limit], "total": len(items)}

    @app.get("/api/videos/{video_id}")
    def video(video_id: str, request: Request, download: bool = False):
        path = request.app.state.engine.history.video_path(video_id)
        if path is None:
            raise HTTPException(404, "動画が未完成、保存失敗、または存在しません")
        return FileResponse(path, media_type="video/mp4", filename=path.name,
                            content_disposition_type="attachment" if download else "inline")

    @app.get("/api/logs/{format}")
    def logs(format: str, request: Request):
        items = request.app.state.engine.history.items()
        if format == "jsonl":
            body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items)
            media = "application/x-ndjson"
        elif format == "csv":
            output = io.StringIO(newline="")
            columns = ["recorded_at", "id", "kind", "status", "predicted_name", "water_score",
                       "sparkling_water_score", "total_frames", "valid_frames", "facemesh_success_rate",
                       "effective_fps", "processing_seconds", "model_id", "video_id", "video_status",
                       "duration_seconds", "error_message", "video_error"]
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for item in items:
                prediction = item.get("prediction", {})
                row = {**item, **item.get("quality", {}), "predicted_name": prediction.get("predicted_name"),
                       "model_id": prediction.get("model_id", item.get("model_id")),
                       "water_score": prediction.get("probabilities", {}).get("water"),
                       "sparkling_water_score": prediction.get("probabilities", {}).get("sparkling_water")}
                values = {key: row.get(key) for key in columns}
                for key, value in values.items():
                    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                        values[key] = "'" + value
                writer.writerow(values)
            body = "\ufeff" + output.getvalue()
            media = "text/csv"
        else:
            raise HTTPException(404, "csvまたはjsonlを指定してください")
        return Response(body, media_type=media,
                        headers={"Content-Disposition": f'attachment; filename="detector-history.{format}"'})

    @app.get("/stream.mjpg")
    async def stream(request: Request):
        runtime = request.app.state.engine
        async def frames():
            sequence = -1
            while not runtime.stop_event.is_set() and not await request.is_disconnected():
                with runtime.lock:
                    current, jpeg = runtime.preview_sequence, runtime.preview_jpeg
                if jpeg is not None and current != sequence:
                    sequence = current
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(0.1)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


if __name__ == "__main__":
    arguments = parse_args()
    print(f"Androidから http://<PCのWi-Fi IPv4アドレス>:{arguments.web_port} を開いてください")
    # MJPEG responses stay open indefinitely; bound their drain time before lifespan cleanup.
    uvicorn.run(create_app(arguments), host=arguments.web_host, port=arguments.web_port,
                workers=1, timeout_graceful_shutdown=3)
