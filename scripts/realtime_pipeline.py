"""Shared processing for one real-time FaceMesh trial."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .normalize_facemesh import normalize_dataframe, validate_normalized_output
    from .realtime_predictors import TrialPrediction, TrialPredictor
except ImportError:
    from normalize_facemesh import normalize_dataframe, validate_normalized_output
    from realtime_predictors import TrialPrediction, TrialPredictor


class TrialQualityError(ValueError):
    """A trial that should return no classification result."""


@dataclass(frozen=True)
class TrialQuality:
    total_frames: int
    valid_frames: int
    facemesh_success_rate: float
    elapsed_seconds: float
    effective_fps: float


@dataclass(frozen=True)
class ProcessedTrial:
    trial_id: str
    prediction: TrialPrediction
    quality: TrialQuality
    normalized_frames: int
    processing_seconds: float


def prepare_normalized_trial(
    raw_trial: pd.DataFrame,
    *,
    trial_id: str,
    minimum_valid_frames: int,
    minimum_success_rate: float,
) -> tuple[pd.DataFrame, TrialQuality]:
    if raw_trial.empty:
        raise TrialQualityError("収集フレームがありません")
    if not 0 <= minimum_success_rate <= 1:
        raise ValueError("minimum_success_rateは0～1で指定してください")
    if minimum_valid_frames < 2:
        raise ValueError("minimum_valid_framesは2以上にしてください")
    if "facemesh_ok" not in raw_trial:
        raise TrialQualityError("収集データにfacemesh_okがありません")

    detection = pd.to_numeric(raw_trial["facemesh_ok"], errors="coerce").eq(1)
    total_frames = len(raw_trial)
    valid_frames = int(detection.sum())
    success_rate = valid_frames / total_frames
    if "elapsed_sec" in raw_trial:
        elapsed = pd.to_numeric(raw_trial["elapsed_sec"], errors="coerce")
    else:
        elapsed = pd.Series(dtype=float)
    if elapsed.notna().any():
        elapsed_seconds = float(elapsed.max())
        elapsed_span = float(elapsed.max() - elapsed.min())
    else:
        elapsed_seconds = 0.0
        elapsed_span = 0.0
    effective_fps = (
        (total_frames - 1) / elapsed_span
        if total_frames >= 2 and elapsed_span > 0
        else 0.0
    )

    quality = TrialQuality(
        total_frames=total_frames,
        valid_frames=valid_frames,
        facemesh_success_rate=success_rate,
        elapsed_seconds=elapsed_seconds,
        effective_fps=effective_fps,
    )
    if valid_frames < minimum_valid_frames:
        raise TrialQualityError(
            f"FaceMesh有効フレーム不足: {valid_frames} < {minimum_valid_frames}"
        )
    if success_rate < minimum_success_rate:
        raise TrialQualityError(
            "FaceMesh成功率不足: "
            f"{success_rate:.1%} < {minimum_success_rate:.1%}"
        )

    normalized = normalize_dataframe(
        raw_trial,
        trial_id=trial_id,
        source_file="realtime_memory",
        frames=None,
    )
    validate_normalized_output(normalized)
    return normalized, quality


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def process_trial(
    raw_trial: pd.DataFrame,
    *,
    trial_id: str,
    predictor: TrialPredictor,
    minimum_valid_frames: int,
    minimum_success_rate: float,
    log_path: Path,
    save_trials_dir: Path | None = None,
) -> ProcessedTrial:
    started = time.perf_counter()
    if save_trials_dir is not None:
        raw_dir = save_trials_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_trial.to_csv(raw_dir / f"{trial_id}.csv", index=False, encoding="utf-8-sig")

    try:
        normalized, quality = prepare_normalized_trial(
            raw_trial,
            trial_id=trial_id,
            minimum_valid_frames=minimum_valid_frames,
            minimum_success_rate=minimum_success_rate,
        )
        prediction = predictor.predict_normalized(normalized)
        processing_seconds = time.perf_counter() - started
        processed = ProcessedTrial(
            trial_id=trial_id,
            prediction=prediction,
            quality=quality,
            normalized_frames=len(normalized),
            processing_seconds=processing_seconds,
        )

        if save_trials_dir is not None:
            normalized_dir = save_trials_dir / "normalized"
            normalized_dir.mkdir(parents=True, exist_ok=True)
            normalized.to_csv(
                normalized_dir / f"{trial_id}.csv",
                index=False,
                encoding="utf-8-sig",
            )

        append_jsonl(
            log_path,
            {
                "recorded_at": datetime.now().astimezone().isoformat(),
                "status": "success",
                "trial_id": trial_id,
                "prediction": asdict(prediction),
                "quality": asdict(quality),
                "normalized_frames": len(normalized),
                "processing_seconds": processing_seconds,
            },
        )
        return processed
    except Exception as exc:
        detection = pd.to_numeric(
            raw_trial.get("facemesh_ok", pd.Series(dtype=float)), errors="coerce"
        ).eq(1)
        append_jsonl(
            log_path,
            {
                "recorded_at": datetime.now().astimezone().isoformat(),
                "status": "rejected",
                "trial_id": trial_id,
                "model_id": predictor.model_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "total_frames": len(raw_trial),
                "valid_frames": int(detection.sum()),
                "minimum_valid_frames": minimum_valid_frames,
                "minimum_success_rate": minimum_success_rate,
                "processing_seconds": time.perf_counter() - started,
            },
        )
        raise
