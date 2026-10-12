#!/usr/bin/env python3
"""Plot participant accuracy from saved LOSO fold metrics without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_LABELS = {
    "random_forest": "ランダムフォレスト",
    "extra_trees": "Extra Trees",
    "dummy": "ダミー分類器",
}


def project_path(path):
    return path if path.is_absolute() else PROJECT_ROOT / path


def japanese_font():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Meiryo", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "Yu Gothic", "MS Gothic"):
        if name in available:
            return name
    raise RuntimeError("日本語フォントがありません。Noto Sans JPなどをインストールしてください。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/ml/initial_loso_v1"))
    parser.add_argument("--model", choices=list(MODEL_LABELS), default="random_forest")
    parser.add_argument("--output-dir", type=Path, help="Default: <run-dir>/figures")
    args = parser.parse_args()
    run_dir = project_path(args.run_dir)
    source = run_dir / f"fold_metrics_{args.model}.csv"
    data = pd.read_csv(source).sort_values("test_participant").reset_index(drop=True)
    required = {"test_participant", "test_trials", "accuracy", "tn", "fp", "fn", "tp"}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError("Input must contain nonempty LOSO fold metrics with confusion counts.")
    if data.test_participant.isna().any() or data.test_participant.duplicated().any():
        raise ValueError("Expected one fold per participant.")
    counts = data[["tn", "fp", "fn", "tp"]].to_numpy(float)
    trials = data.test_trials.to_numpy(float)
    if (not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any()
            or not np.isfinite(trials).all() or (trials <= 0).any()
            or not np.array_equal(counts.sum(axis=1), trials)):
        raise ValueError("Invalid trial/confusion counts.")
    accuracy = (data.tn + data.tp).to_numpy(float) / trials
    if not np.allclose(accuracy, data.accuracy.to_numpy(float), rtol=0, atol=1e-12):
        raise ValueError("Saved accuracy does not match correct predictions / test trials.")

    output_dir = project_path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"participant_accuracy_{args.model}_ja"
    table = pd.DataFrame({
        "participant": data.test_participant,
        "correct": (data.tn + data.tp).astype(int),
        "trials": trials.astype(int),
        "accuracy_percent": accuracy * 100,
    })
    table.to_csv(output_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    with plt.rc_context({
        "font.family": japanese_font(), "font.size": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }):
        fig, ax = plt.subplots(figsize=(160 / 25.4, 95 / 25.4), layout="constrained")
        positions = np.arange(len(data))
        bars = ax.bar(positions, accuracy * 100, width=0.62, color="#777777",
                      edgecolor="#333333", linewidth=0.6, zorder=3)
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in accuracy * 100], padding=4)
        ax.set_xticks(positions, [f"{participant}\n（{int(n)}試行）"
                                 for participant, n in zip(data.test_participant, trials)])
        ax.set(ylim=(0, 100), ylabel="正解率（%）", xlabel="被験者")
        ax.set_yticks(np.arange(0, 101, 20))
        ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.5, zorder=0)
        ax.tick_params(axis="x", length=0)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output_dir / f"{stem}.{extension}", dpi=600, facecolor="white")
        plt.close(fig)
    caption = (
        f"図X　{MODEL_LABELS[args.model]}による被験者別の正解率\n"
        "被験者を1名ずつ評価対象とする交差検証（LOSO）を行い、"
        "各被験者の試行を、当該被験者を除いたデータで学習したモデルで分類した。"
        "正解率は評価試行数に対する正解試行数の割合である。"
        "棒上の数値は正解率、被験者ID下の括弧内は評価試行数を示す。\n\n"
        "作図情報（論文のキャプションには含めない）：\n"
        "各棒は1回のLOSO評価の結果を表し、信頼区間や統計的有意差を示すものではない。\n"
        "図幅160 mm、高さ95 mm、白黒印刷対応。PDFはフォント埋め込み、"
        "SVGは文字をパス化、PNGは600 dpi。\n"
        f"データ出典：{source}\n"
    )
    (output_dir / f"{stem}_caption.txt").write_text(caption, encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Saved: {output_dir / stem}")


if __name__ == "__main__":
    main()
