#!/usr/bin/env python3
"""Restricted-label permutation tests of fixed-model participant LOSO scores.

Each draw permutes trial labels within exchangeability blocks BEFORE all folds
are refitted. The statistic is the unweighted mean of participant fold scores.
See docs/permutation_test_loso.md for assumptions and publication guidance.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import get_scorer

try:
    from . import train_loso_classifier as loso
except ImportError:
    import train_loso_classifier as loso


MODEL_LABELS = {"dummy": "Dummy", "extra_trees": "Extra Trees", "random_forest": "Random Forest"}
METRIC_LABELS = {"balanced_accuracy": "Balanced accuracy", "roc_auc": "ROC AUC",
                 "macro_f1": "Macro F1", "f1": "F1", "accuracy": "Accuracy",
                 "precision": "Precision", "recall": "Recall"}


def exchangeability_blocks(df, columns, *, target_column):
    """Return row positions, including constant-label blocks unchanged by a draw."""
    if target_column in columns:
        raise ValueError("The target column cannot define permutation blocks.")
    for column in columns:
        if column not in df or df[column].isna().any():
            raise ValueError(f"Missing permutation block values/column: {column}")
    blocks = [np.asarray(indices, dtype=int)
              for indices in df.groupby(columns, sort=True, dropna=False).indices.values()]
    if not any(df.iloc[indices][target_column].nunique() > 1 for indices in blocks):
        raise ValueError("No block has both labels; no informative permutation is possible.")
    return blocks


def permute_labels(y, blocks, *, seed, permutation_id):
    # Independent deterministic streams allow exact restart and any worker count.
    rng = np.random.default_rng(np.random.SeedSequence([seed, permutation_id]))
    permuted = y.copy()
    for indices in blocks:
        permuted[indices] = rng.permutation(y[indices])
    return permuted


def score_loso(x, y, folds, estimator, metric):
    scorer = get_scorer("f1_macro" if metric == "macro_f1" else metric)
    scores = []
    for fold in folds:
        fitted = clone(estimator).fit(x[fold.train_indices], y[fold.train_indices])
        scores.append(scorer(fitted, x[fold.test_indices], y[fold.test_indices]))
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite fold score; inspect labels and metric.")
    return float(np.mean(scores))


def permutation_row(permutation_id, *, x, y, blocks, seed, folds, models, metric):
    shuffled = permute_labels(y, blocks, seed=seed, permutation_id=permutation_id)
    return {"permutation_id": permutation_id,
            **{name: score_loso(x, shuffled, folds, model, metric)
               for name, model in models.items()}}


def monte_carlo_pvalue(observed, null_scores):
    null_scores = np.asarray(null_scores, dtype=float)
    if not np.isfinite(observed) or not len(null_scores) or not np.isfinite(null_scores).all():
        raise ValueError("Observed and null scores must be finite; null scores cannot be empty.")
    # Treat machine-precision ties as ties, including at observed == 0.
    tolerance = 100 * np.finfo(float).eps * max(1.0, abs(observed))
    exceedances = int(np.count_nonzero(null_scores >= observed - tolerance))
    return (exceedances + 1) / (len(null_scores) + 1), exceedances


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return adjusted


def summarize(observed, null_scores):
    rows = []
    for name, score in observed.items():
        values = null_scores[name].to_numpy(float)
        pvalue, exceedances = monte_carlo_pvalue(score, values)
        rows.append({"model": name, "observed_score": score,
                     "null_mean": float(values.mean()),
                     "null_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                     "n_permutations": len(values), "exceedances": exceedances,
                     "p_raw": pvalue, "minimum_p": 1 / (len(values) + 1)})
    summary = pd.DataFrame(rows)
    summary["p_holm"] = holm_adjust(summary["p_raw"])
    return summary


def save_figure(summary, null_scores, *, metric, output_dir, width_mm=180, dpi=600):
    """Vector output, embedded PDF fonts, fixed score axes, accessible colors."""
    count = len(summary)
    columns = min(count, 3) if width_mm >= 150 else 1
    rows = (count + columns - 1) // columns
    style = {"font.family": "DejaVu Sans", "font.size": 8,
             "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 6.5,
             "xtick.labelsize": 7, "ytick.labelsize": 7,
             "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(rows, columns, squeeze=False,
                                 figsize=(width_mm / 25.4, 2.8 * rows), layout="constrained")
        draws = len(null_scores)
        heading = f"Restricted-label permutation test (B = {draws:,})"
        if draws < 999:
            heading = "SMOKE TEST ONLY | " + heading
        fig.suptitle(heading, fontsize=9)
        for index, (_, result) in enumerate(summary.iterrows()):
            ax = axes.flat[index]
            values = null_scores[result["model"]].to_numpy(float)
            ax.hist(values, bins=np.linspace(0, 1, 26), weights=np.ones(len(values)) / len(values),
                    color="#0072B2", edgecolor="white", linewidth=0.4, label="Permuted labels")
            ax.axvline(result["observed_score"], color="#D55E00", linewidth=1.6,
                       linestyle="--", label="Observed score")
            ax.set(xlim=(-0.015, 1.015), xlabel=f"Mean LOSO {METRIC_LABELS[metric].lower()}",
                   ylabel="Proportion of permutations")
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ymax = ax.get_ylim()[1]
            ax.set_ylim(0, min(1.08, ymax * 1.4))
            digits = max(3, int(np.ceil(np.log10(len(values) + 1))))
            annotation = (f"Observed = {result['observed_score']:.3f}\n"
                          f"$p$ = {result['p_raw']:.{digits}f}")
            if count > 1:
                annotation += f"; Holm $p$ = {result['p_holm']:.{digits}f}"
            ax.set_title(f"({chr(97 + index)}) {MODEL_LABELS[result['model']]}\n{annotation}",
                         loc="left", fontsize=8, linespacing=1.4)
            ax.tick_params(direction="out", width=0.6)
        for ax in list(axes.flat)[count:]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
        for extension in ("pdf", "svg", "png"):
            fig.savefig(output_dir / f"permutation_test.{extension}", dpi=dpi, facecolor="white")
        plt.close(fig)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_checkpoint(path, names):
    if not path.exists():
        return pd.DataFrame(columns=["permutation_id", *names])
    table = pd.read_csv(path, float_precision="round_trip")
    if list(table.columns) != ["permutation_id", *names]:
        raise ValueError("Checkpoint model columns do not match.")
    if not np.array_equal(table["permutation_id"], np.arange(1, len(table) + 1)):
        raise ValueError("Checkpoint IDs must be consecutive from 1; checkpoint may be damaged.")
    values = table[names].to_numpy(float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Checkpoint contains invalid/incomplete scores.")
    return table


def render_outputs(output_dir, manifest, *, width_mm, dpi):
    null_scores = read_checkpoint(output_dir / "permutation_scores.csv", manifest["model_names"])
    expected = manifest["analysis"]["n_permutations"]
    if len(null_scores) != expected:
        raise ValueError(f"Incomplete run: {len(null_scores)}/{expected}; use --resume.")
    summary = summarize(manifest["observed_scores"], null_scores)
    summary.to_csv(output_dir / "permutation_summary.csv", index=False, encoding="utf-8-sig")
    metric = manifest["analysis"]["metric"]
    save_figure(summary, null_scores, metric=metric, output_dir=output_dir, width_mm=width_mm, dpi=dpi)
    blocks = ", ".join(manifest["analysis"]["block_columns"])
    caption = (
        f"Figure. Restricted-label permutation tests of participant LOSO classification. "
        f"The statistic was the unweighted mean {METRIC_LABELS[metric]} across "
        f"{manifest['participants']} held-out participants ({manifest['trials']} trials). "
        f"Blue histograms show {expected:,} Monte Carlo permutations sampled with replacement "
        f"within blocks defined by {blocks}, preserving block-specific class counts. "
        "Dashed orange lines indicate observed scores. Each permutation used the same fold "
        "assignments and refitted every model on permuted training labels, evaluating against "
        "permuted test labels. One-sided p-values were computed as (C + 1)/(B + 1), where C "
        "counts permuted scores at least as large as observed and B is the number of draws. "
        f"Holm correction covers the {len(summary)} models tested here for this single metric. "
        "Histograms describe the null distribution, not confidence intervals for performance.\n\n"
        "Assumptions: labels are exchangeable within the specified blocks under the null. "
        "Feature definitions, exclusions, models and metric must be fixed independently of "
        "these evaluation results. This analysis does not repeat a data-driven selection "
        "procedure or correct for unreported models/metrics. If selection was performed, "
        "repeat it within training folds and within every permutation.\n"
    )
    if expected < 999:
        caption = "SMOKE TEST ONLY: fewer than 999 permutations; not a publication run.\n\n" + caption
    (output_dir / "figure_caption.txt").write_text(caption, encoding="utf-8")
    return summary


def run(args):
    output_dir = loso.project_path(args.output)
    manifest_path = output_dir / "permutation_manifest.json"
    if args.plot_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return render_outputs(output_dir, manifest, width_mm=args.width_mm, dpi=args.dpi)
    config_path = loso.project_path(args.config)
    config = loso.load_config(config_path)
    names = args.models or list(config["models"])
    if len(set(names)) != len(names) or any(name not in config["models"] for name in names):
        raise ValueError("--models must be distinct names present in the config.")
    metric = args.metric or config["primary_metric"]
    seed = config["random_seed"] if args.seed is None else args.seed
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Random seed must be a nonnegative integer.")
    group, target = config["group_column"], config["target_column"]
    block_columns = list(dict.fromkeys([group, *(args.block_columns or [])]))
    features_path = loso.project_path(config["input_path"])
    exclusions_path = loso.project_path(config["excluded_trials_path"])
    # Validate before the legacy loader casts labels to integers.
    raw_labels = pd.read_csv(features_path)[target]
    if not pd.to_numeric(raw_labels, errors="coerce").isin([0, 1]).all():
        raise ValueError("Labels must be exactly 0 or 1, without missing/fractional values.")
    data = loso.load_and_validate_features(
        features_path, group_column=group, target_column=target, positive_label=1,
        active_exclusions=loso.load_active_exclusions(exclusions_path))
    folds = loso.build_loso_folds(data, group_column=group, target_column=target)
    blocks = exchangeability_blocks(data, block_columns, target_column=target)
    models = {name: model for name, model in loso.build_models(config).items() if name in names}
    models = {name: models[name] for name in names}
    for model in models.values():
        if model.get_params().get("random_state") is None:
            raise ValueError("Set an explicit model random_state in the config for reproducibility.")
        if "n_jobs" in model.get_params():
            model.set_params(n_jobs=1)  # Parallelize draws, avoiding nested oversubscription.
    analysis = {"config": config, "metric": metric, "seed": seed,
                "n_permutations": args.n_permutations, "block_columns": block_columns,
                "model_names": names, "model_parameters": {n: m.get_params() for n, m in models.items()},
                "features_sha256": loso.sha256_file(features_path),
                "exclusions_sha256": loso.sha256_file(exclusions_path),
                "source_sha256": {p.name: loso.sha256_file(p) for p in
                                  [Path(__file__), Path(loso.__file__), Path(loso.__file__).with_name("feature_schema.py")]},
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__,
                             "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}}
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["analysis"] != analysis:
            raise ValueError("Resume settings, data, code or versions differ from the saved run.")
    else:
        loso.prepare_output_dir(output_dir, overwrite=False)
        manifest = {"analysis": analysis, "model_names": names, "status": "observing",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "git_commit": loso.git_commit(), "git_worktree_dirty": loso.git_worktree_dirty(),
                    "trials": len(data), "participants": len(folds), "observed_scores": {}}
        write_json(manifest_path, manifest)
    loso.save_dataset_audit(data, folds, group_column=group, target_column=target, output_dir=output_dir)
    data[["trial_id", *block_columns]].to_csv(output_dir / "permutation_blocks.csv", index=False)
    for name, model in models.items():
        if name in manifest["observed_scores"]:
            continue
        print(f"Observed LOSO: {name}", flush=True)
        result = loso.run_model_loso(data, folds, model, model_name=name, group_column=group,
                                     target_column=target, positive_label=1)
        result.predictions.to_csv(output_dir / f"observed_predictions_{name}.csv", index=False)
        result.fold_metrics.to_csv(output_dir / f"observed_fold_metrics_{name}.csv", index=False)
        manifest["observed_scores"][name] = float(result.fold_metrics[metric].mean())
        write_json(manifest_path, manifest)
    checkpoint_path = output_dir / "permutation_scores.csv"
    checkpoint = read_checkpoint(checkpoint_path, names)
    if len(checkpoint) > args.n_permutations:
        raise ValueError("Checkpoint contains too many permutations.")
    manifest["status"] = "permuting"
    write_json(manifest_path, manifest)
    x = data[loso.FACIAL_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=int)
    print(f"Permutations: {len(checkpoint)}/{args.n_permutations}; workers={args.jobs}", flush=True)
    if len(checkpoint) < args.n_permutations:
        with checkpoint_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["permutation_id", *names])
            if handle.tell() == 0:
                writer.writeheader()
                handle.flush()
            with Parallel(n_jobs=args.jobs, return_as="generator", batch_size=1) as parallel:
                results = parallel(delayed(permutation_row)(
                    index, x=x, y=y, blocks=blocks, seed=seed, folds=folds, models=models, metric=metric)
                    for index in range(len(checkpoint) + 1, args.n_permutations + 1))
                for row in results:
                    writer.writerow(row)
                    handle.flush()
                    index = row["permutation_id"]
                    if index == 1 or index % 10 == 0 or index == args.n_permutations:
                        print(f"Permutations: {index}/{args.n_permutations}", flush=True)
    summary = render_outputs(output_dir, manifest, width_mm=args.width_mm, dpi=args.dpi)
    manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(manifest_path, manifest)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=loso.DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=Path("results/ml/permutation_loso_v1"))
    parser.add_argument("--models", nargs="+", choices=list(MODEL_LABELS))
    parser.add_argument("--metric", choices=loso.METRIC_COLUMNS)
    parser.add_argument("--n-permutations", type=int, default=999)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel draws; -1 uses all CPUs.")
    parser.add_argument("--block-columns", nargs="+", help="Additional within-participant exchangeability columns.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from a completed run.")
    parser.add_argument("--width-mm", type=float, default=180)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    if args.n_permutations < 1 or args.jobs == 0 or args.jobs < -1:
        parser.error("Use n-permutations >= 1 and jobs >= 1 (or -1).")
    if not np.isfinite(args.width_mm) or args.width_mm < 85 or args.dpi < 72:
        parser.error("Use width-mm >= 85 and dpi >= 72.")
    if args.resume and args.plot_only:
        parser.error("Choose --resume or --plot-only.")
    return args


def main():
    args = parse_args()
    summary = run(args)
    print(summary.to_string(index=False))
    print(f"Saved to: {loso.project_path(args.output)}")


if __name__ == "__main__":
    main()
