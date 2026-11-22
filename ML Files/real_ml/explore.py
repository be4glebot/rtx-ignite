"""Generate track-level exploratory plots and a reproducible quality report."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rtx-ignite-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preprocess import BASE, MEASUREMENTS, CLASSES


def validate_tracks(tracks):
    if tracks.empty or tracks.track_uid.duplicated().any():
        raise ValueError("Expected nonempty, unique tracks")
    if not set(tracks.label).issubset(CLASSES):
        raise ValueError("Unexpected supervised class")


def explore(processed_dir=BASE / "data/processed", reports_dir=BASE / "reports"):
    reports_dir = Path(reports_dir)
    tracks = pd.read_csv(Path(processed_dir) / "tracks.csv")
    validate_tracks(tracks)
    summary = json.loads((reports_dir / "preprocessing_summary.json").read_text())
    inputs = pd.read_csv(reports_dir / "input_audit.csv")
    exclusions = pd.read_csv(reports_dir / "exclusions.csv")
    raw = pd.read_csv(reports_dir / "full_log_track_audit.csv")
    sns.set_theme(style="whitegrid", context="notebook")

    def save(name):
        plt.tight_layout()
        plt.savefig(reports_dir / name, dpi=160, bbox_inches="tight")
        plt.close()

    counts = tracks.label.value_counts().sort_index()
    counts.rename_axis("class").to_csv(reports_dir / "class_counts.csv", header=["tracks"])
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax, color="#267a9e", rot=0)
    ax.bar_label(ax.containers[0])
    ax.set(title="Unique labeled radar tracks", xlabel="Observed object class", ylabel="Tracks")
    save("class_distribution.png")

    missing = pd.DataFrame({"measurement": list(MEASUREMENTS),
                            "observation_missing_rate": [1 - tracks[f"{m}_n_valid"].sum() / tracks.n_observations.sum() for m in MEASUREMENTS],
                            "tracks_without_measurement": [int(tracks[f"{m}_mean"].isna().sum()) for m in MEASUREMENTS]})
    missing.to_csv(reports_dir / "missing_values.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=missing, x="measurement", y="observation_missing_rate", ax=ax, color="#d89538")
    ax.set(title="Missing or invalid measurements in retained observations", ylabel="Fraction missing", xlabel="Measurement")
    ax.tick_params(axis="x", rotation=35)
    save("missing_values.png")

    means = [f"{m}_mean" for m in MEASUREMENTS]
    for suffix in ["mean", "std", "range", "missing_rate"]:
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        for ax, (m, unit) in zip(axes.flat, MEASUREMENTS.items()):
            sns.histplot(data=tracks, x=f"{m}_{suffix}", bins=25, ax=ax, color="#267a9e")
            ax.set_xlabel(f"{m} {suffix} [{unit if suffix != 'missing_rate' else 'fraction'}]")
        fig.suptitle(f"Per-track {suffix.replace('_', ' ')} distributions", y=1.02)
        save(f"feature_{suffix}_distributions.png")
    features = tracks.select_dtypes(include=np.number).drop(columns=["system_track_id", "recording_day"], errors="ignore")
    features.describe().T.to_csv(reports_dir / "feature_statistics.csv")
    features.corr().to_csv(reports_dir / "feature_correlations.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(tracks[means].corr(), vmin=-1, vmax=1, cmap="vlag", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Pearson correlations of track means (descriptive)")
    save("feature_correlations.png")

    fig, axes = plt.subplots(2, 5, figsize=(19, 8))
    for ax, (m, unit) in zip(axes.flat, MEASUREMENTS.items()):
        sns.boxplot(data=tracks, x="label", y=f"{m}_mean", order=sorted(CLASSES), ax=ax, color="#90bbc9", fliersize=2)
        ax.set(xlabel="", ylabel=f"{m} mean [{unit}]")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Observed class differences; not evidence of intent", y=1.02)
    save("class_measurements.png")
    tracks.groupby("label")[means].agg(["median", "mean", "std"]).to_csv(reports_dir / "class_feature_statistics.csv")
    cross = pd.crosstab(tracks.recording_day, tracks.label)
    cross.to_csv(reports_dir / "classes_by_day.csv")
    pd.crosstab(tracks.recording_session, tracks.label).to_csv(reports_dir / "classes_by_session.csv")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(cross, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set(title="Class coverage by recording day", xlabel="Class", ylabel="Recording day")
    save("class_session_coverage.png")
    ratio = counts.max() / counts.min()
    lines = ["# NRC radar data quality report", "", f"Source revision: `{summary['revision']}`.", "",
             f"- Imported source files: {summary['source_files']}; source observations including repeated extracts: {int(inputs.rows.sum()):,}.",
             f"- Official metadata entries: {summary['metadata_label_rows']} (not unique system tracks).",
             f"- Usable unique labeled tracks: **{len(tracks)}**; retained observations: **{int(tracks.n_observations.sum()):,}**.",
             f"- Recording groups: {tracks.recording_session.nunique()}; recording days: {tracks.recording_day.nunique()}.",
             f"- Classes: {', '.join(f'{c}={n}' for c, n in counts.items())}.",
             f"- Largest/smallest class ratio: {ratio:.2f}:1. Accuracy alone would obscure minority-class errors.",
             f"- Removed duplicate track representations: {summary['duplicate_track_representations']}.",
             f"- Repeated identical metadata annotations collapsed: {int(inputs.duplicate_label_rows.sum())}.",
             f"- Exact repeated rows detected in annotated inputs: {int(inputs.duplicate_rows.sum())}; CSV repeats removed, Excel repeats retained because timestamps are absent.",
             f"- Invalid/unassociated ID rows in source files: {int(inputs.invalid_id_rows.sum())}.",
             f"- Full-log sensor groups excluded from direct supervision: {len(raw):,}, covering {int(raw.n_observations.sum()):,} rows. This includes tracks also present in curated extracts; do not add these counts to unique labeled tracks.", "",
             "## Exclusion audit", "", "Counts below are source-level audit entries, not necessarily unique physical objects:", ""]
    for reason, group in exclusions.groupby("reason"):
        lines.append(f"- {reason}: {len(group)} entries, {int(group.n_observations.sum()):,} affected observations/values.")
    lines += ["", "## Measurement quality", "",
              "Missing, nonnumeric, infinite, negative range, out-of-bounds elevation (±90°) and azimuth (±360°) values become NaN. Signed positions, velocities and negative dBsm remain valid. No arbitrary outlier clipping or global imputation is used. Statistical summaries skip NaNs; missing rates retain them in the denominator. Angular statistics are linear and may mislead near wraparound; inspect before training.", "",
              "CSV metadata IDs are sensor `id` values despite the column name `system_track_id`. Each is checked against its extract; the extract's SystemTrackID must agree with its filename. Multiple sensor segments collapse into one system track. Workbook annotations are independently checked against normalized metadata. Some annotation headers are blank. Full raw logs provide no new supervised labels.", "",
              "## Leakage and sampling limitations", "",
              "Use recording_session for grouped splitting, keeping R1/R2 together. Prefer recording_day or leave-one-trial-out evaluation for stronger separation of shared weather, scene and possibly repeated flights. Demo-7 is a curated collection, not a distinct recording session. Demo-1 has no published parent start time and is conservatively grouped by day. System-track uniqueness does not establish unique physical-object identity across recordings.", "",
              "Do not use IDs, paths, session, day, radar, source format, label or built-in radar classifications/probabilities as features. Absolute positions, range, RCS availability, observation counts and duration can reveal acquisition conditions rather than object physics. Compare feature ablations and inspect classes_by_session.csv before committing to a split. Entire-track summaries are offline features, not a validated early-warning system.", "",
              "The upstream README headline class counts differ from the pinned metadata (e.g. BIRD); this report derives counts from verified files, never from README totals. Human/dog validates as HUMAN under the official taxonomy but is excluded here because the raw annotation is ambiguous. MIXED and UNKNOWN are also excluded. Curated UAV selection creates selection bias. Quebec 2019–2021 field trials with one radar family do not establish Kyle Field performance or represent all aircraft, weather or crowds.", "",
              "## Next phase (no model trained)", "",
              "Freeze a grouped holdout before tuning. Compare Logistic Regression (train-fold median imputation and scaling), a shallow Decision Tree, and a constrained Random Forest using the same grouped folds. Check class presence in every fold; rare classes may prevent meaningful estimates. Fit imputation, scaling, selection and weighting on training folds only. Report macro-F1, balanced accuracy, per-class precision/recall, confusion matrices and group-level uncertainty. Explain coefficients/tree rules and evaluate permutation importance on held-out groups. Consider a documented UAV-versus-bird task if four-class evaluation lacks independent support.", "",
              "These labels describe observed object type. They do not establish authorization, hostile behavior or malicious intent. No intent-recognition claim is supported.", ""]
    (reports_dir / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated reports for {len(tracks)} unique labeled tracks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=BASE / "data/processed")
    parser.add_argument("--reports-dir", type=Path, default=BASE / "reports")
    args = parser.parse_args()
    explore(args.processed_dir, args.reports_dir)
