"""Validate official annotations and build one row per unique system radar track."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from download_data import DEFAULT_RAW, REVISION, git_blob_sha, safe_path

BASE = Path(__file__).resolve().parent
MEASUREMENTS = {"azimuth": "deg", "elevation": "deg", "range": "m",
                "posX": "m", "posY": "m", "posZ": "m",
                "veloX": "m/s", "veloY": "m/s", "veloZ": "m/s", "rcs": "dBsm"}
CLASSES = {"UAV", "BIRD", "HUMAN", "VEHICLE"}


def normalize_label(value):
    if pd.isna(value) or not str(value).strip():
        return "UNKNOWN"
    value = str(value).strip().upper()
    aliases = {"BIRD": "BIRD", "BURD": "BIRD", "UAV": "UAV", "PEOPLE": "HUMAN",
               "HUMAN": "HUMAN", "HUMAN/DOG": "HUMAN", "CAR": "VEHICLE",
               "VEHICLE": "VEHICLE", "TRUCK": "VEHICLE", "SMALL 4X4": "VEHICLE",
               "UNKNOWN": "UNKNOWN", "MIXED": "MIXED"}
    if value.startswith("MIX OF BIRDS"):
        return "MIXED"
    if value not in aliases:
        raise ValueError(f"Undocumented ground-truth label: {value}")
    return aliases[value]


def track_id(value):
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or not np.isfinite(number) or number < 0 or number != int(number):
        return None
    return str(int(number))


def ambiguous_label(value):
    return normalize_label(value) not in CLASSES or str(value).strip().upper() == "HUMAN/DOG"


def curated_truth(frame, expected, filename):
    # Metadata system_track_id is actually the CSV's sensor-level id.
    if set(frame.id.map(track_id).dropna()) != set(expected):
        raise ValueError(f"CSV sensor IDs disagree with metadata: {filename}")
    match = re.search(r"_(UAV|TRUCK)_(\d+)\.csv$", filename)
    if not match:
        match = re.search(r"(DRONE)_Id_(\d+)\.csv$", filename)
    if not match:
        raise ValueError(f"Unknown curated filename: {filename}")
    label = "UAV" if match[1] == "DRONE" else normalize_label(match[1])
    tid = track_id(match[2])
    if set(expected.values()) != {label}:
        raise ValueError(f"Filename and metadata labels disagree: {filename}")
    if set(frame.SystemTrackID.map(track_id).dropna()) != {tid}:
        raise ValueError(f"Curated file contains unrelated system tracks: {filename}")
    return {tid: label}


def session_identity(filename):
    match = re.match(r"tracks_(\d{4}-\d{2}-\d{2})T(\d{6})-(R[12])", filename)
    if match:
        return match[1].replace("-", "") + "T" + match[2], match[3]
    match = re.match(r"Echodyne_Log_(\d{8})_(\d{6})", filename)
    if match:
        return match[1] + "T" + match[2], "R1"
    if re.fullmatch(r"DRONE_Id_\d+\.csv", filename):
        # No parent start time is published. Group the entire known trial date.
        return "20201109_session_unknown", "R1"
    raise ValueError(f"Unknown recording identity: {filename}")


def verify_labels(annotation, metadata):
    normalized = {k: normalize_label(v) for k, v in annotation.items()}
    if normalized != metadata:
        raise ValueError("Annotation/metadata label or track-ID disagreement")
    return normalized


def summarize(frame):
    result, invalid = {}, {}
    for name in MEASUREMENTS:
        original = frame[name] if name in frame else pd.Series(np.nan, index=frame.index)
        values = pd.to_numeric(original, errors="coerce").astype(float)
        bad = original.notna() & (values.isna() | ~np.isfinite(values))
        if name == "range":
            bad |= values < 0
        if name == "elevation":
            bad |= values.abs() > 90
        if name == "azimuth":
            bad |= values.abs() > 360
        invalid[name] = int(bad.sum())
        values = values.mask(bad)
        result.update({f"{name}_mean": values.mean(), f"{name}_std": values.std(ddof=0),
                       f"{name}_min": values.min(), f"{name}_max": values.max(),
                       f"{name}_range": values.max() - values.min(),
                       f"{name}_missing_rate": values.isna().mean(),
                       f"{name}_n_valid": int(values.count())})
    return result, invalid


def deduplicate_tracks(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["track_uid"]].append(row)
    kept, audit = [], []
    for uid, candidates in sorted(groups.items()):
        candidates.sort(key=lambda r: (r["source_format"] != "xlsx", r["source_file"]))
        conflict = len({r["label"] for r in candidates}) > 1
        if not conflict:
            kept.append(candidates[0])
        for row in candidates if conflict else candidates[1:]:
            audit.append({"track_uid": uid, "source_file": row["source_file"],
                          "reason": "conflicting_labels" if conflict else "duplicate_track",
                          "retained_source": "" if conflict else candidates[0]["source_file"]})
    return kept, audit


def read_csv(path, **kwargs):
    import gzip
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        header = handle.readline()
    return pd.read_csv(path, sep=";" if header.count(";") > header.count(",") else ",",
                       low_memory=False, **kwargs)


def label_map(frame, id_column, label_column):
    result = {}
    for tid, label in zip(frame[id_column], frame[label_column]):
        key = track_id(tid)
        if key is None:
            raise ValueError("Invalid annotated track ID")
        value = normalize_label(label)
        if key in result and result[key] != value:
            raise ValueError(f"Conflicting annotation for ID {key}")
        result[key] = value
    return result


def annotation_map(frame):
    # Official example uses two positional columns; several label headers are blank.
    if frame.shape[1] != 2 or frame.columns[0] != "SystemTrackID":
        raise ValueError("Unexpected annotation sheet layout")
    return label_map(frame, frame.columns[0], frame.columns[1])


def feature_dictionary(columns):
    identity = {"track_uid": "Globally unique NRC recording/radar/system-track key; never a feature",
                "recording_session": "Recording start, shared across radars; split group, never a feature",
                "recording_day": "Trial day; stronger split group, never a feature",
                "radar_id": "Radar unit within recording; provenance, never a feature",
                "system_track_id": "SystemTrackID (not sensor id); provenance, never a feature",
                "label": "Verified expert object class; supervised target, never an input",
                "source_file": "Preferred source path; provenance, never a feature",
                "source_format": "Preferred source format; provenance, never a feature",
                "n_observations": "Retained updates; candidate feature subject to acquisition-duration bias"}
    stats = {"mean": "Arithmetic mean of valid observations", "std": "Population standard deviation (ddof=0)",
             "min": "Minimum valid measurement", "max": "Maximum valid measurement",
             "range": "Maximum minus minimum valid measurement", "missing_rate": "Missing or invalid / retained observations",
             "n_valid": "Number of valid measurements"}
    rows = []
    for column in columns:
        if column in identity:
            rows.append(dict(column=column, unit="count" if column == "n_observations" else "n/a", description=identity[column]))
        else:
            name, stat = next((m, column[len(m)+1:]) for m in MEASUREMENTS if column.startswith(m + "_"))
            unit = "fraction" if stat == "missing_rate" else "count" if stat == "n_valid" else MEASUREMENTS[name]
            rows.append(dict(column=column, unit=unit, description=f"{name}: {stats[stat]}"))
    return pd.DataFrame(rows)


def preprocess(raw_dir=DEFAULT_RAW, processed_dir=BASE / "data/processed", reports_dir=BASE / "reports"):
    raw_dir, processed_dir, reports_dir = map(Path, (raw_dir, processed_dir, reports_dir))
    manifest = json.loads((raw_dir / "download_manifest.json").read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("Unexpected source revision; review schema before upgrading")
    for entry in manifest["files"]:
        if git_blob_sha(safe_path(raw_dir, entry["path"]).read_bytes()) != entry["sha"]:
            raise ValueError(f"Corrupt or modified source: {entry['path']}")
    labels = pd.read_csv(raw_dir / "metadata/track_labels.csv")
    inventory = pd.read_csv(raw_dir / "metadata/file_manifest.csv")
    candidates, exclusions, input_audit, raw_audit = [], [], [], []
    for item in inventory.sort_values(["folder", "file"]).itertuples(index=False):
        relative = f"data/{item.folder}/{item.file}"
        path = raw_dir / relative
        session, radar = session_identity(item.file)
        indexed = labels[(labels.folder == item.folder) & (labels.file == item.file)]
        expected = label_map(indexed, "system_track_id", "label")
        if path.suffix == ".gz":
            # Deliberately do not use radar labels or propagate curated labels to full logs.
            frame = read_csv(path, usecols=["id", "SystemTrackID"])
            for tid, count in frame.groupby("id", dropna=False).size().items():
                raw_audit.append(dict(recording_session=session, source_file=relative, sensor_track_id=tid,
                                      n_observations=int(count), reason="full_log_not_used_for_supervision"))
            if len(frame) != item.rows:
                raise ValueError(f"Manifest row count mismatch: {relative}")
            input_audit.append(dict(source_file=relative, rows=len(frame), duplicate_rows=None,
                                    invalid_id_rows=int(frame.id.isna().sum()), role="unlabeled_full_log"))
            continue
        if path.suffix == ".xlsx":
            with pd.ExcelFile(path) as workbook:
                frame = pd.read_excel(workbook, sheet_name=0)
                annotation = pd.read_excel(workbook, sheet_name=1)
            truth = verify_labels(annotation_map(annotation), expected)
            ambiguous_ids = {track_id(row.system_track_id) for row in indexed.itertuples()
                             if ambiguous_label(row.label_raw) and normalize_label(row.label_raw) in CLASSES}
        else:
            frame = read_csv(path)
            frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
            truth = curated_truth(frame, expected, item.file)
            ambiguous_ids = set()
        missing_columns = set(MEASUREMENTS) - set(frame.columns)
        if missing_columns:
            raise ValueError(f"Missing measurements {missing_columns}: {relative}")
        if len(frame) != item.rows or len(indexed) != item.n_labeled_tracks:
            raise ValueError(f"Manifest counts disagree: {relative}")
        frame["_tid"] = frame.SystemTrackID.map(track_id)
        invalid_ids = int(frame._tid.isna().sum())
        # Excel has no time axis: identical values may be legitimate stationary updates.
        # Only remove exact raw CSV records, which include timestamps.
        duplicates = int(frame.duplicated().sum())
        if path.suffix == ".csv":
            frame = frame.drop_duplicates()
        input_audit.append(dict(source_file=relative, rows=item.rows, duplicate_rows=duplicates,
                                invalid_id_rows=invalid_ids, duplicate_label_rows=len(indexed) - len(expected), role="annotated"))
        if invalid_ids:
            exclusions.append(dict(source_file=relative, track_uid="", reason="invalid_track_id_rows", n_observations=invalid_ids))
        observed = set(frame._tid.dropna())
        for missing in sorted(set(truth) - observed):
            exclusions.append(dict(source_file=relative, track_uid=f"NRC:{session}:{radar}:{missing}", reason="label_without_observations", n_observations=0))
        for tid, group in frame.groupby("_tid"):
            uid = f"NRC:{session}:{radar}:{tid}"
            label = truth.get(tid, "UNLABELED")
            if tid in ambiguous_ids:
                exclusions.append(dict(source_file=relative, track_uid=uid, reason="ambiguous_raw_label", n_observations=len(group)))
                continue
            if label not in CLASSES:
                exclusions.append(dict(source_file=relative, track_uid=uid, reason=label.lower(), n_observations=len(group)))
                continue
            features, invalid = summarize(group)
            if not any(features[f"{m}_n_valid"] for m in MEASUREMENTS):
                exclusions.append(dict(source_file=relative, track_uid=uid, reason="no_valid_measurements", n_observations=len(group)))
                continue
            candidates.append(dict(track_uid=uid, recording_session=session, recording_day=session[:8], radar_id=radar,
                                   system_track_id=tid, label=label, source_file=relative, source_format=path.suffix[1:],
                                   n_observations=len(group), **features))
            for name, count in invalid.items():
                if count:
                    exclusions.append(dict(source_file=relative, track_uid=uid, reason=f"invalid_{name}_values", n_observations=count))
        print(f"Validated {relative}", flush=True)
    kept, duplicates = deduplicate_tracks(candidates)
    if not kept:
        raise ValueError("No usable tracks; inspect input annotations")
    tracks = pd.DataFrame(kept).sort_values("track_uid").reset_index(drop=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(processed_dir / "tracks.csv", index=False, float_format="%.12g")
    feature_dictionary(tracks.columns).to_csv(reports_dir / "feature_dictionary.csv", index=False)
    pd.DataFrame(exclusions, columns=["source_file", "track_uid", "reason", "n_observations"]).to_csv(reports_dir / "exclusions.csv", index=False)
    pd.DataFrame(duplicates, columns=["track_uid", "source_file", "reason", "retained_source"]).to_csv(reports_dir / "duplicate_tracks.csv", index=False)
    pd.DataFrame(input_audit).to_csv(reports_dir / "input_audit.csv", index=False)
    pd.DataFrame(raw_audit).to_csv(reports_dir / "full_log_track_audit.csv", index=False)
    summary = dict(revision=REVISION, source_files=len(inventory), metadata_label_rows=len(labels),
                   usable_tracks=len(tracks), usable_observations=int(tracks.n_observations.sum()),
                   recording_sessions=int(tracks.recording_session.nunique()),
                   classes=tracks.label.value_counts().sort_index().to_dict(),
                   duplicate_track_representations=len(duplicates),
                   exclusion_reasons=pd.Series([r["reason"] for r in exclusions]).value_counts().to_dict())
    (reports_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return tracks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--processed-dir", type=Path, default=BASE / "data/processed")
    parser.add_argument("--reports-dir", type=Path, default=BASE / "reports")
    args = parser.parse_args()
    preprocess(args.raw_dir, args.processed_dir, args.reports_dir)
