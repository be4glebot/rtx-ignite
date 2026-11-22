# NRC radar data quality report

Source revision: `df3cebae895f4f12d23cd50480dd4c47b09d422e`.

- Imported source files: 81; source observations including repeated extracts: 2,042,407.
- Official metadata entries: 1117 (not unique system tracks).
- Usable unique labeled tracks: **713**; retained observations: **367,112**.
- Recording groups: 51; recording days: 9.
- Classes: BIRD=492, HUMAN=63, UAV=146, VEHICLE=12.
- Largest/smallest class ratio: 41.00:1. Accuracy alone would obscure minority-class errors.
- Removed duplicate track representations: 15.
- Repeated identical metadata annotations collapsed: 1.
- Exact repeated rows detected in annotated inputs: 72375; CSV repeats removed, Excel repeats retained because timestamps are absent.
- Invalid/unassociated ID rows in source files: 95857.
- Full-log sensor groups excluded from direct supervision: 8,134, covering 1,271,758 rows. This includes tracks also present in curated extracts; do not add these counts to unique labeled tracks.

## Exclusion audit

Counts below are source-level audit entries, not necessarily unique physical objects:

- ambiguous_raw_label: 1 entries, 167 affected observations/values.
- invalid_track_id_rows: 40 entries, 95,857 affected observations/values.
- label_without_observations: 4 entries, 0 affected observations/values.
- mixed: 376 entries, 22,275 affected observations/values.
- unknown: 3 entries, 236 affected observations/values.
- unlabeled: 1284 entries, 192,604 affected observations/values.

## Measurement quality

Missing, nonnumeric, infinite, negative range, out-of-bounds elevation (±90°) and azimuth (±360°) values become NaN. Signed positions, velocities and negative dBsm remain valid. No arbitrary outlier clipping or global imputation is used. Statistical summaries skip NaNs; missing rates retain them in the denominator. Angular statistics are linear and may mislead near wraparound; inspect before training.

CSV metadata IDs are sensor `id` values despite the column name `system_track_id`. Each is checked against its extract; the extract's SystemTrackID must agree with its filename. Multiple sensor segments collapse into one system track. Workbook annotations are independently checked against normalized metadata. Some annotation headers are blank. Full raw logs provide no new supervised labels.

## Leakage and sampling limitations

Use recording_session for grouped splitting, keeping R1/R2 together. Prefer recording_day or leave-one-trial-out evaluation for stronger separation of shared weather, scene and possibly repeated flights. Demo-7 is a curated collection, not a distinct recording session. Demo-1 has no published parent start time and is conservatively grouped by day. System-track uniqueness does not establish unique physical-object identity across recordings.

Do not use IDs, paths, session, day, radar, source format, label or built-in radar classifications/probabilities as features. Absolute positions, range, RCS availability, observation counts and duration can reveal acquisition conditions rather than object physics. Compare feature ablations and inspect classes_by_session.csv before committing to a split. Entire-track summaries are offline features, not a validated early-warning system.

The upstream README headline class counts differ from the pinned metadata (e.g. BIRD); this report derives counts from verified files, never from README totals. Human/dog validates as HUMAN under the official taxonomy but is excluded here because the raw annotation is ambiguous. MIXED and UNKNOWN are also excluded. Curated UAV selection creates selection bias. Quebec 2019–2021 field trials with one radar family do not establish Kyle Field performance or represent all aircraft, weather or crowds.

## Next phase (no model trained)

Freeze a grouped holdout before tuning. Compare Logistic Regression (train-fold median imputation and scaling), a shallow Decision Tree, and a constrained Random Forest using the same grouped folds. Check class presence in every fold; rare classes may prevent meaningful estimates. Fit imputation, scaling, selection and weighting on training folds only. Report macro-F1, balanced accuracy, per-class precision/recall, confusion matrices and group-level uncertainty. Explain coefficients/tree rules and evaluate permutation importance on held-out groups. Consider a documented UAV-versus-bird task if four-class evaluation lacks independent support.

These labels describe observed object type. They do not establish authorization, hostile behavior or malicious intent. No intent-recognition claim is supported.
