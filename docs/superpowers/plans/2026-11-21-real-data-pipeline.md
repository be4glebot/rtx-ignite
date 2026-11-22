# Real data pipeline implementation plan

**Goal:** Validate NRC real radar data independently of the archived simulation pipeline; no model training or intent inference.
**Architecture:** Pin upstream revision, verify downloaded Git blob hashes, verify annotations against metadata, canonicalize sessions and deduplicate curated extracts, aggregate a strict measurement allowlist, generate audit tables and EDA.
**Stack:** Python, pandas, NumPy, openpyxl, Matplotlib, Seaborn; standard-library unittest.
**Spec:** User's RTX Ignite Phase 1 requirements in this chat.

The user explicitly requests implementation and testing without stopping at a plan. Proceed inline in the fresh clone on feature/real-data-pipeline. Existing simulation files are preserved. No AGENTS.md or pre-existing test suite was found. Existing commits use descriptive imperative messages.

## Tasks
- [x] Download: test integrity validation; implement download_data.py with pinned source manifest, atomic downloads, caching and attribution.
- [x] Preprocess: test normalization, session identities, invalid values, metadata conflicts and duplicate handling; implement preprocess.py producing tracks.csv, exclusions, input audit and feature dictionary.
- [x] Explore: implement explore.py with distributions, correlations, class comparisons, missingness and data quality report. Run on actual downloads.
- [x] Verify: run automated tests and complete pipeline, inspect plots and reproducibility; document exact counts, limitations and future session-grouped model comparison; review and commit.

## Review focus
CSV sensor IDs may differ from SystemTrackID; filename labels must never label unrelated tracks. Workbook sheets may disagree with metadata. Curated extracts repeat sessions and tracks. Missing numerical values must not become zero. Two radars and multiple tracks from the same recording must not cross evaluation partitions.

## Decisions
Use only the ten documented common measurements, excluding radar classification/probability/confidence fields. Prefer annotated workbook measurements over repeated curated CSV tracks; do not concatenate two representations of the same track. Keep full raw logs for an unlabeled-data audit, not as a source of additional ground truth. Preserve track length as a candidate feature with an explicit acquisition-bias warning. Use population standard deviation (ddof=0), including zero for one valid observation.

## Validation ledger
Official CSV metadata uses sensor IDs despite its column name. Verified CSV sensor-ID sets, filename class and system ID before canonicalization. Some workbook annotation headers are blank; accept the official two-column layout and verify its content against metadata. One duplicate HUMAN annotation collapses. Independent review identified Human/dog ambiguity; preserve official normalization for validation but exclude this raw annotation from supervision. Reviewer verified the fix and reported no remaining findings.

Complete pipeline: 713 tracks, 367,112 observations, 51 recording groups, 17 passing tests. A second cached download/preprocess/explore run reproduced all 24 generated data/report files byte-for-byte. Raw/processed/venv paths verified Git-ignored. GitHub push dry run returned 403 for configured account junehwang0930-lang; no remote branch or PR was created.
