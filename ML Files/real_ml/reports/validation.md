# Phase 1 validation record

Validated on 2026-11-21 using Python 3.12 on Windows and `requirements-lock.txt`.

- Download completed: 87 hash-verified files, including 81 measurement files and six documentation/metadata files. Cached download verified successfully without network access.
- Preprocessing and EDA completed successfully against all files in the pinned official manifest.
- `python -m unittest discover -s tests -v`: **17 tests passed**, including four checks against the real source/output artifacts; no skips in the complete run.
- Compilation and `git diff --check` passed.
- A second complete cached pipeline run reproduced `tracks.csv` and all 23 generated report artifacts **byte-for-byte** (24 SHA-256 comparisons, zero differences). This validation note is separate from those generated reports.
- `tracks.csv` SHA-256: `4d76f82554c2d394683b0a9f4114093ed08f74ee4af8a0bbafa7173b3793e520`.
- Result: 713 unique tracks, 367,112 observations; 492 BIRD, 146 UAV, 63 HUMAN, 12 VEHICLE. 51 recording groups across nine days.
- Inspected class-distribution, class-comparison, correlation and feature-distribution PNGs for readable labels/layout.
- Independent code review verified the sensor/system-ID relationship and duplicate handling. Its ambiguous Human/dog finding was fixed and re-reviewed with no remaining findings.
- Raw downloads, processed data and the virtual environment are ignored by Git. Existing simulation, CAD and coverage components are unchanged.
- No models were trained and no intent-recognition claims were made.

The first exploratory runs exposed blank workbook label headers and a duplicate annotation; these were investigated against source files and incorporated into regression coverage. The data dictionary remains authoritative for measurement units. Source version, license, attribution and transformations are documented in the module README.
