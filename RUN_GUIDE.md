# DR-Train study pipeline: run guide

Version 1.2.0. Everything is in one file, `railtrack_study_pipeline.py`. It starts from the three verified foundation files and ends with the single scoring of the frozen 2016 holdout. It does not predict maintenance. It forecasts and monitors a measured track-response indicator, and it reports honestly how well that works.

## The six commands

| Step | Command | Reads | Main output | Time on one core |
|---|---|---|---|---|
| 0 (optional) | `self-test` | Foundation files only | `STUDY_v1/self_test/` and SELF-TEST PASS in the log | 30 to 45 minutes |
| 1 | `develop` | Foundation files only | `STUDY_v1/develop/DEVELOPMENT_REPORT.html` | 60 to 90 minutes |
| 2 | `freeze` | Foundation files and step 1 results | `STUDY_v1/FROZEN_PROTOCOL.json` | About 10 minutes |
| 3 | `verify-extraction` | Raw data for four development passages | `STUDY_v1/verification/extraction_check.json` (must say PASS) | A few minutes |
| 4 | `extract-holdout --confirm-single-use` | Raw data for the 89 holdout passages | `STUDY_v1/holdout/holdout_vibration_features.csv.gz` | Under an hour |
| 5 | `score-holdout` | Frozen protocol, extraction record and holdout features | `STUDY_v1/holdout_results/HOLDOUT_REPORT.html` | A few minutes |

Steps 0 to 3 never open holdout data. Step 4 is the first and only time the holdout recordings are read. Every step from `develop` onwards adds a line to `STUDY_v1/PROTOCOL_LEDGER.jsonl`, an append-only record with times, file hashes and any override used.

## Before you start

1. Copy `railtrack_study_pipeline.py` into `/data2/nr1/V1_RailTrack/`.
2. Leave `build_predictive_maintenance_foundation.py` unchanged in the same folder. Steps 3 and 4 import it so that holdout features are computed by exactly the same code as the development features. The program checks its SHA-256 (`e8c5b403...`).
3. The foundation files are expected in `/data2/nr1/V1_RailTrack/FOUNDATION_for_new_chat/`. Results go to `/data2/nr1/V1_RailTrack/STUDY_v1/`. Other locations can be given with `--foundation-dir`, `--output-dir`, `--data-root` and `--builder`.

Requirements: Python 3.9 or later with NumPy and SciPy. Matplotlib (figures) and scikit-learn (random forest) are used when installed. Steps 3 and 4 also need h5py, as the builder does. The program was tested with Python 3.13 (NumPy 2.5, SciPy 1.18) and Python 3.11 (NumPy 1.26, SciPy 1.13, which matches the cluster's Conda environment).

## Commands for the cluster

Submit each step through SLURM, one at a time, and check its log before the next. The thread settings keep NumPy to the single core that SLURM allocates; without them the job can run markedly slower.

```bash
cd /data2/nr1/V1_RailTrack

# Step 0 (optional): develop, freeze, a stand-in extraction record and scoring on a pseudo-holdout cut from
# development data, plus the refusal checks (re-freeze, second scoring, replaced protocol).
sbatch --job-name=rt_selftest --time=02:00:00 --mem=8G --cpus-per-task=1 --output=selftest-%j.log \
  --wrap="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u railtrack_study_pipeline.py self-test"

# Step 1: development analyses and report.
sbatch --job-name=rt_develop --time=04:00:00 --mem=8G --cpus-per-task=1 --output=develop-%j.log \
  --wrap="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u railtrack_study_pipeline.py develop"
```

Read `STUDY_v1/develop/DEVELOPMENT_REPORT.html`. Its key results should match the report supplied with this guide, because the inputs and the code are identical. Only freeze once you are content with the primary indicator, the model and the hypotheses in the configuration at the top of the file. Any later change to the file means re-running step 1.

```bash
# Step 2: freeze the protocol. The log prints the protocol SHA-256.
sbatch --job-name=rt_freeze --time=01:00:00 --mem=8G --cpus-per-task=1 --output=freeze-%j.log \
  --wrap="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u railtrack_study_pipeline.py freeze"
```

After step 2, register the protocol SHA-256 outside the cluster before any holdout data are read: for example a short OSF or Zenodo registration, or at least an e-mail to a co-author with `FROZEN_PROTOCOL.json` attached. This is the only safeguard that a determined person could not undo, and reviewers will ask for it.

```bash
# Step 3: re-extract four development passages from raw files and compare with the foundation table.
sbatch --job-name=rt_verify --time=01:00:00 --mem=24G --cpus-per-task=1 --output=verify-%j.log \
  --wrap="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u railtrack_study_pipeline.py verify-extraction"

# Step 4: extract the holdout features (single use; refuses to run twice). Put the registration reference in quotes.
sbatch --job-name=rt_holdout --time=03:00:00 --mem=24G --cpus-per-task=1 --output=holdout-%j.log \
  --wrap="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u railtrack_study_pipeline.py extract-holdout --confirm-single-use --registration 'OSF link or e-mail date'"

# Step 5: the single prospective evaluation.
sbatch --job-name=rt_score --time=01:00:00 --mem=8G --cpus-per-task=1 --output=score-%j.log \
  --wrap="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u railtrack_study_pipeline.py score-holdout"
```

## What is pre-declared for the holdout

These are fixed in `FROZEN_PROTOCOL.json` at step 2:

- **Primary indicator:** the cabin low-frequency response, i.e. log band power from 0.5 to 5 Hz, taken as the geometric mean of the front and back cabin sensors.
- **Primary model:** the local-level Kalman model, updated by every new observation day (sequential monitoring).
- **Secondary indicators:** bogie vertical broadband RMS, cabin broadband RMS, and bogie longitudinal 0.5 to 5 Hz band power.
- **Independent units:** calendar episodes, i.e. holdout observation days of both directions separated by more than seven days. The holdout holds 13 dates in five episodes; eight dates fall in the first.
- **Decision rule:** an exact one-sided sign test over the episodes at the 0.05 level. With five episodes, a hypothesis is supported only if every episode favours the model (p = 0.031, one chance in 32). Pooled skill and 95% episode-bootstrap intervals are reported as descriptions, not as decision rules.

The hypotheses are:

- **H1 (primary):** sequential forecasts of the primary model have a lower mean absolute error than persistence (the last observed daily anomaly), that is, the filtered monitor tracks the track response better than raw day-to-day monitoring.
- **H2 (key secondary):** the same against the static development fingerprint, that is, there is forecastable change in the track response beyond the fixed profile.
- **H3:** the forecasts rank bins with a positive mean Spearman correlation.
- **H4:** the 80% prediction interval covers between 70 and 90% of holdout bin-days.
- **H5 (secondary):** forecasts made once from the last development day beat the static fingerprint.
- **D1 and D2 (descriptive):** the share of tested segments that exceed the frozen alarm thresholds (5% expected without change) and the share of alarms in the corridor; all metrics for the secondary indicators, the other models (including the mean-reverting model) and each episode.

Why persistence is the primary comparison: in the development data the primary model beat persistence in every calendar episode (13 of 13), but the static fingerprint in only about three of four (10 of 13). Because all five holdout episodes must favour the model, the estimated power is about 0.83 for H1 and about 0.24 for H2. H2 is therefore reported as a key secondary result, with its low power stated in advance. If you prefer the fingerprint comparison as the primary hypothesis, set `'primary_comparison': 'fingerprint'` in `holdout_plan`, rename the two hypotheses to `H1_sequential_vs_fingerprint` and `H2_sequential_vs_persistence` with their texts (the program refuses mismatched names), and re-run `develop` before freezing.

## Safeguards built into the program

- **Input checks:** the program stops if the foundation files do not match their manifest, or if any holdout identifier appears in the development table.
- **Freeze checks:** `freeze` refuses development results that came from a different code version, different settings, or the reduced `--quick` mode. It refuses to run once an extraction has started or finished (holdout file, partial file, extraction log or ledger entry).
- **Extraction checks:** `extract-holdout` needs the frozen protocol, the frozen pipeline file (or a documented `--protocol-amendment`), a PASS from `verify-extraction`, and the same holdout identifiers and digest. It excludes any holdout recording that duplicates a development recording, records every override flag, and will not run twice.
- **Scoring checks:** `score-holdout` requires the frozen pipeline file (or a documented `--protocol-amendment`), and the protocol hash, holdout-file hash and accepted passage list recorded at extraction, in both the extraction log and the ledger. It refuses a second scoring unless `--rescore-reason` is given; earlier results are then kept in a dated folder and both scorings stay in the ledger.

## Changes in version 1.2.0 (after an independent code review)

- Intervals and tests now use calendar episodes; version 1.1.0 counted each episode once per direction, which made intervals too narrow.
- The holdout hypotheses are decided by exact sign tests over episodes, which remain valid with five units.
- Holdout extraction and scoring are tied together by the extraction record and the append-only ledger.
- The scheduled-work tests use a random-walk-plus-noise null for the corridor-minus-control contrast, fitted outside scheduled-work windows and checked on clean placebo windows; the earlier placebo null rested on heavily overlapping windows. The composite uses exactly the channels that are valid for each record.
- The regime-aware model needs a jump on two consecutive days before it resets, and its forecasts include the variance of future resets.
- The classification re-analysis adds placebos that shift the real label pattern in time, which is the exact comparison.
- Holdout alarms are computed exactly as the development alarm null (same origins and horizons).

## If something fails

- **"Builder SHA-256 differs":** the builder file was edited. Restore the original. If you must use another version, add `--allow-builder-mismatch` and record why.
- **"Foundation counts differ from the manifest":** the foundation files are not the verified ones.
- **"Current settings differ" or "pipeline file differs" at freeze:** re-run `develop` with the final file, then freeze.
- **"Holdout extraction has started or finished" at freeze:** the protocol can no longer be changed for this output folder.
- **Extraction stopped part way (for example a time limit):** no holdout file or extraction record exists, so nothing can be scored. Submit step 4 again; any partial file is overwritten and the ledger keeps both attempts.
- **No figures:** Matplotlib is unavailable. All tables and the report are still written.
- **No random forest rows:** scikit-learn is unavailable. The kNN and shrinkage LDA results remain.
