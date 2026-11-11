# drtrain-track-forecasting

Code for the paper *From Repeated Passages to Inspection Priorities: Predictive Monitoring of Railway Track Response with Quantified Uncertainty and Detection Limits*, by Nima Rezazadeh, Donato Perfetto, Francesco Caputo and Alessandro De Luca, under revision in *Measurement*, 2026.

The code forecasts a regime-robust vibration indicator of railway track condition from an in-service light-rail vehicle (DR-Train, LRV4306, Region 5, Pittsburgh) and evaluates it on a chronological 2016 holdout under a frozen, pre-registered protocol.

## Data
The raw data are not included. Download DR-Train from the original archive:
Liu et al. (2019), *Scientific Data* 6:146, https://doi.org/10.1038/s41597-019-0148-3

## Files

| File | Purpose |
|---|---|
| `build_predictive_maintenance_foundation.py` | Builds the 100 m feature table from the raw DR-Train files |
| `railtrack_study_pipeline.py` | Frozen confirmatory pipeline: develop, freeze, verify-extraction, extract-holdout and score-holdout (unchanged since the protocol was frozen) |
| `railtrack_revision_analyses.py` | Post hoc analyses for the first revision; imports the frozen pipeline without modifying it and writes to `STUDY_v1/revision_R1/` |
| `run_revision.sbatch` | SLURM job used to run the first-revision analyses |
| `railtrack_revision2_checks.py` | Post hoc checks for the second revision (observation variance by number of passages, day-level tests allowing for serial dependence, inspection effort); reuses the frozen pipeline and the first-revision script without modifying them and writes to `STUDY_v1/revision_R2/` |
| `run_revision2.sbatch` | SLURM job used to run the second-revision checks |
| `replot_figures.py` | Publication versions of the channel-level, scheduled-work and detection-probability figures from the pipeline outputs |
| `FROZEN_PROTOCOL.json` | The frozen protocol used for the holdout evaluation |
| `PROTOCOL_LEDGER.jsonl` | Append-only record of every protocol step, including both post hoc revision runs |
| `RUN_GUIDE.md` | Step-by-step instructions for running the pipeline |

## Requirements
Python 3.9 or later, NumPy, SciPy and h5py; Matplotlib and scikit-learn for the figures and the random forest.

## Integrity (SHA-256)

| File | SHA-256 |
|---|---|
| `railtrack_study_pipeline.py` (v1.2.0, frozen) | `4ead75ca41d98d3de7744d23ed54253d1ffb57171bc75a9185d73e187a219a46` |
| `build_predictive_maintenance_foundation.py` (v1.0) | `e8c5b403306691c1f101cfce75f28714903ce87682ebdb024f687f342da38852` |
| `FROZEN_PROTOCOL.json` | `4f263236ff2a7686bcfdf0a98b6624031fbd34df542e96cc5f54282370562f3b` |
| `railtrack_revision_analyses.py` (R1.0, post hoc) | `48d40d8403500ac8762b90c1e211f3b4b64a43e259278344ed23bd140c1dac04` |
| `railtrack_revision2_checks.py` (R2.0, post hoc) | `1313903ef108d4253e96e6e157e917d275ed3fd061e2b87d17654d1c68bd5e85` |

The digests of the frozen pipeline and the builder match those recorded in `FROZEN_PROTOCOL.json` before the holdout evaluation. Both revision scripts were added after the holdout had been scored. Each verifies the digests of the files it reuses before running and imports them read-only, so the confirmatory results are unchanged.

Check a file with `sha256sum <file>` (Linux) or `Get-FileHash <file>` (Windows PowerShell).

The default folder paths in the scripts and SLURM files refer to the authors' computing cluster; set your own paths with the command-line options described in `RUN_GUIDE.md`.

## Citation
[Paper reference, added after acceptance]

Code archive (Zenodo):
- v1.0.1, confirmatory analyses: https://doi.org/10.5281/zenodo.23576526
- v1.1.0, first-revision analyses: https://doi.org/10.5281/zenodo.24044584
- v1.2.0, second-revision checks: [DOI, added after this release]

## Licence
MIT (see LICENSE).
