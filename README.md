# drtrain-track-forecasting

Code for the paper: *[Paper title]*, by Nima Rezazadeh, Donato Perfetto, Francesco Caputo and Alessandro De Luca, under review in *Journal of Civil Structural Health Monitoring*, 2026.

The code forecasts a regime-robust vibration indicator of railway track condition from an
in-service light-rail vehicle (DR-Train, LRV4306, Region 5, Pittsburgh) and evaluates it
on a chronological 2016 holdout under a frozen, pre-registered protocol.

## Data
The raw data are not included. Download DR-Train from the original archive:
Liu et al. (2019), *Scientific Data* 6:146, https://doi.org/10.1038/s41597-019-0148-3

## Files
| File | Purpose |
|---|---|
| `build_predictive_maintenance_foundation.py` | Builds the 100 m feature table from the raw DR-Train files |
| `railtrack_study_pipeline.py` | All analyses: develop, freeze, verify-extraction, extract-holdout, score-holdout |
| `replot_figures.py` | Publication versions of Figs. 1, 5 and 6a from the pipeline outputs (figure numbers of the Results draft) |
| `FROZEN_PROTOCOL.json` | The frozen protocol used for the holdout evaluation |
| `PROTOCOL_LEDGER.jsonl` | Append-only record of every protocol step |
| `RUN_GUIDE.md` | Step-by-step instructions for running the pipeline |

## Requirements
Python 3.9 or later, NumPy, SciPy, h5py; Matplotlib and scikit-learn for figures and the random forest.

## Integrity (SHA-256)
| File | SHA-256 |
|---|---|
| `railtrack_study_pipeline.py` (v1.2.0) | `4ead75ca41d98d3de7744d23ed54253d1ffb57171bc75a9185d73e187a219a46` |
| `build_predictive_maintenance_foundation.py` (v1.0) | `e8c5b403306691c1f101cfce75f28714903ce87682ebdb024f687f342da38852` |
| `FROZEN_PROTOCOL.json` | `4f263236ff2a7686bcfdf0a98b6624031fbd34df542e96cc5f54282370562f3b` |

Check a file with `sha256sum <file>` (Linux) or `Get-FileHash <file>` (Windows PowerShell).
The default folder paths inside the scripts refer to the authors' computing cluster;
set your own paths with the command-line options described in RUN_GUIDE.md.

## Citation
[Paper reference, added after acceptance]
Code archive: [Zenodo DOI, added after the first release]

## Licence
MIT (see LICENSE).
