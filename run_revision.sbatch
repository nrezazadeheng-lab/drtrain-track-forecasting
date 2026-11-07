#!/bin/bash
#SBATCH --job-name=rt_revision
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=revision-%j.log

# Post hoc analyses for the first revision (Measurement). The frozen pipeline, protocol, foundation files and earlier
# results are only read. Outputs go to STUDY_v1/revision_R1/; two lines are appended to STUDY_v1/PROTOCOL_LEDGER.jsonl.
set -euo pipefail
cd /data2/nr1/V1_RailTrack
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# If the study's Python environment is not active by default, activate it here, for example:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate <environment name>

echo "Frozen pipeline SHA-256 (must start with 4ead75ca):"
sha256sum railtrack_study_pipeline.py

# --allow-holdout-diagnostics re-reads the digest-checked holdout features only to describe the frozen models
# (calibration, residuals, operating conditions, inspection decisions). Delete it to use development data only.
ARGS=(--allow-holdout-diagnostics)

# Optional daily temperatures (NOAA GHCN-Daily CSV with DATE and TAVG, or TMAX and TMIN), used only if the file exists.
TEMP=/data2/nr1/V1_RailTrack/pittsburgh_daily_temperature.csv
if [ -f "$TEMP" ]; then ARGS+=(--temperature-csv "$TEMP"); fi

python3 -u railtrack_revision_analyses.py "${ARGS[@]}"

# One archive of the results to bring back for the revision.
cd STUDY_v1 && rm -f revision_R1.zip && python3 -m zipfile -c revision_R1.zip revision_R1/
echo "Archive: /data2/nr1/V1_RailTrack/STUDY_v1/revision_R1.zip"
