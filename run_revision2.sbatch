#!/bin/bash
#SBATCH --job-name=rt_revision2
#SBATCH --time=00:45:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=revision2-%j.log

# Post hoc checks for the second revision (Measurement, Reviewer 8). The frozen pipeline, the R1 revision module,
# the protocol, the foundation files and all earlier results are only read. Outputs go to STUDY_v1/revision_R2/;
# two lines are appended to STUDY_v1/PROTOCOL_LEDGER.jsonl. STUDY_v1/revision_R1/ is not touched.
set -euo pipefail
cd /data2/nr1/V1_RailTrack
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# If the study's Python environment is not active by default, activate it here, for example:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate <environment name>

echo "Frozen pipeline SHA-256 (must start with 4ead75ca):"
sha256sum railtrack_study_pipeline.py
echo "R1 revision module SHA-256 (must start with 48d40d84):"
sha256sum railtrack_revision_analyses.py

ARGS=(--allow-holdout-diagnostics
      --foundation-dir /data2/nr1/railtrack_study/foundation_cache/ec683380532885913160947d/build_20261018T095715146004Z)

python3 -u railtrack_revision2_checks.py "${ARGS[@]}"

# One archive of the results to bring back.
cd STUDY_v1 && rm -f revision_R2.zip && python3 -m zipfile -c revision_R2.zip revision_R2/
echo "Archive: /data2/nr1/V1_RailTrack/STUDY_v1/revision_R2.zip"
