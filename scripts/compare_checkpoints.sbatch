#!/bin/bash -l
#SBATCH --job-name=t2t-compare
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/compare-%j.log

set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the Touch2Touch repository root}"
module load conda

limit="$(squeue -h -j "$SLURM_JOB_ID" -o '%L')"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
exec conda run --no-capture-output -n touch2touch-smoke \
    python -u scripts/compare_checkpoints.py \
    --output "results/comparison-$SLURM_JOB_ID" --time-left "$limit" \
    --expected-pairs 552 "$@"
