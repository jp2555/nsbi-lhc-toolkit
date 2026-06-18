#!/bin/bash
# PARALLEL kappa_lambda cloud conversion: a Slurm job array over (point x file-shard).
# 6 points x NSHARDS shards  ->  array indices 0 .. (6*NSHARDS - 1).
#
#   mkdir -p logs
#   # full run (20 shards/point -> 120 tasks, 32 concurrent):
#   NSHARDS=20 sbatch -A <NERSC_PROJECT> --array=0-119%32 \
#       examples/HH_bbtautau_kappalambda_sophon/scripts/convert_shards.sbatch
#   # FAST subset test today (8 files/point, 4 shards -> 24 tasks):
#   NSHARDS=4 MAXFILES=8 sbatch -A <NERSC_PROJECT> --array=0-23%24 \
#       examples/HH_bbtautau_kappalambda_sophon/scripts/convert_shards.sbatch
#
# IMPORTANT: keep --array upper bound = 6*NSHARDS - 1.  Then merge with merge_clouds.py.
#
#SBATCH -J dihiggs_convert
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/convert_%A_%a.out
#SBATCH -e logs/convert_%A_%a.err

set -euo pipefail
ENV=${ENV:-nsbi-env}
NSHARDS=${NSHARDS:-20}
MAXFILES=${MAXFILES:-0}
RAW=${RAW:-$SCRATCH/dihiggs/raw}
OUT=${OUT:-$SCRATCH/dihiggs/clouds}
REPO=${REPO:-$SCRATCH/nsbi-lhc-toolkit}

echo "ENV=$ENV NSHARDS=$NSHARDS MAXFILES=$MAXFILES task=$SLURM_ARRAY_TASK_ID"
cd "$REPO"
srun pixi run -e "$ENV" python \
    examples/HH_bbtautau_kappalambda_sophon/scripts/convert_shard.py \
    --raw-dir "$RAW" --out-dir "$OUT" \
    --task-id "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS" --max-files "$MAXFILES"
