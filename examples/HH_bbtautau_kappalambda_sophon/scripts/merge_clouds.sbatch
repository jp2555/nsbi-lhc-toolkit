#!/bin/bash
# Merge per-shard partial clouds into one <label>.npz per kappa_lambda point.
# Run after the convert_shards array finishes (or chain with --dependency):
#   jid=$(NSHARDS=20 sbatch -A <PROJ> --array=0-119%32 --parsable .../convert_shards.sbatch)
#   sbatch -A <PROJ> --dependency=afterok:$jid .../merge_clouds.sbatch
#
# --mem must hold ONE point's full cloud array. Fine for subset tests; for the full
# 500k-event points bump it (e.g. --mem=180G) or skip merge and train from shards.
#
#SBATCH -J dihiggs_merge
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/merge_%j.out
#SBATCH -e logs/merge_%j.err

set -euo pipefail
ENV=${ENV:-nsbi-env}
OUT=${OUT:-$SCRATCH/dihiggs/clouds}
REPO=${REPO:-$SCRATCH/nsbi-lhc-toolkit}

cd "$REPO"
srun pixi run -e "$ENV" python \
    examples/HH_bbtautau_kappalambda_sophon/scripts/merge_clouds.py --out-dir "$OUT"
