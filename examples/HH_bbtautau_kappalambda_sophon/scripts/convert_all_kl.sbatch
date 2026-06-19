#!/bin/bash
# Convert all kappa_lambda HH->bbtautau Delphes datasets to sophon-ak4 clouds.
# Pure I/O + numpy (no GPU) -> run on Perlmutter CPU nodes, single-process "shared" QOS.
#
# Submit (override the account; paths default to $SCRATCH):
#   sbatch -A <NERSC_PROJECT> examples/HH_bbtautau_kappalambda_sophon/scripts/convert_all_kl.sbatch
# Override paths if needed:
#   RAW=/path/raw OUT=/path/clouds REPO=/path/nsbi-lhc-toolkit \
#     sbatch -A <NERSC_PROJECT> .../convert_all_kl.sbatch
#
# Full production = all files (MAXFILES=0, the default) for all six points; this can be
# long, so bump the walltime at submit: `sbatch --time=24:00:00 ...`. Set POINTS to
# convert a subset, or to fan the points out into parallel per-point jobs (see README).
#
#SBATCH -J dihiggs_convert
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o convert_%j.out
#SBATCH -e convert_%j.err

set -euo pipefail

# CPU env by default: the GPU env won't activate on a CPU node (cuda system requirement),
# and the converter is pure uproot/numpy (no GPU). Build it once: `pixi install -e nsbi-env`.
ENV=${ENV:-nsbi-env}
RAW=${RAW:-$SCRATCH/dihiggs/raw}
OUT=${OUT:-$SCRATCH/dihiggs/clouds}
REPO=${REPO:-$SCRATCH/nsbi-lhc-toolkit}
MAXFILES=${MAXFILES:-0}
POINTS=${POINTS:-}            # e.g. "kl0 kl5"; empty = all six points

echo "ENV=$ENV  RAW=$RAW  OUT=$OUT  REPO=$REPO  MAXFILES=$MAXFILES  POINTS=${POINTS:-<all>}"

cd "$REPO"
# ${POINTS:+...} expands to nothing when POINTS is empty (all points), else word-splits.
srun pixi run -e "$ENV" python \
    examples/HH_bbtautau_kappalambda_sophon/scripts/convert_all_kl.py \
    --raw-dir "$RAW" --out-dir "$OUT" --max-files "$MAXFILES" \
    ${POINTS:+--points $POINTS}
