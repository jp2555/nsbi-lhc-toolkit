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
#SBATCH -J dihiggs_convert
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o convert_%j.out
#SBATCH -e convert_%j.err

set -euo pipefail

RAW=${RAW:-$SCRATCH/dihiggs/raw}
OUT=${OUT:-$SCRATCH/dihiggs/clouds}
REPO=${REPO:-$SCRATCH/nsbi-lhc-toolkit}

echo "RAW=$RAW"
echo "OUT=$OUT"
echo "REPO=$REPO"

cd "$REPO"
srun pixi run -e nsbi-env-gpu python \
    examples/HH_bbtautau_kappalambda_sophon/scripts/convert_all_kl.py \
    --raw-dir "$RAW" --out-dir "$OUT"
