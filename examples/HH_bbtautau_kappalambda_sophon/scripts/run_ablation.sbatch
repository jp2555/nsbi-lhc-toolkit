#!/bin/bash
# Stats data-efficiency ablation for the sophon-ak4 NSBI diagnostic.
# Trains all 3 controls (scratch / sophon_frozen / sophon_finetune) at several
# per-point event counts and writes ablation.json + ablation.png (best val-loss vs N).
#
# This is GPU training (the diagnostic pins devices=1) -> one GPU on a "shared"
# GPU node. Use the *_g account variant.
#
# Submit (subset clouds must already exist; see convert_all_kl.py --points kl0 kl5):
#   mkdir -p logs
#   CKPT=$SCRATCH/sophon-ak4/models/JetClassII_SophonAK4/model.pt \
#     sbatch -A <NERSC_PROJECT>_g \
#       examples/HH_bbtautau_kappalambda_sophon/scripts/run_ablation.sbatch
#
# Override anything via env (defaults shown below):
#   CLOUDS=... OUTDIR=... CKPT=... POINT_A=kl5 POINT_B=kl0 \
#   SIZES="20000 50000 100000" EPOCHS=30 LR=2e-4 BATCH=256 \
#     sbatch -A <NERSC_PROJECT>_g .../run_ablation.sbatch
#
#SBATCH -J dihiggs_ablation
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
# GPU "shared" charges max(cores, mem-equiv-cores) and requires exactly 32 cores/GPU.
# Request 32 cores and let memory auto-allocate proportionally (~60 GB for 1 GPU) --
# an explicit --mem can push the mem-equivalent above 32 cores and get rejected.
#SBATCH --cpus-per-task=32
#SBATCH -t 06:00:00
#SBATCH -o logs/ablation_%j.out
#SBATCH -e logs/ablation_%j.err

set -euo pipefail

# GPU env: training needs CUDA. Build it once: `pixi install -e nsbi-env-gpu`.
ENV=${ENV:-nsbi-env-gpu}
REPO=${REPO:-$SCRATCH/nsbi-lhc-toolkit}
CLOUDS=${CLOUDS:-$SCRATCH/dihiggs/clouds_subset}
POINT_A=${POINT_A:-kl5}          # denominator / fixed reference (kappa_lambda=5)
POINT_B=${POINT_B:-kl0}          # numerator
OUTDIR=${OUTDIR:-$SCRATCH/dihiggs/ablation_${POINT_B}_vs_${POINT_A}}
# default sweep pushes into the low-stats stress regime (where pretraining should
# help most); REPEATS>1 retrains each point with different seeds for error bars.
SIZES=${SIZES:-"2000 5000 10000 20000 50000 100000"}
REPEATS=${REPEATS:-1}
EPOCHS=${EPOCHS:-30}
LR=${LR:-2e-4}
BATCH=${BATCH:-256}
# checkpoint: CKPT (or pre-exported SOPHON_AK4_CKPT). Without it the two sophon_* controls
# are skipped and only `scratch` runs.
export SOPHON_AK4_CKPT=${CKPT:-${SOPHON_AK4_CKPT:-}}

echo "ENV=$ENV REPO=$REPO CLOUDS=$CLOUDS"
echo "task=$POINT_B vs $POINT_A  SIZES=$SIZES  REPEATS=$REPEATS  EPOCHS=$EPOCHS  LR=$LR  BATCH=$BATCH"
echo "OUTDIR=$OUTDIR  SOPHON_AK4_CKPT=${SOPHON_AK4_CKPT:-<unset>}"

cd "$REPO"
# SIZES is intentionally unquoted so it word-splits into the nargs="*" --ablation-sizes list.
srun pixi run -e "$ENV" python \
    examples/HH_bbtautau_kappalambda_sophon/scripts/run_kappa_diagnostic.py \
    --clouds-dir "$CLOUDS" \
    --point-a "$POINT_A" --point-b "$POINT_B" \
    --ablation-sizes $SIZES --repeats "$REPEATS" \
    --epochs "$EPOCHS" --learning-rate "$LR" --batch-size "$BATCH" \
    --out-dir "$OUTDIR"
