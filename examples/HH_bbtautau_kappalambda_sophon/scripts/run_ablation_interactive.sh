#!/bin/bash
# Interactive ablation runner -- no batch queue wait. Loops the whole stats sweep
# (smallest N first) in one process; the diagnostic now writes ablation.json +
# ablation.png after EVERY size, so you can Ctrl-C and keep the finished points.
#
# 1) grab an interactive GPU node (interactive QOS, fast turnaround, max 4 h):
#      salloc -N 1 -C gpu -q interactive -t 02:00:00 -A m5295_g
# 2) from the repo root, run this (CKPT points at the sophon-ak4 weights):
#      CKPT=$SCRATCH/sophon-ak4/models/JetClassII_SophonAK4/model.pt \
#        bash examples/HH_bbtautau_kappalambda_sophon/scripts/run_ablation_interactive.sh
#
# Smoke-test the small sizes first (~1-2 min each) before committing to 50k/100k:
#      SIZES="2000 5000" CKPT=... bash .../run_ablation_interactive.sh
# Error bars at the low end:
#      SIZES="2000 5000 10000" REPEATS=4 CKPT=... bash .../run_ablation_interactive.sh
set -euo pipefail

ENV=${ENV:-nsbi-env-gpu}
REPO=${REPO:-$SCRATCH/nsbi-lhc-toolkit}
CLOUDS=${CLOUDS:-$SCRATCH/dihiggs/clouds_subset}
POINT_A=${POINT_A:-kl5}          # denominator / reference
POINT_B=${POINT_B:-kl0}          # numerator
OUTDIR=${OUTDIR:-$SCRATCH/dihiggs/ablation_${POINT_B}_vs_${POINT_A}}
SIZES=${SIZES:-"2000 5000 10000 20000 50000 100000"}
REPEATS=${REPEATS:-1}
EPOCHS=${EPOCHS:-30}
LR=${LR:-2e-4}
BATCH=${BATCH:-256}
# checkpoint via CKPT (or a pre-exported SOPHON_AK4_CKPT); without it the sophon_* controls
# are skipped and only `scratch` runs.
export SOPHON_AK4_CKPT=${CKPT:-${SOPHON_AK4_CKPT:-}}

echo "task=$POINT_B vs $POINT_A  SIZES=$SIZES  REPEATS=$REPEATS  EPOCHS=$EPOCHS  LR=$LR"
echo "CLOUDS=$CLOUDS  ->  OUTDIR=$OUTDIR  CKPT=${SOPHON_AK4_CKPT:-<unset>}"

cd "$REPO"
# SIZES unquoted so it word-splits into the nargs="*" --ablation-sizes list.
# Running directly (no srun): we are already inside the salloc allocation.
pixi run -e "$ENV" python \
    examples/HH_bbtautau_kappalambda_sophon/scripts/run_kappa_diagnostic.py \
    --clouds-dir "$CLOUDS" --point-a "$POINT_A" --point-b "$POINT_B" \
    --ablation-sizes $SIZES --repeats "$REPEATS" \
    --epochs "$EPOCHS" --learning-rate "$LR" --batch-size "$BATCH" \
    --out-dir "$OUTDIR"
