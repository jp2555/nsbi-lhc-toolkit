#!/usr/bin/env bash
# Option A driver: EveNet fine-tune test on CMS bbtautau samples (Perlmutter).
# Stages the EVENET_INTEGRATION_PLAN.md sec 10 runbook; gates in EVENET_FEASIBILITY_NOTE sec 8.
#
# Usage:
#   ./run_option_a.sh smoke                      # local sanity, no ROOT/GPU needed (any machine)
#   ./run_option_a.sh setup                      # clone EveNet + download checkpoints
#   ./run_option_a.sh check <file.root>          # verify NanoAOD branch names before converting
#   ./run_option_a.sh convert                    # NanoAOD -> NPZ (kl0, kl1, kl5) + jes_mhh inject
#   ./run_option_a.sh preprocess                 # EveNet preprocess (shifter) for the chosen pair
#   ./run_option_a.sh configs                    # generate the 75 sweep configs + run scripts
#   ./run_option_a.sh train                      # submit sbatch array (or train-local, sequential)
#   ./run_option_a.sh predict                    # sequential predictions (GPU node)
#   ./run_option_a.sh eval                       # AUC money plot + closure gates
#
# Required env (convert/preprocess/train):
#   NANO=/path/to/cms_nanoaod          with kl0/ kl1/ kl5/ subdirs of *.root
#   BTAG_WP=<float>                    era WP for BTAG_BRANCH from the BTV tables
#   ACCOUNT=<mXXXX>                    Slurm allocation (train stage only)
# Optional env (defaults):
#   STORE=$PSCRATCH/evenet-klambda     outputs (npz/, evenet-train/, checkpoints/, predictions/)
#   EVENET_SRC=$HOME/EveNet_Public     EveNet code checkout (working_dir for train/predict)
#   IMAGE=avencast1994/evenet:1.5      shifter image
#   TAU_ENCODING=anonymous             or corner (G1 A/B: writes npz under npz-$TAU_ENCODING/)
#   KL_HYP=5                           hypothesis class vs the SM kl=1 reference (0 or 5)
#   TASK=kl                            or syst (nominal-vs-jes_mhh, Money Plot 2)
#   BTAG_BRANCH=Jet_btagUParTAK4B      NanoAOD b-tag discriminant branch
#   CONVERT_PY=python3                 python with numpy+uproot for the adapter steps
#   NGPU=1  TIME=04:00:00              per training task (sbatch array)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STORE="${STORE:-${PSCRATCH:-/tmp}/evenet-klambda}"
EVENET_SRC="${EVENET_SRC:-$HOME/EveNet_Public}"
IMAGE="${IMAGE:-avencast1994/evenet:1.5}"
TAU_ENCODING="${TAU_ENCODING:-anonymous}"
KL_HYP="${KL_HYP:-5}"
TASK="${TASK:-kl}"
BTAG_BRANCH="${BTAG_BRANCH:-Jet_btagUParTAK4B}"
CONVERT_PY="${CONVERT_PY:-python3}"
NPZ="$STORE/npz-$TAU_ENCODING"
FARM="$HERE/config_farm-$TASK-$TAU_ENCODING"

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "== $*"; }

pair_files() {  # the two NPZ classes for the chosen TASK
    if [ "$TASK" = "kl" ]; then
        echo "$NPZ/kl1.npz $NPZ/kl$KL_HYP.npz"
    else
        echo "$NPZ/kl1.npz $NPZ/kl1_jesmhh.npz"
    fi
}

stage_smoke() {
    cd "$HERE"
    note "adapter smoke (both encodings) + closure self-test (no ROOT/torch needed)"
    $CONVERT_PY scripts/nanoaod_to_evenet_npz.py --smoke 512 --class-id 0 --output /tmp/oa_smoke_a.npz
    $CONVERT_PY scripts/nanoaod_to_evenet_npz.py --smoke 512 --class-id 0 --tau-encoding corner \
        --output /tmp/oa_smoke_c.npz
    $CONVERT_PY scripts/inject_systematic.py --input /tmp/oa_smoke_a.npz \
        --output /tmp/oa_smoke_jes.npz --kind jes_mhh --alpha 0.05
    $CONVERT_PY scripts/eval_closure.py --self-test
    note "smoke OK"
}

stage_setup() {
    [ -d "$EVENET_SRC" ] || git clone https://github.com/UW-EPE-ML/EveNet_Public "$EVENET_SRC"
    mkdir -p "$STORE/pretrain-weights"
    if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi
    $HF download Avencast/EveNet --local-dir "$STORE/pretrain-weights"
    ls -la "$STORE/pretrain-weights" | grep -E "20M" || die "20M checkpoints not found after download"
    note "setup OK: $EVENET_SRC + $STORE/pretrain-weights"
}

stage_check() {
    local f="${1:-}"; [ -n "$f" ] || die "usage: $0 check <real NanoAOD file.root>"
    $CONVERT_PY - "$f" <<'EOF'
import sys, uproot
e = uproot.open(sys.argv[1])["Events"]
for pat in ("Tau_id*", "Jet_btag*", "PuppiMET_*", "MET_p*", "genWeight"):
    print(f"{pat:12s} ->", e.keys(filter_name=pat))
EOF
    echo "Confirm: (a) the Tau_id VSjet branch matches --tau-id-branch (default"
    echo "Tau_idDeepTau2018v2p5VSjet), (b) $BTAG_BRANCH exists, (c) BTAG_WP is the"
    echo "BTV-table WP for THAT discriminant. Then run: convert"
}

stage_convert() {
    [ -n "${NANO:-}" ] || die "set NANO=/path/to/nanoaod (kl0/ kl1/ kl5/ subdirs)"
    [ -n "${BTAG_WP:-}" ] || die "set BTAG_WP (era WP for $BTAG_BRANCH from the BTV tables)"
    mkdir -p "$NPZ"
    cd "$HERE"
    for kl in 0 1 5; do
        cls=1; [ "$kl" = "1" ] && cls=0            # kl=1 is the SM reference class
        note "convert kl=$kl (class $cls, tau-encoding $TAU_ENCODING)"
        $CONVERT_PY scripts/nanoaod_to_evenet_npz.py --input "$NANO/kl$kl/*.root" \
            --class-id $cls --btag-wp "$BTAG_WP" --btag-branch "$BTAG_BRANCH" \
            --tau-encoding "$TAU_ENCODING" --output "$NPZ/kl$kl.npz"
    done
    note "inject jes_mhh systematic on the nominal (Money Plot 2 input)"
    $CONVERT_PY scripts/inject_systematic.py --input "$NPZ/kl1.npz" \
        --output "$NPZ/kl1_jesmhh.npz" --kind jes_mhh --alpha 0.05
    ls -la "$NPZ"
}

stage_preprocess() {
    [ -d "$EVENET_SRC" ] || die "EVENET_SRC=$EVENET_SRC missing (run: setup)"
    local files; files=$(pair_files)
    for f in $files; do [ -f "$f" ] || die "$f missing (run: convert)"; done
    note "preprocess $files -> $STORE/evenet-train ($TASK task)"
    cd "$EVENET_SRC"
    shifter --image="$IMAGE" python3 preprocessing/preprocess.py \
        --files $files --split_ratio 0.8,0.1,0.1 \
        --store_dir "$STORE/evenet-train" --config "$HERE/configs/event_info_klambda.yaml"
}

stage_configs() {
    cd "$HERE"
    ! grep -q "PLACEHOLDER" configs/workflow_klambda.yaml \
        || die "fill the <PLACEHOLDER> paths in configs/workflow_klambda.yaml first \
(working_dir=$EVENET_SRC, image, network/resonance/option yamls, pretrain ckpt under \
$STORE/pretrain-weights/) — see HANDOFF.md"
    python3 scripts/make_klambda_configs.py configs/workflow_klambda.yaml \
        --farm "$FARM" --store_dir "$STORE" --ray_dir "${PSCRATCH:-/tmp}/ray-$USER"
    [ -f "$FARM/predict-evenet.sh" ] \
        || echo "WARN: no predict-evenet.sh — set workflow_klambda.yaml:predict_yaml to also emit predict configs"
}

stage_train() {
    [ -f "$FARM/train-evenet.sh" ] || die "run: configs"
    [ -n "${ACCOUNT:-}" ] || die "set ACCOUNT=<mXXXX> for sbatch (or use: train-local)"
    local n; n=$(wc -l < "$FARM/train-evenet.sh")
    cat > "$FARM/train-array.sbatch" <<EOF
#!/bin/bash
#SBATCH -A $ACCOUNT -C gpu -q regular -t ${TIME:-04:00:00}
#SBATCH -N 1 --gpus-per-task=${NGPU:-1} --ntasks=1 -c 32
#SBATCH --array=1-$n%16
#SBATCH -o $FARM/slurm-%A_%a.out
eval "\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" $FARM/train-evenet.sh)"
EOF
    sbatch "$FARM/train-array.sbatch"
    note "submitted $n-task array (throttle %16); watch: squeue --me"
}

stage_train_local() {   # sequential, for an interactive GPU node (salloc)
    [ -f "$FARM/train-evenet.sh" ] || die "run: configs"
    bash "$FARM/train-evenet.sh"
}

stage_predict() {
    [ -f "$FARM/predict-evenet.sh" ] || die "no predict-evenet.sh (see configs stage WARN)"
    bash "$FARM/predict-evenet.sh"
}

stage_eval() {
    cd "$HERE"
    note "G0: AUC money plot"
    python3 scripts/plot_data_efficiency.py --store_dir "$STORE" \
        --output "data_efficiency-$TASK-$TAU_ENCODING.png" \
        ${CEILING:+--ceiling "$CEILING"}
    note "G0.5: ratio-closure gates (|IC| vs stat, SC_rms < 0.05, chi2/ndf ~ 1)"
    python3 scripts/eval_closure.py --store_dir "$STORE" \
        --output-prefix "closure-$TASK-$TAU_ENCODING"
}

case "${1:-}" in
    smoke)       stage_smoke ;;
    setup)       stage_setup ;;
    check)       stage_check "${2:-}" ;;
    convert)     stage_convert ;;
    preprocess)  stage_preprocess ;;
    configs)     stage_configs ;;
    train)       stage_train ;;
    train-local) stage_train_local ;;
    predict)     stage_predict ;;
    eval)        stage_eval ;;
    *) grep "^#" "$0" | sed -n '2,32p'; exit 1 ;;
esac
