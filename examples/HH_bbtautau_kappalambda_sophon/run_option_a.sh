#!/usr/bin/env bash
# Option A driver: EveNet fine-tune test on CMS bbtautau samples (Perlmutter).
# Stages the EVENET_INTEGRATION_PLAN.md sec 10 runbook; gates in EVENET_FEASIBILITY_NOTE sec 8.
#
# Usage:
#   ./run_option_a.sh smoke                      # local sanity, no ROOT/GPU needed (any machine)
#   ./run_option_a.sh setup                      # clone EveNet + download checkpoints
#   ./run_option_a.sh ceiling                    # kinematic ceiling from the FEATURE ntuple (CPU)
#   ./run_option_a.sh ceiling-array              # same, as a 6-task sbatch array + merge job
#   ./run_option_a.sh check <file.root>          # verify NanoAOD branch names before converting
#   ./run_option_a.sh convert                    # ntuples -> NPZ (kl0, kl1, kl5) + jes_mhh inject
#   ./run_option_a.sh preprocess                 # EveNet preprocess (shifter) for the chosen pair
#   ./run_option_a.sh configs                    # generate the 90 sweep configs + run scripts
#   ./run_option_a.sh train                      # submit sbatch array (or train-local, sequential)
#   ./run_option_a.sh predict                    # 90-task GPU sbatch array (or predict-local)
#   ./run_option_a.sh eval                       # AUC money plot (ceiling overlaid) + closure gates
#
# INPUT NOTE: the flat feature ntuples (dihiggs_powheg_data.root: tree_sbi_lam*, 12
# features) feed the CEILING ONLY. The sweep (convert stage) takes object-level input,
# dispatched on INPUT_FORMAT:
#   crown   (default) the CROWN analysis ntuples convert_powheg_to_sbi.py reads
#           (b-pair + tautau-leg four-vectors; published mt/et selection) -- set NTUPLES
#   nanoaod raw CMS NanoAOD with kl0/ kl1/ kl5/ subdirs -- set NANO and BTAG_WP
# ttbar (CROWN, same branch contract):
#   python3 scripts/crown_to_evenet_npz.py --input '<globs>' --class-id 1 --output ...
#
# Required env (convert/preprocess/train):
#   NTUPLES=/ceph/jpan/saved_datasets/ntuple_bbtt_24   CROWN base (KIT; override elsewhere)
#   NANO=/path/to/cms_nanoaod          only for INPUT_FORMAT=nanoaod (with BTAG_WP)
#   ACCOUNT=<mXXXX>                    Slurm allocation (train stage only)
# Optional env (defaults):
#   STORE=$PSCRATCH/evenet-klambda     outputs (npz/, evenet-train/, checkpoints/, predictions/)
#   EVENET_SRC=$HOME/EveNet_Public     EveNet code checkout (working_dir for train/predict)
#   IMAGE=avencast1994/evenet:1.5      shifter image
#   INPUT_FORMAT=crown                 or nanoaod (convert stage input format)
#   TAU_ENCODING=anonymous             or corner (G1 A/B: writes npz under npz-$TAU_ENCODING/)
#   VARIANT=<none>|nonoise             nonoise = classification trained on CLEAN inputs
#                                      (noise_prob [0,0]) at fraction 1.0 only; outputs go to
#                                      $STORE/variant-nonoise/ and config_farm-*-nonoise/,
#                                      reading the SHARED preprocessed data from $STORE
#   SIZES_YAML=[1.0]                   override the workflow's dataset_size_choice
#   SEEDS_YAML=[5,6,...,19]            override the workflow's seeds. Seed extension: set
#                                      this + SIZES_YAML + a fresh FARM, rerun configs/train/
#                                      predict -- outputs join the same STORE, so eval
#                                      ensembles automatically grow to every seed found
#   FARM=<config_farm-$TASK-...>       config-farm dir (override for seed-extension batches)
#   NBOOT=200                          Poisson test-set bootstrap replicas (eval stage)
#   RETRAIN=1                          allow train to resubmit tags that already have
#                                      checkpoints (guard against a stale/forgotten FARM)
#   PREDICT_ANYWAY=1                   allow predict while a train-array is still queued
#   KL_HYP=5                           hypothesis class vs the SM kl=1 reference (0 or 5)
#   TASK=kl                            or syst (nominal-vs-jes_mhh, Money Plot 2)
#   BTAG_BRANCH=Jet_btagUParTAK4B      NanoAOD b-tag discriminant branch
#   CONVERT_PY=python3                 python with numpy+uproot for the adapter steps
#   NGPU=1  TIME=04:00:00              per training task (sbatch array)
#   QOS=regular                        train-array QOS; QOS=shared bills the used node
#                                      FRACTION (1 GPU + 32c) instead of the whole
#                                      4-GPU node -- use for small-fraction tiers
#   FEATURES=.../dihiggs_powheg_data.root   prelim-result feature ntuple (ceiling stage)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STORE="${STORE:-${PSCRATCH:-/tmp}/evenet-klambda}"
EVENET_SRC="${EVENET_SRC:-$HOME/EveNet_Public}"
IMAGE="${IMAGE:-avencast1994/evenet:1.5}"
TAU_ENCODING="${TAU_ENCODING:-anonymous}"
INPUT_FORMAT="${INPUT_FORMAT:-crown}"
KL_HYP="${KL_HYP:-5}"
TASK="${TASK:-kl}"
BTAG_BRANCH="${BTAG_BRANCH:-Jet_btagUParTAK4B}"
CONVERT_PY="${CONVERT_PY:-python3}"
NTUPLES="${NTUPLES:-/ceph/jpan/saved_datasets/ntuple_bbtt_24}"   # CROWN ntuples (KIT)
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
if [ -z "${FEATURES:-}" ]; then      # feature ntuple: repo-local copies first, then the original
    for c in "$HERE"/{samples,saved_datasets}/{dihiggs,diHiggs}_powheg_data.root \
             "$REPO_ROOT"/{samples,saved_datasets}/{dihiggs,diHiggs}_powheg_data.root \
             /pscratch/sd/j/jing/NSBI-irishep/dihiggs_bbtautau/dihiggs_powheg_data.root; do
        [ -f "$c" ] && FEATURES="$c" && break
    done
    FEATURES="${FEATURES:-/pscratch/sd/j/jing/NSBI-irishep/dihiggs_bbtautau/dihiggs_powheg_data.root}"
fi
NPZ="$STORE/npz-$TAU_ENCODING"
CEILING_JSON="$STORE/ceiling-kl$KL_HYP.json"
# VARIANT: rerun a slice of the sweep with ONE config value changed, fully separated from
# the baseline. Separation is mandatory, not cosmetic: the run-tag regex in
# plot_data_efficiency.py uses re.search, so a suffixed tag would parse as a baseline run
# and be silently merged into the baseline cells as an extra seed. Everything except the
# flipped knob is shared -- notably the preprocessed data, read from the MAIN store.
VARIANT="${VARIANT:-}"
case "$VARIANT" in
    "")        STORE_RUN="$STORE"; SFX="" ;;
    nonoise)   STORE_RUN="$STORE/variant-nonoise"; SFX="-nonoise"
               SIZES_YAML="${SIZES_YAML:-[1.0]}" ;;   # calibration-mechanism test @ full stats
    *)         echo "ERROR: unknown VARIANT=$VARIANT (known: nonoise)" >&2; exit 1 ;;
esac
FARM="${FARM:-$HERE/config_farm-$TASK-$TAU_ENCODING$SFX}"
RESOLVED="resolved$SFX"

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

stage_ceiling() {
    [ -f "$FEATURES" ] || die "FEATURES=$FEATURES not found (feature ntuple)"
    mkdir -p "$STORE"
    cd "$HERE"
    note "kinematic ceiling from $FEATURES (tree_sbi_lam1 vs tree_sbi_lam$KL_HYP)"
    $CONVERT_PY scripts/feature_ceiling.py --input "$FEATURES" \
        --tree-ref tree_sbi_lam1 --tree-hyp "tree_sbi_lam$KL_HYP" \
        --out-prefix "${CEILING_JSON%.json}" ${CEILING_OPTS:-}
    # quick pass: CEILING_OPTS="--max-events 100000 --seeds 2 --max-iter 100"
}

stage_ceiling_array() {   # one shared-QOS CPU task per size tier + dependent merge job
    [ -f "$FEATURES" ] || die "FEATURES=$FEATURES not found (feature ntuple)"
    [ -n "${ACCOUNT:-}" ] || die "set ACCOUNT=<project> for sbatch"
    mkdir -p "$STORE"
    local prefix="${CEILING_JSON%.json}"
    cat > "$STORE/ceiling-array.sbatch" <<EOF
#!/bin/bash
#SBATCH -A $ACCOUNT -q shared -C cpu -c 32 -t ${TIME:-02:00:00}
#SBATCH --array=0-5
#SBATCH -o $STORE/ceiling-%A_%a.out
export PATH="\$HOME/.pixi/bin:\$PATH"
SIZES=(0.003 0.01 0.02 0.05 0.1 1.0)
cd $HERE
$CONVERT_PY scripts/feature_ceiling.py --input "$FEATURES" \\
    --tree-ref tree_sbi_lam1 --tree-hyp "tree_sbi_lam$KL_HYP" \\
    --sizes \${SIZES[\$SLURM_ARRAY_TASK_ID]} \\
    --out-prefix "$prefix.part\$SLURM_ARRAY_TASK_ID" ${CEILING_OPTS:-}
EOF
    local jid
    jid=$(sbatch --parsable "$STORE/ceiling-array.sbatch")
    sbatch -A "$ACCOUNT" -q shared -C cpu -c 1 -t 00:10:00 \
        --dependency=afterok:"$jid" -o "$STORE/ceiling-merge-%j.out" \
        --wrap "export PATH=\$HOME/.pixi/bin:\$PATH; cd $HERE && $CONVERT_PY scripts/feature_ceiling.py --merge '$prefix'"
    note "submitted array $jid (6 tiers) + merge job; result -> $CEILING_JSON; watch: squeue --me"
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
    mkdir -p "$NPZ"
    cd "$HERE"
    for kl in 0 1 5; do
        cls=1; [ "$kl" = "1" ] && cls=0            # kl=1 is the SM reference class
        note "convert kl=$kl (class $cls, $INPUT_FORMAT, tau-encoding $TAU_ENCODING)"
        if [ "$INPUT_FORMAT" = "crown" ]; then
            [ -d "$NTUPLES" ] || die "NTUPLES=$NTUPLES not found (CROWN base dir \
containing GluGluHHto2B2Tau_*kl-*/{mt,et}/*.root; default = the KIT location)"
            klp="${kl}p00"
            $CONVERT_PY scripts/crown_to_evenet_npz.py \
                --input "$NTUPLES/GluGluHHto2B2Tau_Par-c2-0p00-kl-$klp-kt-1p00_*PowhegBugFix*/mt/*.root" \
                        "$NTUPLES/GluGluHHto2B2Tau_Par-c2-0p00-kl-$klp-kt-1p00_*PowhegBugFix*/et/*.root" \
                --class-id $cls --tau-encoding "$TAU_ENCODING" --output "$NPZ/kl$kl.npz"
        else
            [ -n "${NANO:-}" ] || die "set NANO=/path/to/nanoaod (kl0/ kl1/ kl5/ subdirs)"
            [ -n "${BTAG_WP:-}" ] || die "set BTAG_WP (era WP for $BTAG_BRANCH from the BTV tables)"
            $CONVERT_PY scripts/nanoaod_to_evenet_npz.py --input "$NANO/kl$kl/*.root" \
                --class-id $cls --btag-wp "$BTAG_WP" --btag-branch "$BTAG_BRANCH" \
                --tau-encoding "$TAU_ENCODING" --output "$NPZ/kl$kl.npz"
        fi
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
    # merge the pair into ONE shuffled NPZ: upstream preprocess assumes every file is
    # class-mixed (bincount crash on single-class), and the shuffle prevents the
    # mt-then-et file ordering from leaking channel composition into the test split
    local merged="$NPZ/merged-$TASK-kl$KL_HYP.npz"
    $CONVERT_PY scripts/merge_npz.py --inputs $files --output "$merged"
    note "preprocess $merged -> $STORE/evenet-train ($TASK task)"
    cd "$EVENET_SRC"
    shifter --image="$IMAGE" env PYTHONPATH="$EVENET_SRC" python3 preprocessing/preprocess.py \
        --files "$merged" --split_ratio 0.8,0.1,0.1 \
        --store_dir "$STORE/evenet-train" --config "$HERE/configs/preprocess_klambda.yaml"
    # EveNet's loader globs *.parquet per dir: test.parquet MUST NOT sit next to train/val
    # (training would ingest it; predict on the combined dir would see train events).
    [ -f "$STORE/evenet-train/test.parquet" ] || die "test.parquet not produced — check the log"
    mkdir -p "$STORE/evenet-test"
    mv "$STORE/evenet-train/test.parquet" "$STORE/evenet-test/test.parquet"
    [ -f "$STORE/evenet-train/shape_metadata.json" ] && \
        cp "$STORE/evenet-train/shape_metadata.json" "$STORE/evenet-test/"
    note "store layout:"; ls -la "$STORE/evenet-train" "$STORE/evenet-test"
}

stage_configs() {
    cd "$HERE"
    # everything the resolved workflow will reference must exist (vendored in configs/)
    for f in configs/event_info_klambda.yaml configs/resonance_klambda.yaml \
             configs/predict_klambda.yaml configs/options_finetune.yaml \
             configs/options_frozen.yaml configs/options_scratch.yaml \
             configs/network_20M.yaml; do
        [ -f "$f" ] || die "missing $f (vendored config; git pull?)"
    done
    [ -d "$EVENET_SRC" ] || die "EVENET_SRC=$EVENET_SRC missing (run: setup)"
    # resolve the templates: <PLACEHOLDER> tokens -> this run's env (no hand-editing)
    sed -e "s|<PATH_TO>/EveNet-Full/share/options/options_pretrain.yaml|$HERE/configs/options_finetune.yaml|" \
        -e "s|<PATH_TO>/EveNet-Full/share/options/options.yaml|$HERE/configs/options_scratch.yaml|" \
        -e "s|<PATH_TO>/options_frozen.yaml|$HERE/configs/options_frozen.yaml|" \
        -e "s|<PATH_TO>/Exotic-Higgs-Study/configs/network.yaml|$HERE/configs/network_20M.yaml|" \
        -e "s|<PATH_TO>/Exotic-Higgs-Study/configs/resonance.yaml|$HERE/configs/resonance_klambda.yaml|" \
        -e "s|<PATH_TO>/EveNet-Full|$EVENET_SRC|" \
        -e "s|<NERSC_ACCOUNT>|${ACCOUNT:-m5295_g}|" \
        -e "s|^predict_yaml: null.*|predict_yaml: $HERE/configs/predict_klambda.$RESOLVED.yaml|" \
        -e "s|^train_yaml: train_klambda.yaml|train_yaml: train_klambda.$RESOLVED.yaml|" \
        -e "s|<STORE>|$STORE|g" \
        ${SIZES_YAML:+-e "s|^dataset_size_choice: .*|dataset_size_choice: $SIZES_YAML|"} \
        ${SEEDS_YAML:+-e "s|^seeds: .*|seeds: $SEEDS_YAML|"} \
        configs/workflow_klambda.yaml > configs/workflow_klambda.$RESOLVED.yaml
    # the ONLY knob the nonoise variant flips: classification trained on clean inputs, to
    # match the t=0 endpoint used at predict time (see EVENET_G0_RESULTS sec 7)
    sed -e "s|<STORE>|$STORE|g" -e "s|<WANDB_ENTITY>|${WANDB_ENTITY:-$USER}|" \
        ${VARIANT:+-e "s|noise_prob: *\[1.0, 1.0\]|noise_prob:     [0.0, 0.0]|"} \
        configs/train_klambda.yaml > configs/train_klambda.$RESOLVED.yaml
    sed -e "s|<STORE>|$STORE|g" \
        configs/predict_klambda.yaml > configs/predict_klambda.$RESOLVED.yaml
    ! grep -qE "<(STORE|PATH_TO|NERSC|WANDB)" configs/*.$RESOLVED.yaml \
        || die "unresolved <tokens> remain in configs/*.$RESOLVED.yaml"
    if [ -n "${SEEDS_YAML:-}" ] && [ "$FARM" = "$HERE/config_farm-$TASK-$TAU_ENCODING$SFX" ]; then
        die "SEEDS_YAML is set but FARM is the baseline farm -- a seed extension must go to a fresh dir (e.g. FARM=$FARM-ext) or it overwrites the baseline run scripts"
    fi
    $CONVERT_PY scripts/make_klambda_configs.py configs/workflow_klambda.$RESOLVED.yaml \
        --farm "$FARM" --store_dir "$STORE_RUN" --data_store_dir "$STORE" \
        --ray_dir "${PSCRATCH:-/tmp}/ray-$USER"
    [ -f "$FARM/predict-evenet.sh" ] || die "predict-evenet.sh not emitted — check predict_yaml"
    if [ -n "$VARIANT" ]; then
        note "VARIANT=$VARIANT -> $STORE_RUN (data read from $STORE)"
        note "VERIFY THE FLIPPED KNOB IS THE ONE IN EFFECT before submitting (train_klambda.yaml"
        note "and the options_*.yaml files both set noise_prob, with different values):"
        note "  cd $EVENET_SRC && shifter --image=$IMAGE env PYTHONPATH=\$PWD python3 -c \\"
        note "    \"from evenet.control.global_config import global_config as g; \\"
        note "     g.load_yaml('$FARM/evenet-klambda-finetune-size1.0-seed0.yaml'); \\"
        note "     print(g.options.Training.ProgressiveTraining.stages[0].train_parameters.noise_prob)\""
        note "  expect [0.0, 0.0] -- anything else means the baseline sweep ran a different"
        note "  recipe than assumed, which is itself a finding. STOP and report."
    fi
    note "LOAD-TEST one config inside the shifter image before submitting the array:"
    note "  shifter --image=$IMAGE python3 -c \"import yaml; yaml.safe_load(open('$FARM/\$(ls $FARM | head -1)'))\" && echo yaml-ok"
}

stage_train() {
    [ -f "$FARM/train-evenet.sh" ] || die "run: configs"
    [ -n "${ACCOUNT:-}" ] || die "set ACCOUNT=<mXXXX> for sbatch (or use: train-local)"
    local n; n=$(wc -l < "$FARM/train-evenet.sh")
    # a run script whose tags already have checkpoints is almost always the WRONG farm
    # (e.g. a seed extension submitted without FARM= pointing at the extension dir);
    # resubmitting retrains into the same checkpoint dirs and contaminates the
    # best-val selection of already-published runs
    local n_exist=0 tag
    while IFS= read -r line; do
        tag=$(grep -o "evenet-klambda-[a-z]*-size[0-9.]*-seed[0-9]*" <<< "$line" | head -1)
        [ -n "$tag" ] && [ -d "$STORE_RUN/checkpoints/$tag" ] && n_exist=$((n_exist + 1))
    done < "$FARM/train-evenet.sh"
    if [ "$n_exist" -gt 0 ] && [ -z "${RETRAIN:-}" ]; then
        die "$n_exist of $n runs in $FARM already have checkpoints under $STORE_RUN/checkpoints -- wrong FARM? (a seed extension needs FARM=<extension dir> on EVERY stage). RETRAIN=1 to really retrain."
    fi
    cat > "$FARM/train-array.sbatch" <<EOF
#!/bin/bash
#SBATCH -A $ACCOUNT -C gpu -q ${QOS:-regular} -t ${TIME:-04:00:00}
#SBATCH -N 1 --gpus-per-task=${NGPU:-1} --ntasks=1 -c 32
#SBATCH --array=1-$n%16
#SBATCH -o $FARM/slurm-%A_%a.out
export WANDB_API_KEY=\${WANDB_API_KEY:-dummy} WANDB_MODE=\${WANDB_MODE:-offline}
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
    [ -f "$FARM/predict-evenet.sh" ] || die "no predict-evenet.sh (run: configs)"
    [ -n "${ACCOUNT:-}" ] || die "set ACCOUNT=<mXXXX_g> for sbatch (or use: predict-local on a GPU node)"
    if [ -z "${PREDICT_ANYWAY:-}" ] && command -v squeue >/dev/null 2>&1 \
       && squeue --me -h -o %j 2>/dev/null | grep -q "train-array"; then
        die "a train-array job is still queued/running -- best-val selection would snapshot partial runs (PREDICT_ANYWAY=1 to override)"
    fi
    # pre-registered selection rule: predict loads the newest ckpt (mtime), so mark the
    # BEST-val checkpoint newest in every run dir (identical rule for all arms)
    $CONVERT_PY scripts/select_best_ckpt.py --store "$STORE_RUN"
    local n; n=$(wc -l < "$FARM/predict-evenet.sh")
    cat > "$FARM/predict-array.sbatch" <<EOF
#!/bin/bash
#SBATCH -A $ACCOUNT -C gpu -q shared -t ${PTIME:-00:30:00}
#SBATCH -N 1 --gpus-per-task=1 --ntasks=1 -c 32
#SBATCH --array=1-$n%32
#SBATCH -o $FARM/predict-%A_%a.out
export WANDB_API_KEY=\${WANDB_API_KEY:-dummy} WANDB_MODE=\${WANDB_MODE:-offline}
eval "\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" $FARM/predict-evenet.sh)"
EOF
    sbatch "$FARM/predict-array.sbatch"
    note "submitted $n-task predict array (shared GPU QOS); watch: squeue --me"
}

stage_predict_local() {   # sequential, for an interactive GPU node (salloc)
    [ -f "$FARM/predict-evenet.sh" ] || die "run: configs"
    bash "$FARM/predict-evenet.sh"
}

stage_eval() {
    cd "$HERE"
    [ -n "${CEILING:-}" ] || { [ -f "$CEILING_JSON" ] && CEILING="$CEILING_JSON"; }
    note "G0: AUC money plot (ceiling: ${CEILING:-none — run: ceiling})"
    $CONVERT_PY scripts/plot_data_efficiency.py --store_dir "$STORE_RUN" \
        --output "data_efficiency-$TASK-$TAU_ENCODING$SFX.png" \
        ${CEILING:+--ceiling "$CEILING"}
    if [ -n "${NORM_PT:-}" ]; then
        [ -f "$NORM_PT" ] || die "NORM_PT=$NORM_PT not found"
    elif [ -f "$STORE/evenet-train/normalization.pt" ]; then
        NORM_PT="$STORE/evenet-train/normalization.pt"
    else
        NORM_PT=""                      # balanced-prior fallback if preprocess absent
    fi
    note "G0.5: ratio-closure gates (|IC| vs stat, SC_rms < 0.05, chi2/ndf ~ 1)"
    $CONVERT_PY scripts/eval_closure.py --store_dir "$STORE_RUN" \
        --output-prefix "closure-$TASK-$TAU_ENCODING$SFX" \
        ${NORM_PT:+--normalization "$NORM_PT"}
    if [ -n "$NORM_PT" ]; then
        note "G0.5 ensemble: trained-prior closure + Poisson(1) test-set bootstrap"
        $CONVERT_PY scripts/eval_ensemble.py --store_dir "$STORE_RUN" \
            --normalization "$NORM_PT" --bootstrap "${NBOOT:-200}" \
            --output-prefix "ensemble-$TASK-$TAU_ENCODING$SFX"
    else
        note "no normalization.pt under $STORE/evenet-train -- skipped ensemble eval"
    fi
}

case "${1:-}" in
    smoke)       stage_smoke ;;
    setup)       stage_setup ;;
    ceiling)     stage_ceiling ;;
    ceiling-array) stage_ceiling_array ;;
    check)       stage_check "${2:-}" ;;
    convert)     stage_convert ;;
    preprocess)  stage_preprocess ;;
    configs)     stage_configs ;;
    train)       stage_train ;;
    train-local) stage_train_local ;;
    predict)     stage_predict ;;
    predict-local) stage_predict_local ;;
    eval)        stage_eval ;;
    *) awk 'NR>1 { if ($0 !~ /^#/) exit; print }' "$0"; exit 1 ;;
esac
