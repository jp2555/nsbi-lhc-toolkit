# HANDOFF — EveNet Option A: κ_λ data-efficiency test on CMS bbττ

**For:** a fresh session continuing on **Perlmutter**. **Repo/branch:** `jp2555/nsbi-lhc-toolkit @ fm`,
code in `examples/HH_bbtautau_kappalambda_sophon/`. **Date:** 2026-07-13.
Supersedes the 2026-06-26 handoff (its still-valid content is folded in below).

## Mission (one paragraph)
Money plots showing a pretrained event-level FM (**EveNet**, arXiv:2601.17126) improves training in
the **low-statistics** regime — especially systematic variations — for κ_λ in HH→bbττ. Metric =
**data efficiency** (AUC *and ratio closure* vs training fraction), bar = the kinematic ceiling.
**No σ_κλ significance** (MC-stat/wifi deferred). Decision gates G0/G0.5/G1 with kill criteria:
`EVENET_FEASIBILITY_NOTE.pdf` §8; runbook: `EVENET_INTEGRATION_PLAN.md` §10; driver: `run_option_a.sh`.

## Where we are (2026-07-11)
- ✅ Scratch smoke pipeline end-to-end (2026-06-27, `5485efb`): NPZ→preprocess→train→predict→plot,
  prediction key `classification/klambda` **confirmed**; NPZ schema hardened (18 tokens,
  `x_mask/conditions_mask/num_vectors/subprocess_id`); `predict_klambda.yaml` + `resonance_klambda.yaml` added.
- ✅ Git recovery done: diverged Perlmutter clone merged (cherry-pick), **LFS pointer-flap fixed**
  (`.gitattributes` exempts `docs/teaching/*.png`), run artifacts gitignored.
- ✅ Feasibility analysis + Option A harness built and locally verified (see file table).
- ✅ **Ceiling measured at full statistics** (2026-07-12, 6-task sbatch array on m5295, merged to
  `$STORE/ceiling-kl5.json`): AUC 0.792 (1%) → 0.817 (100%), saturates by 10%; closure gates first
  pass at the 10% fraction; method-limited |IC| floor ≈ 1.7% at ≥30%. Decision anchors set.
- ✅ **Sweep machinery complete** (2026-07-13): fraction grid re-cut to
  `[0.003, 0.01, 0.02, 0.05, 0.1, 1.0]` (ceiling showed 0.3≈1.0; low side refined; 0.003 ≈
  1.4k/1.1k events/class = the EveNet-paper regime) → **90 configs** (3 arms × 6 × 5 seeds).
  Options files VENDORED into `configs/` (`options_{finetune,frozen,scratch}.yaml` from the
  Exotic-Higgs recipes, frozen = backbone GlobalEmbedding/PET/ObjectEncoder freeze.type=full,
  heads trainable; + `network_20M.yaml`). `configs` stage now AUTO-RESOLVES every `<PLACEHOLDER>`
  from env (writes `configs/*.resolved.yaml`, gitignored) — no hand-editing. Config generation
  verified end-to-end locally (90 train + 90 predict, per-arm assertions).
- ⏭ **next: `convert`** (blocked ONLY on NTUPLES location — open item 1) → `preprocess` →
  `configs` → `train` (ACCOUNT=m5295_g) → `predict` → `eval` (prints the adoption verdict).

## New since 2026-06-27 (all verified locally where stated)
| File | Role | Verified |
|---|---|---|
| `EVENET_FEASIBILITY_NOTE.{tex,pdf}` | 12pp: τ/γ 3-tier gap, ratio-asymmetry argument, corner encoding, options A/B/C, gates G0–G4, open questions Q1–Q10 | compiles clean |
| `scripts/crown_to_evenet_npz.py` | **DEFAULT sweep input.** CROWN mt/et ntuples → 4-token cloud (b1,b2,τ_h,ℓ) + MET globals; genWeight×puweight; sentinel filtering | ✅ vs fabricated CROWN tree |
| `scripts/nanoaod_to_evenet_npz.py` | raw NanoAOD v15 path (`INPUT_FORMAT=nanoaod`); needs `--btag-wp` (BTV) | ✅ vs fabricated NanoAOD tree incl. fallbacks |
| `scripts/delphes_to_evenet_npz.py` | + `--tau-encoding corner` (G1); Delphes cross-check path | ✅ corner/anon kinematics identical |
| `scripts/eval_closure.py` | **G0.5**: integral closure ±stat err, stat-debiased shape RMS, χ²/ndf from `prediction.pt` | ✅ analytic self-test, multi-seed/size |
| `scripts/feature_ceiling.py` | ceiling from the 12-feature file (`dihiggs_powheg_data.root`, lowercase) | ✅ on the real file |
| `run_option_a.sh` | staged driver: smoke/setup/ceiling/check/convert/preprocess/configs/train/predict/eval | ✅ smoke+ceiling stages |
| `scripts/inject_systematic.py` | + `subprocess_id` sync (was stale vs new schema → would mislabel predictions) | ✅ on 18-token schema |

Ceiling result on the real file (kl1-vs-kl5, full stats): **AUC ≈ 0.816**; at 3% training fraction
AUC barely drops but **calibration collapses** (|IC| 0.34, SC_rms 0.50 → 0.008/0.065 at 30%) —
the G0.5 phenomenon in miniature: low-stat failure is calibration, not ordering.

## Decision log (settled — do not re-open)
- **Input = CROWN analysis ntuples** (supersedes "Delphes first"): the files
  `NSBI-pheno/dihiggs_bbtautau/convert_powheg_to_sbi.py` reads
  (`<base>/GluGluHHto2B2Tau_*kl-{0p00,1p00,2p45,5p00}*/{mt,et}/*.root`, tree `ntuple`).
  They carry per-object four-vectors + MET + charges; the flat 12-feature file is an
  **irreversible aggregate** (no MET/charges/object splits) → **ceiling only**. ttbar shares the contract.
- **Pairing:** class 0 = κλ=1 (SM ref), class 1 = `KL_HYP` (default 5; 0 available).
- **τ encodings:** `anonymous` (stock, G0 default); `corner` (0,0,±1) = G1 A/B (one env var;
  identical kinematics/conditions by construction; injector treats both as jets).
- **Closure gates:** |IC| < max(0.01, ~3×stat), SC_rms < 0.05 (=`CLOSURE_TOL`), χ²/ndf≈1 ⇒ noise-only.
  Compare configs at the FIXED shared test split; seed spread = noise floor. Do NOT gate on raw max-bin.
- Carried: binary head + LR trick; 3-way contrast via options files; wifi/MC-stat deferred;
  bbγγ = later increment (photon corner (0,1,0)); G0 adopt-iff finetune > scratch beyond compute
  penalty AND ≥ ceiling; a G0 null with scratch passing closure ⇒ objective-mismatch verdict
  (also condemns a like-for-like new τ/γ FM — feasibility note §8).
- **Decision primacy (2026-07-12, fixed pre-unblinding): closure left-shift is the win metric.**
  Finetune passing both gates at a smaller fraction than scratch (e.g. 3% vs 30% = 10×
  equivalent-data multiplier) = adoption-grade win even at AUC parity; AUC-only win with failing
  closure = not a win. `eval_closure.py` prints the first-passing fractions + multiplier.
  Ceiling anchors (full stats, kl1-vs-kl5): AUC 0.792→0.817; gates first pass at the 10%
  fraction; method-limited |IC| floor ≈ 1.7%.

## Env setup (Perlmutter)
```bash
pixi install -e nsbi-env                            # one-time, CPU env (uproot/sklearn/yaml/mpl/torch)
export CONVERT_PY="pixi run -e nsbi-env python"     # per shell — covers ALL CPU stages of the driver
```
| Stages | Env |
|---|---|
| smoke, ceiling, convert, configs, eval | pixi `nsbi-env` via `$CONVERT_PY` (login node OK) |
| preprocess, train, predict | shifter `avencast1994/evenet:1.5` (driver invokes it) |
| setup | plain shell (`git clone` + `hf download Avencast/EveNet`) |

Driver env knobs: `STORE` (default `$PSCRATCH/evenet-klambda`), `NTUPLES` (CROWN base, required for
convert), `KL_HYP=0|5`, `TAU_ENCODING=corner`, `TASK=syst` (Money Plot 2), `ACCOUNT` (sbatch), `NGPU`, `TIME`.

## Open items (priority order)
1. **NTUPLES: RESOLVED 2026-07-13** — CROWN ntuples live at KIT:
   `/ceph/jpan/saved_datasets/ntuple_bbtt_24` (now the driver default). Plan: run `convert` at
   KIT (both encodings), rsync the NPZs to `$PSCRATCH/evenet-klambda/npz-*/` on Perlmutter via
   dtn01.nersc.gov, continue there from `preprocess`.
2. **Load-test ONE config inside the shifter image before the 90-job array** (schema drift between
   the vendored options/network yamls and the installed EveNet is caught exactly there); verify
   EveNet reads `options.Training.seed`. The `configs` stage prints the load-test command.
3. Ceiling for `KL_HYP=0` too if running the kl0 pair (`KL_HYP=0 ./run_option_a.sh ceiling-array`);
   also worth re-running `ceiling-array` once on the NEW grid so overlay x-points match the sweep.

## Conversion record (2026-07-13, KIT: /ceph/jpan/saved_datasets/ntuple_bbtt_24)
NPZs produced at /ceph/jpan/evenet-klambda/npz-{anonymous,corner}/ and shipped to
$PSCRATCH/evenet-klambda/. Selected events (identical across encodings): kl0 438k, kl1 488k,
kl5 339k; negative weights 3.9/6.1/1.0%; all optional branches (q_1/q_2, met/metphi,
genWeight, puweight) resolved. **Sentinel drops 18.6–24.5%, κλ-dependent — verified to be
the valid-b-pair requirement** (per-branch census: pt_1/pt_2 0% bad, bpair_pt_1 14.2%,
bpair_pt_2 25.4% in kl5/mt; softer κλ=5 spectrum → more failed second b). Population is
therefore "events with a valid H→bb candidate" — correct phase space; NOTE the ceiling was
measured on the flat file WITHOUT this cut (kept as NaN features), so the sweep population
is slightly easier: a marginal AUC-over-ceiling result carries this caveat; the closure
left-shift metric (arms-internal) is immune.

## Gotchas (hard-won — read before committing/debugging)
- **git-lfs**: the Mac clone has NO git-lfs. `.gitattributes` LFS-tracks `*.root/*.npy/*.h5/*.onnx/*.png`
  (exempted: `docs/_images/`, `docs/teaching/` pngs). Committing matching binaries from the Mac
  recreates the pointer-flap that blocked pulls for a day. Perlmutter-side commits are safe.
- `deta_hh` in the feature file carries **inf sentinels (~10%)** (handled in `feature_ceiling.py`;
  already excluded from the NSBI TrainingFeatures upstream).
- Weights: ceiling trains AND evaluates with |w| class-normalised (AUC invariant; closure then
  measures calibration, not the κλ cross-section ratio W0/W1≈0.35). EveNet uses stored signed weights.
- Local-smoke quirks (sandbox only): `WANDB_API_KEY=dummy WANDB_MODE=offline`; wandb needs
  `sentry_sdk gitpython`; run from `EveNet-Full/` with `PYTHONPATH` set.
- Commits: brief (6–10 words), **no Claude attribution**, push `origin fm` after each change.

## Pointers
- Feasibility/gates: `EVENET_FEASIBILITY_NOTE.pdf` · runbook: `EVENET_INTEGRATION_PLAN.md` §10 ·
  landscape: `FOUNDATION_MODEL_BRIEF.md`.
- Prelim physics: Ghosh–Klute–Pan note (`NSBI-pheno/dihiggs_bbtautau`); feature file
  `/pscratch/sd/j/jing/NSBI-irishep/dihiggs_bbtautau/dihiggs_powheg_data.root` (+ `ttbar_powheg_data.root`).
- EveNet: code `github.com/UW-EPE-ML/EveNet_Public`, ckpts `hf.co/Avencast/EveNet`,
  image `avencast1994/evenet:1.5`.

## Suggested skills
- `superpowers:verification-before-completion` — evidence before claiming any EveNet step works.
- `superpowers:systematic-debugging` — for the first config-load/preprocess failure (expected at open item 4).
- `remember` — save state at session end.
