"""Delphes ROOT (HH->bbtautau) -> EveNet NPZ (high-level event-object schema).

EveNet ingests HIGH-LEVEL objects, not constituents. This adapter emits the exact
NPZ contract its preprocessor expects (EveNet-Full/preprocessing/helper.py):

  x                       (N, N_OBJ_MAX, 7)  Source objects, columns in event_info order:
                                             [energy, pt, eta, phi, btag, isLepton, charge]
  conditions              (N, 10)            Globals, columns in event_info order:
                                             [met, met_phi, nLepton, nbJet, nJet, HT, HT_lep,
                                              M_all, M_leps, M_bjets]
  num_sequential_vectors  (N,)  int          number of real objects per event (padding/mask)
  event_weight            (N,)  float        Event.Weight (NLO sign preserved)
  classification          (N,)  int          class id (set per-file via --class-id;
                                             binary kappa_lambda task: 0=reference, 1=hypothesis)

Then on Perlmutter (shifter docker:avencast1994/evenet:1.5):
  python EveNet-Full/preprocessing/preprocess.py --files *.npz --split_ratio 0.8,0.1,0.1 \
         --store_dir <out> --config <global_klambda.yaml>

bb->tautau object mapping (STOCK schema -> preserves pretrained weights; see EVENET_INTEGRATION_PLAN.md sec 3):
  * b-jets  -> Source object, btag=Jet.BTag, isLepton=0, charge=0
  * tau_h   -> Source object as a jet (btag from Jet.BTag ~0, isLepton=0). Tau-ID is LOST
              (EveNet has no tau type); kinematics are exact. Documented limitation, not silent.
  * e / mu  -> Source object, isLepton=1, charge=+-1
  * MET     -> conditions[met, met_phi]

CONSTRAINT enforced by EveNet's preprocessor: log-scaled columns (energy, pt, met, HT, HT_lep,
M_all, M_leps, M_bjets) must be NON-NEGATIVE and finite. The builders below guarantee this.

Local-testable smoke mode (no ROOT/uproot needed): `--smoke 256` fabricates events so the NPZ
schema can be validated. The real path uses uproot (imported lazily) and the verified Delphes
branch names from delphes_to_sophon_clouds.py.
"""
import argparse
import numpy as np

N_OBJ_MAX = 16
SOURCE_FEATURES = ["energy", "pt", "eta", "phi", "btag", "isLepton", "charge"]
CONDITION_FEATURES = ["met", "met_phi", "nLepton", "nbJet", "nJet",
                      "HT", "HT_lep", "M_all", "M_leps", "M_bjets"]
# Columns the EveNet preprocessor will log1p-scale -> must stay >= 0.
_NONNEG_COND = [0, 5, 6, 7, 8, 9]   # met, HT, HT_lep, M_all, M_leps, M_bjets
_EPS = 1e-9


def _inv_mass(E, px, py, pz):
    m2 = E * E - (px * px + py * py + pz * pz)
    return np.sqrt(np.clip(m2, 0.0, None))


def build_conditions(x, mask, met, met_phi):
    """Compute the 10 globals from the object array + MET. Vectorised over events.

    x: (N, M, 7) in SOURCE_FEATURES order; mask: (N, M) bool; met, met_phi: (N,).
    """
    energy, pt, eta, phi = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    btag, is_lep_f = x[..., 4], x[..., 5]
    m = mask.astype(bool)
    is_jet = m & (is_lep_f < 0.5)
    is_lep = m & (is_lep_f >= 0.5)
    is_bjet = is_jet & (btag >= 0.5)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    def grp_mass(sel):
        E = np.where(sel, energy, 0.0).sum(1)
        sx = np.where(sel, px, 0.0).sum(1)
        sy = np.where(sel, py, 0.0).sum(1)
        sz = np.where(sel, pz, 0.0).sum(1)
        return _inv_mass(E, sx, sy, sz)

    cond = np.stack([
        np.clip(met.astype(np.float64), 0.0, None),
        met_phi.astype(np.float64),
        is_lep.sum(1).astype(np.float64),                       # nLepton
        is_bjet.sum(1).astype(np.float64),                      # nbJet
        is_jet.sum(1).astype(np.float64),                       # nJet
        np.where(is_jet, pt, 0.0).sum(1),                       # HT
        np.where(is_lep, pt, 0.0).sum(1),                       # HT_lep
        grp_mass(m),                                            # M_all
        grp_mass(is_lep),                                       # M_leps
        grp_mass(is_bjet),                                      # M_bjets
    ], axis=1).astype(np.float32)
    return cond


def _pack_event(objs):
    """objs: list of (energy, pt, eta, phi, btag, isLepton, charge), pt-sorted within caller.
    Returns padded (N_OBJ_MAX, 7) float32 row and the count of real objects."""
    row = np.zeros((N_OBJ_MAX, 7), dtype=np.float32)
    n = min(len(objs), N_OBJ_MAX)
    if n:
        row[:n] = np.asarray(objs[:n], dtype=np.float32)
    return row, n


def _finalize(x, counts, met, met_phi, weight, class_id):
    x = np.asarray(x, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int64)
    mask = np.arange(N_OBJ_MAX)[None, :] < counts[:, None]
    # energy, pt must be >= 0 for the log-scaler.
    x[..., 0] = np.clip(x[..., 0], 0.0, None)
    x[..., 1] = np.clip(x[..., 1], 0.0, None)
    conditions = build_conditions(x, mask, np.asarray(met), np.asarray(met_phi))
    return {
        "x": x,
        "conditions": conditions,
        "num_sequential_vectors": counts,
        "event_weight": np.asarray(weight, dtype=np.float32),
        "classification": np.full(len(x), int(class_id), dtype=np.int64),
    }


# --------------------------------------------------------------------------- real Delphes path
_BRANCHES = [
    "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass", "Jet.BTag",
    "Electron.PT", "Electron.Eta", "Electron.Phi", "Electron.Charge",
    "Muon.PT", "Muon.Eta", "Muon.Phi", "Muon.Charge",
    "MissingET.MET", "MissingET.Phi", "Event.Weight",
]


def _get(arr, name, i, default=None):
    if name not in arr.fields:
        return default
    import awkward as ak
    return ak.to_numpy(arr[name][i])


def _scalar(x):
    try:
        return float(x[0])
    except (TypeError, IndexError, KeyError):
        return float(x)


def _event_objects(arr, i):
    """Build the pt-sorted object list for event i: jets then leptons."""
    objs = []
    jpt = _get(arr, "Jet.PT", i, np.zeros(0))
    if jpt is not None and len(jpt):
        jeta = _get(arr, "Jet.Eta", i); jphi = _get(arr, "Jet.Phi", i)
        jm = _get(arr, "Jet.Mass", i, np.zeros_like(jpt))
        jb = _get(arr, "Jet.BTag", i, np.zeros_like(jpt))
        jE = np.sqrt((jpt * np.cosh(jeta)) ** 2 + np.clip(jm, 0, None) ** 2)
        order = np.argsort(-jpt)
        for k in order:
            objs.append((jE[k], jpt[k], jeta[k], jphi[k],
                         1.0 if jb[k] > 0.5 else 0.0, 0.0, 0.0))
    leps = []
    for coll in ("Electron", "Muon"):
        pt = _get(arr, f"{coll}.PT", i)
        if pt is None or not len(pt):
            continue
        eta = _get(arr, f"{coll}.Eta", i); phi = _get(arr, f"{coll}.Phi", i)
        q = _get(arr, f"{coll}.Charge", i, np.zeros_like(pt))
        E = pt * np.cosh(eta)
        for k in range(len(pt)):
            leps.append((E[k], pt[k], eta[k], phi[k], 0.0, 1.0, float(q[k])))
    leps.sort(key=lambda o: -o[1])
    objs.extend(leps)
    return objs


def convert_delphes(path, tree, class_id, step_size=20000):
    import uproot
    xs, counts, mets, mphis, weights = [], [], [], [], []
    for arr in uproot.iterate(f"{path}:{tree}", step_size=step_size, library="ak",
                              filter_name=_BRANCHES):
        if "Jet.PT" not in arr.fields:
            raise KeyError(f"'Jet.PT' missing among {list(arr.fields)}; check Delphes leaf names.")
        n = len(arr["Jet.PT"])
        for i in range(n):
            row, c = _pack_event(_event_objects(arr, i))
            xs.append(row); counts.append(c)
            met = _get(arr, "MissingET.MET", i); mphi = _get(arr, "MissingET.Phi", i)
            mets.append(_scalar(met) if met is not None else 0.0)
            mphis.append(_scalar(mphi) if mphi is not None else 0.0)
            w = arr["Event.Weight"][i] if "Event.Weight" in arr.fields else 1.0
            weights.append(_scalar(w))
    return _finalize(np.stack(xs), counts, mets, mphis, weights, class_id)


# --------------------------------------------------------------------------- smoke (no uproot)
def make_smoke(n_events, class_id, seed=0):
    rng = np.random.default_rng(seed)
    xs, counts, mets, mphis, weights = [], [], [], [], []
    for _ in range(n_events):
        njet = int(rng.integers(2, 7)); nlep = int(rng.integers(0, 3))
        objs = []
        for j in range(njet):
            pt = float(rng.uniform(20, 300)); eta = float(rng.uniform(-2.5, 2.5))
            phi = float(rng.uniform(-np.pi, np.pi)); mass = float(rng.uniform(0, 25))
            E = np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2)
            btag = 1.0 if j < 2 else 0.0           # 2 b-jets per event
            objs.append((E, pt, eta, phi, btag, 0.0, 0.0))
        for _ in range(nlep):
            pt = float(rng.uniform(15, 150)); eta = float(rng.uniform(-2.5, 2.5))
            phi = float(rng.uniform(-np.pi, np.pi))
            objs.append((pt * np.cosh(eta), pt, eta, phi, 0.0, 1.0, float(rng.choice([-1, 1]))))
        objs.sort(key=lambda o: (o[5], -o[1]))     # jets first, then pt-desc
        row, c = _pack_event(objs)
        xs.append(row); counts.append(c)
        mets.append(float(rng.uniform(0, 200))); mphis.append(float(rng.uniform(-np.pi, np.pi)))
        weights.append(1.0)
    return _finalize(np.stack(xs), counts, mets, mphis, weights, class_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="Delphes ROOT path (real mode)")
    ap.add_argument("--tree", default="Delphes")
    ap.add_argument("--output", required=True, help="output .npz path")
    ap.add_argument("--class-id", type=int, required=True, help="0=reference, 1=hypothesis")
    ap.add_argument("--smoke", type=int, default=0, help="fabricate N synthetic events (no ROOT)")
    ap.add_argument("--step-size", type=int, default=20000)
    args = ap.parse_args()

    if args.smoke:
        out = make_smoke(args.smoke, args.class_id)
    else:
        if not args.input:
            ap.error("--input required unless --smoke is given")
        out = convert_delphes(args.input, args.tree, args.class_id, args.step_size)
    np.savez_compressed(args.output, **out)
    print(f"wrote {args.output}: " + ", ".join(f"{k}{v.shape}" for k, v in out.items()))


if __name__ == "__main__":
    main()
