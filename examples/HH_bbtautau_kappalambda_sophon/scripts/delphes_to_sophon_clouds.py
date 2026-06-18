"""Real-Delphes ROOT -> sophon-ak4 constituent clouds (HH->bbtautau).

Reads the full Delphes output written by ``delphes_card_CMS_hhbbtt_v0.tcl`` and emits,
per AK4 jet, a constituent cloud in **sophon-ak4's exact 17-feature schema** plus the
4 ``pf_vectors`` (px,py,pz,energy), so the pretrained sophon-ak4 encoder receives
schema-matched inputs. Supersedes the simplified ``delphes_to_clouds.py`` for the
sophon path.

Pipeline (streamed in chunks via ``uproot.iterate`` so memory is bounded):
  1. select AK4 jets (pt > JET_PT_MIN), keep the leading N_JETS_MAX;
  2. gather constituents from EFlowTrack (charged: hadrons/e/mu, with d0/dz),
     EFlowPhoton (photons) and EFlowNeutralHadron (neutral hadrons);
  3. associate each constituent to its NEAREST jet within DeltaR < JET_DR (=0.4);
  4. per jet, keep the leading N_PART_MAX constituents and compute the 17 features
     (with sophon-ak4's manual standardization) + the 4-vectors;
  5. add event object tokens for isolated electrons / muons / MET;
  6. write a compressed .npz.

Output arrays (leading axis = events):
  parts        (N, N_JETS_MAX, N_PART_MAX, 21)  [17 standardized pf_features | 4 pf_vectors
                                                 (px,py,pz,energy)]; encoder slices 0:17 / 17:21
  part_mask    (N, N_JETS_MAX, N_PART_MAX)      1 = real constituent
  jet_mask     (N, N_JETS_MAX)                  1 = real jet
  obj          (N, N_OBJ_MAX, 6)               [log_pt, eta, sin_phi, cos_phi, val, type_id]
  obj_mask     (N, N_OBJ_MAX)
  w            (N,)                              Event.Weight (NLO sign preserved)

Approximations to refine against the real files (documented, not silent):
  * constituent energy uses the massless form E = pt*cosh(eta) for all collections
    (Delphes EFlow towers are ~massless); if your files store Tower.E, swap it in.
  * branch names follow the card's TreeWriter; verify with `uproot.open(f)["Delphes"].keys()`.
"""
import argparse
import numpy as np
import awkward as ak
import uproot

N_JETS_MAX = 8
N_PART_MAX = 128          # sophon-ak4 trained with 128 constituents
F_PART = 17              # sophon-ak4 pf_features
F_TOTAL = F_PART + 4     # parts stores 17 features + 4 pf_vectors (px,py,pz,energy)
N_OBJ_MAX = 6
F_OBJ = 6
JET_DR = 0.4
JET_PT_MIN = 20.0
_EPS = 1e-9

# Read ONLY these leaves. Reading the whole Delphes tree makes uproot choke on TObject
# members (e.g. `Particle.fBits` -> "wrong number of bytes") and needlessly loads the huge
# Particle/Track/Tower collections. Optional leaves that are absent are simply ignored.
_BRANCHES = [
    "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass",
    "EFlowTrack.PT", "EFlowTrack.Eta", "EFlowTrack.Phi", "EFlowTrack.Charge", "EFlowTrack.PID",
    "EFlowTrack.D0", "EFlowTrack.DZ", "EFlowTrack.ErrorD0", "EFlowTrack.ErrorDZ",
    "EFlowPhoton.ET", "EFlowPhoton.Eta", "EFlowPhoton.Phi",
    "EFlowNeutralHadron.ET", "EFlowNeutralHadron.Eta", "EFlowNeutralHadron.Phi",
    "Electron.PT", "Electron.Eta", "Electron.Phi", "Electron.Charge",
    "Muon.PT", "Muon.Eta", "Muon.Phi", "Muon.Charge",
    "MissingET.MET", "MissingET.Phi",
    "Event.Weight",
]

# sophon-ak4 manual standardization: name -> (subtract, multiply, clip_min, clip_max).
# Features not listed use identity with clip [-5, 5] (matches the data config's defaults).
_STD = {
    "pt_log":   (1.7, 0.7, -5.0, 5.0),
    "e_log":    (2.0, 0.7, -5.0, 5.0),
    "logptrel": (-4.7, 0.7, -5.0, 5.0),
    "logerel":  (-4.7, 0.7, -5.0, 5.0),
    "deltaR":   (0.2, 4.0, -5.0, 5.0),
    "d0err":    (0.0, 1.0, 0.0, 1.0),
    "dzerr":    (0.0, 1.0, 0.0, 1.0),
}


def _std(name, x):
    sub, mul, lo, hi = _STD.get(name, (0.0, 1.0, -5.0, 5.0))
    return np.clip((np.asarray(x, dtype=np.float64) - sub) * mul, lo, hi)


def _dphi(a, b):
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


def _scalar(x):
    """Return a python float whether ``x`` is a bare scalar or a length>=1 array
    (Delphes per-event collections like Event.Weight/MissingET are length-1)."""
    try:
        return float(x[0])
    except (TypeError, IndexError, KeyError):
        return float(x)


def _get(arr, name, i, default=None):
    """Per-event branch access; returns a numpy array (or `default` if branch absent)."""
    if name not in arr.fields:
        return default
    return ak.to_numpy(arr[name][i])


def _collect_constituents(arr, i):
    """Combine the three EFlow collections for event i into per-constituent numpy arrays.

    Returns a dict of equal-length arrays: pt, eta, phi, energy, charge, d0, d0err,
    dz, dzerr, is_ch, is_nh, is_ph, is_e, is_mu (all float64).
    """
    cols = {k: [] for k in ("pt", "eta", "phi", "energy", "charge", "d0", "d0err",
                            "dz", "dzerr", "is_ch", "is_nh", "is_ph", "is_e", "is_mu")}

    # --- charged: EFlowTrack ---
    tpt = _get(arr, "EFlowTrack.PT", i)
    if tpt is not None and len(tpt):
        teta = _get(arr, "EFlowTrack.Eta", i); tphi = _get(arr, "EFlowTrack.Phi", i)
        tq = _get(arr, "EFlowTrack.Charge", i, np.zeros_like(tpt))
        pid = _get(arr, "EFlowTrack.PID", i, np.zeros_like(tpt))
        d0 = _get(arr, "EFlowTrack.D0", i, np.zeros_like(tpt))
        dz = _get(arr, "EFlowTrack.DZ", i, np.zeros_like(tpt))
        d0e = _get(arr, "EFlowTrack.ErrorD0", i, np.zeros_like(tpt))
        dze = _get(arr, "EFlowTrack.ErrorDZ", i, np.zeros_like(tpt))
        apid = np.abs(pid)
        is_e = (apid == 11).astype(np.float64)
        is_mu = (apid == 13).astype(np.float64)
        is_ch = (1.0 - is_e - is_mu)        # remaining charged tracks = charged hadrons
        cols["pt"].append(tpt); cols["eta"].append(teta); cols["phi"].append(tphi)
        cols["energy"].append(tpt * np.cosh(teta))
        cols["charge"].append(tq)
        cols["d0"].append(np.tanh(d0)); cols["dz"].append(np.tanh(dz))
        cols["d0err"].append(d0e); cols["dzerr"].append(dze)
        cols["is_ch"].append(is_ch); cols["is_e"].append(is_e); cols["is_mu"].append(is_mu)
        cols["is_nh"].append(np.zeros_like(tpt)); cols["is_ph"].append(np.zeros_like(tpt))

    # --- neutral towers: EFlowPhoton (is_ph) and EFlowNeutralHadron (is_nh) ---
    for coll, flag in (("EFlowPhoton", "is_ph"), ("EFlowNeutralHadron", "is_nh")):
        et = _get(arr, f"{coll}.ET", i)
        if et is None or not len(et):
            continue
        eta = _get(arr, f"{coll}.Eta", i); phi = _get(arr, f"{coll}.Phi", i)
        z = np.zeros_like(et)
        cols["pt"].append(et); cols["eta"].append(eta); cols["phi"].append(phi)
        cols["energy"].append(et * np.cosh(eta))
        cols["charge"].append(z)
        cols["d0"].append(z); cols["dz"].append(z); cols["d0err"].append(z); cols["dzerr"].append(z)
        cols["is_ch"].append(z); cols["is_e"].append(z); cols["is_mu"].append(z)
        cols["is_ph"].append(np.ones_like(et) if flag == "is_ph" else z)
        cols["is_nh"].append(np.ones_like(et) if flag == "is_nh" else z)

    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}


def _jet_features(c, idx, jpt, jeta, jphi, jE):
    """Build the 17 standardized features + 4-vector for the constituents `idx` of one jet."""
    pt = c["pt"][idx]; eta = c["eta"][idx]; phi = c["phi"][idx]; E = c["energy"][idx]
    deta = eta - jeta
    dphi = _dphi(phi, jphi)
    feats = np.stack([
        _std("pt_log",   np.log(np.clip(pt, _EPS, None))),
        _std("e_log",    np.log(np.clip(E, _EPS, None))),
        _std("logptrel", np.log(np.clip(pt / max(jpt, _EPS), _EPS, None))),
        _std("logerel",  np.log(np.clip(E / max(jE, _EPS), _EPS, None))),
        _std("deltaR",   np.hypot(deta, dphi)),
        _std("charge",   c["charge"][idx]),
        _std("is_ch",    c["is_ch"][idx]),
        _std("is_nh",    c["is_nh"][idx]),
        _std("is_ph",    c["is_ph"][idx]),
        _std("is_e",     c["is_e"][idx]),
        _std("is_mu",    c["is_mu"][idx]),
        _std("d0",       c["d0"][idx]),
        _std("d0err",    c["d0err"][idx]),
        _std("dz",       c["dz"][idx]),
        _std("dzerr",    c["dzerr"][idx]),
        _std("deta",     deta),
        _std("dphi",     dphi),
    ], axis=1).astype(np.float32)                       # (n_const, 17)
    px = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta)
    vecs = np.stack([px, py, pz, E], axis=1).astype(np.float32)   # (n_const, 4)
    # parts = [17 standardized features | 4 raw pf_vectors]; the encoder slices cols
    # 0:17 as ParT x and 17:21 as ParT v. Keeping them in one tensor means no separate
    # part_vectors needs to flow through the dataset/model/ONNX.
    return np.concatenate([feats, vecs], axis=1)                  # (n_const, 21)


def _object_tokens(arr, i):
    """Event object tokens: isolated electrons (type 1), muons (type 2), MET (type 3)."""
    obj = np.zeros((N_OBJ_MAX, F_OBJ), dtype=np.float32)
    mask = np.zeros((N_OBJ_MAX,), dtype=np.float32)
    o = 0

    def _add(pt, eta, phi, val, type_id):
        nonlocal o
        if o >= N_OBJ_MAX:
            return
        obj[o] = [np.log(max(pt, _EPS)), eta, np.sin(phi), np.cos(phi), val, float(type_id)]
        mask[o] = 1.0
        o += 1

    for coll, tid in (("Electron", 1), ("Muon", 2)):
        pt = _get(arr, f"{coll}.PT", i)
        if pt is None:
            continue
        eta = _get(arr, f"{coll}.Eta", i); phi = _get(arr, f"{coll}.Phi", i)
        order = np.argsort(-pt)
        for k in order:
            _add(float(pt[k]), float(eta[k]), float(phi[k]), 0.0, tid)
    met = _get(arr, "MissingET.MET", i); mphi = _get(arr, "MissingET.Phi", i)
    if met is not None and len(np.atleast_1d(met)):
        m = float(np.atleast_1d(met)[0]); p = float(np.atleast_1d(mphi)[0])
        _add(m, 0.0, p, m, 3)
    return obj, mask


def _process_chunk(arr):
    n = len(arr["Jet.PT"])
    parts = np.zeros((n, N_JETS_MAX, N_PART_MAX, F_TOTAL), dtype=np.float32)  # 17 feats + 4 vecs
    part_mask = np.zeros((n, N_JETS_MAX, N_PART_MAX), dtype=np.float32)
    jet_mask = np.zeros((n, N_JETS_MAX), dtype=np.float32)
    obj = np.zeros((n, N_OBJ_MAX, F_OBJ), dtype=np.float32)
    obj_mask = np.zeros((n, N_OBJ_MAX), dtype=np.float32)

    for i in range(n):
        jpt = ak.to_numpy(arr["Jet.PT"][i]); jeta = ak.to_numpy(arr["Jet.Eta"][i])
        jphi = ak.to_numpy(arr["Jet.Phi"][i])
        jmass = _get(arr, "Jet.Mass", i, np.zeros_like(jpt))
        keep = np.where(jpt > JET_PT_MIN)[0]
        keep = keep[np.argsort(-jpt[keep])][:N_JETS_MAX]    # leading jets above threshold
        if len(keep) == 0:
            obj[i], obj_mask[i] = _object_tokens(arr, i)
            continue
        jpt, jeta, jphi, jmass = jpt[keep], jeta[keep], jphi[keep], jmass[keep]
        jE = np.sqrt((jpt * np.cosh(jeta)) ** 2 + jmass ** 2)

        c = _collect_constituents(arr, i)
        if len(c["pt"]):
            # nearest-jet association within DeltaR < JET_DR
            d_eta = c["eta"][:, None] - jeta[None, :]
            d_phi = _dphi(c["phi"][:, None], jphi[None, :])
            dr = np.hypot(d_eta, d_phi)                      # (n_const, n_jet)
            nearest = np.argmin(dr, axis=1)
            within = dr[np.arange(len(nearest)), nearest] < JET_DR
            for j in range(len(jpt)):
                jet_mask[i, j] = 1.0
                sel = np.where(within & (nearest == j))[0]
                if len(sel) == 0:
                    continue
                sel = sel[np.argsort(-c["pt"][sel])][:N_PART_MAX]   # leading constituents
                combined = _jet_features(c, sel, jpt[j], jeta[j], jphi[j], jE[j])  # (nsel, 21)
                nsel = len(sel)
                parts[i, j, :nsel] = combined
                part_mask[i, j, :nsel] = 1.0
        else:
            jet_mask[i, :len(jpt)] = 1.0

        obj[i], obj_mask[i] = _object_tokens(arr, i)

    # Event.Weight may be a length-1 collection (Delphes Event TClonesArray) or flat;
    # _scalar() handles both. Preserve NLO sign; fall back to 1.0 only if truly absent.
    if "Event.Weight" in arr.fields:
        w = np.array([_scalar(arr["Event.Weight"][i]) for i in range(n)], dtype=np.float64)
    else:
        w = np.ones(n, dtype=np.float64)
    return {"parts": parts, "part_mask": part_mask,
            "jet_mask": jet_mask, "obj": obj, "obj_mask": obj_mask, "w": w}


def convert_tree(path, tree_name, out_path, step_size=20000):
    """``path`` may be a single file/glob string or a list of file paths."""
    keys = ("parts", "part_mask", "jet_mask", "obj", "obj_mask", "w")
    source = ({p: tree_name for p in path} if isinstance(path, (list, tuple))
              else f"{path}:{tree_name}")
    chunks = {k: [] for k in keys}
    _checked = False
    for arr in uproot.iterate(source, step_size=step_size, library="ak", filter_name=_BRANCHES):
        if not _checked:
            if "Jet.PT" not in arr.fields:
                raise KeyError(
                    f"'Jet.PT' not found among read branches {list(arr.fields)}. The Delphes "
                    f"leaf names differ from _BRANCHES — inspect with uproot.open(f)['{tree_name}'].keys().")
            _checked = True
        out = _process_chunk(arr)
        for k in keys:
            chunks[k].append(out[k])
    merged = {k: (np.concatenate(v, axis=0) if v else np.empty((0,), dtype=np.float32))
              for k, v in chunks.items()}
    np.savez_compressed(out_path, **merged)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Delphes ROOT path (local or root://...)")
    ap.add_argument("--tree", default="Delphes")
    ap.add_argument("--output", required=True)
    ap.add_argument("--step-size", type=int, default=20000)
    args = ap.parse_args()
    convert_tree(args.input, args.tree, args.output, step_size=args.step_size)
