"""Inject a CONTROLLED, known parametric systematic into an EveNet NPZ.

Produces a "varied" copy of a nominal EveNet NPZ by applying a known smooth distortion to the
jet objects, then recomputing the affected globals consistently (shared build_conditions). The
varied file carries classification=--class-id (e.g. 1) so that, paired with the nominal file
(classification=0), it forms a nominal-vs-varied discriminator. The data-efficiency of learning
that discriminator at small N_var is Money Plot 2 (see EVENET_INTEGRATION_PLAN.md sec 6).

Because the distortion is known exactly, closure is exact -- no real systematic samples needed.

Kinds:
  jes        constant jet-energy scale: jet (pt, energy) *= (1 + alpha)
  jes_mhh    m_HH-dependent scale: jet (pt, energy) *= (1 + alpha * tanh((M_all - m0)/w))
             (a smooth shape systematic correlated with the kappa_lambda observable)

MET is left unchanged (a JES->MET propagation could be added later); all jet-derived globals
(HT, nbJet, M_all, M_bjets, ...) are recomputed from the scaled objects.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delphes_to_evenet_npz import build_conditions, N_OBJ_MAX  # noqa: E402


def inject(data, kind, alpha, m0, w, class_id):
    x = np.array(data["x"], dtype=np.float32, copy=True)
    counts = np.asarray(data["num_sequential_vectors"], dtype=np.int64)
    mask = np.arange(N_OBJ_MAX)[None, :] < counts[:, None]
    cond = np.asarray(data["conditions"], dtype=np.float32)
    met, met_phi = cond[:, 0], cond[:, 1]                       # raw (npz conditions are pre-log)

    is_jet = mask & (x[..., 5] < 0.5)                           # isLepton == 0

    if kind == "jes":
        scale = np.full(len(x), 1.0 + alpha, dtype=np.float32)
    elif kind == "jes_mhh":
        m_all = cond[:, 7]                                      # M_all proxy for m_HH
        scale = (1.0 + alpha * np.tanh((m_all - m0) / w)).astype(np.float32)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    s = scale[:, None]
    x[..., 0] = np.where(is_jet, x[..., 0] * s, x[..., 0])      # energy
    x[..., 1] = np.where(is_jet, x[..., 1] * s, x[..., 1])      # pt
    x[..., 0] = np.clip(x[..., 0], 0.0, None)
    x[..., 1] = np.clip(x[..., 1], 0.0, None)

    out = dict(data)
    out["x"] = x
    out["conditions"] = build_conditions(x, mask, met, met_phi)
    out["classification"] = np.full(len(x), int(class_id), dtype=np.int64)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="nominal EveNet .npz")
    ap.add_argument("--output", required=True, help="varied .npz")
    ap.add_argument("--kind", choices=["jes", "jes_mhh"], default="jes")
    ap.add_argument("--alpha", type=float, default=0.05, help="systematic strength")
    ap.add_argument("--m0", type=float, default=350.0, help="jes_mhh: M_all pivot [GeV]")
    ap.add_argument("--w", type=float, default=150.0, help="jes_mhh: M_all width [GeV]")
    ap.add_argument("--class-id", type=int, default=1, help="class label for the varied sample")
    args = ap.parse_args()

    data = {k: v for k, v in np.load(args.input, allow_pickle=True).items()}
    out = inject(data, args.kind, args.alpha, args.m0, args.w, args.class_id)
    np.savez_compressed(args.output, **out)
    moved = float(np.mean(np.abs(out["conditions"][:, 5] - data["conditions"][:, 5])))  # <|dHT|>
    print(f"wrote {args.output}: kind={args.kind} alpha={args.alpha} mean|dHT|={moved:.3f}")


if __name__ == "__main__":
    main()
