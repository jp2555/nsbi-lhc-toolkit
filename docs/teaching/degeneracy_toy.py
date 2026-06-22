"""Toy: how the kappa_lambda <-> JES degeneracy (and the profiled kappa_lambda
interval) shrinks as you add features.

Companion to Lesson 4. The point it makes numerically:

  * In ONE feature (m_HH) the kappa_lambda signal and a JES shift are *fully*
    degenerate -> profiling JES destroys the kappa_lambda measurement.
  * Adding features that respond DIFFERENTLY to kappa_lambda vs JES breaks the
    degeneracy -> the profiled kappa_lambda interval shrinks back toward the
    stat-only ideal.

Model (the Gaussian / Asimov limit of an NSBI profiled fit, so it's exact and
noise-free). Each event is a feature vector x ~ Normal(mean, I). The mean shifts
linearly with two parameters:
        mean(mu, alpha) = mu * s  +  alpha * j
  - mu    = kappa_lambda deviation (the POI)         -> direction s in feature space
  - alpha = JES nuisance (constrained, sigma=1)      -> direction j in feature space
With unit covariance the per-event Fisher information is the outer product of the
parameter gradients, so for N events:
        I = N * [[s.s, s.j],[j.s, j.j]]   (+ prior 1/sigma_alpha^2 on the alpha-alpha term)
The profiled sigma(mu) = sqrt( (I+prior)^{-1} )_{mu,mu}.
The data correlation rho = s.j / (|s||j|) over the USED features IS the degeneracy:
rho -> 1 means kappa_lambda and JES point the same way (indistinguishable).

Only the *trends* matter (absolute sigma scales as 1/sqrt(N)); rho and the
inflation = sigma_profiled/sigma_statonly are N-independent.

Usage:  python degeneracy_toy.py
"""
import json
import numpy as np

# --- per-feature sensitivity of each parameter (arbitrary units; physically motivated) ---
FEATURES = ["m_HH", "cos_theta*", "pT_HH", "m_bb", "dR_bb",
            "m_tautau", "dphi_HH", "dR_tautau", "pT_H1", "pT_H2"]

# kappa_lambda signature: strong in m_HH (interference reshapes the spectrum) and in the
# angular / pT_HH variables; weak in the jet-mass / single-leg variables.
s_kl = np.array([1.0, 0.6, 0.5, 0.05, 0.10, 0.05, 0.30, 0.10, 0.20, 0.20])
# JES signature: strong in m_HH (it propagates) AND in m_bb / jet pT (it scales jet energies);
# ~zero in the angular variables (dR, dphi robust to energy scale) and in m_tautau (taus, not jets).
j_jes = np.array([0.9, 0.05, 0.30, 0.90, 0.05, 0.0, 0.05, 0.0, 0.70, 0.70])

N = 200.0           # effective events (sets the absolute info scale; trend is N-independent)
SIGMA_ALPHA = 1.0   # Gaussian prior width on the JES nuisance, in its own sigma units


def analyze(idx):
    """idx = list of feature indices used. Returns (rho, sigma_stat, sigma_prof, inflation)."""
    s, j = s_kl[idx], j_jes[idx]
    I = N * np.array([[s @ s, s @ j], [j @ s, j @ j]])        # data Fisher for (mu, alpha)
    rho = I[0, 1] / np.sqrt(I[0, 0] * I[1, 1])                # degeneracy = cos(angle s,j)
    sigma_stat = 1.0 / np.sqrt(I[0, 0])                       # JES fixed (no systematic)
    Ic = I.copy(); Ic[1, 1] += 1.0 / SIGMA_ALPHA**2          # add the JES prior
    sigma_prof = np.sqrt(np.linalg.inv(Ic)[0, 0])            # JES profiled (systematic on)
    return float(rho), float(sigma_stat), float(sigma_prof), float(sigma_prof / sigma_stat)


def main():
    # cumulative prefixes in the lesson's feature order -> "add features one at a time"
    rows = []
    for k in range(1, len(FEATURES) + 1):
        idx = list(range(k))
        rho, st, pr, infl = analyze(idx)
        rows.append({"n_features": k, "added": FEATURES[k - 1],
                     "rho_degeneracy": rho, "sigma_stat": st,
                     "sigma_profiled": pr, "syst_inflation": infl})

    best = rows[-1]["sigma_stat"]   # stat-only with all features = the ideal
    print(f"{'n':>2} {'+feature':<11} {'degeneracy rho':>14} "
          f"{'sigma_kl (prof)':>16} {'vs ideal':>9} {'syst infl':>10}")
    print("-" * 66)
    for r in rows:
        print(f"{r['n_features']:>2} {r['added']:<11} {r['rho_degeneracy']:>14.3f} "
              f"{r['sigma_profiled']:>16.4f} {r['sigma_profiled']/best:>8.2f}x "
              f"{r['syst_inflation']:>9.2f}x")
    print(f"\n1-feature (m_HH only): degeneracy rho={rows[0]['rho_degeneracy']:.3f}, "
          f"profiled sigma is {rows[0]['sigma_profiled']/best:.1f}x the all-feature ideal.")
    print(f"all {len(FEATURES)} features:   degeneracy rho={rows[-1]['rho_degeneracy']:.3f}, "
          f"systematic inflation down to {rows[-1]['syst_inflation']:.2f}x.")

    with open("degeneracy_toy.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    _plot(rows)


def _plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        print(f"[plot] skipped ({exc})")
        return
    n = [r["n_features"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(n, [r["sigma_profiled"] for r in rows], "o-", color="#7a2d2d", label="profiled (JES floats)")
    ax1.plot(n, [r["sigma_stat"] for r in rows], "s--", color="#234e70", label="stat-only (JES fixed)")
    ax1.set_xlabel("number of features used")
    ax1.set_ylabel(r"$\sigma(\kappa_\lambda)$  (a.u.)")
    ax1.set_title("kappa_lambda uncertainty shrinks as features are added")
    ax1.legend()
    ax2.plot(n, [r["rho_degeneracy"] for r in rows], "o-", color="#1c5fa8", label=r"degeneracy $\rho$ (k$\lambda$–JES)")
    ax2.plot(n, [r["syst_inflation"] for r in rows], "^-", color="#1f6b3b", label="systematic inflation")
    ax2.axhline(1.0, ls=":", color="gray", lw=1)
    ax2.set_xlabel("number of features used")
    ax2.set_ylabel("degeneracy / inflation")
    ax2.set_title("the degeneracy breaks as features are added")
    ax2.legend()
    fig.tight_layout()
    fig.savefig("degeneracy_toy.png", dpi=120)
    print("[plot] -> degeneracy_toy.png")


if __name__ == "__main__":
    main()
