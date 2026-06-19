"""Confidence-interval extraction from a profile-likelihood scan."""
import numpy as np


def ci_from_scan(points, delta_nll, level=1.0):
    """Confidence interval from a (minimum-subtracted) profile-likelihood scan.

    Finds where the curve crosses ``level`` on each side of the minimum by linear
    interpolation. With the inference module's convention the scanned curve is the test
    statistic t = -2 dlnL (Minuit ``errordef=LEAST_SQUARES``), so ``level=1.0`` is the
    ~68% (1 sigma) interval and ``level=3.84`` is ~95%.

    Returns (lo, hi, half_width) with half_width = (hi - lo) / 2. A bound is ``None`` if
    the curve never reaches ``level`` on that side (interval open within the scan range).
    """
    p = np.asarray(points, dtype=float)
    d = np.asarray(delta_nll, dtype=float)
    order = np.argsort(p)
    p, d = p[order], d[order]
    imin = int(np.argmin(d))

    def _cross(indices):
        prev = imin
        for i in indices:
            if d[i] >= level:
                x0, x1, y0, y1 = p[prev], p[i], d[prev], d[i]
                if y1 == y0:
                    return float(x1)
                return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
            prev = i
        return None

    lo = _cross(range(imin - 1, -1, -1))
    hi = _cross(range(imin + 1, len(p)))
    half = (hi - lo) / 2.0 if (lo is not None and hi is not None) else None
    return lo, hi, half
