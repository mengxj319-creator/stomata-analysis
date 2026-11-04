"""
Ripley's K / L for replicated point patterns (several images of one sampling unit, e.g. one plant).

K estimator (per image, window = whole image):
  - focal points = points at least max_d from every image edge; if fewer than MIN_POINTS, the buffer is
    halved; if still too few, all points are used. All points serve as neighbours.
  - lambda = n / window area; K(r) = (focal-to-any pairs with distance <= r) / (lambda * n_focal)
  - L(r) = sqrt(K(r) / pi) - r

CSR envelope for the mean L of a unit: in each simulation one CSR pattern is generated per image (same n,
same window) and the simulated curves are averaged as the observed ones. Regularity is tested with a global
rank envelope test (extreme rank length; Myllymaki et al. 2017, J R Stat Soc B 79:381-404).
RPSD (regular point spacing distance) = smallest r at which the observed mean L is below the envelope.

Also provided:
  - density-scaled curves L*(r*) = L * sqrt(lambda) at r* = r * sqrt(lambda), which remove the effect of density
  - studentized permutation test for two groups of curves (Hahn 2012, J Am Stat Assoc 107:754-764)
  - a size-aware (hard-core) null model: non-overlapping identical ellipses (median stomatal length and width
    of the unit, long axis along the image x axis) placed by random sequential inhibition
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

MIN_POINTS = 5


def window_area(bounds):
    return (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])


def focal_mask(points, bounds, max_d):
    xmin, xmax, ymin, ymax = bounds
    inside = lambda b: ((points[:, 0] >= xmin + b) & (points[:, 0] <= xmax - b) &
                        (points[:, 1] >= ymin + b) & (points[:, 1] <= ymax - b))
    mask = inside(max_d)
    if mask.sum() < MIN_POINTS:
        mask = inside(max_d / 2)
        if mask.sum() < MIN_POINTS:
            mask = np.ones(len(points), dtype=bool)
    return mask


def k_function(points, bounds, distances, max_d):
    n = len(points)
    focal = focal_mask(points, bounds, max_d)
    idx = np.flatnonzero(focal)
    d = cdist(points[idx], points)
    d[np.arange(len(idx)), idx] = np.inf  # exclude self-pairs
    pair_d = np.sort(d.ravel())
    counts = np.searchsorted(pair_d, distances, side="right")
    lam = n / window_area(bounds)
    return counts / (lam * len(idx))


def l_function(points, bounds, distances, max_d):
    k = k_function(points, bounds, distances, max_d)
    return np.sqrt(np.maximum(k, 0) / np.pi) - distances


def csr(n, bounds, rng):
    xmin, xmax, ymin, ymax = bounds
    return np.column_stack([rng.uniform(xmin, xmax, n), rng.uniform(ymin, ymax, n)])


def simulate_mean_curves(patterns, distances, max_d, nsim, rng):
    return np.array([np.mean([l_function(csr(len(p), b, rng), b, distances, max_d) for p, b in patterns], axis=0)
                     for _ in range(nsim)])


def global_envelope_erl(observed, sims, alpha=0.05):
    """Two-sided global rank envelope test, extreme rank length ordering.
    observed: (m,) curve; sims: (s, m) curves under H0 (remove columns identical in all curves, e.g. r = 0).
    Returns (p_value, lower, upper) of the 100(1 - alpha)% global envelope."""
    X = np.vstack([observed, sims])  # row 0 = observed
    n = X.shape[0]
    lo = rankdata(X, method="max", axis=0)
    hi = rankdata(-X, method="max", axis=0)
    R_sorted = np.sort(np.minimum(lo, hi), axis=1)
    order = np.lexsort(R_sorted.T[::-1])  # lexicographic order: smaller = more extreme
    erl = np.empty(n)
    rank = 0
    for k, i in enumerate(order):
        if k == 0 or not np.array_equal(R_sorted[i], R_sorted[order[k - 1]]):
            rank = k + 1
        erl[i] = rank
    measure = np.array([(erl <= e).sum() for e in erl]) / n
    p_value = (measure <= measure[0]).sum() / n
    keep = measure > alpha
    return p_value, X[keep].min(axis=0), X[keep].max(axis=0)


def mnnd(points):
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1).mean()


def analyze_unit(patterns, distances, max_d, nsim, rng):
    """patterns: list of (points_um, bounds_um), one per image. Images with < MIN_POINTS points are skipped.
    Returns the mean L curve, its global CSR envelope, the global test P value, RPSD and summary values."""
    patterns = [(p, b) for p, b in patterns if len(p) >= MIN_POINTS]
    if not patterns:
        return None
    L_mean = np.mean([l_function(p, b, distances, max_d) for p, b in patterns], axis=0)
    sims = simulate_mean_curves(patterns, distances, max_d, nsim, rng)
    use = distances > 0  # r = 0 is identical for all curves
    p, lo, hi = global_envelope_erl(L_mean[use], sims[:, use])
    lower = np.r_[np.zeros((~use).sum()), lo]
    upper = np.r_[np.zeros((~use).sum()), hi]
    regular, cluster = L_mean < lower, L_mean > upper
    density = np.mean([len(p) / (window_area(b) / 1e6) for p, b in patterns])
    rpsd = distances[regular].min() if regular.any() else np.nan
    return dict(
        n_images=len(patterns),
        n_points_mean=np.mean([len(p) for p, _ in patterns]),
        density_mm2=density,
        mnnd_um=np.mean([mnnd(p) for p, _ in patterns]),
        global_p=p,
        regular=bool(p <= 0.05 and regular.any()),
        rpsd_um=rpsd,
        rpsd_star=rpsd * np.sqrt(density / 1e6),
        regular_max_um=distances[regular].max() if regular.any() else np.nan,
        clustered_at_any_r=bool(cluster.any()),
        L_mean=L_mean, lower=lower, upper=upper,
    )


def scaled_curve(patterns, rstar, max_d):
    """Mean L*(r*) of a unit: each image evaluated at r = r* / sqrt(lambda_unit), multiplied by sqrt(lambda_unit)."""
    pats = [(p, b) for p, b in patterns if len(p) >= MIN_POINTS]
    lam = np.mean([len(p) / window_area(b) for p, b in pats])
    r = rstar / np.sqrt(lam)
    return np.mean([l_function(p, b, r, max_d) for p, b in pats], axis=0) * np.sqrt(lam)


def studentized_perm_test(curves, labels, nperm, rng):
    """Two groups of curves (rows of `curves`, group given by boolean `labels`). Returns (statistic, P).
    Distances at which all curves are equal (e.g. L = -r below the smallest inter-point distance) are left out;
    curves that differ only by floating-point rounding count as equal."""
    tol = 1e-12 * curves.var(axis=0).max()

    def stat(lab):
        a, b = curves[lab], curves[~lab]
        v = a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b)
        ok = v > tol
        return np.sum((a.mean(0) - b.mean(0))[ok] ** 2 / v[ok])
    t0 = stat(labels)
    perm = np.array([stat(rng.permutation(labels)) for _ in range(nperm)])
    return t0, (1 + np.sum(perm >= t0)) / (nperm + 1)


def ssi_ellipses(n, lx, ly, bounds, rng, max_tries=200000):
    """Random sequential inhibition of n identical ellipses (axes lx along x, ly along y) in the window."""
    xmin, xmax, ymin, ymax = bounds
    a = ly / lx
    pts = np.empty((n, 2))
    k, tries = 0, 0
    while k < n:
        cand = np.column_stack([rng.uniform(xmin, xmax, 256), rng.uniform(ymin, ymax, 256)])
        for c in cand:
            tries += 1
            if k == 0 or np.min((a * (pts[:k, 0] - c[0])) ** 2 + (pts[:k, 1] - c[1]) ** 2) >= ly ** 2:
                pts[k] = c
                k += 1
                if k == n:
                    break
        if tries > max_tries:
            raise RuntimeError(f"Random sequential inhibition jammed at {k}/{n} points")
    return pts


def size_null_model(patterns, lx, ly, distances, max_d, nsim, rng, csr_lower):
    """Compare the observed mean L of a unit with the hard-core null model.
    csr_lower: lower global CSR envelope of the unit (from analyze_unit).
    Returns the global test P value against the null, the r range where the observed curve is below the null
    envelope (more regular than size exclusion alone), the RPSD expected from size exclusion alone, and the
    mean deviation of the observed curve from the null mean."""
    patterns = [(p, b) for p, b in patterns if len(p) >= MIN_POINTS]
    use = distances > 0
    L_obs = np.mean([l_function(p, b, distances, max_d) for p, b in patterns], axis=0)
    hc = np.array([np.mean([l_function(ssi_ellipses(len(p), lx, ly, b, rng), b, distances, max_d)
                            for p, b in patterns], axis=0) for _ in range(nsim)])
    p_hc, lo, hi = global_envelope_erl(L_obs[use], hc[:, use])
    lo = np.r_[np.zeros((~use).sum()), lo]
    hi = np.r_[np.zeros((~use).sum()), hi]
    more_regular = (L_obs < lo) & use
    hc_mean = hc.mean(axis=0)
    expected = (hc_mean < csr_lower) & use
    return dict(
        p_vs_size_null=p_hc,
        more_regular_than_size_null=bool(more_regular.any()),
        more_regular_range_um=(f"{distances[more_regular].min():.0f}-{distances[more_regular].max():.0f}"
                               if more_regular.any() else ""),
        more_clustered_than_size_null=bool(((L_obs > hi) & use).any()),
        rpsd_expected_from_size_um=distances[expected].min() if expected.any() else np.nan,
        mean_deviation_from_size_null=float(np.mean((L_obs - hc_mean)[use])),
        L_size_null=hc_mean,
    )
