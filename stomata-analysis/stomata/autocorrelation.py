"""
Spatial autocorrelation of a stomatal trait within an image: global Moran's I and local Getis-Ord Gi*.

Neighbourhood definitions (names used in the configuration file):
  knnK        the K nearest neighbours (K = n - 1 if fewer points), e.g. knn5
  bandM       all points within M x the image's mean nearest-neighbour distance, e.g. band1.5
  delaunay    Delaunay natural neighbours

Moran's I: row-standardised weights, permutation inference (pseudo P as in PySAL esda, one-tailed in the
direction of the observed I).

Gi* (Getis & Ord 1995): binary weights with the focal point included (w_ii = 1), analytical z-scores:
  z_i = (sum_j w_ij x_j - xbar W_i) / (S sqrt((n S1_i - W_i^2) / (n - 1))),
  W_i = sum_j w_ij, S1_i = sum_j w_ij^2, S^2 = mean(x^2) - xbar^2.
Because neighbouring z-scores are correlated and many points are tested, the fraction of points flagged is
compared with the fraction flagged after randomly permuting the trait values within the image (the same test
on a pattern without spatial structure); Benjamini-Hochberg FDR counts are also given.
"""

import re

import numpy as np
from scipy import stats
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

MIN_POINTS = 5


def parse_neighbourhood(name):
    m = re.fullmatch(r"knn(\d+)", name)
    if m:
        return ("knn", int(m.group(1)))
    m = re.fullmatch(r"band([\d.]+)", name)
    if m:
        return ("band", float(m.group(1)))
    if name == "delaunay":
        return ("delaunay", None)
    raise ValueError(f"Unknown neighbourhood '{name}' (use knnK, bandM or delaunay)")


def weights(points, name):
    kind, val = parse_neighbourhood(name)
    if kind == "delaunay":
        W = np.zeros((len(points), len(points)))
        for s in Delaunay(points).simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    W[s[i], s[j]] = W[s[j], s[i]] = 1.0
        return W
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    if kind == "knn":
        k = min(val, len(points) - 1)
        W = np.zeros_like(d)
        W[np.arange(len(points))[:, None], np.argsort(d, axis=1)[:, :k]] = 1.0
        return W
    return (d <= val * d.min(axis=1).mean()).astype(float)


def row_standardise(W):
    rs = W.sum(axis=1, keepdims=True)
    return np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)  # points without neighbours keep zero rows


def morans_i(values, W, nperm, rng):
    """Returns I, E[I], pseudo P and the fraction of points without neighbours."""
    n = len(values)
    W = row_standardise(W)
    s0 = W.sum()
    z = values - values.mean()
    den = (z ** 2).sum()
    I = n / s0 * (z @ W @ z) / den
    Z = np.array([rng.permutation(z) for _ in range(nperm)])
    I_perm = n / s0 * np.einsum("ij,jk,ik->i", Z, W, Z) / den
    larger = (I_perm >= I).sum()
    if nperm - larger < larger:
        larger = nperm - larger
    p = (larger + 1) / (nperm + 1)
    return I, -1 / (n - 1), p, float((W.sum(axis=1) == 0).mean())


def gistar_z(x, W):
    W = W.copy()
    np.fill_diagonal(W, 1.0)
    n = len(x)
    xbar = x.mean()
    S = np.sqrt((x ** 2).mean() - xbar ** 2)
    Wi = W.sum(axis=1)
    S1 = (W ** 2).sum(axis=1)
    num = W @ x - xbar * Wi
    den = S * np.sqrt((n * S1 - Wi ** 2) / (n - 1))
    return np.divide(num, den, out=np.zeros(n), where=den > 0)


def bh_count(z, q=0.05):
    """Number of points significant after Benjamini-Hochberg FDR control (two-sided)."""
    p = np.sort(2 * stats.norm.sf(np.abs(z)))
    ok = p <= q * np.arange(1, len(p) + 1) / len(p)
    return int(np.max(np.nonzero(ok)[0]) + 1) if ok.any() else 0
