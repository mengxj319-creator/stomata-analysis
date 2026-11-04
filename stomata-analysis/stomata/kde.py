"""
Kernel density estimation (KDE) of stomatal centroids.

Per image: centroids normalised to [0, 1] x [0, 1] by image width and height; scipy gaussian_kde with
bw_method = bandwidth (a factor applied to the data covariance; for n < 15 points, max(bandwidth, 0.5 / sqrt(n)));
evaluated on a grid x grid lattice and min-max normalised to 0-1.
High-density extent = proportion of grid cells with normalised density > threshold.

Per group: mean KDE map (average landscape) and std KDE map (cell-wise SD across images = variability between
images); the std maps of all groups are normalised together, and the high-variability extent is the proportion
of cells > threshold.
"""

import numpy as np
from scipy.stats import gaussian_kde


def grid_points(grid):
    xx, yy = np.mgrid[0:1:grid * 1j, 0:1:grid * 1j]
    return np.vstack([xx.ravel(), yy.ravel()])


def kde_map(xy, bandwidth, grid):
    """xy: (2, n) normalised centroids. Returns a grid x grid map normalised to 0-1 (min-max)."""
    n = xy.shape[1]
    bw = bandwidth
    if bw != "scott" and n < 15:
        bw = max(bw, 0.5 / np.sqrt(n))
    z = gaussian_kde(xy, bw_method=bw)(grid_points(grid)).reshape(grid, grid)
    return (z - z.min()) / (z.max() - z.min())


def extent(map_, threshold):
    return float((map_ > threshold).mean())
