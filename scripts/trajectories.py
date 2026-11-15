"""Compositional directions and pairwise cosine relationships (A_XY)."""
import numpy as np
import pandas as pd

from data_io import atomic_order

EDGES = np.arange(0.0, 1.0 + 1e-9, 0.2)


def fit_trajectory(values, fractions, min_bin_count=20):
    """Fit a line through fraction-interval centroids; the slope is the direction v_X."""
    values, fractions = np.asarray(values, float), np.asarray(fractions, float)
    bins, centroids, centers = [], [], []
    for k, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        mask = (fractions > lo) & (fractions <= hi)
        count = int(mask.sum())
        used = count >= min_bin_count
        bins.append(dict(bin=k, lower=float(lo), upper=float(hi), center=float((lo + hi) / 2),
                         count=count, used=used))
        if used:
            centroids.append(values[mask].mean(axis=0))
            centers.append((lo + hi) / 2)
    if len(centroids) < 3:
        return None, bins, centroids
    x = np.array(centers)
    coef = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), np.vstack(centroids), rcond=None)[0]
    slope = coef[1]
    if not np.isfinite(slope).all() or np.linalg.norm(slope) == 0:
        return None, bins, centroids
    return slope, bins, centroids


def pairwise(vectors):
    """Full precision cosine values; stable atomic-number ties, no self rankings."""
    elements = atomic_order(vectors)
    pairs, ranking = [], []
    if not elements:
        raise ValueError('No valid elemental vectors or trajectory directions')
    matrix = np.vstack([vectors[e] for e in elements])
    valid = np.isfinite(matrix).all(axis=1) & (np.linalg.norm(matrix, axis=1) > 0)
    normed = np.zeros_like(matrix)
    normed[valid] = matrix[valid] / np.linalg.norm(matrix[valid], axis=1, keepdims=True)
    similarities = np.clip(normed @ normed.T, -1, 1)
    similarities[~valid, :] = np.nan
    similarities[:, ~valid] = np.nan
    np.fill_diagonal(similarities, np.nan)
    for i, x in enumerate(elements):
        for j, y in enumerate(elements):
            pairs.append(dict(leaveout=x, element_X=x, element_Y=y, A_XY=similarities[i, j]))
        order = np.argsort(-similarities[i], kind='stable')
        order = [j for j in order if np.isfinite(similarities[i, j])]
        ranking.append(dict(leaveout=x, support_elements=[elements[j] for j in order],
                            support_A_XY=[float(similarities[i, j]) for j in order],
                            n_support_elements=len(order)))
    return pd.DataFrame(pairs), pd.DataFrame(ranking)
