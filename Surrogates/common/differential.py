"""
Differential (per parton-pT bin) CRPS ratios on shared bins and floors.
"""

import numpy as np

from . import settings
from .features import TARGET_NAMES
from .metrics import crps_floor_within


def bin_centers(edges) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    return np.sqrt(edges[:-1] * edges[1:])


def bin_index(values, edges) -> np.ndarray:
    return np.digitize(np.asarray(values), np.asarray(edges)) - 1


def _counts(idx: np.ndarray, n_bins: int) -> np.ndarray:
    return np.array([int(np.sum(idx == b)) for b in range(n_bins)], dtype=np.int64)


def knn_differential(
    knn,
    X,
    y,
    binvar,
    edges=settings.PT_EDGES,
    ks=settings.K_LADDER,
    min_count=settings.MIN_COUNT,
    query_ids=None,
) -> dict:
    X, y = np.asarray(X), np.asarray(y)
    edges = np.asarray(edges, dtype=float)
    n_bins, D = len(edges) - 1, y.shape[1]
    idx = bin_index(binvar, edges)
    counts = _counts(idx, n_bins)
    floors, intercept, slope, ratio = (np.full((n_bins, D), np.nan) for _ in range(4))
    ids = None if query_ids is None else np.asarray(query_ids)
    for b in range(n_bins):
        if counts[b] < min_count:
            continue
        mask = idx == b
        sweep = knn.sweep_k(
            X[mask], y[mask], ks=tuple(ks), query_ids=None if ids is None else ids[mask]
        )
        floors[b] = crps_floor_within(y[mask])
        intercept[b] = sweep["crps_extrapolated"]
        slope[b] = sweep["slope"]
        ratio[b] = intercept[b] / floors[b]
    return {
        "edges": edges,
        "centers": bin_centers(edges),
        "counts": counts,
        "floors": floors,
        "intercept": intercept,
        "slope": slope,
        "ratio": ratio,
    }


def series_differential(
    crps, y, binvar, edges=settings.PT_EDGES, min_count=settings.MIN_COUNT
) -> np.ndarray:
    crps, y = np.asarray(crps, dtype=float), np.asarray(y)
    edges = np.asarray(edges, dtype=float)
    n_bins, D = len(edges) - 1, y.shape[1]
    idx = bin_index(binvar, edges)
    counts = _counts(idx, n_bins)
    ratio = np.full((n_bins, D), np.nan)
    for b in range(n_bins):
        if counts[b] < min_count:
            continue
        mask = idx == b
        ratio[b] = crps[mask].mean(axis=0) / crps_floor_within(y[mask])
    return ratio


def median_over_bins(ratio) -> np.ndarray:
    r = np.asarray(ratio, dtype=float)
    out = np.full(r.shape[1], np.nan)
    for d in range(r.shape[1]):
        col = r[:, d][np.isfinite(r[:, d])]
        if len(col):
            out[d] = float(np.median(col))
    return out


def nan_to_none(a) -> list:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        return [None if not np.isfinite(v) else float(v) for v in arr]
    return [nan_to_none(row) for row in arr]


def by_target(arr2d, names=TARGET_NAMES) -> dict:
    arr = np.asarray(arr2d, dtype=float)
    return {n: nan_to_none(arr[:, d]) for d, n in enumerate(names)}


def ceiling_record(
    knn,
    X,
    y,
    binvar,
    query_ids=None,
    min_count=settings.MIN_COUNT,
    ks=settings.K_LADDER,
    edges=settings.PT_EDGES,
) -> dict:
    """JSON-ready per-bin k-NN ceiling with its median over populated bins."""
    diff = knn_differential(knn, X, y, binvar, edges, ks, min_count, query_ids)
    return {
        "n_query": int(len(X)),
        "centers": [float(c) for c in diff["centers"]],
        "counts": [int(c) for c in diff["counts"]],
        "ratio": by_target(diff["ratio"]),
        "floors": by_target(diff["floors"]),
        "median_ratio": dict(
            zip(TARGET_NAMES, nan_to_none(median_over_bins(diff["ratio"])))
        ),
    }


def analysis_settings(n_query, min_count, max_train_per_flavour, **extra) -> dict:
    """The settings a truth-only ceiling output was computed with (its skip key)."""
    return {
        "ks": list(settings.K_LADDER),
        "edges": [float(e) for e in settings.PT_EDGES],
        "n_query": int(n_query),
        "min_count": int(min_count),
        "max_train_per_flavour": int(max_train_per_flavour),
        **extra,
    }
