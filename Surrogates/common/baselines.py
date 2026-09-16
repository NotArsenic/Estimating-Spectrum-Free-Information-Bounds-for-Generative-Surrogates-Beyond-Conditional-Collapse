"""
Reference samplers for the model comparison.

All three implement the same contract as a trained model's sampler:
    sample(X, m) -> (N, m, 4)
"""

import numpy as np

from .features import FLAVOUR_NAMES, TARGET_NAMES, _N_FLAVOURS

_D = len(TARGET_NAMES)

# Column layout of the 26-D conditioner.
_CONT_COLS = slice(0, 4)
_FLAV_COLS = slice(4, 4 + _N_FLAVOURS)


def _flavour_index(X: np.ndarray) -> np.ndarray:
    """Recover the flavour stratum from the one-hot block."""
    return np.asarray(X[:, _FLAV_COLS]).argmax(axis=1)


class MarginalSampler:
    """
    Draws from the unconditional p(y), ignoring the conditioner entirely.
    """

    def __init__(self, y_train: np.ndarray, seed: int = 0):
        self.y = np.asarray(y_train, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def sample(self, X: np.ndarray, m: int) -> np.ndarray:
        n = len(X)
        idx = self.rng.integers(0, len(self.y), size=(n, m))
        return self.y[idx]


class LookupTableSampler:
    """
    Conditional histogram over a binned conditioner: bin the training rows by (log pt, |eta|, flavour, nTrueInt), then sample a stored training target from the matching cell.

    Cells that are empty (or below ``min_count``) fall back to the flavour marginal, and then to the global marginal. Without a fallback the sampler silently fails on exactly the sparse high-pT region the differential figure is about.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_pt_bins: int = 40,
        n_eta_bins: int = 12,
        n_pu_bins: int = 6,
        min_count: int = 20,
        seed: int = 0,
    ):
        self.y = np.asarray(y_train, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.min_count = min_count

        cont = np.asarray(X_train[:, _CONT_COLS])

        self.pt_edges = self._quantile_edges(cont[:, 0], n_pt_bins)
        self.eta_edges = self._quantile_edges(cont[:, 1], n_eta_bins)
        self.pu_edges = self._quantile_edges(cont[:, 3], n_pu_bins)

        keys = self._keys(X_train)
        order = np.argsort(keys, kind="stable")
        self.sorted_rows = order
        sorted_keys = keys[order]
        uniq, starts = np.unique(sorted_keys, return_index=True)
        ends = np.append(starts[1:], len(sorted_keys))
        self.cells = {
            int(k): (int(s), int(e))
            for k, s, e in zip(uniq, starts, ends)
            if e - s >= min_count
        }

        # Fallbacks
        flav = _flavour_index(X_train)
        self.flav_rows = {int(f): np.flatnonzero(flav == f) for f in np.unique(flav)}
        self.all_rows = np.arange(len(self.y))

    @staticmethod
    def _quantile_edges(v: np.ndarray, n_bins: int) -> np.ndarray:
        e = np.quantile(v, np.linspace(0, 1, n_bins + 1))
        return np.unique(e)[1:-1]  # interior edges for np.searchsorted

    def _keys(self, X: np.ndarray) -> np.ndarray:
        cont = np.asarray(X[:, _CONT_COLS])
        pt_b = np.searchsorted(self.pt_edges, cont[:, 0])
        eta_b = np.searchsorted(self.eta_edges, cont[:, 1])
        pu_b = np.searchsorted(self.pu_edges, cont[:, 3])
        flav = _flavour_index(X)
        n_eta = len(self.eta_edges) + 1
        n_pu = len(self.pu_edges) + 1
        return ((pt_b * n_eta + eta_b) * n_pu + pu_b) * _N_FLAVOURS + flav

    def sample(self, X: np.ndarray, m: int) -> np.ndarray:
        keys = self._keys(X)
        flav = _flavour_index(X)
        out = np.empty((len(X), m, _D), dtype=np.float32)
        for i, k in enumerate(keys):
            cell = self.cells.get(int(k))
            if cell is not None:
                pool = self.sorted_rows[cell[0] : cell[1]]
            else:
                pool = self.flav_rows.get(int(flav[i]), self.all_rows)
                if len(pool) < self.min_count:
                    pool = self.all_rows
            out[i] = self.y[pool[self.rng.integers(0, len(pool), size=m)]]
        return out


class KNNCeiling:
    """
    Per-flavour k-nearest-neighbour sampler over the standardised continuous conditioner
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k: int = 50,
        seed: int = 0,
        max_train_per_flavour: int | None = 400_000,
        train_ids: np.ndarray | None = None,
        self_exclude: str = "auto",
    ):
        """
        Parameters
        ----------
        train_ids : (n_train,) unique row identifiers
        self_exclude : {"auto", "ids", "distance", "off", "raise"}
        """

        from scipy.spatial import cKDTree

        valid = ("auto", "ids", "distance", "off", "raise")
        if self_exclude not in valid:
            raise ValueError(
                f"self_exclude must be one of {valid}, got {self_exclude!r}"
            )
        if self_exclude == "auto":
            self_exclude = "ids" if train_ids is not None else "off"
        if self_exclude in ("ids", "raise") and train_ids is None:
            raise ValueError(
                f"self_exclude={self_exclude!r} requires train_ids "
                "(pass a unique row id array such as global_match_id)"
            )

        self.k = k
        self.self_exclude = self_exclude
        self.self_match_fraction = None
        self.rng = np.random.default_rng(seed)
        self.y = np.asarray(y_train, dtype=np.float32)
        self.ids = None if train_ids is None else np.asarray(train_ids)
        flav = _flavour_index(X_train)
        cont = np.asarray(X_train[:, _CONT_COLS], dtype=np.float64)

        self._trees, self._rows = {}, {}
        for f in np.unique(flav):
            rows = np.flatnonzero(flav == f)
            if max_train_per_flavour and len(rows) > max_train_per_flavour:
                rows = self.rng.choice(rows, max_train_per_flavour, replace=False)
            if len(rows) <= k:
                continue
            self._rows[int(f)] = rows
            self._trees[int(f)] = cKDTree(cont[rows])

    def _neighbours(self, X, k, query_ids=None, tol: float = 1e-12):
        """
        Return ``(neighbour_row_indices (N,k), neighbour_distances (N,k))``.

        ``self_match_fraction`` records the fraction of rows that actually hit a self-match, so an overlap is visible rather than merely handled.
        """
        flav = _flavour_index(X)
        self._check_flavours(flav)
        cont = np.asarray(X[:, _CONT_COLS], dtype=np.float64)
        idx = np.zeros((len(X), k), dtype=np.int64)
        dist = np.full((len(X), k), np.nan)

        mode = self.self_exclude
        by_ids = mode in ("ids", "raise") and query_ids is not None
        by_dist = mode == "distance"
        check = by_ids or by_dist
        kq = k + 1 if check else k
        n_self = 0
        query_ids = None if query_ids is None else np.asarray(query_ids)

        for f, tree in self._trees.items():
            sel = np.flatnonzero(flav == f)
            if len(sel) == 0:
                continue
            kq_f = min(kq, tree.n)
            d, j = tree.query(cont[sel], k=kq_f, workers=-1)
            if kq_f == 1:  # cKDTree drops the trailing axis at k=1
                d, j = d[:, None], j[:, None]
            rows_f = self._rows[f][j]  # (n_sel, kq_f) global train rows

            if check and kq_f > k:
                if by_ids:
                    hit = self.ids[rows_f] == query_ids[sel][:, None]
                else:
                    hit = d <= tol
                    hit[:, 1:] = False  # only the first can be the self-match
                has_self = hit.any(axis=1)
                n_self += int(has_self.sum())
                # Drop the matched column by pushing it to the end, then take k.
                order = np.argsort(hit.astype(np.int8), kind="stable", axis=1)
                d = np.take_along_axis(d, order, axis=1)[:, :k]
                rows_f = np.take_along_axis(rows_f, order, axis=1)[:, :k]
            else:
                d, rows_f = d[:, :k], rows_f[:, :k]

            idx[sel] = rows_f
            dist[sel] = d

        if check:
            self.self_match_fraction = n_self / max(len(X), 1)
            if n_self and mode == "raise":
                raise AssertionError(
                    f"KNNCeiling: {n_self:,} of {len(X):,} query rows "
                    f"({self.self_match_fraction:.2%}) are also fit rows. The "
                    "fit and query sets must be disjoint — a self-match leaks "
                    "each row's own target into its own predictive ensemble "
                    "and deflates CRPS, which is the direction that fakes "
                    "'the model beat the ceiling'."
                )
        return idx, dist

    def sample(
        self, X: np.ndarray, m: int, k: int | None = None, query_ids=None
    ) -> np.ndarray:
        k = k or self.k
        idx, _ = self._neighbours(X, k, query_ids=query_ids)
        pick = self.rng.integers(0, k, size=(len(X), m))
        return self.y[np.take_along_axis(idx, pick, axis=1)]

    def _check_flavours(self, flav: np.ndarray) -> None:
        missing = sorted(set(np.unique(flav).tolist()) - set(self._trees))
        if missing:
            raise ValueError(
                "KNNCeiling: no reference tree for query flavour(s) "
                f"{[FLAVOUR_NAMES[f] for f in missing]} (the reference set has too "
                "few rows of that flavour). Those rows would silently receive "
                "another row's neighbours."
            )

    def zero_distance_fraction(self, X: np.ndarray, query_ids=None) -> float:
        """Fraction of queries whose nearest reference neighbour is at distance 0."""
        _, dist = self._neighbours(X, 1, query_ids=query_ids)
        return float(np.mean(dist[:, 0] == 0.0))

    def self_overlap_deflation(
        self, X_train: np.ndarray, n_rows: int, ks=(5, 10, 20, 50, 100), seed: int = 0
    ) -> dict:
        """
        How much CRPS drops when a query's own row is left in its neighbourhood.

        ``n_rows`` reference rows are queried against their own trees twice, with the self row included and excluded by id.
        Returns
        ``{str(k): mean over targets of 1 - CRPS_included / CRPS_excluded}``.
        """
        from .metrics import crps_per_event

        if self.ids is None:
            raise ValueError("self_overlap_deflation needs train_ids")
        pool = np.concatenate([self._rows[f] for f in sorted(self._rows)])
        rng = np.random.default_rng(seed)
        rows = rng.choice(pool, size=min(n_rows, len(pool)), replace=False)
        Xq, yq, idq = np.asarray(X_train)[rows], self.y[rows], self.ids[rows]
        out = {}
        for k in ks:
            idx_in, _ = self._neighbours(Xq, k, query_ids=None)
            idx_ex, _ = self._neighbours(Xq, k, query_ids=idq)
            c_in = crps_per_event(yq, self.y[idx_in]).mean(axis=0)
            c_ex = crps_per_event(yq, self.y[idx_ex]).mean(axis=0)
            out[str(k)] = float(np.mean(1.0 - c_in / c_ex))
        return out

    def sweep_k(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ks=(5, 10, 20, 50, 100),
        query_ids=None,
    ) -> dict:
        """
        CRPS vs k, the mean neighbourhood radius at each k, and a linear
        extrapolation to radius -> 0.

        Returns ``{"ks", "radius", "crps", "crps_extrapolated", "slope",
        "slope_se", "intercept_se", "target_names"}``
        """
        from .metrics import crps_per_event

        radii, crps_by_k = [], []
        for k in ks:
            if k < 2:
                raise ValueError("k must be >= 2 for the fair CRPS estimator")
            idx, dist = self._neighbours(X, k, query_ids=query_ids)
            radii.append(float(np.nanmean(dist[:, -1])))
            s = self.y[idx]  # (N, k, D) - ensemble size k
            crps_by_k.append(crps_per_event(y, s).mean(axis=0))

        radius = np.asarray(radii)
        crps = np.asarray(crps_by_k)  # (n_k, D)
        D = crps.shape[1]
        extrap = np.empty(D)
        slope = np.full(D, np.nan)
        slope_se = np.full(D, np.nan)
        intercept_se = np.full(D, np.nan)
        n = len(radius)
        xm = float(radius.mean())
        sxx = float(np.sum((radius - xm) ** 2))
        for d in range(D):
            m_d, b_d = np.polyfit(radius, crps[:, d], 1)
            extrap[d] = b_d  # value at radius = 0
            slope[d] = m_d
            if n > 2 and sxx > 0:
                resid = crps[:, d] - (m_d * radius + b_d)
                s2 = float(np.sum(resid**2) / (n - 2))  # residual variance
                slope_se[d] = float(np.sqrt(s2 / sxx))
                intercept_se[d] = float(np.sqrt(s2 * (1.0 / n + xm**2 / sxx)))
        return {
            "ks": list(ks),
            "radius": radius,
            "crps": crps,
            "crps_extrapolated": extrap,
            "slope": slope,
            "slope_se": slope_se,
            "intercept_se": intercept_se,
            "target_names": list(TARGET_NAMES),
        }
