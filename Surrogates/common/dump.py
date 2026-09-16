"""
Per-event evaluation dump.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

from .features import TARGET_NAMES, residual_cone_radius, _DR_CONE
from .metrics import crps_per_event, compute_pit_sampled
from .provenance import write_json

_INDEX_PASSTHROUGH = [
    "run",
    "luminosityBlock",
    "event",
    "global_match_id",
    "parton_pt",
    "parton_eta",
    "parton_phi",
    "parton_pdgId",
    "pileup_nTrueInt",
    "rho",
    "pt_hat",
    "generator_weight",
    "ps_isr_up",
    "ps_fsr_up",
    "ps_isr_down",
    "ps_fsr_down",
    "delta_R",
    "has_genjet_match",
    "genjet_pt",
    "genjet_eta",
    "genjet_phi",
    "genjet_mass",
]


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _hash_column(values: np.ndarray) -> str:
    """
    Stable content hash of the alignment key.

    Recorded in meta.json so a cross-model mismatch surfaces as an immediate hard failure rather than as a subtly wrong scatter in a figure.
    """
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def write_dump(
    dump_dir: str | Path,
    *,
    model_name: str,
    seed: int,
    df_test,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_fn,
    m: int = 200,
    subset_size: int = 100_000,
    batch_size: int = 4096,
    ll_fn=None,
    hdr_fn=None,
    internals_fn=None,
    limit: int | None = None,
    pit_seed: int = 7,
    extra_meta: dict | None = None,
) -> dict:
    """
    Generate the per-event dump.

    Parameters
    ----------
    sample_fn : callable(X_batch: np.ndarray, m: int) -> (B, m, 4) np.ndarray
    ll_fn : callable(X_sub, y_sub) -> (n_sub,) or None
    hdr_fn : callable(X_sub, y_sub) -> (log_p_truth, log_p_samples) or None
    internals_fn : callable(X_sub) -> dict[str, np.ndarray] or None
    limit : truncate to the first ``limit`` events (verification runs only).

    Returns
    -------
    The meta dict that was written.
    """
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    N_full = len(y_test)
    N = min(limit, N_full) if limit else N_full
    n_sub = min(subset_size, N)
    D = len(TARGET_NAMES)

    print(f"[dump] {model_name} seed={seed} -> {dump_dir}")
    print(f"[dump] events={N:,} (of {N_full:,})  m={m}  subset={n_sub:,}")

    # samples + per-event CRPS/PIT/cone audit, streamed in batches
    samples_path = dump_dir / "samples.npy"
    samples_mm = np.lib.format.open_memmap(
        samples_path, mode="w+", dtype=np.float32, shape=(N, m, D)
    )
    crps = np.empty((N, D), dtype=np.float32)
    pit = np.empty((N, D), dtype=np.float32)
    frac_outside = np.empty(N, dtype=np.float32)

    import time as _time_dump

    _t0_dump = _time_dump.perf_counter()

    for bi, start in enumerate(range(0, N, batch_size)):
        end = min(start + batch_size, N)
        s = np.asarray(sample_fn(X_test[start:end], m), dtype=np.float32)
        if s.shape != (end - start, m, D):
            raise ValueError(
                f"sample_fn returned {s.shape}, expected {(end - start, m, D)}"
            )
        samples_mm[start:end] = s

        yb = y_test[start:end]
        crps[start:end] = crps_per_event(yb, s)
        # PIT is seeded per batch so the dump is reproducible
        pit[start:end] = compute_pit_sampled(yb, s, seed=pit_seed + bi)

        frac_outside[start:end] = np.mean(residual_cone_radius(s) > _DR_CONE, axis=1)

        if bi % 5 == 0:
            elapsed = _time_dump.perf_counter() - _t0_dump
            if bi > 0:
                eta = elapsed / end * (N - end)
                print(
                    f"[dump]   sampled {end:,}/{N:,}  "
                    f"({100*end/N:.1f}%  elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m)"
                )
            else:
                print(f"[dump]   sampled {end:,}/{N:,}")

    samples_mm.flush()
    del samples_mm

    sub_idx = np.arange(n_sub, dtype=np.int64)
    np.save(dump_dir / "sub_idx.npy", sub_idx)
    X_sub, y_sub = X_test[sub_idx], y_test[sub_idx]

    if ll_fn is not None:
        print(f"[dump] log-likelihood on {n_sub:,} events...")
        ll = np.asarray(ll_fn(X_sub, y_sub), dtype=np.float32)
        np.save(dump_dir / "ll.npy", ll)

    if hdr_fn is not None:
        print(f"[dump] HDR rank on {n_sub:,} events...")
        from .metrics import hdr_rank_per_event

        lp_truth, lp_samples = hdr_fn(X_sub, y_sub)
        rank = hdr_rank_per_event(np.asarray(lp_truth), np.asarray(lp_samples)).astype(
            np.float32
        )
        np.save(dump_dir / "hdr_rank.npy", rank)

    if internals_fn is not None:
        print(f"[dump] model internals on {n_sub:,} events...")
        np.savez_compressed(dump_dir / "internals.npz", **internals_fn(X_sub))

    # index.parquet
    cols = {}
    for name in _INDEX_PASSTHROUGH:
        if name in df_test.columns:
            cols[name] = df_test[name].to_numpy()[:N]
    for d, tname in enumerate(TARGET_NAMES):
        cols[f"y_true_{d}"] = y_test[:N, d]
        cols[f"crps_{d}"] = crps[:, d]
        cols[f"pit_{d}"] = pit[:, d]
    cols["frac_outside_disk"] = frac_outside
    index_df = pl.DataFrame(cols)
    index_df.write_parquet(dump_dir / "index.parquet")

    # meta.json
    meta = {
        "model": model_name,
        "seed": seed,
        "git_sha": _git_sha(),
        "n_events": int(N),
        "n_events_full_test_split": int(N_full),
        "limit": limit,
        "m": int(m),
        "subset_size": int(n_sub),
        "target_names": list(TARGET_NAMES),
        "pit_seed": pit_seed,
        "dr_cone": _DR_CONE,
        "has_ll": ll_fn is not None,
        "has_hdr": hdr_fn is not None,
        "has_internals": internals_fn is not None,
    }
    if "global_match_id" in cols:
        meta["global_match_id_sha256"] = _hash_column(cols["global_match_id"])
    else:
        meta["global_match_id_sha256"] = None
    if extra_meta:
        meta.update(extra_meta)

    write_json(dump_dir / "meta.json", meta)

    print(f"[dump] wrote {dump_dir}")
    for k in ("crps", "pit"):
        arr = {"crps": crps, "pit": pit}[k]
        print(
            f"[dump]   mean {k}: "
            + "  ".join(f"{TARGET_NAMES[d]}={arr[:, d].mean():.5f}" for d in range(D))
        )
    print(f"[dump]   mean frac_outside_disk: {frac_outside.mean():.6f}")
    return meta


# Reading
class Dump:
    """Lazy reader for one dump directory. ``samples`` is memory-mapped."""

    def __init__(self, dump_dir: str | Path):
        self.dir = Path(dump_dir)
        with open(self.dir / "meta.json") as f:
            self.meta = json.load(f)
        self._index = None
        self._samples = None

    @property
    def index(self):
        if self._index is None:
            self._index = pl.read_parquet(self.dir / "index.parquet")
        return self._index

    @property
    def samples(self) -> np.ndarray:
        if self._samples is None:
            self._samples = np.load(self.dir / "samples.npy", mmap_mode="r")
        return self._samples

    def _opt(self, name):
        p = self.dir / name
        return np.load(p) if p.exists() else None

    @property
    def sub_idx(self):
        return self._opt("sub_idx.npy")

    @property
    def ll(self):
        return self._opt("ll.npy")

    @property
    def hdr_rank(self):
        return self._opt("hdr_rank.npy")

    @property
    def internals(self):
        p = self.dir / "internals.npz"
        return np.load(p) if p.exists() else None

    def crps(self) -> np.ndarray:
        """(N, D) per-event CRPS."""
        return np.column_stack(
            [self.index[f"crps_{d}"].to_numpy() for d in range(len(TARGET_NAMES))]
        )

    def pit(self) -> np.ndarray:
        """(N, D) per-event PIT."""
        return np.column_stack(
            [self.index[f"pit_{d}"].to_numpy() for d in range(len(TARGET_NAMES))]
        )

    def y_true(self) -> np.ndarray:
        """(N, D) truth targets."""
        return np.column_stack(
            [self.index[f"y_true_{d}"].to_numpy() for d in range(len(TARGET_NAMES))]
        )

    def __repr__(self):
        return (
            f"Dump({self.meta.get('model')}, seed={self.meta.get('seed')}, "
            f"N={self.meta.get('n_events'):,}, m={self.meta.get('m')})"
        )


def assert_dumps_aligned(*dumps) -> None:
    """
    Hard-fail unless every dump covers exactly the same events in the same
    order.
    """
    if len(dumps) < 2:
        return
    ref = dumps[0]
    ref_ids = ref.index["global_match_id"].to_numpy()
    for other in dumps[1:]:
        if other.meta.get("n_events") != ref.meta.get("n_events"):
            raise AssertionError(
                f"dump length mismatch: {ref!r} has {ref.meta.get('n_events')} "
                f"events, {other!r} has {other.meta.get('n_events')}"
            )
        other_ids = other.index["global_match_id"].to_numpy()
        if not np.array_equal(ref_ids, other_ids):
            n_bad = int(np.sum(ref_ids != other_ids))
            raise AssertionError(
                f"dump event alignment mismatch between {ref!r} and {other!r}: "
                f"{n_bad:,} of {len(ref_ids):,} global_match_id values differ. "
                "These dumps are NOT per-event comparable — check that seed, "
                "pT window and data path match."
            )
        rh, oh = ref.meta.get("global_match_id_sha256"), other.meta.get(
            "global_match_id_sha256"
        )
        if rh and oh and rh != oh:
            raise AssertionError(
                f"global_match_id hash mismatch between {ref!r} and {other!r} "
                "despite element-wise equality — meta.json is stale."
            )
