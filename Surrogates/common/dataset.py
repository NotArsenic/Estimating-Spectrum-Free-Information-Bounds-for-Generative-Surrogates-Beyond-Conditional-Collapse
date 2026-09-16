"""
Dataset loading, splitting, and standardisation for V2 surrogates.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)


from .settings import PT_MIN as DEFAULT_PT_MIN, PT_MAX as DEFAULT_PT_MAX


from .features import (
    eng_conditioner,
    eng_targets,
    residual_cone_radius,
    CONDITIONER_NAMES,
    TARGET_NAMES,
    STAGE_CONFIGS,
    ALL_CONDITIONER_BLOCKS,
    _DR_CONE,
)
from .provenance import write_json


# PyTorch Dataset wrapper
class JetDataset(Dataset):
    """In-memory dataset returning (conditioner, target) tensor pairs."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Data loading & filtering
def load_parquet(
    path: str,
    drop_negative_genjet_mass: bool = True,
    pt_min: float | None = DEFAULT_PT_MIN,
    pt_max: float | None = DEFAULT_PT_MAX,
    pt_col: str = "parton_pt",
    require_common_event_set: bool = False,
) -> pl.DataFrame:
    """
    Load the parquet dataset and apply universal filters.
    """
    df = pl.read_parquet(path)

    if require_common_event_set:
        n_before = len(df)
        df = df.filter(
            pl.col("has_genjet_match").cast(pl.Boolean)
            & (pl.col("genjet_mass") > 0)
            & (pl.col("genjet_partonFlavour") != 0)
        )
        print(
            f"[dataset] common event-set filter: {len(df):,} / {n_before:,} "
            "rows kept (has_genjet_match & genjet_mass>0 & "
            "genjet_partonFlavour!=0)"
        )

    if (pt_min is not None or pt_max is not None) and pt_col in df.columns:
        n_before = len(df)
        if pt_min is not None:
            df = df.filter(pl.col(pt_col) >= pt_min)
        if pt_max is not None:
            df = df.filter(pl.col(pt_col) < pt_max)
        lo = "-inf" if pt_min is None else f"{pt_min:g}"
        hi = "inf" if pt_max is None else f"{pt_max:g}"
        print(
            f"[dataset] {pt_col} window [{lo}, {hi}): "
            f"{len(df):,} / {n_before:,} rows kept"
        )
        if len(df) == 0:
            raise ValueError(f"pT window [{lo}, {hi}) on '{pt_col}' selected 0 rows.")

    if drop_negative_genjet_mass and "genjet_mass" in df.columns:
        # Only filter rows that have a genjet match AND bad mass
        mask = df["has_genjet_match"].cast(pl.Boolean) & (df["genjet_mass"] <= 0)
        n_bad = mask.sum()
        if n_bad > 0:
            print(f"[dataset] Dropping {n_bad} rows with genjet_mass <= 0")
            df = df.filter(~mask)

    return df


# Grouped splitting on `event`
def split_by_event(
    df: pl.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a DataFrame into train / val / test by unique event.

    Two jets from the same event always land in the same split.

    Parameters
    ----------
    df : polars DataFrame with ``run``, ``luminosityBlock``, ``event`` columns.
    val_frac, test_frac : fractions of *events* held out.
    seed : random seed for reproducibility.

    Returns
    -------
    df_train, df_val, df_test
    """
    df = df.with_columns(
        pl.concat_str(
            [
                pl.col("run").cast(pl.Utf8),
                pl.col("luminosityBlock").cast(pl.Utf8),
                pl.col("event").cast(pl.Utf8),
            ],
            separator="_",
        ).alias("_event_key")
    )

    keys = df.select("_event_key").unique().sort("_event_key").to_series().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_test - n_val

    test_keys = keys[:n_test]
    val_keys = keys[n_test : n_test + n_val]
    train_keys = keys[n_test + n_val :]

    df_test = df.filter(pl.col("_event_key").is_in(test_keys)).drop("_event_key")
    df_val = df.filter(pl.col("_event_key").is_in(val_keys)).drop("_event_key")
    df_train = df.filter(pl.col("_event_key").is_in(train_keys)).drop("_event_key")

    shuffle_seed = seed + 10_000
    df_test = df_test.sample(fraction=1.0, shuffle=True, seed=shuffle_seed)
    df_val = df_val.sample(fraction=1.0, shuffle=True, seed=shuffle_seed + 1)

    print(
        f"[dataset] Split rows:   train={len(df_train):,}  "
        f"val={len(df_val):,}  test={len(df_test):,}"
    )
    print(
        f"[dataset] Split events: train={n_train:,}  " f"val={n_val:,}  test={n_test:,}"
    )

    return df_train, df_val, df_test


# Full pipeline: load -> split -> featurise -> standardise -> wrap
def prepare_datasets(
    data_path: str,
    source_prefix: str | None = None,
    target_prefix: str | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    pt_min: float | None = DEFAULT_PT_MIN,
    pt_max: float | None = DEFAULT_PT_MAX,
    stage: str = "parton2reco",
    require_common_event_set: bool | None = None,
    conditioner_blocks: list[str] | None = None,
):
    """
    End-to-end data preparation.

    Returns
    -------
    datasets : dict  {"train": JetDataset, "val": JetDataset, "test": JetDataset}
    stats : dict  standardisation parameters (computed on train only)
    raw_splits : dict  {"train": pl.DataFrame, ...}  for evaluation / plotting
    """
    if stage not in STAGE_CONFIGS:
        raise ValueError(
            f"Unknown stage {stage!r}; expected one of {sorted(STAGE_CONFIGS)}"
        )
    cfg = STAGE_CONFIGS[stage]
    source_prefix = source_prefix or cfg["source_prefix"]
    target_prefix = target_prefix or cfg["target_prefix"]
    flavour_col = cfg["flavour_col"] or f"{source_prefix}_pdgId"
    if require_common_event_set is None:
        require_common_event_set = cfg["dr_cone_fixed"] is None

    df = load_parquet(
        data_path,
        pt_min=pt_min,
        pt_max=pt_max,
        pt_col="parton_pt",
        require_common_event_set=require_common_event_set,
    )
    df_train, df_val, df_test = split_by_event(
        df,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    # Conditioner (train stats used for all splits)
    X_train, cond_names, cond_stats = eng_conditioner(
        df_train,
        source_prefix=source_prefix,
        flavour_col=flavour_col,
        blocks=conditioner_blocks,
    )
    X_val, _, _ = eng_conditioner(
        df_val,
        source_prefix=source_prefix,
        stats=cond_stats,
        flavour_col=flavour_col,
        blocks=conditioner_blocks,
    )
    X_test, _, _ = eng_conditioner(
        df_test,
        source_prefix=source_prefix,
        stats=cond_stats,
        flavour_col=flavour_col,
        blocks=conditioner_blocks,
    )

    # Targets
    y_train, tgt_names = eng_targets(
        df_train,
        source_prefix=source_prefix,
        target_prefix=target_prefix,
    )
    y_val, _ = eng_targets(
        df_val,
        source_prefix=source_prefix,
        target_prefix=target_prefix,
    )
    y_test, _ = eng_targets(
        df_test,
        source_prefix=source_prefix,
        target_prefix=target_prefix,
    )

    # Target standardisation stats (train only)
    y_mean = y_train.astype(np.float64).mean(axis=0).astype(np.float32)
    y_std = y_train.astype(np.float64).std(axis=0).astype(np.float32)

    # Disk-clamp cone
    if cfg["dr_cone_fixed"] is not None:
        dr_cone = float(cfg["dr_cone_fixed"])
    else:
        dr_cone = float(residual_cone_radius(y_train).max()) * 1.05

    stats = {
        **cond_stats,
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "cond_names": cond_names,
        "conditioner_blocks": (
            list(conditioner_blocks)
            if conditioner_blocks is not None
            else list(ALL_CONDITIONER_BLOCKS)
        ),
        "input_dim": X_train.shape[1],
        "target_names": tgt_names,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "pt_min": pt_min,
        "pt_max": pt_max,
        "pt_col": "parton_pt",
        "stage": stage,
        "source_prefix": source_prefix,
        "target_prefix": target_prefix,
        "flavour_col": flavour_col,
        "dr_cone": dr_cone,
        "require_common_event_set": require_common_event_set,
    }

    # Pileup profile for inference sampling
    nTrueInt_train = np.asarray(df_train["pileup_nTrueInt"], dtype=np.float64)
    stats["nTrueInt_quantiles"] = (
        np.quantile(nTrueInt_train, np.linspace(0.0, 1.0, 2001))
        .astype(np.float32)
        .tolist()
    )

    # Sanity print
    print("[dataset] Target moments (train):")
    for i, name in enumerate(tgt_names):
        print(f"  {name:20s}  mean={y_mean[i]:.4f}  std={y_std[i]:.4f}")

    datasets = {
        "train": JetDataset(X_train, y_train),
        "val": JetDataset(X_val, y_val),
        "test": JetDataset(X_test, y_test),
    }
    raw_splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }

    return datasets, stats, raw_splits


def make_loader(
    dataset: JetDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader that fetches whole batches.
    """
    sampler = BatchSampler(
        RandomSampler(dataset) if shuffle else SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=drop_last,
    )
    return DataLoader(
        dataset,
        batch_size=None,  # sampler already yields batches
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def save_stats(stats: dict, path: str):
    """Save standardisation stats to JSON (numpy arrays -> lists)."""
    out = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v

    write_json(path, out, allow_nan=True)


def load_stats(path: str) -> dict:
    """Load standardisation stats from JSON."""
    with open(path) as f:
        return json.load(f)


def sample_nTrueInt(
    stats: dict,
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample pileup (``nTrueInt``) from the compact quantile grid in ``stats``.
    """
    if rng is None:
        rng = np.random.default_rng()
    q = np.asarray(stats.get("nTrueInt_quantiles", [21.8, 21.8]), dtype=np.float64)
    grid = np.linspace(0.0, 1.0, len(q))
    u = rng.random(n)
    return np.interp(u, grid, q)
