"""
Write a small, event-grouped sample of the dataset for the smoke run.

Keeps every row of the events whose (run, luminosityBlock, event) key falls in a
deterministic hash bucket, so train/val/test splits stay event-grouped and all
six flavours are represented.

    python Scripts/make_smoke_data.py [--fraction 0.02]
"""

import argparse
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Surrogates"))

from common import paths as _paths  # noqa: E402

N_BUCKETS = 10_000


def select_events(frame, fraction: float):
    key = pl.concat_str(
        [
            pl.col("run").cast(pl.Utf8),
            pl.col("luminosityBlock").cast(pl.Utf8),
            pl.col("event").cast(pl.Utf8),
        ],
        separator="_",
    )
    return frame.filter(key.hash(seed=0) % N_BUCKETS < int(round(fraction * N_BUCKETS)))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default=_paths.rel(_paths.DEFAULT_DATA))
    ap.add_argument("--out", default="Smoke/Data/dataset/smoke.parquet")
    ap.add_argument("--fraction", type=float, default=0.02)
    args = ap.parse_args(argv)

    out = _paths.REPO_ROOT / args.out
    if out.exists():
        print(f"[smoke_data] {args.out} exists, keeping it")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    df = select_events(
        pl.scan_parquet(_paths.REPO_ROOT / args.data), args.fraction
    ).collect()
    df.write_parquet(out)
    print(f"[smoke_data] wrote {args.out}: {df.height:,} rows")


if __name__ == "__main__":
    main()
