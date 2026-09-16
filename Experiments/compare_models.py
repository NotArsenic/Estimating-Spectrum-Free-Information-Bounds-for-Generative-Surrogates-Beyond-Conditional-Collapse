"""
Per-seed cross-model comparison on the paper's scoring events.

    python Experiments/compare_models.py --seed 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Surrogates"))
sys.path.insert(0, str(REPO / "Experiments"))

import paper_runs  # noqa: E402
from common import paths as _paths  # noqa: E402
from common import settings  # noqa: E402
from common.baselines import (
    KNNCeiling,
    LookupTableSampler,
    MarginalSampler,
)  # noqa: E402
from common.dataset import prepare_datasets  # noqa: E402
from common.differential import (
    by_target,
    knn_differential,
    nan_to_none,  # noqa: E402
    series_differential,
)
from common.dump import Dump, assert_dumps_aligned  # noqa: E402
from common.features import TARGET_NAMES  # noqa: E402
from common.metrics import crps_per_event, crps_unconditional_floor  # noqa: E402
from common.provenance import write_json  # noqa: E402

MODEL_NAMES = ("MDN_V2", "FlowMatching_SingleStage_V2")
MODEL_SHORT = {"MDN_V2": "MDN", "FlowMatching_SingleStage_V2": "CFM"}


def summary_path(seed: int) -> Path:
    return _paths.RESULTS_COMPARISON / f"seed_{seed}" / "comparison_summary.json"


def compute_summary(
    dumps,
    X_train,
    y_train,
    train_ids,
    X_test,
    test_ids,
    seed,
    *,
    scoring_events=settings.SCORING_EVENTS,
    max_train_per_flavour=settings.MAX_TRAIN_PER_FLAVOUR,
    min_count=settings.MIN_COUNT,
    overlap_rows=settings.OVERLAP_ROWS,
    ks=settings.K_LADDER,
    edges=settings.PT_EDGES,
):
    ref = dumps["MDN_V2"]
    n = min(scoring_events, len(ref.index))
    y = ref.y_true()[:n]
    ids = ref.index["global_match_id"].to_numpy()[:n]
    pt = ref.index["parton_pt"].to_numpy()[:n]
    if not np.array_equal(np.asarray(test_ids)[:n], ids):
        raise SystemExit(
            "[compare] dump events differ from this seed's test split -- "
            "the dump was written under another seed or pT window"
        )
    X = np.asarray(X_test)[:n]
    m = int(ref.meta["m"])

    fl = crps_unconditional_floor(y)
    floors = {t: float(fl[f"crps_floor_{t}"]) for t in TARGET_NAMES}
    floor_arr = np.array([floors[t] for t in TARGET_NAMES])

    crps = {name: dumps[name].crps()[:n] for name in MODEL_NAMES}
    samples = {
        "Marginal": MarginalSampler(y_train, seed=seed).sample(X, m),
        "LookupTable": LookupTableSampler(X_train, y_train, seed=seed).sample(X, m),
    }
    for name, s in samples.items():
        crps[name] = crps_per_event(y, s)
    ratio = {
        name: dict(zip(TARGET_NAMES, (c.mean(axis=0) / floor_arr).tolist()))
        for name, c in crps.items()
    }

    knn = KNNCeiling(
        X_train,
        y_train,
        seed=seed,
        max_train_per_flavour=max_train_per_flavour,
        train_ids=train_ids,
        self_exclude="ids",
    )
    sweep = knn.sweep_k(X, y, ks=tuple(ks), query_ids=ids)
    diff = knn_differential(knn, X, y, pt, edges, ks, min_count, ids)
    series = {
        name: series_differential(c, y, pt, edges, min_count)
        for name, c in crps.items()
    }

    inputs = {}
    for name in MODEL_NAMES:
        meta = dumps[name].meta
        inputs[name] = {
            "checkpoint_sha256": meta.get("checkpoint_sha256"),
            "global_match_id_sha256": meta.get("global_match_id_sha256"),
        }
        if name == "FlowMatching_SingleStage_V2":
            inputs[name].update(
                {"solver": meta.get("solver"), "n_steps": meta.get("n_steps")}
            )

    summary = {
        "seed": seed,
        "crps_floor_estimator": settings.FLOOR_ESTIMATOR,
        "events": {
            "n": int(n),
            "m": m,
            "global_match_id_sha256": ref.meta.get("global_match_id_sha256"),
        },
        "inputs": inputs,
        "floors": floors,
        "spectrum_averaged_crps_ratio": ratio,
        "knn": {
            "ks": list(ks),
            "max_train_per_flavour": int(max_train_per_flavour),
            "self_exclusion": "ids",
            "global_sweep": {
                "radius": [float(r) for r in sweep["radius"]],
                "per_target": {
                    t: {
                        "crps": [float(v) for v in sweep["crps"][:, d]],
                        "slope": float(sweep["slope"][d]),
                        "slope_se": float(sweep["slope_se"][d]),
                        "intercept": float(sweep["crps_extrapolated"][d]),
                        "intercept_se": float(sweep["intercept_se"][d]),
                        "intercept_ratio": float(
                            sweep["crps_extrapolated"][d] / floor_arr[d]
                        ),
                    }
                    for d, t in enumerate(TARGET_NAMES)
                },
            },
            "differential": {
                "edges": [float(e) for e in diff["edges"]],
                "centers": [float(c) for c in diff["centers"]],
                "counts": [int(c) for c in diff["counts"]],
                "min_count": int(min_count),
                "floors": by_target(diff["floors"]),
                "per_target": {
                    t: {
                        "intercept": nan_to_none(diff["intercept"][:, d]),
                        "slope": nan_to_none(diff["slope"][:, d]),
                        "ratio": nan_to_none(diff["ratio"][:, d]),
                    }
                    for d, t in enumerate(TARGET_NAMES)
                },
            },
            "diagnostics": {
                "zero_distance_fraction": knn.zero_distance_fraction(X, ids),
                "self_overlap_deflation": knn.self_overlap_deflation(
                    X_train, overlap_rows, ks=tuple(ks), seed=seed
                ),
            },
        },
        "differential_series": {name: by_target(r) for name, r in series.items()},
    }
    extras = {
        "crps": crps,
        "samples": samples,
        "sweep": sweep,
        "diff": diff,
        "floor": floor_arr,
    }
    return summary, extras


def summary_is_current(path, dumps, scoring_events, min_count, ks) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    s = json.loads(path.read_text())
    n_expected = min(scoring_events, int(dumps["MDN_V2"].meta["n_events"]))
    return (
        all(
            s["inputs"].get(name, {}).get("checkpoint_sha256")
            == dumps[name].meta.get("checkpoint_sha256")
            for name in MODEL_NAMES
        )
        and s["events"]["n"] == n_expected
        and s["knn"]["ks"] == list(ks)
        and s["knn"]["differential"]["min_count"] == min_count
    )


def write_figures(
    dumps, extras, out_dir, wandb_project=None, wandb_entity=None, seed=None
):
    from common import plots_dump as pd_

    ordered = [dumps[name] for name in MODEL_NAMES]
    n_dump = len(ordered[0].index)

    def pad(a):
        out = np.full((n_dump, a.shape[1]), np.nan)
        out[: len(a)] = a
        return out

    pd_.plot_all_dump(
        ordered,
        save_dir=str(out_dir),
        losses_paths={
            name: str(Path(d.dir).parent / "losses.json") for name, d in dumps.items()
        },
        baselines={k: pad(extras["crps"][k]) for k in ("Marginal", "LookupTable")},
        baseline_samples=extras["samples"],
        knn_curve={
            "centers": extras["diff"]["centers"],
            "ratio": extras["diff"]["ratio"],
        },
        knn_sweep=extras["sweep"],
        knn_floor=extras["floor"],
        p1_only=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=None if seed is None else f"comparison_seed{seed}",
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", default=settings.DATA_PATH)
    ap.add_argument("--scoring_events", type=int, default=settings.SCORING_EVENTS)
    ap.add_argument(
        "--knn_max_train_per_flavour", type=int, default=settings.MAX_TRAIN_PER_FLAVOUR
    )
    ap.add_argument("--min_count", type=int, default=settings.MIN_COUNT)
    ap.add_argument("--overlap_rows", type=int, default=settings.OVERLAP_ROWS)
    ap.add_argument(
        "--skip_if_current",
        action="store_true",
        help="Do nothing if the summary already matches both dumps and settings.",
    )
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args(argv)

    dirs = {
        "MDN_V2": paper_runs.run_dir("parton2reco", "MDN", args.seed) / "dump",
        "FlowMatching_SingleStage_V2": paper_runs.run_dir(
            "parton2reco", "CFM", args.seed
        )
        / "dump",
    }
    missing = [_paths.rel(d) for d in dirs.values() if not (d / "meta.json").exists()]
    if missing:
        raise SystemExit(f"[compare] missing dump(s): {missing}")
    dumps = {name: Dump(d) for name, d in dirs.items()}
    out = summary_path(args.seed)
    if args.skip_if_current and summary_is_current(
        out, dumps, args.scoring_events, args.min_count, settings.K_LADDER
    ):
        print(f"[compare] {_paths.rel(out)} matches the dumps and settings, skipping")
        return
    assert_dumps_aligned(dumps["MDN_V2"], dumps["FlowMatching_SingleStage_V2"])

    datasets, _, raw = prepare_datasets(args.data, seed=args.seed)
    summary, extras = compute_summary(
        dumps,
        datasets["train"].X.numpy(),
        datasets["train"].y.numpy(),
        raw["train"]["global_match_id"].to_numpy(),
        datasets["test"].X.numpy(),
        raw["test"]["global_match_id"].to_numpy(),
        args.seed,
        scoring_events=args.scoring_events,
        max_train_per_flavour=args.knn_max_train_per_flavour,
        min_count=args.min_count,
        overlap_rows=args.overlap_rows,
    )
    for name, d in dirs.items():
        summary["inputs"][name]["dump"] = _paths.rel(d)
    write_json(out, summary)
    print(f"[compare] wrote {_paths.rel(out)}")
    if not args.no_figures:
        write_figures(
            dumps,
            extras,
            out.parent,
            os.environ.get("WANDB_PROJECT"),
            os.environ.get("WANDB_ENTITY"),
            args.seed,
        )


if __name__ == "__main__":
    main()
