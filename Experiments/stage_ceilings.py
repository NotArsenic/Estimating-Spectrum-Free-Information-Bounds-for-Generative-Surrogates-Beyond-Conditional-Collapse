"""
Model-free differential k-NN ceilings for the three conditioning stages, per seed.

    python Experiments/stage_ceilings.py --seed 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Surrogates"))

from common import paths as _paths  # noqa: E402
from common import settings  # noqa: E402
from common.baselines import KNNCeiling  # noqa: E402
from common.dataset import prepare_datasets  # noqa: E402
from common.differential import analysis_settings, ceiling_record  # noqa: E402
from common.features import TARGET_NAMES  # noqa: E402
from common.provenance import write_json  # noqa: E402

STAGES = ("parton2reco", "parton2genjet", "genjet2reco")


def output_path(seed: int) -> Path:
    return _paths.RESULTS_CEILINGS / f"stage_ceilings_seed_{seed}.json"


def plot_diagnostic(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(TARGET_NAMES), figsize=(4.6 * len(TARGET_NAMES), 4.2)
    )
    for ax, t in zip(axes, TARGET_NAMES):
        for stage, rec in result["stages"].items():
            r = np.array([np.nan if v is None else v for v in rec["ratio"][t]])
            ax.plot(rec["centers"], r, "o-", ms=3, label=stage)
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(t)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", default=settings.DATA_PATH)
    ap.add_argument("--n_query", type=int, default=settings.SCORING_EVENTS)
    ap.add_argument("--min_count", type=int, default=settings.MIN_COUNT)
    ap.add_argument(
        "--max_train_per_flavour", type=int, default=settings.MAX_TRAIN_PER_FLAVOUR
    )
    ap.add_argument("--skip_if_current", action="store_true")
    args = ap.parse_args(argv)

    out = output_path(args.seed)
    run_settings = analysis_settings(
        args.n_query, args.min_count, args.max_train_per_flavour
    )
    if args.skip_if_current and out.exists():
        old = json.loads(out.read_text())
        if old.get("seed") == args.seed and old.get("settings") == run_settings:
            print(f"[stages] {_paths.rel(out)} matches the settings, skipping")
            return

    result = {
        "seed": args.seed,
        "crps_floor_estimator": settings.FLOOR_ESTIMATOR,
        "settings": run_settings,
        "stages": {},
    }
    for stage in STAGES:
        datasets, _, raw = prepare_datasets(
            args.data, seed=args.seed, stage=stage, require_common_event_set=True
        )
        X_train, y_train = datasets["train"].X.numpy(), datasets["train"].y.numpy()
        n_q = min(args.n_query, len(datasets["test"]))
        knn = KNNCeiling(
            X_train,
            y_train,
            seed=args.seed,
            max_train_per_flavour=args.max_train_per_flavour,
            train_ids=raw["train"]["global_match_id"].to_numpy(),
            self_exclude="ids",
        )
        rec = ceiling_record(
            knn,
            datasets["test"].X.numpy()[:n_q],
            datasets["test"].y.numpy()[:n_q],
            raw["test"]["parton_pt"].to_numpy()[:n_q],
            raw["test"]["global_match_id"].to_numpy()[:n_q],
            min_count=args.min_count,
        )
        rec["n_train"] = int(len(X_train))
        result["stages"][stage] = rec
        print(f"[stages] {stage}: median ratio {rec['median_ratio']}")

    write_json(out, result)
    plot_diagnostic(result, out.with_suffix(".png"))
    print(f"[stages] wrote {_paths.rel(out)}")


if __name__ == "__main__":
    main()
