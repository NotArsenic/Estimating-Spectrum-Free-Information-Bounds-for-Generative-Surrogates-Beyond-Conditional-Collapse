"""
Cone-narrowing positive control, per seed.
    python Experiments/cone_narrowing_control.py --seed 1
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


def output_path(seed: int) -> Path:
    return _paths.RESULTS_CEILINGS / f"cone_control_seed_{seed}.json"


def cone_key(cone: float) -> str:
    return f"{cone:.2f}"


def cone_drift(per_cone: dict, cones) -> dict:
    first, last = per_cone[cone_key(cones[0])], per_cone[cone_key(cones[-1])]
    out = {}
    for t in TARGET_NAMES:
        a, b = first["median_ratio"][t], last["median_ratio"][t]
        out[t] = None if a is None or b is None else float(b - a)
    return out


def plot_diagnostic(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cones = result["cones"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for t in TARGET_NAMES:
        ys = [result["per_cone"][cone_key(c)]["median_ratio"][t] for c in cones]
        ax.plot(cones, [np.nan if v is None else v for v in ys], "o-", label=t)
    ax.invert_xaxis()
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.legend(fontsize=8)
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

    cones = tuple(settings.CONES)
    out = output_path(args.seed)
    run_settings = analysis_settings(
        args.n_query, args.min_count, args.max_train_per_flavour, cones=list(cones)
    )
    if args.skip_if_current and out.exists():
        old = json.loads(out.read_text())
        if old.get("seed") == args.seed and old.get("settings") == run_settings:
            print(f"[cone] {_paths.rel(out)} matches the settings, skipping")
            return

    datasets, _, raw = prepare_datasets(args.data, seed=args.seed, stage="parton2reco")
    X_train, y_train = datasets["train"].X.numpy(), datasets["train"].y.numpy()
    X_test, y_test = datasets["test"].X.numpy(), datasets["test"].y.numpy()
    dR_train, dR_test = (
        raw["train"]["delta_R"].to_numpy(),
        raw["test"]["delta_R"].to_numpy(),
    )
    ids_train = raw["train"]["global_match_id"].to_numpy()
    ids_test = raw["test"]["global_match_id"].to_numpy()
    pt_test = raw["test"]["parton_pt"].to_numpy()

    per_cone = {}
    for cone in cones:
        tr = dR_train < cone
        te = np.flatnonzero(dR_test < cone)[: args.n_query]
        knn = KNNCeiling(
            X_train[tr],
            y_train[tr],
            seed=args.seed,
            max_train_per_flavour=args.max_train_per_flavour,
            train_ids=ids_train[tr],
            self_exclude="ids",
        )
        rec = ceiling_record(
            knn,
            X_test[te],
            y_test[te],
            pt_test[te],
            ids_test[te],
            min_count=args.min_count,
        )
        rec["n_train"] = int(tr.sum())
        rec["target_std"] = {
            t: float(y_test[te, d].std()) for d, t in enumerate(TARGET_NAMES)
        }
        per_cone[cone_key(cone)] = rec
        print(f"[cone] dR < {cone_key(cone)}: median ratio {rec['median_ratio']}")

    result = {
        "seed": args.seed,
        "crps_floor_estimator": settings.FLOOR_ESTIMATOR,
        "settings": run_settings,
        "cones": list(cones),
        "per_cone": per_cone,
        "drift": cone_drift(per_cone, cones),
    }
    write_json(out, result)
    plot_diagnostic(result, out.with_suffix(".png"))
    print(f"[cone] wrote {_paths.rel(out)}")


if __name__ == "__main__":
    main()
