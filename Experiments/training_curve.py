"""
Per-seed validation curves and the overfit-arc numbers.
    python Experiments/training_curve.py --seed 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Surrogates"))
sys.path.insert(0, str(REPO / "Experiments"))

import paper_runs  # noqa: E402
from common import paths as _paths  # noqa: E402
from common.differential import nan_to_none  # noqa: E402
from common.features import TARGET_NAMES  # noqa: E402
from common.provenance import checkpoint_sha256, write_json  # noqa: E402


def output_path(seed: int) -> Path:
    return _paths.RESULTS_COMPARISON / f"training_curve_seed_{seed}.json"


def derive(losses: dict) -> dict:
    val = np.asarray(losses["val_losses"], dtype=float)
    per_target = {
        t: [float(h[f"crps_{t}"]) for h in losses["val_crps_history"]]
        for t in TARGET_NAMES
    }
    mean_crps = np.mean([per_target[t] for t in TARGET_NAMES], axis=0)
    n = len(val)
    best = int(np.nanargmin(val))
    nll_cost = float((val[-1] - val[best]) / abs(val[best]) * 100.0)
    crps_cost = float((mean_crps[-1] - mean_crps[best]) / mean_crps[best] * 100.0)
    return {
        "n_epochs": n,
        "best_val_loss_epoch": best + 1,
        "epochs_past_best": n - 1 - best,
        "val_loss_cost_pct": nll_cost,
        "mean_val_crps_cost_pct": crps_cost,
        "sensitivity_ratio": None if crps_cost == 0 else nll_cost / crps_cost,
        "val_losses": nan_to_none(val),
        "mean_val_crps": nan_to_none(mean_crps),
        "per_target_val_crps": per_target,
        "val_crps_config": losses.get("val_crps_config"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--skip_if_current", action="store_true")
    args = ap.parse_args(argv)

    out = output_path(args.seed)
    run_dirs = {
        s: paper_runs.run_dir("parton2reco", s, args.seed) for s in paper_runs.MODELS
    }
    for s, d in run_dirs.items():
        for f in ("losses.json", "best_model.pt"):
            if not (d / f).exists():
                raise SystemExit(f"[training_curve] missing {_paths.rel(d / f)}")
    shas = {s: checkpoint_sha256(d / "best_model.pt") for s, d in run_dirs.items()}
    if args.skip_if_current and out.exists():
        old = json.loads(out.read_text())["per_model"]
        if all(
            old.get(s, {}).get("checkpoint_sha256") == sha for s, sha in shas.items()
        ):
            print(
                f"[training_curve] {_paths.rel(out)} matches the checkpoints, skipping"
            )
            return

    result = {"seed": args.seed, "per_model": {}}
    for s, d in run_dirs.items():
        rec = derive(json.loads((d / "losses.json").read_text()))
        rec["checkpoint_sha256"] = shas[s]
        rec["losses"] = _paths.rel(d / "losses.json")
        result["per_model"][s] = rec
    write_json(out, result)
    print(f"[training_curve] wrote {_paths.rel(out)}")


if __name__ == "__main__":
    main()
