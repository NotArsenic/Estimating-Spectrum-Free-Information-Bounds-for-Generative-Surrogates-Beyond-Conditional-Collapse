"""
Metric trap, per seed: a condition-blind Marginal sampler against the trained
models on marginal KL and 1-Wasserstein.

    python Experiments/dummy_sampler_baseline.py --seed 1
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Surrogates"))
sys.path.insert(0, str(REPO / "Experiments"))

import paper_runs  # noqa: E402
from common import paths as _paths  # noqa: E402
from common import settings  # noqa: E402
from common.baselines import MarginalSampler  # noqa: E402
from common.dataset import prepare_datasets  # noqa: E402
from common.features import TARGET_NAMES  # noqa: E402
from common.metrics import compute_kl_divergence, compute_wasserstein  # noqa: E402
from common.provenance import write_json  # noqa: E402


def output_path(seed: int) -> Path:
    return _paths.RESULTS_COMPARISON / f"dummy_sampler_seed_{seed}.json"


def metric_trap_ratios(dummy: dict, metrics: dict) -> dict:
    out = {}
    for t in TARGET_NAMES:
        rec = {"kl_dummy": float(dummy[f"kl_{t}"]), "w1_dummy": float(dummy[f"w1_{t}"])}
        for short, m in metrics.items():
            key = short.lower()
            for div in ("kl", "w1"):
                rec[f"{div}_{key}"] = float(m[f"{div}_{t}"])
                rec[f"{div}_ratio_{key}_over_dummy"] = (
                    rec[f"{div}_{key}"] / rec[f"{div}_dummy"]
                )
        out[t] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", default=settings.DATA_PATH)
    ap.add_argument("--skip_if_current", action="store_true")
    args = ap.parse_args(argv)

    paths = {
        s: paper_runs.run_dir("parton2reco", s, args.seed) / "metrics.json"
        for s in paper_runs.MODELS
    }
    missing = [_paths.rel(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"[dummy] missing metrics: {missing}")
    metrics = {s: json.loads(p.read_text()) for s, p in paths.items()}
    shas = {s: m.get("checkpoint_sha256") for s, m in metrics.items()}
    out = output_path(args.seed)
    if (
        args.skip_if_current
        and out.exists()
        and json.loads(out.read_text()).get("checkpoint_sha256") == shas
    ):
        print(f"[dummy] {_paths.rel(out)} matches the checkpoints, skipping")
        return

    datasets, _, _ = prepare_datasets(args.data, seed=args.seed)
    y_train = datasets["train"].y.numpy()
    y_test = datasets["test"].y.numpy()
    for s, m in metrics.items():
        if m.get("n_test") != len(y_test):
            raise SystemExit(
                f"[dummy] {s} metrics.json has n_test={m.get('n_test')}, "
                f"test split has {len(y_test)}"
            )
    draw = MarginalSampler(y_train, seed=args.seed).sample(y_test, 1)[:, 0, :]
    dummy = {**compute_kl_divergence(y_test, draw), **compute_wasserstein(y_test, draw)}

    write_json(
        out,
        {
            "description": "model / Marginal-sampler ratios of marginal KL and W1 "
            "on the full test split; > 1 means the model is worse",
            "seed": args.seed,
            "n_test": int(len(y_test)),
            "metrics": {s: _paths.rel(p) for s, p in paths.items()},
            "checkpoint_sha256": shas,
            "per_target": metric_trap_ratios(dummy, metrics),
        },
    )
    print(f"[dummy] wrote {_paths.rel(out)}")


if __name__ == "__main__":
    main()
