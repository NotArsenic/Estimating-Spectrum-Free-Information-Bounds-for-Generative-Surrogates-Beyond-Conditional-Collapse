"""
Write ablation configs: a copy of a model's baseline config with one key changed.

    python Experiments/gen_ablation_configs.py --model mdn --param width --values 256 1024
    python Experiments/gen_ablation_configs.py --model fm --param conditioner_blocks \
        --values '["kinematics"]' '["kinematics","pileup"]'
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Surrogates"))

from common import paths as _paths  # noqa: E402

MODEL_ONLY_PARAMS = {
    "n_components": "mdn",
    "lambda_calib": "mdn",
    "n_steps": "fm",
    "solver": "fm",
}
RESERVED = ("model", "stage", "seeds", "save_root", "run_tag")


def tag(value) -> str:
    if isinstance(value, list):
        return "+".join(str(x) for x in value)
    return str(value).replace(".", "p").replace("-", "m")


def build_ablation_config(baseline: dict, param: str, value, seeds=(1, 2, 3)) -> dict:
    if param in RESERVED:
        raise ValueError(f"cannot sweep {param!r}")
    owner = MODEL_ONLY_PARAMS.get(param)
    if owner is not None and owner != baseline["model"]:
        raise ValueError(f"{param!r} is a {owner}-only parameter")
    if param == "conditioner_blocks":
        if not isinstance(value, list) or "kinematics" not in value:
            raise ValueError(
                "conditioner_blocks must be a JSON list that contains 'kinematics'"
            )
        if "continuous" in value:
            raise ValueError(
                "use the fine-grained block names; 'continuous' is retired"
            )
    cfg = dict(baseline)
    cfg[param] = value
    cfg["seeds"] = list(seeds)
    cfg["save_root"] = "ablation"
    cfg["run_tag"] = f"{param}{tag(value)}"
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", choices=["mdn", "fm"], required=True)
    ap.add_argument("--stage", default="parton2reco")
    ap.add_argument("--param", required=True)
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args(argv)

    baseline = json.loads(
        (_paths.CONFIGS / f"{args.model}_{args.stage}.json").read_text()
    )
    out_dir = _paths.CONFIGS / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for raw in args.values:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        cfg = build_ablation_config(baseline, args.param, value, args.seeds)
        path = out_dir / f"{args.model}_{args.stage}_{args.param}_{tag(value)}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"[gen_ablation] wrote {_paths.rel(path)}")


if __name__ == "__main__":
    main()
