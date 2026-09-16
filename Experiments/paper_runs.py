"""
The runs the paper uses, and where each one's outputs live.

    python Experiments/paper_runs.py list [--group parton2reco|genjet2reco|ladder]
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Surrogates"))
sys.path.insert(0, str(REPO / "Experiments"))

from common import paths as _paths  # noqa: E402
from run_experiment import resolve_config, run_dir_for  # noqa: E402

MODELS = ("MDN", "CFM")
RUNGS = (
    "kinematics",
    "kinematics+pileup",
    "kinematics+pileup+fourier_phi",
    "kinematics+pileup+flavour",
)
LADDER_RUNGS = RUNGS + ("full",)
GROUPS = ("parton2reco", "genjet2reco", "ladder")


def load() -> dict:
    return json.loads((_paths.CONFIGS / "paper_runs.json").read_text())


def config_path(group: str, model: str, rung: str | None = None) -> Path:
    if group == "ladder" and rung == "full":
        group, rung = "parton2reco", None
    entry = load()[group][model]
    return _paths.CONFIGS / (entry[rung] if group == "ladder" else entry)


def config_paths(group: str | None = None) -> list:
    out = []
    for g in ([group] if group else GROUPS):
        for m in MODELS:
            if g == "ladder":
                out += [config_path(g, m, r) for r in RUNGS]
            else:
                out.append(config_path(g, m))
    return out


def resolved(group: str, model: str, rung: str | None = None):
    return resolve_config(config_path(group, model, rung))


def run_dir(group: str, model: str, seed: int, rung: str | None = None) -> Path:
    model_key, stage, _, merged = resolved(group, model, rung)
    return run_dir_for(model_key, stage, merged, seed)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--group", choices=GROUPS, default=None)
    args = ap.parse_args(argv)
    for p in config_paths(args.group):
        print(_paths.rel(p))


if __name__ == "__main__":
    main()
