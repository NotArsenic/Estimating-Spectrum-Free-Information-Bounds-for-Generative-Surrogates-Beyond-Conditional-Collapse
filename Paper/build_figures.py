#!/usr/bin/env python
"""
Draw every paper figure from Paper/figure_data/*.json (written by build_numbers.py).

    python Paper/build_figures.py

Nothing here computes a number. Figures are written to Paper/figures/<name>.pdf
(under ARTIFACT_ROOT when it is set).
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _sub in ("Surrogates", "Experiments", "Paper"):
    if str(REPO / _sub) not in sys.path:
        sys.path.insert(0, str(REPO / _sub))

from common import paths as _paths  # noqa: E402
from common.paper_figures import DRAW  # noqa: E402
from paperkit.figure_data import FIGURES  # noqa: E402


def build_all(verbose: bool = True) -> dict:
    data_dir = _paths.PAPER_OUT / "figure_data"
    out_dir = _paths.PAPER_OUT / "figures"
    missing = [n for n in FIGURES if not (data_dir / f"{n}.json").exists()]
    if missing:
        raise SystemExit(
            f"[build_figures] missing figure data {missing}; "
            "run Paper/build_numbers.py first"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {
        name: DRAW[name](
            json.loads((data_dir / f"{name}.json").read_text()), out_dir / f"{name}.pdf"
        )
        for name in FIGURES
    }
    if verbose:
        print(f"\n{'figure':<28s} {'w (in)':>8s} {'h (in)':>8s}")
        for name, (w, h) in sizes.items():
            print(f"{name:<28s} {w:>8.3f} {h:>8.3f}")
    return sizes


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build_all()
