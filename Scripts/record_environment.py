"""
Record the software and hardware a pipeline run used (no hostname, user or paths).

    python Scripts/record_environment.py [--require_cuda]
"""

import argparse
import importlib
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Surrogates"))

from common import paths as _paths  # noqa: E402
from common.provenance import write_json  # noqa: E402

PACKAGES = ("numpy", "polars", "scipy", "matplotlib", "torch")


def environment_record() -> dict:
    rec = {"python": platform.python_version()}
    for name in PACKAGES:
        rec[name] = importlib.import_module(name).__version__
    import torch
    rec["cuda"] = torch.version.cuda
    rec["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        rec["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                                 stderr=subprocess.DEVNULL).decode().strip()
        rec["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO,
                                                        stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        rec["git_sha"], rec["git_dirty"] = None, None
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require_cuda", action="store_true")
    args = ap.parse_args(argv)
    rec = environment_record()
    if args.require_cuda and rec["gpu"] is None:
        raise SystemExit("[environment] CUDA is not available but GPU phases were requested")
    write_json(_paths.RESULTS / "environment.json", rec)
    print(f"[environment] {rec}")


if __name__ == "__main__":
    main()
