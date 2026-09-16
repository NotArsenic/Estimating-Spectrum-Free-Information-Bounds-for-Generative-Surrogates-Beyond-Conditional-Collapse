"""
Every path in the repository, resolved once, here.
"""

from __future__ import annotations

import os
from pathlib import Path

# common/ -> Surrogates/ -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]


def _artifact_root() -> Path:
    raw = os.environ.get("ARTIFACT_ROOT", "")
    if raw and Path(raw).is_absolute():
        raise ValueError("ARTIFACT_ROOT must be a repo-relative path")
    return REPO_ROOT / raw


ARTIFACTS = _artifact_root()

SURROGATES = REPO_ROOT / "Surrogates"
DATA_DIR = REPO_ROOT / "Data"
CONFIGS = REPO_ROOT / "Experiments" / "configs"
SCRIPTS = REPO_ROOT / "Scripts"
PAPER = REPO_ROOT / "Paper"
PAPER_OUT = ARTIFACTS / "Paper"

# training artifacts
OUTPUT = ARTIFACTS / "Surrogates" / "Output"
RUNS = OUTPUT / "runs"
REGISTRY_PATH = RUNS / "registry.jsonl"
RESOLVED_CONFIGS = RUNS / "resolved_configs"

# analysis artifacts
RESULTS = ARTIFACTS / "Results"
RESULTS_COMPARISON = RESULTS / "comparison"
RESULTS_CEILINGS = RESULTS / "ceilings"
RESULTS_ABLATION = RESULTS / "ablation"

# the two surrogates
MODEL_DIRNAME = {"mdn": "MDN", "fm": "CFM"}
MODEL_ENTRY = {
    "mdn": SURROGATES / "MDN" / "main.py",
    "fm": SURROGATES / "CFM" / "main.py",
}
MODEL_LONGNAME = {"mdn": "MDN_V2", "fm": "FlowMatching_SingleStage_V2"}

DEFAULT_DATA = DATA_DIR / "dataset" / "PJ_dataset_qcd_flat_15to7000.parquet"


def rel(p) -> str:
    """POSIX path of ``p`` relative to the repository root."""
    p = Path(p)
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        raise ValueError(f"path is outside the repository: {p.name}") from None


def model_output_root(model: str) -> Path:
    """``Output/MDN`` or ``Output/CFM``. Accepts 'mdn'/'fm' or 'MDN'/'CFM'."""
    return OUTPUT / MODEL_DIRNAME.get(model.lower(), model)


def save_root(model: str, stage: str = "parton2reco", ablation: bool = False) -> Path:
    base = model_output_root(model)
    if ablation:
        return base / "Experiments" / "ablation"
    if stage == "parton2reco":
        return base
    return base / "Experiments" / stage


def seed_dir(
    model: str,
    seed: int,
    stage: str = "parton2reco",
    run_tag: str = "",
    ablation: bool = False,
) -> Path:
    name = f"seed_{seed}" + (f"_{run_tag}" if run_tag else "")
    return save_root(model, stage, ablation) / name


def ensure_dirs() -> None:
    for p in (
        OUTPUT,
        RUNS,
        RESOLVED_CONFIGS,
        RESULTS,
        RESULTS_COMPARISON,
        RESULTS_CEILINGS,
        PAPER_OUT,
    ):
        p.mkdir(parents=True, exist_ok=True)
