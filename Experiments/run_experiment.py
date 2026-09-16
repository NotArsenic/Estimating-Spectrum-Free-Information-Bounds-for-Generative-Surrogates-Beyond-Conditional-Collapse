"""
Experiment runner and run registry.

Usage
-----
    python Experiments/run_experiment.py Experiments/configs/mdn_parton2reco.json
    python Experiments/run_experiment.py Experiments/configs/fm_parton2reco.json --mode dump --seeds 1
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Surrogates"))

from common import paths as _paths  # noqa: E402
from common import settings as _settings  # noqa: E402
from common.provenance import (
    check_no_abs_paths,
    checkpoint_sha256,
    write_json,
)  # noqa: E402

MODEL_MAIN = dict(_paths.MODEL_ENTRY)


def git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def canonical_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def identity_hash(merged: dict, model: str, stage: str) -> str:
    """Config hash identifying a run; W&B keys and the loader worker count are excluded."""
    ident = {
        k: v
        for k, v in merged.items()
        if not k.startswith("wandb_") and k != "num_workers"
    }
    return canonical_hash({**ident, "model": model, "stage": stage})


def parse_overrides(pairs) -> dict:
    out = {}
    for kv in pairs:
        key, _, value = kv.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def resolve_config(delta_path, base_path=None, epochs=None, overrides=None, data=None):
    """Return ``(model, stage, seeds, merged)``; ``merged`` has no model/seeds keys."""
    base = json.loads(Path(base_path or _paths.CONFIGS / "base.json").read_text())
    delta = json.loads(Path(delta_path).read_text())
    model = delta.get("model")
    if model not in MODEL_MAIN:
        raise ValueError(
            f'{Path(delta_path).name}: "model" must be one of {sorted(MODEL_MAIN)}'
        )
    stage = delta.get("stage")
    if not stage:
        raise ValueError(f'{Path(delta_path).name}: "stage" is required')
    merged = {**base, **delta, **dict(_settings.TRAIN_OVERRIDES)}
    if epochs is not None:
        merged["epochs"] = epochs
    merged.update(overrides or {})
    if data is not None:
        merged["data"] = data
    if Path(merged["data"]).is_absolute():
        raise ValueError("config 'data' must be a repo-relative path")
    seeds = list(merged.pop("seeds", None) or [])
    merged.pop("model", None)
    return model, stage, seeds, merged


def run_dir_for(model: str, stage: str, merged: dict, seed: int) -> Path:
    """Mirror of main.py's run-directory resolution."""
    save_root = merged.get("save_root")
    if save_root:
        root = _paths.model_output_root(model) / "Experiments" / Path(save_root).name
    else:
        root = _paths.save_root(model, stage)
    run_tag = merged.get("run_tag")
    return root / (f"seed_{seed}" + (f"_{run_tag}" if run_tag else ""))


def load_registry(path=None) -> list:
    path = Path(path or _paths.REGISTRY_PATH)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_registry(entry: dict, path=None) -> None:
    check_no_abs_paths(entry)
    path = Path(path or _paths.REGISTRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _read_json(path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def train_state(registry, config_hash, seed, run_dir, epochs) -> str:
    run_dir = Path(run_dir)
    ckpt, metrics, losses = (
        run_dir / "best_model.pt",
        run_dir / "metrics.json",
        run_dir / "losses.json",
    )
    registered = any(
        r.get("config_hash") == config_hash
        and r.get("seed") == seed
        and r.get("status") == "done"
        for r in registry
    )
    if registered and ckpt.exists() and metrics.exists():
        return "done"
    if ckpt.exists() and losses.exists() and not metrics.exists():
        n_epochs_logged = len(_read_json(losses).get("val_losses", []))
        trained_under = _read_json(run_dir / "stats.json").get("config_hash")
        if n_epochs_logged == epochs and trained_under == config_hash:
            return "eval_only"
    return "train"


def eval_common_current(run_dir) -> bool:
    ckpt, out = Path(run_dir) / "best_model.pt", Path(run_dir) / "metrics_common.json"
    if not (ckpt.exists() and out.exists()):
        return False
    return json.loads(out.read_text()).get("checkpoint_sha256") == checkpoint_sha256(
        ckpt
    )


def dump_current(run_dir, model, merged, limit, m) -> bool:
    ckpt, meta_path = (
        Path(run_dir) / "best_model.pt",
        Path(run_dir) / "dump" / "meta.json",
    )
    if not (ckpt.exists() and meta_path.exists()):
        return False
    meta = json.loads(meta_path.read_text())
    ok = (
        meta.get("checkpoint_sha256") == checkpoint_sha256(ckpt)
        and meta.get("limit") == limit
        and meta.get("m") == m
    )
    if model == "fm":
        ok = (
            ok
            and meta.get("n_steps") == merged.get("n_steps")
            and meta.get("solver") == merged.get("solver")
        )
    return ok


def build_command(
    model, resolved_path, config_hash, stage, seed, launch, merged
) -> list:
    cmd = [
        sys.executable,
        _paths.rel(MODEL_MAIN[model]),
        "--config",
        _paths.rel(resolved_path),
        "--config_hash",
        config_hash,
        "--stage",
        stage,
        "--seeds",
        str(seed),
    ]
    if launch == "eval_only":
        cmd += ["--eval_only"]
    elif launch == "eval_common":
        cmd += [
            "--eval_only",
            "--force_common_event_set",
            "--metrics_filename",
            "metrics_common.json",
        ]
    elif launch == "dump":
        cmd += [
            "--dump_only",
            "--dump_limit",
            str(_settings.SCORING_EVENTS),
            "--dump_m",
            str(_settings.DUMP_M),
            "--dump_subset",
            str(_settings.DUMP_SUBSET),
        ]
        if model == "fm":
            cmd += [
                "--dump_n_steps",
                str(merged["n_steps"]),
                "--solver",
                merged["solver"],
            ]
    elif launch != "train":
        raise ValueError(f"unknown launch mode {launch!r}")
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("delta", help="Experiments/configs/<id>.json")
    ap.add_argument("--base", default=None)
    ap.add_argument("--mode", choices=("train", "eval_common", "dump"), default="train")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--override", nargs="+", default=[], metavar="KEY=VALUE")
    ap.add_argument("--force", action="store_true", help="Ignore the skip rules.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    model, stage, seeds, merged = resolve_config(
        args.delta,
        args.base,
        args.epochs,
        parse_overrides(args.override),
        data=os.environ.get("DATA"),
    )
    seeds = args.seeds or seeds or list(_settings.SEEDS)
    config_hash = identity_hash(merged, model, stage)
    resolved = _paths.RESOLVED_CONFIGS / f"{config_hash}.json"
    write_json(resolved, merged)
    sha = git_sha()
    registry = load_registry()
    print(
        f"[run_experiment] mode={args.mode} model={model} stage={stage} "
        f"hash={config_hash} config={_paths.rel(args.delta)}"
    )

    for seed in seeds:
        run_dir = run_dir_for(model, stage, merged, seed)
        tag = f"[run_experiment] seed={seed} {_paths.rel(run_dir)}:"
        if args.mode == "train":
            state = (
                "train"
                if args.force
                else train_state(registry, config_hash, seed, run_dir, merged["epochs"])
            )
            if state == "done":
                shas = {
                    r.get("git_sha")
                    for r in registry
                    if r.get("config_hash") == config_hash
                    and r.get("seed") == seed
                    and r.get("status") == "done"
                }
                if sha not in shas:
                    print(f"{tag} registered under a different git SHA; not retraining")
                print(f"{tag} done, skipping")
                continue
            launch = state
        elif args.mode == "eval_common":
            if not args.force and eval_common_current(run_dir):
                print(f"{tag} metrics_common.json matches the checkpoint, skipping")
                continue
            launch = "eval_common"
        else:
            if not args.force and dump_current(
                run_dir, model, merged, _settings.SCORING_EVENTS, _settings.DUMP_M
            ):
                print(f"{tag} dump matches the checkpoint and settings, skipping")
                continue
            launch = "dump"

        cmd = build_command(model, resolved, config_hash, stage, seed, launch, merged)
        print(f"{tag} {launch}: python " + " ".join(cmd[1:]))
        if args.dry_run:
            continue
        t0 = time.time()
        returncode = subprocess.run(cmd, cwd=REPO).returncode
        if args.mode == "train":
            entry = {
                "config_hash": config_hash,
                "git_sha": sha,
                "model": model,
                "stage": stage,
                "seed": seed,
                "launch": launch,
                "status": "done" if returncode == 0 else "failed",
                "returncode": returncode,
                "run_dir": _paths.rel(run_dir),
                "metrics_path": _paths.rel(run_dir / "metrics.json"),
                "wall_time_s": round(time.time() - t0, 1),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            append_registry(entry)
            registry.append(entry)
        if returncode != 0:
            raise SystemExit(f"{tag} {launch} failed with exit code {returncode}")


if __name__ == "__main__":
    main()
