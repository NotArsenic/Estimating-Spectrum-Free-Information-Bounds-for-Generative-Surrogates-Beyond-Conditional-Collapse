"""
Plot a metric vs. swept hyperparameter

Usage
-----
    python plot_ablation.py --model mdn --stage parton2reco --param width \
        --extra_point 512=Surrogates/Output/MDN/seed_1/metrics.json \
        --out Results/ablation/mdn_parton2reco_width.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Surrogates"))

from common import paths as _paths
from common.provenance import write_json

REGISTRY_PATH = _paths.REGISTRY_PATH
RESOLVED_DIR = _paths.RESOLVED_CONFIGS

_DEFAULT_METRICS = [
    "mean_log_likelihood",
    "crps_ratio_log_pT_resp",
    "crps_ratio_log_mass_frac",
]
_MODEL_COLORS = {"mdn": "#d62728", "fm": "#1f77b4"}

_BASELINE_CONDITIONER_BLOCKS_LEGACY = ["continuous", "flavour", "fourier_phi"]
_BASELINE_CONDITIONER_BLOCKS_NEW = ["kinematics", "pileup", "flavour", "fourier_phi"]

_ABLATION_DEFAULTS = {
    "mdn": {
        "width": 512,
        "depth": 6,
        "n_components": 8,
        "lambda_calib": 0.1,
        "lr": 0.0003,
        "conditioner_blocks": _BASELINE_CONDITIONER_BLOCKS_NEW,
    },
    "fm": {
        "width": 512,
        "depth": 6,
        "lr": 0.0003,
        "conditioner_blocks": _BASELINE_CONDITIONER_BLOCKS_NEW,
    },
}


def _conditioner_blocks_match(cfg_val, default_val):
    """Check if a config's conditioner_blocks matches the baseline default,
    accepting both legacy (continuous) and fine-grained (kinematics+pileup)
    representations as equivalent."""
    if cfg_val == default_val:
        return True
    s = set(cfg_val) if isinstance(cfg_val, list) else {cfg_val}
    return s == set(_BASELINE_CONDITIONER_BLOCKS_LEGACY) or s == set(
        _BASELINE_CONDITIONER_BLOCKS_NEW
    )


_CONDITIONER_RUNG_ORDER = [
    ("kinematics",),
    ("kinematics", "pileup"),
    ("kinematics", "pileup", "fourier_phi"),
    ("kinematics", "pileup", "flavour"),
    ("kinematics", "pileup", "fourier_phi", "flavour"),
]

_CONDITIONER_RUNG_ORDER_LEGACY = [
    ("continuous",),
    ("continuous", "fourier_phi"),
    ("continuous", "flavour"),
    ("continuous", "flavour", "fourier_phi"),
]

_GENJET_ORACLE_LABEL = "+ genjet oracle"


def _canon(value):
    """Dict keys must be hashable -- conditioner_blocks values are JSON
    lists, so canonicalise to a tuple. Scalars pass through unchanged."""
    return tuple(value) if isinstance(value, list) else value


def _label(value) -> str:
    """Short display form for an x-axis tick / legend entry."""
    if isinstance(value, (list, tuple)):
        return "+".join(str(x) for x in value)
    if isinstance(value, str) and value == _GENJET_ORACLE_LABEL:
        return value
    return str(value)


def _normalise_conditioner_blocks(value):
    """Normalise a canonicalised conditioner_blocks tuple to the fine-grained
    representation, so legacy and new configs sort to the same rung."""
    if not isinstance(value, tuple):
        return value
    out = []
    for b in value:
        if b == "continuous":
            out.extend(["kinematics", "pileup"])
        else:
            out.append(b)
    return tuple(out)


def _conditioner_rung_index(value):
    """Return a sort key for conditioner_blocks values that follows the
    information-content ladder rather than lexicographic order."""
    normed = _normalise_conditioner_blocks(value)
    # Check against the canonical rung order
    for i, rung in enumerate(_CONDITIONER_RUNG_ORDER):
        if set(normed) == set(rung):
            return i
    # Check legacy
    if isinstance(value, tuple):
        for i, rung in enumerate(_CONDITIONER_RUNG_ORDER_LEGACY):
            if set(value) == set(rung):
                return i
    # Genjet oracle goes at the end
    if isinstance(value, str) and value == _GENJET_ORACLE_LABEL:
        return len(_CONDITIONER_RUNG_ORDER) + 1
    # Unknown: put at the end, sorted by length then string
    return len(_CONDITIONER_RUNG_ORDER) + 100


_BLOCK_ORDER = ("kinematics", "pileup", "flavour", "fourier_phi")


def _canonical_block_key(k):
    """Order-insensitive canonical key for a conditioner_blocks tuple."""
    if not isinstance(k, tuple):
        return k
    normed = _normalise_conditioner_blocks(k)
    return tuple(
        sorted(
            normed,
            key=lambda b: (_BLOCK_ORDER.index(b) if b in _BLOCK_ORDER else 99, b),
        )
    )


def _dedupe_conditioner_dict(d: dict) -> dict:
    """Merge conditioner_blocks keys that denote the same block set."""
    merged: dict = {}
    for k, vals in d.items():
        nk = _canonical_block_key(k)
        merged.setdefault(nk, [])
        merged[nk].extend(vals)
    return merged


def _localise(path_str: str) -> Path:
    """Registry and provenance paths are repo-relative; resolve them against the repo root."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    rows = []
    with open(REGISTRY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect_points(
    model: str, stage: str, param: str, extra_points: list[tuple], metrics: list[str]
) -> tuple:
    """Returns ``(values, provenance)``."""
    out = {m: {} for m in metrics}
    prov = {m: {} for m in metrics}

    for entry in _load_registry():
        if entry.get("model") != model or entry.get("stage") != stage:
            continue
        if entry.get("status") != "done":
            continue
        resolved_path = RESOLVED_DIR / f"{entry['config_hash']}.json"
        if not resolved_path.exists():
            print(
                f"[plot_ablation] WARNING: no resolved config for "
                f"{entry['config_hash']} (run predates the resolved-config "
                "store?) -- skipping."
            )
            continue
        with open(resolved_path) as f:
            cfg = json.load(f)
        if param not in cfg:
            continue  # this run didn't sweep this param -- not part of it
        value = _canon(cfg[param])

        defaults = _ABLATION_DEFAULTS.get(model, {})
        off_baseline = False
        for other_param, default_val in defaults.items():
            if other_param == param:
                continue
            if other_param not in cfg:
                continue

            if other_param == "conditioner_blocks":
                if not _conditioner_blocks_match(cfg[other_param], default_val):
                    off_baseline = True
                    break
            elif cfg[other_param] != default_val:
                off_baseline = True
                break
        if off_baseline:
            continue

        metrics_path = _localise(entry["metrics_path"])
        if not metrics_path.exists():
            print(
                f"[plot_ablation] WARNING: {entry['metrics_path']} missing "
                f"(registry says done, not found on this checkout) -- skipping."
            )
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        for metric in metrics:
            if metric in m and m[metric] is not None:
                out[metric].setdefault(value, []).append(m[metric])
                prov[metric].setdefault(value, []).append(
                    {
                        "source": str(metrics_path),
                        "key_path": metric,
                        "config_hash": entry.get("config_hash"),
                        "seed": entry.get("seed"),
                        "git_sha": entry.get("git_sha"),
                        "origin": "registry",
                    }
                )

    for metric in metrics:
        for value, entries in list(prov[metric].items()):
            seen, keep = set(), []
            for i, e in enumerate(entries):
                if e["source"] in seen:
                    continue
                seen.add(e["source"])
                keep.append(i)
            if len(keep) != len(entries):
                prov[metric][value] = [entries[i] for i in keep]
                out[metric][value] = [out[metric][value][i] for i in keep]

    for value, path in extra_points:
        value = _canon(value)
        if not Path(path).exists():
            raise SystemExit(f"--extra_point path does not exist: {path}")
        with open(path) as f:
            m = json.load(f)
        for metric in metrics:
            if metric in m and m[metric] is not None:
                out[metric].setdefault(value, []).append(m[metric])
                prov[metric].setdefault(value, []).append(
                    {
                        "source": _paths.rel(path),
                        "key_path": metric,
                        "config_hash": None,
                        "seed": None,
                        "git_sha": None,
                        "origin": "extra_point",
                    }
                )

    return out, prov


def _floor_estimator_of(path) -> str:
    """Which CRPS-floor estimator wrote this metrics.json."""
    try:
        with open(_localise(str(path))) as f:
            return json.load(f).get("crps_floor_estimator", "mc_1M_legacy")
    except Exception:
        return "unknown"


def _panel_provenance_note(rows, is_ratio: bool) -> str:
    """One line stating seed count, n_runs and floor estimator for a panel."""
    seeds = sorted({sd for r in rows for sd in (r["seeds"] or [])})
    nruns = sorted({r["n_runs"] for r in rows})
    bits = [
        f"seed{'s' if len(seeds) != 1 else ''} "
        f"{','.join(map(str, seeds)) if seeds else 'n/a'}",
        (
            f"n_runs/rung {min(nruns)}"
            if len(nruns) == 1
            else f"n_runs/rung {min(nruns)}-{max(nruns)}"
        ),
    ]
    if is_ratio:
        est = sorted(
            {_floor_estimator_of(q["source"]) for r in rows for q in r["provenance"]}
        )
        bits.append("floor " + "/".join(est))
    return " | ".join(bits)


def _parse_extra_point(s: str) -> tuple:
    value_str, _, path = s.partition("=")
    try:
        value = json.loads(value_str)
    except json.JSONDecodeError:
        value = value_str
    return value, path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", choices=["mdn", "fm"], required=True)
    ap.add_argument(
        "--stage",
        default="parton2reco",
        choices=["parton2reco", "parton2genjet", "genjet2reco"],
    )
    ap.add_argument(
        "--param",
        required=True,
        help="Swept hyperparameter name, must match a key in "
        "runs/resolved_configs/<hash>.json (e.g. width, "
        "depth, n_components, lambda_calib).",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=_DEFAULT_METRICS,
        help="metrics.json keys to plot, one panel each.",
    )
    ap.add_argument(
        "--extra_point",
        action="append",
        default=[],
        metavar="VALUE=metrics.json",
        help="Inject a point that predates the run_experiment.py "
        "registry, e.g. 512=Surrogates/Output/MDN/"
        "seed_1/metrics.json. Repeatable.",
    )
    ap.add_argument(
        "--oracle_point",
        action="append",
        default=[],
        metavar="LABEL=metrics.json",
        help="Inject a point that is only drawn on the panels "
        "named by --oracle_metrics. Use for the genjet "
        "oracle rung: eng_targets redefines y0/y1/y2 by "
        "conditioning stage, so only log_mass_frac is a "
        "like-for-like comparison. Repeatable.",
    )
    ap.add_argument(
        "--oracle_metrics",
        nargs="+",
        default=["crps_ratio_log_mass_frac"],
        help="metrics.json keys whose panels the --oracle_point(s) "
        "are drawn on (default: crps_ratio_log_mass_frac).",
    )
    ap.add_argument(
        "--oracle_ref_point",
        default=None,
        metavar="metrics.json",
        help="On the --oracle_metrics panels only, replace the "
        "top non-oracle rung's value with this file's value, "
        "so the oracle is compared against a number measured "
        "on the same (common) event set. E.g. "
        "metrics_common.json for the p2r->g2r "
        "log_mass_frac comparison.",
    )
    ap.add_argument(
        "--homogeneous_seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Keep only registry points from this seed, so every "
        "rung of every panel is a single run at one seed. "
        "Without it the baseline config silently picks up "
        "seeds 2 and 3 from the registry while every "
        "off-baseline config is seed 1, so the top rung of "
        "the ladder is a 3-seed mean and the rest are not "
        "(audit N4). Points injected with --extra_point / "
        "--oracle_point carry no seed and are never dropped.",
    )
    ap.add_argument(
        "--oracle_panel",
        action="store_true",
        help="Draw --oracle_point / --oracle_ref_point in their "
        "OWN panel with their own axis and event-set label, "
        "instead of injecting them into the "
        "--oracle_metrics panel. The oracle is a different "
        "conditioning stage measured on the common event "
        "set; mixing it into a panel whose other points are "
        "on the unrestricted set puts two event sets on one "
        "axis with nothing telling the reader (audit N4).",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--out_json",
        default=None,
        help="Also write the plotted values to this JSON path, "
        "with the source metrics.json and key path behind "
        "every point. Without it the sweep numbers exist "
        "only inside the PNG and cannot be quoted in a "
        "paper without being re-typed by hand.",
    )
    ap.add_argument(
        "--logx",
        action="store_true",
        help="Force log-scale x-axis (auto-detected by default "
        "when values span > 10x).",
    )
    args = ap.parse_args(argv)

    extra_points = [_parse_extra_point(s) for s in args.extra_point]
    points, provenance = _collect_points(
        args.model, args.stage, args.param, extra_points, args.metrics
    )

    if args.homogeneous_seed is not None:
        dropped = 0
        for metric in list(points):
            for key in list(points[metric]):
                keep_v, keep_p = [], []
                for val, prov in zip(points[metric][key], provenance[metric][key]):
                    if (
                        prov.get("origin") == "registry"
                        and prov.get("seed") != args.homogeneous_seed
                    ):
                        dropped += 1
                        continue
                    keep_v.append(val)
                    keep_p.append(prov)
                if keep_v:
                    points[metric][key] = keep_v
                    provenance[metric][key] = keep_p
                else:
                    del points[metric][key]
                    del provenance[metric][key]
        print(
            f"[plot_ablation] --homogeneous_seed {args.homogeneous_seed}: "
            f"dropped {dropped} registry point(s) from other seeds"
        )

    oracle_points = [_parse_extra_point(s) for s in args.oracle_point]
    oracle_vals: dict = {}  # {metric: {label: [value]}}
    oracle_prov: dict = {}  # {metric: {label: [provenance]}}
    for label, path in oracle_points:
        if not Path(path).exists():
            raise SystemExit(f"--oracle_point path does not exist: {path}")
        with open(path) as f:
            mj = json.load(f)
        for metric in args.oracle_metrics:
            if metric in mj and mj[metric] is not None:
                oracle_vals.setdefault(metric, {})[label] = [mj[metric]]
                oracle_prov.setdefault(metric, {})[label] = [
                    {
                        "source": _paths.rel(path),
                        "key_path": metric,
                        "config_hash": None,
                        "seed": None,
                        "git_sha": None,
                        "origin": "oracle_point",
                    }
                ]

    oracle_ref_vals: dict = {}  # {metric: value}
    if args.oracle_ref_point:
        if not Path(args.oracle_ref_point).exists():
            raise SystemExit(
                f"--oracle_ref_point path does not exist: {args.oracle_ref_point}"
            )
        with open(args.oracle_ref_point) as f:
            rj = json.load(f)
        for metric in args.oracle_metrics:
            if metric in rj and rj[metric] is not None:
                oracle_ref_vals[metric] = rj[metric]

    draw_oracle_panel = bool(args.oracle_panel and (oracle_vals or oracle_ref_vals))
    n_panels = len(args.metrics) + (1 if draw_oracle_panel else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.8))
    if n_panels == 1:
        axes = [axes]
    axes = list(axes)
    oracle_ax = axes[-1] if draw_oracle_panel else None
    panel_axes = axes[: len(args.metrics)]

    color = _MODEL_COLORS.get(args.model, "#333333")
    is_conditioner_blocks = args.param == "conditioner_blocks"
    any_data = False
    json_panels: dict = {}
    for ax, metric in zip(panel_axes, args.metrics):
        d = dict(points[metric])

        pv = {k: list(v) for k, v in provenance.get(metric, {}).items()}

        if is_conditioner_blocks:
            d = _dedupe_conditioner_dict(d)
            pv = _dedupe_conditioner_dict(pv)
            if metric in args.oracle_metrics and not draw_oracle_panel:

                for label, vals in oracle_vals.get(metric, {}).items():
                    d[label] = list(vals)
                    pv[label] = list(oracle_prov.get(metric, {}).get(label, []))
                if metric in oracle_ref_vals and d:
                    real = [
                        k
                        for k in d
                        if not (isinstance(k, str) and k == _GENJET_ORACLE_LABEL)
                    ]
                    if real:
                        top = max(real, key=_conditioner_rung_index)
                        d[top] = [oracle_ref_vals[metric]]
                        pv[top] = [
                            {
                                "source": _paths.rel(args.oracle_ref_point),
                                "key_path": metric,
                                "config_hash": None,
                                "seed": None,
                                "git_sha": None,
                                "origin": "oracle_ref_point",
                                "note": "top rung swapped for the common-event-set "
                                "value so the drop to the oracle is "
                                "like-for-like",
                            }
                        ]

        if not d:
            ax.set_title(f"{metric}\n(no completed runs found)", fontsize=10)
            ax.axis("off")
            continue
        any_data = True

        if is_conditioner_blocks:
            xs = sorted(d.keys(), key=_conditioner_rung_index)
        else:

            xs = sorted(
                d.keys(),
                key=lambda v: (
                    isinstance(v, str),
                    len(v) if isinstance(v, (list, tuple)) else 0,
                    v if isinstance(v, (int, float)) else str(v),
                ),
            )
        means = [float(np.mean(d[x])) for x in xs]
        stds = [float(np.std(d[x])) if len(d[x]) > 1 else 0.0 for x in xs]
        n_seeds = [len(d[x]) for x in xs]

        json_panels[metric] = [
            {
                "rung": _label(x),
                "rung_key": list(x) if isinstance(x, (list, tuple)) else x,
                "rung_index": (
                    _conditioner_rung_index(x) if is_conditioner_blocks else None
                ),
                "values": [float(v) for v in d[x]],
                "mean": means[i],
                "sd": stds[i],
                "sd_or_sem": "sd_population_ddof0",
                "n_runs": n_seeds[i],
                "seeds": sorted(
                    {e["seed"] for e in pv.get(x, []) if e.get("seed") is not None}
                ),
                "n_distinct_seeds": len(
                    {e["seed"] for e in pv.get(x, []) if e.get("seed") is not None}
                ),
                "provenance": pv.get(x, []),
            }
            for i, x in enumerate(xs)
        ]

        is_numeric = all(isinstance(x, (int, float)) for x in xs)
        xpos = xs if is_numeric else list(range(len(xs)))

        ax.errorbar(
            xpos,
            means,
            yerr=stds,
            marker="o",
            color=color,
            capsize=4,
            lw=1.5,
            ms=6,
            label=f"{args.model.upper()} ({args.stage})",
        )
        for x, y, n in zip(xpos, means, n_seeds):
            if n == 1:
                ax.annotate(
                    "1 seed",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    fontsize=7,
                    color="gray",
                    ha="center",
                )

        for i, x in enumerate(xs):
            if isinstance(x, str) and x == _GENJET_ORACLE_LABEL:
                ax.plot(
                    xpos[i],
                    means[i],
                    marker="*",
                    ms=16,
                    color="#2ca02c",
                    mec="black",
                    mew=0.6,
                    ls="none",
                    label="genjet oracle (common set)",
                    zorder=5,
                )

        if not is_numeric:
            ax.set_xticks(xpos)
            ax.set_xticklabels([_label(x) for x in xs], rotation=25, ha="right")
        elif args.logx or (max(xpos) / max(min(xpos), 1e-9) > 10):
            ax.set_xscale("log")

        ax.set_xlabel(args.param)
        ax.set_ylabel(metric)
        note = _panel_provenance_note(
            json_panels[metric], metric.startswith("crps_ratio")
        )
        ax.set_title(f"{metric}\n{note}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    oracle_panel_json = None
    if draw_oracle_panel:

        metric = args.oracle_metrics[0]
        labels, vals, provs = [], [], []
        if metric in oracle_ref_vals:
            labels.append("parton -> reco\n(26-d conditioner)")
            vals.append(oracle_ref_vals[metric])
            provs.append(
                {
                    "source": _paths.rel(args.oracle_ref_point),
                    "key_path": metric,
                    "origin": "oracle_ref_point",
                    "seed": None,
                    "config_hash": None,
                    "git_sha": None,
                }
            )
        for lbl, v in oracle_vals.get(metric, {}).items():
            labels.append(str(lbl))
            vals.append(float(v[0]))
            provs.append(oracle_prov.get(metric, {}).get(lbl, [{}])[0])
        if vals:
            any_data = True
            xp = list(range(len(vals)))
            oracle_ax.plot(
                xp,
                vals,
                marker="o",
                ls="--",
                lw=1.5,
                ms=7,
                color="#2ca02c",
                mec="black",
                mew=0.6,
            )
            for x, v in zip(xp, vals):
                oracle_ax.annotate(
                    f"{v:.4f}",
                    (x, v),
                    textcoords="offset points",
                    xytext=(0, 9),
                    fontsize=8,
                    ha="center",
                )
            oracle_ax.set_xticks(xp)
            oracle_ax.set_xticklabels(labels, fontsize=8)
            oracle_ax.set_xlim(-0.5, len(vals) - 0.5)
            # Headroom so the value annotations do not collide with the title.
            _lo, _hi = min(vals), max(vals)
            _pad = max((_hi - _lo) * 0.35, abs(_hi) * 1e-3)
            oracle_ax.set_ylim(_lo - _pad, _hi + _pad)
            oracle_ax.set_ylabel(metric)
            ests = sorted(
                {_floor_estimator_of(q.get("source")) for q in provs if q.get("source")}
            )
            oracle_ax.set_title(
                "genjet oracle (separate estimand)\n"
                "COMMON event set | seed 1 | n_runs/point 1 | floor " + "/".join(ests),
                fontsize=9,
            )
            oracle_ax.grid(alpha=0.3)
            oracle_panel_json = {
                "metric": metric,
                "event_set": "common (has_genjet_match & genjet_mass>0 & "
                "genjet_partonFlavour!=0)",
                "seeds": [1],
                "n_runs_per_point": 1,
                "why_a_separate_panel": (
                    "genjet2reco is a different conditioning stage scored on "
                    "the common event set. Its floor and its target marginal "
                    "both differ from the unrestricted ladder's, so it shares "
                    "no axis with them."
                ),
                "points": [
                    {"label": l, "value": float(v), "provenance": q}
                    for l, v, q in zip(labels, vals, provs)
                ],
            }

    if not any_data:
        raise SystemExit(
            f"No completed runs found for model={args.model} stage={args.stage} "
            f"param={args.param} in {REGISTRY_PATH}, and no --extra_point "
            "given. Nothing to plot."
        )

    _scope = (
        f"seed {args.homogeneous_seed} only"
        if args.homogeneous_seed is not None
        else "seeds as registered"
    )
    fig.suptitle(
        f"{args.model.upper()} / {args.stage} -- sweep over "
        f"{args.param}   [{_scope}; ladder panels on the "
        f"UNRESTRICTED test set]",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[plot_ablation] saved {out_path}")

    if args.out_json:
        jpath = Path(args.out_json)
        write_json(
            jpath,
            {
                "model": args.model,
                "stage": args.stage,
                "param": args.param,
                "figure": _paths.rel(out_path),
                "deduplicated": is_conditioner_blocks,
                "dedupe_note": (
                    "conditioner_blocks rungs are merged on a canonical key: "
                    "'continuous' expands to kinematics+pileup, and block "
                    "order is ignored. Rungs are ordered by information "
                    "content via _conditioner_rung_index."
                    if is_conditioner_blocks
                    else None
                ),
                "oracle_metrics": (
                    list(args.oracle_metrics) if args.oracle_point else []
                ),
                "oracle_ref_point": (
                    _paths.rel(args.oracle_ref_point) if args.oracle_ref_point else None
                ),
                "oracle_panel": oracle_panel_json,
                "homogeneous_seed": args.homogeneous_seed,
                "panel_homogeneity": {
                    m: _panel_provenance_note(rows, m.startswith("crps_ratio"))
                    for m, rows in json_panels.items()
                },
                "panels": json_panels,
            },
        )
        print(f"[plot_ablation] saved {_paths.rel(jpath)}")


if __name__ == "__main__":
    main()
