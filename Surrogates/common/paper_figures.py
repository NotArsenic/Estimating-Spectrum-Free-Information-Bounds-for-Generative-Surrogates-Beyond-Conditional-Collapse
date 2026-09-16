"""
The seven paper figures, drawn from figure-data dicts (Paper/figure_data/*.json).
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from . import paper_style as ps  # noqa: E402

STAGE_ORDER = ("parton2reco", "parton2genjet", "genjet2reco")
RUNG_LABELS = {
    "kinematics": "kin.",
    "kinematics+pileup": "kin.\n+pu",
    "kinematics+pileup+fourier_phi": "kin.\n+pu\n+$\\phi$",
    "kinematics+pileup+flavour": "kin.\n+pu\n+flav.",
    "full": "kin.\n+pu\n+flav.\n+$\\phi$",
}


def _arr(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _pad_lim(lo, hi, frac=0.10):
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        span = max(abs(hi), 1.0) * 0.01
    return lo - frac * span, hi + frac * span


def _differential_ylim(
    series, target, subordinate="MARGINAL", include=(1.0,), max_stretch=2.5
):
    """y-limits set by the informative series; the Marginal sampler may be clipped."""
    core = np.concatenate(
        [_arr(s[target]) for k, s in series.items() if k != subordinate]
        + [np.asarray(include, dtype=float)]
    )
    lo_c, hi_c = np.nanmin(core), np.nanmax(core)
    sub = _arr(series[subordinate][target])
    if not np.isfinite(sub).any():
        return (*_pad_lim(lo_c, hi_c), False)
    lo_f, hi_f = min(lo_c, np.nanmin(sub)), max(hi_c, np.nanmax(sub))
    if (hi_f - lo_f) <= max_stretch * (hi_c - lo_c):
        return (*_pad_lim(lo_f, hi_f), False)
    return (*_pad_lim(lo_c, hi_c), True)


def _sweep_panel(ax, top, target, star_size):
    radius = np.asarray(top["radius"], dtype=float)
    ks = top["ks"]
    fit = top["per_target"][target]
    crps = np.asarray(fit["crps"], dtype=float)
    xs = np.linspace(0.0, radius.max() * 1.05, 32)
    ps.plot_series(
        ax, xs, fit["slope"] * xs + fit["intercept"], "KNN", marker="None", label=None
    )
    ps.plot_series(ax, radius, crps, "KNN", linestyle="None", label=None)
    ax.plot(
        [0.0],
        [fit["intercept"]],
        marker="*",
        markersize=star_size,
        color=ps.SERIES["KNN"]["color"],
        linestyle="None",
        zorder=6,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.35,
    )
    ps.reference_line(ax, 0.0, axis="x")
    ax.set_xlim(-0.055 * radius.max(), radius.max() * 1.10)
    lo, hi = _pad_lim(
        min(fit["intercept"], crps.min()), max(fit["intercept"], crps.max()), 0.22
    )
    ax.set_ylim(lo, hi)
    for i in (0, len(ks) - 1):
        up = crps[i] > 0.5 * (lo + hi)
        ax.annotate(
            f"$k$={ks[i]}",
            (radius[i], crps[i]),
            textcoords="offset points",
            xytext=(3 if i == 0 else -3, 4 if up else -7),
            ha="left" if i == 0 else "right",
            fontsize=5,
            color=ps.NEUTRAL["annot"],
        )
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))


def fig_data_driven_limits(data, save_path):
    ps.use_paper_style()
    targets, top, bottom = data["targets"], data["top"], data["bottom"]
    centers = np.asarray(bottom["centers"], dtype=float)
    series = bottom["series"]
    fig = plt.figure(figsize=(ps.COL_FULL, 3.15), constrained_layout=True)
    sub = fig.subfigures(2, 1, height_ratios=[1.0, 1.0])
    ax_top, ax_bot = sub[0].subplots(1, len(targets)), sub[1].subplots(1, len(targets))
    for di, t in enumerate(targets):
        _sweep_panel(ax_top[di], top, t, star_size=6.0)
        ps.column_header(ax_top[di], t)
    ax_top[0].set_ylabel("CRPS")
    sub[0].supxlabel(r"mean neighbourhood radius $\bar{r}_k$", fontsize=7)
    order = ["MDN", "CFM", "KNN", "LOOKUP", "MARGINAL"]
    for di, t in enumerate(targets):
        ax = ax_bot[di]
        lo, hi, clipped = _differential_ylim(series, t)
        for name in order:
            r = _arr(series[name][t])
            if clipped and name == "MARGINAL":
                off = r > hi
                ps.plot_series(ax, centers, np.where(off, np.nan, r), name, label=None)
                if off.any():
                    ax.plot(
                        centers[off],
                        np.full(int(off.sum()), hi),
                        marker="^",
                        markersize=1.8,
                        linestyle="None",
                        alpha=0.85,
                        color=ps.SERIES["MARGINAL"]["color"],
                        clip_on=False,
                        zorder=3,
                    )
            else:
                ps.plot_series(ax, centers, r, name, label=None)
        ps.reference_line(ax, 1.0)
        ax.set_xscale("log")
        ax.set_ylim(lo, hi)
        ax.set_xlim(centers[0] * 0.85, centers[-1] * 1.18)
    ax_bot[0].set_ylabel("CRPS / within-bin floor")
    sub[1].supxlabel(r"parton $p_{\mathrm{T}}$ [GeV]", fontsize=7)
    fig.legend(
        handles=ps.legend_handles(order),
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.055),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    return ps.save(fig, save_path)


def fig_overfit_arc(data, save_path, figsize=(3.5, 1.75)):
    ps.use_paper_style()
    val, crps = _arr(data["val_nll"]), _arr(data["val_crps_mean"])
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    m1 = {k: v for k, v in ps.METRICS["val_nll"].items() if k != "marker"}
    (l1,) = ax.plot(np.arange(1, len(val) + 1), val, **m1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL [nats]", color=m1["color"])
    ax.tick_params(axis="y", labelcolor=m1["color"], which="both")
    ax.set_xlim(0, len(val) + 1)
    ax2 = ax.twinx()
    m2 = {k: v for k, v in ps.METRICS["val_crps"].items() if k != "marker"}
    (l2,) = ax2.plot(np.arange(1, len(crps) + 1), crps, **m2)
    ax2.set_ylabel("validation CRPS (mean over targets)", color=m2["color"])
    ax2.tick_params(axis="y", labelcolor=m2["color"], which="both")
    ax2.grid(False)
    anchor = data["anchor_epoch"]
    ax.axvline(anchor, **ps.REFERENCE)
    ax.annotate(
        f"epoch {anchor}",
        (anchor, 0.965),
        xycoords=("data", "axes fraction"),
        xytext=(3, 0),
        textcoords="offset points",
        va="top",
        fontsize=6,
        color=ps.NEUTRAL["annot"],
    )
    ax.legend(handles=[l1, l2], loc="best", frameon=False, handlelength=1.8)
    return ps.save(fig, save_path)


def fig_knn_seeds(data, save_path, figsize=(5.5, 4.0)):
    ps.use_paper_style()
    targets, seeds = data["targets"], sorted(data["seeds"], key=int)
    fig, axes = plt.subplots(
        len(seeds),
        len(targets),
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )
    for si, s in enumerate(seeds):
        for di, t in enumerate(targets):
            ax = axes[si, di]
            _sweep_panel(ax, data["seeds"][s], t, star_size=5.0)
            if si == 0:
                ps.column_header(ax, t)
            if si < len(seeds) - 1:
                ax.set_xticklabels([])
        axes[si, 0].set_ylabel(f"seed {s}\nCRPS")
    fig.supxlabel(r"mean neighbourhood radius $\bar{r}_k$", fontsize=7)
    handles = ps.legend_handles(["KNN"])
    handles[0].set_label(r"$k$-NN CRPS, OLS fit, and the $r\to0$ intercept ($\star$)")
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=1,
        bbox_to_anchor=(0.5, 1.035),
        frameon=False,
        handlelength=2.2,
    )
    return ps.save(fig, save_path)


def fig_cone_control(data, save_path, figsize=(3.4, 2.1)):
    ps.use_paper_style()
    cones = data["cones"]
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    handles = []
    for t in data["targets"]:
        for per in data["per_seed"].values():
            ps.plot_faint(ax, cones, _arr(per[t]), t, ps.TARGETS)
        (line,) = ps.plot_series(
            ax,
            cones,
            _arr(data["mean"][t]),
            t,
            ps.TARGETS,
            label=ps.TARGET_HEADERS_TT[t],
            markersize=2.6,
            linewidth=1.0,
        )
        handles.append(line)
    ps.reference_line(ax, 1.0)
    ax.invert_xaxis()
    ax.set_xticks(list(cones))
    ax.set_xlabel(r"match-cone upper cut $\Delta R <$")
    ax.set_ylabel("median differential\n$k$-NN ceiling ratio")
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
        handlelength=1.6,
        columnspacing=0.8,
        labelspacing=0.35,
    )
    return ps.save(fig, save_path)


def fig_stage_ceilings(data, save_path, figsize=(5.5, 1.85)):
    ps.use_paper_style()
    targets = data["targets"]
    fig, axes = plt.subplots(1, len(targets), figsize=figsize, constrained_layout=True)
    for d, t in enumerate(targets):
        ax = axes[d]
        for stage in STAGE_ORDER:
            curve = data["stages"][stage]
            ps.plot_series(
                ax,
                curve["centers"],
                _arr(curve["ratio"][t]),
                stage,
                ps.STAGES,
                label=None,
            )
        ps.reference_line(ax, 1.0)
        ax.set_xscale("log")
        ps.column_header(ax, t)
    axes[0].set_ylabel("k-NN ceiling CRPS /\nwithin-bin floor")
    fig.supxlabel(r"common parton $p_{\mathrm{T}}$ [GeV]", fontsize=7)
    fig.legend(
        handles=ps.legend_handles(list(STAGE_ORDER), ps.STAGES),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6,
    )
    return ps.save(fig, save_path)


def fig_genjet_oracle(data, save_path, figsize=(2.65, 2.0)):
    ps.use_paper_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    x = np.array([0.0, 1.0])
    for model in ("MDN", "CFM"):
        for per in data["per_seed"].values():
            ps.plot_faint(ax, x, per[model], model)
        ps.plot_series(ax, x, data["mean"][model], model, label=None, markersize=3.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            r"parton $\to$ reco" "\n" r"(26-d conditioner)",
            r"genjet $\to$ reco" "\n" "(oracle)",
        ],
        fontsize=6,
    )
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(
        r"CRPS / floor, $\log(m_{\mathrm{jet}}/p_{\mathrm{T}}^{\mathrm{jet}})$"
    )
    fig.legend(
        handles=ps.legend_handles(["MDN", "CFM"]),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.13),
        frameon=False,
        handlelength=1.8,
    )
    return ps.save(fig, save_path)


def fig_conditioner_ladder(data, save_path, figsize=(5.5, 2.0)):
    ps.use_paper_style()
    panels = [
        ("ll", "mean log-likelihood [nats]"),
        (
            "R_mass",
            r"CRPS ratio, $\log(m_{\mathrm{jet}}/p_{\mathrm{T}}^{\mathrm{jet}})$",
        ),
    ]
    rungs = data["rungs"]
    x = np.arange(len(rungs))
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, panels):
        for model in ("MDN", "CFM"):
            for per in data["per_seed"].values():
                ps.plot_faint(ax, x, per[model][metric], model)
            ps.plot_series(ax, x, data["mean"][model][metric], model, label=None)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([RUNG_LABELS[r] for r in rungs], fontsize=5)
        ax.set_xlim(-0.35, len(rungs) - 0.65)
    fig.supxlabel("conditioner blocks", fontsize=7)
    fig.legend(
        handles=ps.legend_handles(["MDN", "CFM"]),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.09),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6,
    )
    return ps.save(fig, save_path)


DRAW = {
    "fig1_data_driven_limits": fig_data_driven_limits,
    "fig2_overfit_arc": fig_overfit_arc,
    "figA_knn_3seeds": fig_knn_seeds,
    "figA_cone_control": fig_cone_control,
    "figA_stage_ceilings": fig_stage_ceilings,
    "figA_genjet_oracle": fig_genjet_oracle,
    "figA_conditioner_ladder": fig_conditioner_ladder,
}
