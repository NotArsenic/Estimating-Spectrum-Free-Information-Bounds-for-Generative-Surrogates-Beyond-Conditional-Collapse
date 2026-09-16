"""
Figures built from per-event dumps
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .features import (
    TARGET_NAMES,
    FLAVOUR_NAMES,
    _FLAVOUR_MAP,
    reconstruct_jet,
    _DR_CONE,
)
from .metrics import crps_per_event, crps_floor_within  # noqa: F401  (re-exported)
from .plots import _TARGET_LABELS

_MODEL_COLORS = {
    "MDN_V2": "#1f77b4",
    "FlowMatching_SingleStage_V2": "#d62728",
    "LookupTable": "#2ca02c",
    "Marginal": "#7f7f7f",
    "kNN ceiling": "#9467bd",
}


def _short(model_name: str) -> str:
    return {
        "MDN_V2": "MDN",
        "FlowMatching_SingleStage_V2": "CFM",
    }.get(model_name, model_name)


def _color(model_name: str) -> str:
    return _MODEL_COLORS.get(model_name, "#333333")


# Core: the within-bin floor
def skill_ratio_binned(
    y: np.ndarray,
    crps: np.ndarray,
    binvar: np.ndarray,
    edges: np.ndarray,
    min_count: int = 200,
):
    """
    CRPS skill ratio per bin, each against **its own** within-bin floor.

    Returns ``(centers, ratio (n_bins, D), counts (n_bins,))``.
    """
    y = np.asarray(y)
    crps = np.asarray(crps)
    binvar = np.asarray(binvar)
    D = crps.shape[1]
    n_bins = len(edges) - 1

    centers = (
        np.sqrt(edges[:-1] * edges[1:])
        if np.all(edges > 0)
        else 0.5 * (edges[:-1] + edges[1:])
    )
    ratio = np.full((n_bins, D), np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)

    idx = np.digitize(binvar, edges) - 1
    for b in range(n_bins):
        mask = idx == b
        finite = mask[:, None] & np.isfinite(crps)
        counts[b] = int(finite.any(axis=1).sum())
        if counts[b] < min_count:
            continue
        floor = crps_floor_within(y[mask])
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col = np.nanmean(np.where(mask[:, None], crps, np.nan), axis=0)
            ratio[b] = col / floor
    return centers, ratio, counts


# C1 / C2 - differential ceiling
def plot_differential_ceiling(
    dumps,
    binvar_name: str = "parton_pt",
    edges: np.ndarray | None = None,
    log_x: bool = True,
    knn_curve: dict | None = None,
    baselines: dict | None = None,
    save_path: str | None = None,
    xlabel: str | None = None,
):
    """
    C1/C2: CRPS skill ratio vs a conditioner variable, against the
    within-bin floor.
    """
    if edges is None:
        if binvar_name == "parton_pt":
            edges = np.logspace(np.log10(50), np.log10(1000), 15)
        else:
            v = dumps[0].index[binvar_name].to_numpy()
            edges = np.linspace(v.min(), v.max(), 13)
            log_x = False

    D = len(TARGET_NAMES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for di, ax in enumerate(axes.flat):
        for dump in dumps:
            y = dump.y_true()
            centers, ratio, counts = skill_ratio_binned(
                y, dump.crps(), dump.index[binvar_name].to_numpy(), edges
            )
            ax.plot(
                centers,
                ratio[:, di],
                "o-",
                ms=4,
                color=_color(dump.meta["model"]),
                label=_short(dump.meta["model"]),
            )
        if baselines:
            y_ref = dumps[0].y_true()
            bv = dumps[0].index[binvar_name].to_numpy()
            for label, bcrps in baselines.items():
                _, bratio, _ = skill_ratio_binned(y_ref, bcrps, bv, edges)
                ax.plot(
                    centers,
                    bratio[:, di],
                    "--",
                    lw=1.2,
                    color=_color(label),
                    label=label,
                    alpha=0.85,
                )
        if knn_curve is not None:
            ax.plot(
                knn_curve["centers"],
                np.asarray(knn_curve["ratio"])[:, di],
                "s--",
                ms=3,
                color="#8c564b",
                label="k-NN intercept (truth only)",
                alpha=0.85,
            )
        ax.axhline(1.0, color="gray", ls=":", lw=1.5)
        ax.set_title(
            _TARGET_LABELS.get(TARGET_NAMES[di], TARGET_NAMES[di]), fontsize=10
        )
        ax.set_ylabel("CRPS / within-bin floor")
        if log_x:
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
        if di == 0:
            ax.legend(fontsize=8)

    lbl = xlabel or (
        r"parton $p_T$ [GeV]" if binvar_name == "parton_pt" else binvar_name
    )
    for ax in axes[1]:
        ax.set_xlabel(lbl)
    fig.suptitle(
        "Differential conditional ceiling — floor recomputed within each bin",
        fontsize=12,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_skill_by_flavour(dumps, save_path: str | None = None):
    """C2: skill ratio per flavour, each against its own within-flavour floor."""
    D = len(TARGET_NAMES)
    flavs = list(_FLAVOUR_MAP.items())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    width = 0.8 / max(len(dumps), 1)
    for di, ax in enumerate(axes.flat):
        for mi, dump in enumerate(dumps):
            pdg = np.abs(dump.index["parton_pdgId"].to_numpy())
            y, crps = dump.y_true(), dump.crps()
            names, vals = [], []
            for abs_id, col in flavs:
                mask = pdg == abs_id
                if mask.sum() < 200:
                    continue
                floor = crps_floor_within(y[mask])
                names.append(FLAVOUR_NAMES[col])
                vals.append(crps[mask].mean(axis=0)[di] / floor[di])
            x = np.arange(len(names))
            ax.bar(
                x + mi * width,
                vals,
                width,
                color=_color(dump.meta["model"]),
                label=_short(dump.meta["model"]),
                alpha=0.85,
            )
            ax.set_xticks(x + width * (len(dumps) - 1) / 2)
            ax.set_xticklabels(names)
        ax.axhline(1.0, color="gray", ls=":", lw=1.5)
        ax.set_title(
            _TARGET_LABELS.get(TARGET_NAMES[di], TARGET_NAMES[di]), fontsize=10
        )
        ax.set_ylabel("CRPS / within-flavour floor")
        ax.grid(alpha=0.3, axis="y")
        if di == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_knn_intercept(sweep, floor=None, save_path: str | None = None):
    radius = np.asarray(sweep["radius"])
    crps = np.asarray(sweep["crps"])  # (n_k, D)
    extrap = np.asarray(sweep["crps_extrapolated"])
    ks = sweep["ks"]
    D = crps.shape[1]

    fig, axes = plt.subplots(1, D, figsize=(4.2 * D, 4.4))
    if D == 1:
        axes = [axes]

    for d in range(D):
        ax = axes[d]
        ax.plot(
            radius, crps[:, d], "o", color="#9467bd", ms=7, label="k-NN CRPS", zorder=4
        )
        for r, c, k in zip(radius, crps[:, d], ks):
            ax.annotate(
                f"k={k}",
                (r, c),
                textcoords="offset points",
                xytext=(4, 5),
                fontsize=7,
                color="#555",
            )

        slope, intercept = np.polyfit(radius, crps[:, d], 1)
        xs = np.linspace(0, radius.max() * 1.05, 50)
        ax.plot(
            xs,
            slope * xs + intercept,
            "--",
            color="#9467bd",
            lw=1.2,
            label="linear fit",
        )
        ax.plot(
            [0],
            [intercept],
            "*",
            color="red",
            ms=14,
            zorder=5,
            label=f"intercept = {intercept:.4f}",
        )
        ax.axvline(0.0, color="gray", ls=":", lw=1)

        if floor is not None:
            ax.axhline(
                floor[d],
                color="black",
                ls="-.",
                lw=1.2,
                label=f"marginal floor = {floor[d]:.4f}",
            )

        ax.set_xlabel("mean neighbourhood radius")
        ax.set_ylabel("CRPS")
        ax.set_xlim(left=-0.02 * radius.max())
        ttl = _TARGET_LABELS.get(TARGET_NAMES[d], TARGET_NAMES[d])
        if floor is not None:
            ttl += f"\nceiling ratio = {intercept / floor[d]:.4f}"
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(
        "k-NN conditional ceiling: extrapolation to zero neighbourhood radius "
        "(intercept bounds CRPS* from ABOVE)",
        fontsize=11,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# C3 — conditional PIT
def plot_conditional_pit(dumps, save_path: str | None = None):
    """
    C3: PIT KS statistic within slices of the conditioner
    (pT tercile x flavour group x pileup tercile).

    Aggregate PIT cannot evidence *conditional* calibration — a marginal-only
    sampler passes it trivially — so this is the figure that closes that hole.
    """
    from scipy import stats as sp_stats

    fig, axes = plt.subplots(
        len(dumps), 1, figsize=(13, 4.2 * len(dumps)), squeeze=False
    )

    for mi, dump in enumerate(dumps):
        ax = axes[mi][0]
        idx = dump.index
        pit = dump.pit()
        pt = idx["parton_pt"].to_numpy()
        pu = idx["pileup_nTrueInt"].to_numpy()
        pdg = np.abs(idx["parton_pdgId"].to_numpy())

        pt_t = np.quantile(pt, [1 / 3, 2 / 3])
        pu_t = np.quantile(pu, [1 / 3, 2 / 3])
        pt_b = np.digitize(pt, pt_t)
        pu_b = np.digitize(pu, pu_t)
        flav_groups = {
            "uds": np.isin(pdg, [1, 2, 3]),
            "c": pdg == 4,
            "b": pdg == 5,
            "g": pdg == 21,
        }

        labels, ks_rows = [], []
        for fname, fmask in flav_groups.items():
            for pb in range(3):
                for ub in range(3):
                    mask = fmask & (pt_b == pb) & (pu_b == ub)
                    if mask.sum() < 200:
                        continue
                    labels.append(f"{fname}|pT{pb}|PU{ub}")
                    ks_rows.append(
                        [
                            sp_stats.kstest(pit[mask, d], "uniform").statistic
                            for d in range(pit.shape[1])
                        ]
                    )
        if not ks_rows:
            continue
        ks = np.asarray(ks_rows).T  # (D, n_slices)
        im = ax.imshow(ks, aspect="auto", cmap="viridis", vmin=0)
        ax.set_yticks(range(len(TARGET_NAMES)))
        ax.set_yticklabels(TARGET_NAMES, fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_title(
            f"{_short(dump.meta['model'])} — PIT KS D per conditioner slice "
            "(lower is better)",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, label="KS D")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# C4 — per-event model vs model
def plot_model_vs_model(
    dump_a, dump_b, save_path: str | None = None, max_points: int = 20_000
):
    """
    C4: per-event predictive mean and width, one model against the other.
    """
    sa, sb = dump_a.samples, dump_b.samples
    n = min(len(sa), len(sb), max_points)
    rng = np.random.default_rng(0)
    sel = rng.choice(min(len(sa), len(sb)), n, replace=False)
    sel.sort()

    mean_a, mean_b = sa[sel].mean(axis=1), sb[sel].mean(axis=1)
    std_a, std_b = sa[sel].std(axis=1), sb[sel].std(axis=1)

    D = len(TARGET_NAMES)
    fig, axes = plt.subplots(2, D, figsize=(4 * D, 8))
    la, lb = _short(dump_a.meta["model"]), _short(dump_b.meta["model"])

    for d in range(D):
        for row, (va, vb, what) in enumerate(
            [(mean_a[:, d], mean_b[:, d], "mean"), (std_a[:, d], std_b[:, d], "width")]
        ):
            ax = axes[row][d]
            ax.hexbin(va, vb, gridsize=60, cmap="Blues", bins="log", mincnt=1)
            lo = float(min(va.min(), vb.min()))
            hi = float(max(va.max(), vb.max()))
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            r = float(np.corrcoef(va, vb)[0, 1])
            ax.set_title(
                f"{TARGET_NAMES[d]} predictive {what}\nr = {r:.4f}", fontsize=9
            )
            ax.set_xlabel(la)
            ax.set_ylabel(lb)
    fig.suptitle("Per-event agreement between models", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# C5 — training curves
def plot_val_crps_curve(losses_paths: dict, save_path: str | None = None):
    """
    C5: validation CRPS and validation loss per epoch (from B1).

    If loss improves while CRPS stays flat, the paper's thesis is visible in the optimisation curve itself rather than only in the final table.
    """
    import json

    fig, axes = plt.subplots(
        1,
        len(losses_paths),
        figsize=(6.5 * len(losses_paths), 4.5),
        squeeze=False,
    )
    for i, (name, path) in enumerate(losses_paths.items()):
        ax = axes[0][i]
        with open(path) as f:
            hist = json.load(f)
        val = hist.get("val_losses", [])
        crps_hist = hist.get("val_crps_history", [])
        epochs = np.arange(1, len(val) + 1)

        ax.plot(epochs, val, color="#1f77b4", label="val loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation loss", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")

        if crps_hist:
            mean_crps = [float(np.mean(list(c.values()))) for c in crps_hist]
            ax2 = ax.twinx()
            ax2.plot(
                np.arange(1, len(mean_crps) + 1),
                mean_crps,
                color="#d62728",
                label="val CRPS",
            )
            ax2.set_ylabel("validation CRPS (mean over dims)", color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")
            if val:
                ax.axvline(
                    int(np.argmin(val)) + 1,
                    color="gray",
                    ls=":",
                    label="best epoch (by loss)",
                )
        ax.set_title(f"{_short(name)} — optimisation vs conditional skill", fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# C7 — effective sample size
def plot_n_eff(dump, save_path: str | None = None, edges=None):
    """
    C7: effective sample size vs pT under flat / physical / FSR-up / FSR-down weighting.
    """
    if edges is None:
        edges = np.logspace(np.log10(50), np.log10(1000), 15)
    idx = dump.index
    pt = idx["parton_pt"].to_numpy()
    centers = np.sqrt(edges[:-1] * edges[1:])
    b = np.digitize(pt, edges) - 1

    schemes = {"flat (unweighted)": np.ones(len(pt))}
    for col, label in [
        ("generator_weight", "physical (xsec)"),
        ("ps_fsr_up", "FSR up"),
        ("ps_fsr_down", "FSR down"),
    ]:
        if col in idx.columns:
            schemes[label] = np.nan_to_num(idx[col].to_numpy(), nan=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for label, w in schemes.items():
        n_eff, frac = [], []
        for i in range(len(centers)):
            mask = b == i
            ww = w[mask]
            if mask.sum() == 0 or ww.sum() <= 0:
                n_eff.append(np.nan)
                frac.append(np.nan)
                continue
            ne = ww.sum() ** 2 / np.sum(ww**2)
            n_eff.append(ne)
            frac.append(ne / mask.sum())
        axes[0].plot(centers, n_eff, "o-", ms=4, label=label)
        axes[1].plot(centers, frac, "o-", ms=4, label=label)

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$N_{\mathrm{eff}}$ per bin")
    axes[1].set_ylabel(r"$N_{\mathrm{eff}} / N$")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"parton $p_T$ [GeV]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Effective sample size under reweighting", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# C8 — support geometry
def plot_support_geometry(dumps, save_path: str | None = None, edges=None):
    """
    C8: matching quality and out-of-cone fraction vs pT and flavour.
    """
    if edges is None:
        edges = np.logspace(np.log10(50), np.log10(1000), 15)
    centers = np.sqrt(edges[:-1] * edges[1:])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ref = dumps[0].index
    pt = ref["parton_pt"].to_numpy()
    b = np.digitize(pt, edges) - 1

    if "delta_R" in ref.columns:
        dr = ref["delta_R"].to_numpy()
        med = [
            np.median(dr[b == i]) if (b == i).sum() > 20 else np.nan
            for i in range(len(centers))
        ]
        axes[0].plot(centers, med, "o-", ms=4, color="black")
        axes[0].set_ylabel(r"median $\Delta R$(parton, jet)")
        axes[0].axhline(0.2, color="red", ls="--", label="matching cone")
        axes[0].legend(fontsize=8)
    axes[0].set_title("Dataset: matching quality", fontsize=10)

    for dump in dumps:
        f = dump.index["frac_outside_disk"].to_numpy()
        vals = [
            f[b == i].mean() if (b == i).sum() > 20 else np.nan
            for i in range(len(centers))
        ]
        axes[1].plot(
            centers,
            vals,
            "o-",
            ms=4,
            color=_color(dump.meta["model"]),
            label=_short(dump.meta["model"]),
        )
    axes[1].set_ylabel("mean fraction of samples outside cone")
    axes[1].set_title("Model: out-of-cone sample fraction", fontsize=10)
    axes[1].legend(fontsize=8)

    width = 0.8 / max(len(dumps), 1)
    flavs = [("uds", [1, 2, 3]), ("c", [4]), ("b", [5]), ("g", [21])]
    for mi, dump in enumerate(dumps):
        pdg = np.abs(dump.index["parton_pdgId"].to_numpy())
        f = dump.index["frac_outside_disk"].to_numpy()
        vals, names = [], []
        for nm, ids in flavs:
            mask = np.isin(pdg, ids)
            if mask.sum() < 100:
                continue
            names.append(nm)
            vals.append(f[mask].mean())
        x = np.arange(len(names))
        axes[2].bar(
            x + mi * width,
            vals,
            width,
            color=_color(dump.meta["model"]),
            label=_short(dump.meta["model"]),
            alpha=0.85,
        )
        axes[2].set_xticks(x + width * (len(dumps) - 1) / 2)
        axes[2].set_xticklabels(names)
    axes[2].set_ylabel("mean fraction outside cone")
    axes[2].set_title("Out-of-cone fraction by flavour", fontsize=10)
    axes[2].legend(fontsize=8)

    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel(r"parton $p_T$ [GeV]")
        ax.grid(alpha=0.3)
    axes[2].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# C8b — geometric distance between true and sampled jets
def plot_jet_delta_r(dumps, save_path: str | None = None, edges=None, seed: int = 0):
    """
    C8b: ΔR(true jet, sampled jet) — how far a drawn jet actually lands from the real one, in eta-phi space, as opposed to the residual-space CRPS numbers everywhere else in this module.
    """
    if edges is None:
        edges = np.logspace(np.log10(50), np.log10(1000), 15)
    centers = np.sqrt(edges[:-1] * edges[1:])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    rng = np.random.default_rng(seed)
    for dump in dumps:
        idx = dump.index
        pt = idx["parton_pt"].to_numpy()
        eta = idx["parton_eta"].to_numpy()
        phi = idx["parton_phi"].to_numpy()
        n = len(pt)

        true_jet = reconstruct_jet(pt, eta, phi, dump.y_true())

        m = dump.samples.shape[1]
        draw = rng.integers(0, m, size=n)
        sample_res = np.asarray(dump.samples[np.arange(n), draw])
        pred_jet = reconstruct_jet(pt, eta, phi, sample_res)

        deta = pred_jet["jet_eta"] - true_jet["jet_eta"]
        dphi = np.arctan2(
            np.sin(pred_jet["jet_phi"] - true_jet["jet_phi"]),
            np.cos(pred_jet["jet_phi"] - true_jet["jet_phi"]),
        )
        dr = np.sqrt(deta**2 + dphi**2)

        c = _color(dump.meta["model"])
        label = _short(dump.meta["model"])
        axes[0].hist(
            dr,
            bins=60,
            range=(0, 2 * _DR_CONE),
            histtype="step",
            density=True,
            color=c,
            lw=1.6,
            label=label,
        )

        b = np.digitize(pt, edges) - 1
        med = [
            np.median(dr[b == i]) if (b == i).sum() > 20 else np.nan
            for i in range(len(centers))
        ]
        axes[1].plot(centers, med, "o-", ms=4, color=c, label=label)

    axes[0].axvline(
        2 * _DR_CONE,
        color="black",
        ls="--",
        lw=1.2,
        label=f"geometric bound (2×cone = {2 * _DR_CONE:.1f})",
    )
    axes[0].set_xlabel(r"$\Delta R$(true jet, sampled jet)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Per-event geometric distance (one draw/event)", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"parton $p_T$ [GeV]")
    axes[1].set_ylabel(r"median $\Delta R$(true, sampled)")
    axes[1].set_title(r"Geometric distance vs parton $p_T$", fontsize=10)
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# C9 — MDN component specialisation
def plot_component_specialisation(dump, save_path: str | None = None):
    """
    C9: mixture weight pi_k profiled by flavour and by pT.

    Flavour carries most of the conditional skill, so if componentsspecialise at all, that is where it shows.
    """
    internals = dump.internals
    if internals is None or "pi" not in internals:
        return None
    pi = internals["pi"]  # (n_sub, K)
    sub = dump.sub_idx
    idx = dump.index
    pdg = np.abs(idx["parton_pdgId"].to_numpy()[sub])
    pt = idx["parton_pt"].to_numpy()[sub]
    K = pi.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    flavs = [("uds", [1, 2, 3]), ("c", [4]), ("b", [5]), ("g", [21])]
    rows, names = [], []
    for nm, ids in flavs:
        mask = np.isin(pdg, ids)
        if mask.sum() < 50:
            continue
        rows.append(pi[mask].mean(axis=0))
        names.append(nm)
    if rows:
        im = axes[0].imshow(np.asarray(rows), aspect="auto", cmap="viridis")
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names)
        axes[0].set_xlabel("component k")
        axes[0].set_title(r"$\langle \pi_k \rangle$ by flavour", fontsize=10)
        fig.colorbar(im, ax=axes[0])

    edges = np.logspace(np.log10(max(pt.min(), 1)), np.log10(pt.max()), 10)
    b = np.digitize(pt, edges) - 1
    prof = np.full((len(edges) - 1, K), np.nan)
    for i in range(len(edges) - 1):
        mask = b == i
        if mask.sum() > 50:
            prof[i] = pi[mask].mean(axis=0)
    im2 = axes[1].imshow(prof, aspect="auto", cmap="viridis", origin="lower")
    axes[1].set_xlabel("component k")
    axes[1].set_ylabel(r"$p_T$ bin")
    axes[1].set_title(r"$\langle \pi_k \rangle$ by $p_T$", fontsize=10)
    fig.colorbar(im2, ax=axes[1])

    # Effective number of components: exp(entropy), per event
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(pi > 0, pi * np.log(pi), 0.0), axis=1)
    axes[2].hist(np.exp(ent), bins=60, color="#1f77b4", alpha=0.8)
    axes[2].axvline(K, color="red", ls="--", label=f"K = {K}")
    axes[2].set_xlabel(r"effective components  $\exp(H[\pi])$")
    axes[2].set_ylabel("events")
    axes[2].set_title("Per-event mixture occupancy", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# C11 — comparative triptych
def plot_triptych(
    series: dict,
    y_true: np.ndarray,
    parton_pt: np.ndarray,
    save_path: str | None = None,
    smooth_sigma: float = 150.0,
    exclude: tuple = ("kNN ceiling",),
):
    """
    C11: response ratio, absolute pT, and residual — each sorted by true pT and Gaussian-smoothed. Stacked vertically (not side-by-side), which frees up the height budget for a large, legible canvas without also stretching the plots wide enough to look empty.
    """
    from scipy.ndimage import gaussian_filter1d

    series = {k: v for k, v in series.items() if k not in set(exclude)}

    order = np.argsort(parton_pt)
    pt_sorted = parton_pt[order]
    true_jet_pt = parton_pt * np.exp(y_true[:, 0])

    fig, axes = plt.subplots(3, 1, figsize=(20, 34))

    axes[1].plot(
        pt_sorted,
        gaussian_filter1d(true_jet_pt[order], smooth_sigma),
        color="black",
        lw=2.6,
        label="truth",
        zorder=5,
    )

    primary = {"MDN_V2", "FlowMatching_SingleStage_V2"}
    ordered_labels = [l for l in series if l not in primary] + [
        l for l in series if l in primary
    ]
    for label in ordered_labels:
        pred = series[label]
        pred_jet_pt = parton_pt * np.exp(np.asarray(pred)[:, 0])
        c = _color(label)
        lw = 2.6 if label in primary else 1.3
        zorder = 4 if label in primary else 2
        axes[0].plot(
            pt_sorted,
            gaussian_filter1d((pred_jet_pt / true_jet_pt)[order], smooth_sigma),
            color=c,
            label=_short(label),
            lw=lw,
            zorder=zorder,
        )
        axes[1].plot(
            pt_sorted,
            gaussian_filter1d(pred_jet_pt[order], smooth_sigma),
            color=c,
            label=_short(label),
            lw=lw,
            zorder=zorder,
        )
        axes[2].plot(
            pt_sorted,
            gaussian_filter1d((pred_jet_pt - true_jet_pt)[order], smooth_sigma),
            color=c,
            label=_short(label),
            lw=lw,
            zorder=zorder,
        )

    axes[0].axhline(1.0, color="black", ls=":", lw=2)
    axes[0].set_ylabel(r"predicted / true jet $p_T$", fontsize=19)
    axes[0].set_title("Response ratio", fontsize=22)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"jet $p_T$ [GeV]", fontsize=19)
    axes[1].set_title(r"Absolute jet $p_T$", fontsize=22)
    axes[2].axhline(0.0, color="black", ls=":", lw=2)
    axes[2].set_ylabel(r"predicted - true jet $p_T$ [GeV]", fontsize=19)
    axes[2].set_title("Residual", fontsize=22)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"parton $p_T$ [GeV]", fontsize=19)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=15, loc="best", framealpha=0.9)
    fig.suptitle(
        f"Point-prediction comparison (sorted by true $p_T$, "
        f"Gaussian-smoothed sigma={smooth_sigma:g})",
        fontsize=24,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=220)
        plt.close(fig)
    return fig


# C11b — triptych, generalised to the other reconstructed observables
_OBS_META = {
    "mass": (r"jet mass", "[GeV]", True),
    "energy": (r"jet $E$", "[GeV]", True),
    "eta": (r"jet $\eta$", "", False),
    "phi": (r"jet $\phi$", "[rad]", False),
}


def plot_triptych_observable(
    series: dict,
    y_true: np.ndarray,
    parton_pt: np.ndarray,
    parton_eta: np.ndarray,
    parton_phi: np.ndarray,
    observable: str,
    save_path: str | None = None,
    smooth_sigma: float = 150.0,
    exclude: tuple = ("kNN ceiling",),
):
    """
    :func:`plot_triptych` (C11) generalised from jet pT to any reconstructed observable: "mass", "energy", "eta", "phi".
    """
    from scipy.ndimage import gaussian_filter1d

    if observable not in _OBS_META:
        raise ValueError(
            f"observable must be one of {sorted(_OBS_META)}, " f"got {observable!r}"
        )
    symbol, unit, positive = _OBS_META[observable]
    key = f"jet_{observable}"
    is_phi = observable == "phi"

    series = {k: v for k, v in series.items() if k not in set(exclude)}

    order = np.argsort(parton_pt)
    pt_sorted = parton_pt[order]
    true_val = reconstruct_jet(parton_pt, parton_eta, parton_phi, y_true)[key]

    fig, axes = plt.subplots(3, 1, figsize=(20, 34))

    axes[1].plot(
        pt_sorted,
        gaussian_filter1d(true_val[order], smooth_sigma),
        color="black",
        lw=2.6,
        label="truth",
        zorder=5,
    )

    primary = {"MDN_V2", "FlowMatching_SingleStage_V2"}
    ordered_labels = [l for l in series if l not in primary] + [
        l for l in series if l in primary
    ]
    for label in ordered_labels:
        pred = np.asarray(series[label])
        pred_val = reconstruct_jet(parton_pt, parton_eta, parton_phi, pred)[key]
        c = _color(label)
        lw = 2.6 if label in primary else 1.3
        zorder = 4 if label in primary else 2

        if is_phi:
            diff = np.arctan2(np.sin(pred_val - true_val), np.cos(pred_val - true_val))
        else:
            diff = pred_val - true_val
        panel0 = (pred_val / true_val) if positive else np.abs(diff)

        axes[0].plot(
            pt_sorted,
            gaussian_filter1d(panel0[order], smooth_sigma),
            color=c,
            label=_short(label),
            lw=lw,
            zorder=zorder,
        )
        axes[1].plot(
            pt_sorted,
            gaussian_filter1d(pred_val[order], smooth_sigma),
            color=c,
            label=_short(label),
            lw=lw,
            zorder=zorder,
        )
        axes[2].plot(
            pt_sorted,
            gaussian_filter1d(diff[order], smooth_sigma),
            color=c,
            label=_short(label),
            lw=lw,
            zorder=zorder,
        )

    if positive:
        axes[0].axhline(1.0, color="black", ls=":", lw=2)
        axes[0].set_ylabel(f"predicted / true {symbol}", fontsize=19)
        axes[0].set_title("Response ratio", fontsize=22)
        if observable == "energy":
            axes[1].set_yscale("log")
    else:
        axes[0].set_ylabel(
            f"smoothed |predicted - true| {symbol} {unit}".strip(), fontsize=19
        )
        axes[0].set_title("Resolution (smoothed |residual|)", fontsize=22)

    axes[1].set_ylabel(f"{symbol} {unit}".strip(), fontsize=19)
    axes[1].set_title(f"Absolute {symbol}", fontsize=22)

    axes[2].axhline(0.0, color="black", ls=":", lw=2)
    axes[2].set_ylabel(f"predicted - true {symbol} {unit}".strip(), fontsize=19)
    axes[2].set_title("Residual" + ("" if positive else " (bias)"), fontsize=22)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"parton $p_T$ [GeV]", fontsize=19)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=15, loc="best", framealpha=0.9)
    fig.suptitle(
        f"Point-prediction comparison — {symbol} (sorted by true $p_T$, "
        f"Gaussian-smoothed sigma={smooth_sigma:g})",
        fontsize=24,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=220)
        plt.close(fig)
    return fig


# C12 — inference speed vs fidelity
def plot_speed_vs_fidelity(points: dict, save_path: str | None = None):
    """
    C12: inference cost vs model fidelity, across models and their configs (batch size for both; solver step count for the flow).

    ``points`` maps a config label -> dict with:
        ``speed_us``  mean wall-clock time per event [microseconds]
        ``fidelity``  mean held-out log-likelihood for that checkpoint
        ``model``     the model-family key used by :func:`_color`/`_short`
    """
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(9, 6.5))

    by_model = {}
    for label, p in points.items():
        by_model.setdefault(p["model"], []).append((label, p))

    for model, entries in by_model.items():
        entries = sorted(entries, key=lambda le: le[1]["speed_us"])
        xs = [p["speed_us"] for _, p in entries]
        ys = [p["fidelity"] for _, p in entries]
        c = _color(model)
        ax.scatter(
            xs,
            ys,
            s=90,
            color=c,
            edgecolor="black",
            linewidth=0.6,
            label=_short(model),
            zorder=3,
        )
        for j, (label, p) in enumerate(entries):
            step = 12 + 10 * (j // 2)
            dy = step if j % 2 == 0 else -step
            ax.annotate(
                label,
                (p["speed_us"], p["fidelity"]),
                fontsize=7,
                color=c,
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                arrowprops=dict(arrowstyle="-", color=c, lw=0.5, alpha=0.6),
            )

    ax.set_xscale("log")
    ax.set_xlabel(r"inference time [$\mu$s/event]  (log scale, lower is faster)")
    ax.set_ylabel("mean held-out log-likelihood  (higher is better)")
    ax.set_title("Inference cost vs. fidelity, by model and config", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# Driver
def upload_to_wandb(
    paths: dict, project=None, entity=None, run_name=None, config=None, run=None
):
    """
    Log every figure in ``paths`` to Weights & Biases as images.
    """
    try:
        import wandb
    except ImportError:
        print("[plots_dump] wandb not installed; skipping upload")
        return None

    own_run = run is None
    try:
        if own_run:
            run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config or {},
                job_type="comparison",
            )
        payload = {}
        for name, p in sorted(paths.items()):
            if p and Path(p).exists():
                payload[f"figures/{name}"] = wandb.Image(p)
            else:
                print(f"[plots_dump] missing figure, not uploaded: {name}")
        if payload:
            run.log(payload)
        print(
            f"[plots_dump] uploaded {len(payload)} figures to wandb"
            f" ({run.url if hasattr(run, 'url') else ''})"
        )
        return run
    except Exception as e:  # noqa: BLE001 - upload must not break the run
        print(f"[plots_dump] wandb upload failed ({e}); figures remain on disk")
        return None
    finally:
        if own_run and run is not None:
            try:
                run.finish()
            except Exception:
                pass


def plot_all_dump(
    dumps,
    save_dir: str = "figures_dump",
    losses_paths=None,
    baselines=None,
    baseline_samples=None,
    knn_curve=None,
    knn_sweep=None,
    knn_floor=None,
    speed_fidelity_points=None,
    p1_only: bool = True,
    wandb_project=None,
    wandb_entity=None,
    wandb_run_name=None,
    wandb_config=None,
):
    """
    Generate every dump-derived figure that the available inputs support.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    def _p(name):
        paths[name] = str(save_dir / f"{name}.png")
        return paths[name]

    plot_differential_ceiling(
        dumps,
        "parton_pt",
        knn_curve=knn_curve,
        baselines=baselines,
        save_path=_p("differential_ceiling_pt"),
    )

    if knn_sweep is not None:
        plot_knn_intercept(
            knn_sweep, floor=knn_floor, save_path=_p("knn_ceiling_intercept")
        )

    if len(dumps) >= 2:
        plot_model_vs_model(dumps[0], dumps[1], save_path=_p("model_vs_model"))

    plot_support_geometry(dumps, save_path=_p("support_geometry"))

    if losses_paths:
        plot_val_crps_curve(losses_paths, save_path=_p("val_crps_curve"))

    if not p1_only:
        for var, fname in [
            ("parton_eta", "skill_vs_eta"),
            ("pileup_nTrueInt", "skill_vs_pileup"),
        ]:
            if var in dumps[0].index.columns:
                plot_differential_ceiling(
                    dumps, var, log_x=False, baselines=baselines, save_path=_p(fname)
                )
        plot_skill_by_flavour(dumps, save_path=_p("skill_by_flavour"))
        plot_conditional_pit(dumps, save_path=_p("conditional_pit"))
        plot_n_eff(dumps[0], save_path=_p("n_eff_vs_pt"))
        plot_jet_delta_r(dumps, save_path=_p("jet_delta_r"))

        if speed_fidelity_points:
            plot_speed_vs_fidelity(
                speed_fidelity_points, save_path=_p("speed_vs_fidelity")
            )

        for dump in dumps:
            if dump.internals is not None and "pi" in dump.internals:
                plot_component_specialisation(
                    dump, save_path=_p("component_specialisation")
                )

        if len(dumps) >= 2 or baseline_samples:

            n_tri = min(
                [len(d.index) for d in dumps]
                + [len(s) for s in (baseline_samples or {}).values()]
            )
            series = {
                d.meta["model"]: np.median(np.asarray(d.samples[:n_tri]), axis=1)
                for d in dumps
            }
            for label, s in (baseline_samples or {}).items():
                series[label] = np.median(np.asarray(s[:n_tri]), axis=1)
            y_true_tri = dumps[0].y_true()[:n_tri]
            pt_tri = dumps[0].index["parton_pt"].to_numpy()[:n_tri]
            plot_triptych(series, y_true_tri, pt_tri, save_path=_p("triptych"))

            eta_tri = dumps[0].index["parton_eta"].to_numpy()[:n_tri]
            phi_tri = dumps[0].index["parton_phi"].to_numpy()[:n_tri]
            for obs in ("mass", "energy", "eta", "phi"):
                plot_triptych_observable(
                    series,
                    y_true_tri,
                    pt_tri,
                    eta_tri,
                    phi_tri,
                    obs,
                    save_path=_p(f"triptych_{obs}"),
                )

    print(f"[plots_dump] wrote {len(paths)} figures to {save_dir}")

    if wandb_project:
        upload_to_wandb(
            paths,
            project=wandb_project,
            entity=wandb_entity,
            run_name=wandb_run_name,
            config=wandb_config,
        )
    return paths
