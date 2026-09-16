"""
Plotting utilities for V2 surrogates.

Generates publication-quality figures for training diagnostics, truth-vs-sampled comparisons, calibration, and physics closure.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

try:
    from skimage.measure import marching_cubes

    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

from .features import TARGET_NAMES, FLAVOUR_NAMES, _FLAVOUR_MAP, reconstruct_jet

# Style defaults
_STYLE = {
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}
plt.rcParams.update(_STYLE)

_TRUE_C = "#2176AE"
_PRED_C = "#F4845F"
_MEAN_C = "#7B2D8B"
_GRID_KW = dict(alpha=0.25, linestyle="--")

_TARGET_LABELS = {
    "log_pT_resp": r"$\log(p_{T}^{\mathrm{jet}} / p_{T}^{\mathrm{parton}})$",
    "delta_eta": r"$\Delta\eta$ (parity-folded)",
    "delta_phi": r"$\Delta\phi$",
    "log_mass_frac": r"$\log(m_{\mathrm{jet}} / p_{T}^{\mathrm{jet}})$",
}

_RATIO_YLIM = (0.8, 1.2)


def _mark_clipped_bins(ax, centers, ratio, ylim):
    """Draw arrow markers at the ylim boundary for out-of-range ratio bins."""
    lo, hi = ylim
    for c, r in zip(centers, ratio):
        if np.isnan(r) or r == 1.0:
            continue
        if r > hi:
            ax.annotate(
                "",
                xy=(c, hi),
                xytext=(c, hi - 0.03 * (hi - lo)),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            )
        elif r < lo:
            ax.annotate(
                "",
                xy=(c, lo),
                xytext=(c, lo + 0.03 * (hi - lo)),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            )


# Fig A: Optimisation curves
def plot_optimization(
    train_losses: list[float],
    val_losses: list[float],
    lr_history: list[float] | None = None,
    grad_norm_history: list[float] | None = None,
    save_path: str | None = None,
):
    """Training and validation loss curves."""
    fig, axes = plt.subplots(1, 2 if lr_history else 1, figsize=(12, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    ax = axes[0]
    ax.plot(train_losses, label="Train", alpha=0.8, linewidth=1.0)
    ax.plot(val_losses, label="Val", alpha=0.8, linewidth=1.0)
    if val_losses:
        best_ep = int(np.argmin(val_losses))
        ax.axvline(
            best_ep,
            color="red",
            ls=":",
            lw=1.5,
            label=f"Best epoch ({best_ep + 1})",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(alpha=0.3)

    if lr_history and len(axes) > 1:
        ax2 = axes[1]
        ax2.plot(lr_history, color="tab:orange", linewidth=1.0, label="LR")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("LR & Grad Norm")

        if grad_norm_history:
            ax3 = ax2.twinx()
            ax3.plot(
                grad_norm_history,
                color="tab:green",
                alpha=0.3,
                linewidth=1.0,
                label="Grad Norm",
            )
            ax3.set_ylabel("Grad Norm")
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines3, labels3 = ax3.get_legend_handles_labels()
            ax2.legend(
                lines2 + lines3, labels2 + labels3, loc="upper right", fontsize=8
            )
        else:
            ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


# Fig C: Truth vs Sampled marginals with ratio panel
def plot_truth_vs_sampled(
    truth: np.ndarray,
    sampled: np.ndarray,
    n_bins: int = 100,
    save_path: str | None = None,
    marginal_ref: np.ndarray | None = None,
):
    """1-D marginal histograms with ratio panels."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, height_ratios=[3, 1, 3, 1], hspace=0.05, wspace=0.3)

    for dim, name in enumerate(TARGET_NAMES):
        row = (dim // 2) * 2
        col = dim % 2

        ax_main = fig.add_subplot(gs[row, col])
        ax_ratio = fig.add_subplot(gs[row + 1, col], sharex=ax_main)

        t = truth[:, dim]
        s = sampled[:, dim]
        lo = np.percentile(np.concatenate([t, s]), 0.5)
        hi = np.percentile(np.concatenate([t, s]), 99.5)
        edges = np.linspace(lo, hi, n_bins + 1)

        ht, _ = np.histogram(t, bins=edges)
        hs, _ = np.histogram(s, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        # Normalise
        ht_norm = ht / (ht.sum() * width + 1e-20)
        hs_norm = hs / (hs.sum() * width + 1e-20)

        ax_main.step(
            centers, ht_norm, where="mid", label="Truth", color="black", linewidth=1.2
        )
        ax_main.step(
            centers,
            hs_norm,
            where="mid",
            label="Model",
            color="tab:blue",
            linewidth=1.2,
            linestyle="--",
        )

        if marginal_ref is not None:
            hm, _ = np.histogram(marginal_ref[:, dim], bins=edges)
            hm_norm = hm / (hm.sum() * width + 1e-20)
            ax_main.step(
                centers,
                hm_norm,
                where="mid",
                label="Marginal-only",
                color="gray",
                linewidth=1.0,
                linestyle=":",
            )

        ax_main.set_ylabel("Density")
        ax_main.set_title(_TARGET_LABELS.get(name, name))
        ax_main.legend(loc="upper right")
        ax_main.grid(alpha=0.2)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Ratio panel
        ratio = np.divide(
            hs_norm, ht_norm, out=np.ones_like(hs_norm), where=ht_norm > 0
        )
        ax_ratio.step(centers, ratio, where="mid", color="tab:blue", linewidth=1.0)
        ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="-")
        ax_ratio.set_ylim(*_RATIO_YLIM)
        _mark_clipped_bins(ax_ratio, centers, ratio, _RATIO_YLIM)
        ax_ratio.set_ylabel("Model/Truth")
        ax_ratio.set_xlabel(_TARGET_LABELS.get(name, name))
        ax_ratio.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


# Fig: Per-flavour marginals
def plot_per_flavour_marginals(
    truth: np.ndarray,
    sampled: np.ndarray,
    pdg_ids: np.ndarray,
    target_dim: int = 0,
    n_bins: int = 80,
    save_path: str | None = None,
):
    """Per-flavour breakdown of one target dimension."""
    abs_pdg = np.abs(pdg_ids)
    groups = {"g": 21, "uds": [1, 2, 3], "c": 4, "b": 5}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    name = TARGET_NAMES[target_dim]

    for ax, (flav_name, flav_ids) in zip(axes, groups.items()):
        if isinstance(flav_ids, list):
            mask = np.isin(abs_pdg, flav_ids)
        else:
            mask = abs_pdg == flav_ids
        if mask.sum() < 100:
            ax.set_title(f"{flav_name}: insufficient data")
            continue

        t = truth[mask, target_dim]
        s = sampled[mask, target_dim]
        lo = np.percentile(np.concatenate([t, s]), 1)
        hi = np.percentile(np.concatenate([t, s]), 99)
        edges = np.linspace(lo, hi, n_bins + 1)

        ax.hist(
            t,
            bins=edges,
            density=True,
            alpha=0.5,
            label="Truth",
            color="black",
            histtype="step",
            linewidth=1.2,
        )
        ax.hist(
            s,
            bins=edges,
            density=True,
            alpha=0.5,
            label="Model",
            color="tab:blue",
            histtype="step",
            linewidth=1.2,
            linestyle="--",
        )
        ax.set_title(f"{flav_name} (n={mask.sum():,})")
        ax.set_xlabel(_TARGET_LABELS.get(name, name))
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.2)

    plt.suptitle(f"Per-Flavour: {_TARGET_LABELS.get(name, name)}", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


# Fig: PIT histograms
def plot_pit_histograms(
    pit_values: np.ndarray,
    n_bins: int = 50,
    save_path: str | None = None,
):
    """PIT histogram per target dimension — should be uniform if calibrated."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for dim, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
        ax.hist(
            pit_values[:, dim],
            bins=n_bins,
            range=(0, 1),
            density=True,
            color="tab:blue",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.axhline(
            1.0,
            color="red",
            linewidth=1.5,
            linestyle="--",
            label="Uniform ≡ Marginal-only",
        )
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.set_title(_TARGET_LABELS.get(name, name))
        ax.legend()
        ax.grid(alpha=0.2)

    plt.suptitle("PIT Calibration Histograms", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


# Fig: Phi profile of jet energy response
def plot_phi_profile(
    truth_phi: np.ndarray,
    truth_response: np.ndarray,
    sampled_phi: np.ndarray,
    sampled_response: np.ndarray,
    n_bins: int = 36,
    save_path: str | None = None,
):
    """φ-binned jet energy response profile."""
    from .metrics import phi_profile

    centers_t, means_t, stds_t, counts_t = phi_profile(
        truth_phi, truth_response, n_bins=n_bins
    )
    centers_s, means_s, stds_s, counts_s = phi_profile(
        sampled_phi, sampled_response, n_bins=n_bins
    )

    # SEM = std / sqrt(n), converted to ×10^-3 units for the plot
    sem_t = stds_t / np.sqrt(np.maximum(counts_t, 1)) * 1e3
    sem_s = stds_s / np.sqrt(np.maximum(counts_s, 1)) * 1e3

    # Subtract global mean to show the modulation
    global_mean_t = np.nanmean(means_t)
    global_mean_s = np.nanmean(means_s)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        centers_t,
        (means_t - global_mean_t) * 1e3,
        yerr=sem_t,
        fmt="ko-",
        markersize=4,
        label="Truth",
        linewidth=1.0,
        capsize=2,
    )
    ax.errorbar(
        centers_s,
        (means_s - global_mean_s) * 1e3,
        yerr=sem_s,
        fmt="b^--",
        markersize=4,
        label="Model",
        linewidth=1.0,
        alpha=0.8,
        capsize=2,
    )
    ax.set_xlabel(r"$\phi_{\mathrm{jet}}$ [rad]")
    ax.set_ylabel(
        r"$\langle \log(p_T^{\mathrm{jet}}/p_T^{\mathrm{parton}}) \rangle - \mathrm{global\ mean}\ [\times 10^{-3}]$"
    )
    ax.set_title("Azimuthal Response Profile")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


# Fig: Conditional independence diagnostic
def plot_conditional_response(
    conditioner_val: np.ndarray,
    truth_target: np.ndarray,
    sampled_target: np.ndarray,
    cond_name: str = r"$\log(p_T^{\mathrm{parton}})$",
    target_name: str = "log_mass_frac",
    n_bins: int = 20,
    save_path: str | None = None,
):
    """
    Binned mean response as a function of a conditioner variable.

    A flat model response (while truth varies) indicates conditional collapse.
    """
    edges = np.linspace(
        np.percentile(conditioner_val, 1),
        np.percentile(conditioner_val, 99),
        n_bins + 1,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(conditioner_val, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    truth_means = np.array(
        [
            np.mean(truth_target[bin_idx == i]) if (bin_idx == i).sum() > 10 else np.nan
            for i in range(n_bins)
        ]
    )
    sampled_means = np.array(
        [
            (
                np.mean(sampled_target[bin_idx == i])
                if (bin_idx == i).sum() > 10
                else np.nan
            )
            for i in range(n_bins)
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, truth_means, "ko-", label="Truth", markersize=4)
    ax.plot(centers, sampled_means, "b^--", label="Model", markersize=4, alpha=0.8)
    ax.set_xlabel(cond_name)
    ax.set_ylabel(f"Mean {_TARGET_LABELS.get(target_name, target_name)}")
    ax.set_title("Conditional Response Check")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


# New figure functions
def plot_abs_kinematics(
    true_jet: dict,
    sampled_jet: dict,
    save_path: str | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    keys = ["jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_mass"]
    labels = [
        r"Jet $p_T$ [GeV]",
        r"Jet $\eta$",
        r"Jet $\phi$ [rad]",
        r"Jet $E$ [GeV]",
        r"Jet $m$ [GeV]",
    ]

    for i, (key, xlabel) in enumerate(zip(keys, labels)):
        row = i // 3
        col = i % 3
        inner_gs = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[row, col], height_ratios=[3, 1], hspace=0.05
        )

        ax_main = fig.add_subplot(inner_gs[0])
        ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)

        vt = true_jet[key]
        vs = sampled_jet[key]

        lo = np.percentile(np.concatenate([vt, vs]), 0.5)
        hi = np.percentile(np.concatenate([vt, vs]), 99.5)
        edges = np.linspace(lo, hi, 60)

        ht, _ = np.histogram(vt, bins=edges)
        hs, _ = np.histogram(vs, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        ht_norm = ht / (ht.sum() * width + 1e-20)
        hs_norm = hs / (hs.sum() * width + 1e-20)

        ax_main.step(
            centers, ht_norm, where="mid", label="Truth", color=_TRUE_C, linewidth=1.5
        )
        ax_main.step(
            centers,
            hs_norm,
            where="mid",
            label="Model",
            color=_PRED_C,
            linewidth=1.5,
            linestyle="--",
        )
        ax_main.set_ylabel("Density")
        ax_main.legend(loc="upper right")
        ax_main.grid(**_GRID_KW)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        ratio = np.divide(
            hs_norm, ht_norm, out=np.ones_like(hs_norm), where=ht_norm > 0
        )
        ax_ratio.step(centers, ratio, where="mid", color=_PRED_C, linewidth=1.5)
        ax_ratio.axhline(1.0, color="black", linewidth=1.0, linestyle="-")
        ax_ratio.set_ylim(*_RATIO_YLIM)
        _mark_clipped_bins(ax_ratio, centers, ratio, _RATIO_YLIM)
        ax_ratio.set_ylabel("Mod/Tru")
        ax_ratio.set_xlabel(xlabel)
        ax_ratio.grid(**_GRID_KW)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_response_linearity(
    truth: np.ndarray,
    sampled: np.ndarray,
    parton_pt: np.ndarray,
    n_bins: int = 15,
    save_path: str | None = None,
    pt_label: str = r"Parton $p_T$ [GeV]",
) -> plt.Figure:
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 1, hspace=0.3)

    lo, hi = np.percentile(parton_pt, 1), np.percentile(parton_pt, 99)
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    bin_idx = np.digitize(parton_pt, edges) - 1

    target_dims = [0, 3]
    target_names = ["log_pT_resp", "log_mass_frac"]

    for i, (dim, name) in enumerate(zip(target_dims, target_names)):
        inner_gs = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[i], height_ratios=[3, 1], hspace=0.05
        )

        ax_main = fig.add_subplot(inner_gs[0])
        ax_res = fig.add_subplot(inner_gs[1], sharex=ax_main)

        t_med, t_lo, t_hi = [], [], []
        s_med, s_lo, s_hi = [], [], []

        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 10:
                t_vals = truth[mask, dim]
                s_vals = sampled[mask, dim]
                t_med.append(np.median(t_vals))
                t_lo.append(np.percentile(t_vals, 16))
                t_hi.append(np.percentile(t_vals, 84))
                s_med.append(np.median(s_vals))
                s_lo.append(np.percentile(s_vals, 16))
                s_hi.append(np.percentile(s_vals, 84))
            else:
                t_med.append(np.nan)
                t_lo.append(np.nan)
                t_hi.append(np.nan)
                s_med.append(np.nan)
                s_lo.append(np.nan)
                s_hi.append(np.nan)

        t_med, t_lo, t_hi = np.array(t_med), np.array(t_lo), np.array(t_hi)
        s_med, s_lo, s_hi = np.array(s_med), np.array(s_lo), np.array(s_hi)

        ax_main.plot(centers, t_med, "o-", color=_TRUE_C, label="Truth Median")
        ax_main.fill_between(centers, t_lo, t_hi, color=_TRUE_C, alpha=0.2)
        ax_main.plot(centers, s_med, "^--", color=_PRED_C, label="Model Median")
        ax_main.fill_between(centers, s_lo, s_hi, color=_PRED_C, alpha=0.2)

        ax_main.set_ylabel(_TARGET_LABELS.get(name, name))
        ax_main.set_xscale("log")
        ax_main.legend()
        ax_main.grid(**_GRID_KW)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        res = s_med - t_med
        ax_res.plot(centers, res, "o-", color="black")
        ax_res.axhline(0, color="gray", linestyle="--")
        ax_res.set_ylabel("Mod - Tru")
        ax_res.set_xlabel(pt_label)
        ax_res.set_xscale("log")
        ax_res.grid(**_GRID_KW)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_joint_coverage(
    hdr_rank: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    cls = np.linspace(0.05, 0.95, 19)
    empirical = np.array([np.mean(hdr_rank <= cl) for cl in cls])

    ax.plot(cls, empirical, "o-", color=_PRED_C, label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Ideal = Marginal-only")
    ax.fill_between(cls, cls, empirical, color=_PRED_C, alpha=0.2)
    ax.set_xlabel("Nominal HDR Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Joint HDR Coverage")
    ax.legend()
    ax.grid(**_GRID_KW)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def _kde_isosurface_panel(
    ax, pt, eta, phi, energy, cmap_name="viridis", axis_ranges=None
):
    if not _HAS_SKIMAGE:
        return None

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n_pts = min(6000, len(pt))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pt), n_pts, replace=False)

    x = pt[idx]
    y = eta[idx]
    z = phi[idx]
    e = energy[idx]

    pts = np.vstack([x, y, z])
    kde = gaussian_kde(pts, bw_method="scott")

    if axis_ranges is not None:
        axes_1d = [np.linspace(lo, hi, 35) for lo, hi in axis_ranges]
    else:
        axes_1d = [np.linspace(v.min(), v.max(), 35) for v in [x, y, z]]
    gx, gy, gz = np.meshgrid(*axes_1d, indexing="ij")
    grid_pts = np.vstack([gx.ravel(), gy.ravel(), gz.ravel()])
    density = kde(grid_pts).reshape(35, 35, 35)

    nz_den = density[density > 0]
    if len(nz_den) == 0:
        return None
    levels = np.percentile(nz_den, np.linspace(30, 85, 4))
    alphas = np.linspace(0.22, 0.80, 4)

    # Energy colouring via weighted KDE proxy
    w = e / e.sum()
    kde_e = gaussian_kde(pts, bw_method="scott", weights=w)
    e_volume = kde_e(grid_pts).reshape(35, 35, 35) / np.clip(density, 1e-30, None)
    e_volume *= e.mean() / w.mean()

    e_interp = RegularGridInterpolator(
        axes_1d, e_volume, method="linear", bounds_error=False, fill_value=None
    )

    cmap = plt.get_cmap(cmap_name)
    e_min, e_max = e_volume.min(), e_volume.max()
    norm = mcolors.Normalize(vmin=e_min, vmax=e_max)
    mappable = None

    for lvl, alpha in zip(levels, alphas):
        try:
            verts, faces, _, _ = marching_cubes(density, level=lvl)
        except (ValueError, RuntimeError):
            continue

        verts_real = np.column_stack(
            [np.interp(verts[:, i], np.arange(35), axes_1d[i]) for i in range(3)]
        )
        e_verts = e_interp(verts_real)
        e_faces = e_verts[faces].mean(axis=1)
        e_norm = (e_faces - e_min) / (e_max - e_min + 1e-8)

        fc = cmap(e_norm)
        fc[:, 3] = alpha

        poly = Poly3DCollection(
            verts_real[faces],
            facecolors=fc,
            edgecolors="none",
            antialiased=False,
            shade=False,
        )
        ax.add_collection3d(poly)

        # Set axis limits from vertices
        for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            setter(axes_1d[i][0], axes_1d[i][-1])

        if mappable is None:
            mappable = mcm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])

    return mappable


def plot_density_3d(
    parton_pt: np.ndarray,
    parton_eta: np.ndarray,
    parton_phi: np.ndarray,
    parton_energy: np.ndarray,
    true_jet: dict,
    sampled_jet: dict,
    save_path: str | None = None,
) -> plt.Figure | None:
    if not _HAS_SKIMAGE:
        print("Warning: skimage not available. Skipping 3D KDE plot.")
        return None

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.set_title("Parton Input")
    sm = _kde_isosurface_panel(
        ax1, parton_pt, parton_eta, parton_phi, parton_energy, "plasma"
    )
    ax1.set_xlabel(r"$p_T$ [GeV]")
    ax1.set_ylabel(r"$\eta$")
    ax1.set_zlabel(r"$\phi$ [rad]")
    if sm:
        fig.colorbar(sm, ax=ax1, label="Mean Energy [GeV]")

    def _shared_range(arr_a, arr_b, pct_lo=1, pct_hi=99, n_pts=6000, seed=42):
        rng = np.random.default_rng(seed)
        a = arr_a[rng.choice(len(arr_a), min(n_pts, len(arr_a)), replace=False)]
        b = arr_b[rng.choice(len(arr_b), min(n_pts, len(arr_b)), replace=False)]
        both = np.concatenate([a, b])
        return (np.percentile(both, pct_lo), np.percentile(both, pct_hi))

    jet_ranges = [
        _shared_range(true_jet[k], sampled_jet[k])
        for k in ["jet_pt", "jet_eta", "jet_phi"]
    ]

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.set_title("True Jet")
    sm = _kde_isosurface_panel(
        ax2,
        true_jet["jet_pt"],
        true_jet["jet_eta"],
        true_jet["jet_phi"],
        true_jet["jet_energy"],
        "plasma",
        axis_ranges=jet_ranges,
    )
    ax2.set_xlabel(r"$p_T$ [GeV]")
    ax2.set_ylabel(r"$\eta$")
    ax2.set_zlabel(r"$\phi$ [rad]")
    if sm:
        fig.colorbar(sm, ax=ax2, label="Mean Energy [GeV]")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.set_title("Sampled Jet")
    sm = _kde_isosurface_panel(
        ax3,
        sampled_jet["jet_pt"],
        sampled_jet["jet_eta"],
        sampled_jet["jet_phi"],
        sampled_jet["jet_energy"],
        "plasma",
        axis_ranges=jet_ranges,
    )
    ax3.set_xlabel(r"$p_T$ [GeV]")
    ax3.set_ylabel(r"$\eta$")
    ax3.set_zlabel(r"$\phi$ [rad]")
    if sm:
        fig.colorbar(sm, ax=ax3, label="Mean Energy [GeV]")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_invariant_mass(
    true_jet: dict,
    sampled_jet: dict,
    save_path: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    t_mass = true_jet["jet_mass"]
    s_mass = sampled_jet["jet_mass"]

    lo = np.percentile(np.concatenate([t_mass, s_mass]), 0.5)
    hi = np.percentile(np.concatenate([t_mass, s_mass]), 99.5)
    edges = np.linspace(lo, hi, 81)

    for ax, log_scale in zip(axes, [False, True]):
        ax.hist(
            t_mass,
            bins=edges,
            histtype="step",
            color=_TRUE_C,
            label="Truth",
            linewidth=1.5,
            density=True,
        )
        ax.hist(
            s_mass,
            bins=edges,
            histtype="step",
            color=_PRED_C,
            label="Model",
            linewidth=1.5,
            linestyle="--",
            density=True,
        )
        ax.set_xlabel(r"Jet $m$ [GeV]")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(**_GRID_KW)
        if log_scale:
            ax.set_yscale("log")

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_correlations_pt_energy(
    true_jet: dict,
    sampled_jet: dict,
    n_subsample: int = 30_000,
    save_path: str | None = None,
) -> plt.Figure:
    """pT-E hexbin for truth and sampled jets.  Panels show the same event
    indices so marginal differences are visible without sampling noise."""
    t_pt = true_jet["jet_pt"]
    t_en = true_jet["jet_energy"]
    s_pt = sampled_jet["jet_pt"]
    s_en = sampled_jet["jet_energy"]

    n = min(len(t_pt), len(s_pt))
    rng = np.random.default_rng(0)
    idx = rng.choice(n, min(n_subsample, n), replace=False)

    xlim = (
        min(t_pt[idx].min(), s_pt[idx].min()),
        max(t_pt[idx].max(), s_pt[idx].max()),
    )
    ylim = (
        min(t_en[idx].min(), s_en[idx].min()),
        max(t_en[idx].max(), s_en[idx].max()),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r"Jet $p_T$ vs $E$ correlations", fontsize=13)

    hb1 = axes[0].hexbin(
        t_pt[idx],
        t_en[idx],
        gridsize=55,
        extent=(*xlim, *ylim),
        cmap="Blues",
        norm=mcolors.LogNorm(),
        mincnt=1,
    )
    axes[0].set_title("Truth")
    axes[0].set_xlabel(r"Jet $p_T$ [GeV]")
    axes[0].set_ylabel(r"Jet $E$ [GeV]")
    fig.colorbar(hb1, ax=axes[0], label="Counts (log)")

    hb2 = axes[1].hexbin(
        s_pt[idx],
        s_en[idx],
        gridsize=55,
        extent=(*xlim, *ylim),
        cmap="Oranges",
        norm=mcolors.LogNorm(),
        mincnt=1,
    )
    axes[1].set_title("Sampled (model)")
    axes[1].set_xlabel(r"Jet $p_T$ [GeV]")
    axes[1].set_ylabel(r"Jet $E$ [GeV]")
    fig.colorbar(hb2, ax=axes[1], label="Counts (log)")

    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_kl_summary(
    kl_divergences: dict,
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = []
    vals = []
    floors = []
    for k, v in kl_divergences.items():
        if k.startswith("kl_floor_"):
            continue
        name = k.replace("kl_", "")
        labels.append(_TARGET_LABELS.get(name, name))
        vals.append(v)
        floors.append(kl_divergences.get(f"kl_floor_{name}"))

    y_pos = np.arange(len(labels))
    colors = plt.cm.tab10.colors[: len(labels)]

    ax.barh(y_pos, vals, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("KL Divergence")

    if any(f is not None for f in floors):
        for i, f in enumerate(floors):
            if f is not None:
                label = "Finite-sample floor" if i == 0 else None
                ax.plot(
                    f,
                    y_pos[i],
                    marker="|",
                    color="black",
                    markersize=18,
                    markeredgewidth=2,
                    label=label,
                )

    mean_kl = np.mean(vals)
    ax.axvline(mean_kl, color="red", linestyle="--", label=f"Mean: {mean_kl:.3f}")
    ax.axvline(0.0, color="gray", ls=":", lw=1.5, label="Marginal-only floor")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_inference_speed(
    speed_results: dict,
    save_path: str | None = None,
) -> plt.Figure | None:
    if not speed_results:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = list(speed_results.keys())
    vals = list(speed_results.values())

    y_pos = np.arange(len(labels))
    colors = plt.cm.Set2.colors[: len(labels)]

    bars = ax.bar(y_pos, vals, color=colors, alpha=0.8)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Inference time [µs/event]")
    ax.set_yscale("log")

    for bar in bars:
        yval = bar.get_height()
        label = f"{yval:.3g}" if yval >= 0.01 else f"{yval:.2e}"
        ax.text(
            bar.get_x() + bar.get_width() / 2, yval, label, ha="center", va="bottom"
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


# Master plot function
def plot_all(
    truth: np.ndarray,
    sampled: np.ndarray,
    pdg_ids: np.ndarray | None = None,
    parton_pt: np.ndarray | None = None,
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
    lr_history: list[float] | None = None,
    pit_values: np.ndarray | None = None,
    parton_phi: np.ndarray | None = None,
    save_dir: str = "figures",
    *,
    parton_eta: np.ndarray | None = None,
    parton_energy: np.ndarray | None = None,
    grad_norm_history: list[float] | None = None,
    hdr_rank: np.ndarray | None = None,
    kl_divergences: dict | None = None,
    speed_results: dict | None = None,
    pt_label: str = r"Parton $p_T$ [GeV]",
):
    """Generate all standard figures and save to ``save_dir``."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Optimisation
    if train_losses is not None and val_losses is not None:
        p = str(save_dir / "figA_optimization.png")
        plot_optimization(
            train_losses, val_losses, lr_history, grad_norm_history, save_path=p
        )
        paths["figA"] = p

    # Truth vs Sampled
    rng_marginal = np.random.default_rng(12345)
    marginal_ref = truth[rng_marginal.permutation(len(truth))]
    p = str(save_dir / "figC_truth_vs_sampled.png")
    plot_truth_vs_sampled(truth, sampled, save_path=p, marginal_ref=marginal_ref)
    paths["figC"] = p

    # Per-flavour marginals (for each target dim)
    if pdg_ids is not None:
        for dim in range(4):
            p = str(save_dir / f"figD_per_flavour_{TARGET_NAMES[dim]}.png")
            plot_per_flavour_marginals(
                truth, sampled, pdg_ids, target_dim=dim, save_path=p
            )
            paths[f"figD_{TARGET_NAMES[dim]}"] = p

    # PIT
    if pit_values is not None:
        p = str(save_dir / "figE_pit_calibration.png")
        plot_pit_histograms(pit_values, save_path=p)
        paths["figE"] = p

    if parton_pt is not None:
        log_pt = np.log(parton_pt)
        for dim, name in [(0, "log_pT_resp"), (3, "log_mass_frac")]:
            p = str(save_dir / f"figF_conditional_{name}.png")
            plot_conditional_response(
                log_pt,
                truth[:, dim],
                sampled[:, dim],
                target_name=name,
                save_path=p,
            )
            paths[f"figF_{name}"] = p

    # Phi profile
    if parton_phi is not None:
        # Use truth targets to get truth jet phi
        p = str(save_dir / "figG_phi_profile.png")
        truth_jet_phi = np.arctan2(
            np.sin(parton_phi + truth[:, 2]),
            np.cos(parton_phi + truth[:, 2]),
        )
        sampled_jet_phi = np.arctan2(
            np.sin(parton_phi + sampled[:, 2]),
            np.cos(parton_phi + sampled[:, 2]),
        )
        plot_phi_profile(
            truth_jet_phi,
            truth[:, 0],
            sampled_jet_phi,
            sampled[:, 0],
            save_path=p,
        )
        paths["figG"] = p

    true_jet = None
    sampled_jet = None

    if parton_pt is not None and parton_eta is not None and parton_phi is not None:
        true_jet = reconstruct_jet(parton_pt, parton_eta, parton_phi, truth)
        sampled_jet = reconstruct_jet(parton_pt, parton_eta, parton_phi, sampled)

        p = str(save_dir / "abs_kinematics.png")
        plot_abs_kinematics(true_jet, sampled_jet, save_path=p)
        paths["abs_kinematics"] = p

        p = str(save_dir / "invariant_mass.png")
        plot_invariant_mass(true_jet, sampled_jet, save_path=p)
        paths["invariant_mass"] = p

        p = str(save_dir / "correlations_pt_energy.png")
        plot_correlations_pt_energy(true_jet, sampled_jet, save_path=p)
        paths["correlations_pt_energy"] = p

    if parton_pt is not None:
        p = str(save_dir / "response_linearity.png")
        plot_response_linearity(
            truth, sampled, parton_pt, save_path=p, pt_label=pt_label
        )
        paths["response_linearity"] = p

    if hdr_rank is not None:
        p = str(save_dir / "joint_coverage.png")
        plot_joint_coverage(hdr_rank, save_path=p)
        paths["joint_coverage"] = p

    if (
        parton_pt is not None
        and parton_eta is not None
        and parton_phi is not None
        and parton_energy is not None
    ):
        p = str(save_dir / "density_3d.png")
        plot_density_3d(
            parton_pt,
            parton_eta,
            parton_phi,
            parton_energy,
            true_jet,
            sampled_jet,
            save_path=p,
        )
        paths["density_3d"] = p

    if kl_divergences is not None:
        p = str(save_dir / "kl_summary.png")
        plot_kl_summary(kl_divergences, save_path=p)
        paths["kl_summary"] = p

    if speed_results is not None:
        p = str(save_dir / "inference_speed.png")
        plot_inference_speed(speed_results, save_path=p)
        paths["inference_speed"] = p

    return paths


def plot_dijet_mass(
    m_true: np.ndarray,
    m_model: np.ndarray,
    n_bins: int = 80,
    save_path: str | None = None,
) -> plt.Figure:
    """Dijet invariant mass figure."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.3)

    for col, log_y in enumerate([False, True]):
        ax_main = fig.add_subplot(gs[0, col])
        ax_ratio = fig.add_subplot(gs[1, col], sharex=ax_main)

        lo = np.percentile(np.concatenate([m_true, m_model]), 0.5)
        hi = np.percentile(np.concatenate([m_true, m_model]), 99.5)
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        ht, _ = np.histogram(m_true, bins=edges)
        hs, _ = np.histogram(m_model, bins=edges)

        # normalize to density
        ht_n = ht / (ht.sum() * width + 1e-20)
        hs_n = hs / (hs.sum() * width + 1e-20)

        ax_main.step(
            centers, ht_n, where="mid", label="Truth", color=_TRUE_C, linewidth=1.5
        )
        ax_main.step(
            centers,
            hs_n,
            where="mid",
            label="Model",
            color=_PRED_C,
            linewidth=1.5,
            linestyle="--",
        )
        ax_main.set_ylabel("Density")
        ax_main.set_title("Dijet Invariant Mass" + (" (log scale)" if log_y else ""))
        ax_main.legend()
        ax_main.grid(**_GRID_KW)
        if log_y:
            ax_main.set_yscale("log")
        plt.setp(ax_main.get_xticklabels(), visible=False)

        ratio = np.divide(hs_n, ht_n, out=np.ones_like(hs_n), where=ht_n > 0)
        ax_ratio.step(centers, ratio, where="mid", color=_PRED_C, linewidth=1.5)
        ax_ratio.axhline(1.0, color="black", linewidth=1.0, linestyle="-")
        ax_ratio.set_ylim(*_RATIO_YLIM)
        _mark_clipped_bins(ax_ratio, centers, ratio, _RATIO_YLIM)
        ax_ratio.set_ylabel("Model/Truth")
        ax_ratio.set_xlabel(r"$m_{jj}$ [GeV]")
        ax_ratio.grid(**_GRID_KW)

    fig.suptitle(f"Dijet Invariant Mass (n={len(m_true):,} events)", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_casimir_scaling(
    bin_centers: np.ndarray,
    truth_ratios: np.ndarray,
    model_ratios: np.ndarray,
    truth_ratios_lowPU: np.ndarray | None = None,
    model_ratios_lowPU: np.ndarray | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Casimir scaling figure (2 panels)."""
    n_panels = 2 if truth_ratios_lowPU is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels + 2, 5))
    if n_panels == 1:
        axes = [axes]

    for ax_i, (ax, label, tr, mr) in enumerate(
        zip(
            axes,
            ["All pileup", f"Low pileup (nTrueInt < median)"],
            [truth_ratios, truth_ratios_lowPU],
            [model_ratios, model_ratios_lowPU],
        )
    ):
        if tr is None:
            continue

        ax.plot(bin_centers, tr, "ko-", label="Truth", markersize=5, lw=1.5)
        ax.plot(bin_centers, mr, "b^--", label="Model", markersize=5, lw=1.5, alpha=0.8)
        ax.axhline(2.25, color="gray", ls=":", lw=1.5, label=r"$C_A/C_F = 9/4$")
        ax.set_xlabel(r"Parton $p_T$ [GeV]")
        ax.set_ylabel(
            r"$\langle (m/p_T)^2 \rangle_g \;/\; \langle (m/p_T)^2 \rangle_{uds}$"
        )
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(**_GRID_KW)
        ax.set_ylim(1.2, 2.5)

    fig.suptitle("Casimir Scaling: Gluon vs Light-Quark Jet Mass", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig
