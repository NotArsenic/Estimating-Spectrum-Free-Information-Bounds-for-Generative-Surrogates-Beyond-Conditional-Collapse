"""
MDN-specific diagnostic figures.
"""

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.features import TARGET_NAMES

# Style (matches common/plots.py)
_TRUE_C = "#2176AE"
_PRED_C = "#F4845F"
_MEAN_C = "#7B2D8B"
_GRID_KW = dict(alpha=0.25, linestyle="--")

_TARGET_LABELS = [
    r"$\log(p_T^{\rm jet}/p_T^{\rm parton})$",
    r"$\Delta\eta$ (parity-folded)",
    r"$\Delta\phi$",
    r"$\log(m^{\rm jet}/p_T^{\rm jet})$",
]


# mixture_internals
def plot_mixture_internals(
    pi: np.ndarray,
    sigma: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualise the MDN mixture weights and component widths.

    Parameters
    ----------
    pi    : (N, K)  mixture weights (post-softmax)
    sigma : (N, K, D) per-component standard deviations. If the full L matrix is passed from the model, extract the diagonal externally and pass it here.
    save_path : optional file path to save the figure.

    Returns
    -------
    fig
    """
    K = pi.shape[1]
    D = sigma.shape[2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Mixture Internals — mean π_k and σ_k distributions", fontsize=13)

    # panel 1: stacked bar of mean π_k
    ax = axes[0]
    mean_pi = pi.mean(axis=0)
    order = np.argsort(-mean_pi)
    cmap = plt.get_cmap("tab20", K)
    bottom = 0.0
    for rank, k in enumerate(order):
        legend_label = f"k={k}" if rank < 8 else "_nolegend_"
        ax.bar(
            0,
            mean_pi[k],
            bottom=bottom,
            color=cmap(rank),
            label=legend_label,
            edgecolor="none",
        )
        bottom += mean_pi[k]
    ax.set_xticks([])
    ax.set_ylabel(r"Mean $\pi_k$")
    ax.set_title(r"Mean Mixing Coefficients $\pi_k$")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", **_GRID_KW)

    # panel 2: per-dimension σ_k distribution
    ax = axes[1]
    dim_colors = [_TRUE_C, _PRED_C, _MEAN_C, "green"]
    for d in range(D):
        sigma_flat = sigma[:, :, d].ravel()
        ax.hist(
            sigma_flat,
            bins=80,
            density=True,
            histtype="step",
            color=dim_colors[d % len(dim_colors)],
            linewidth=1.8,
            label=_TARGET_LABELS[d],
        )
    ax.set_xlabel(r"Component $\sigma_k$")
    ax.set_ylabel("Density")
    ax.set_title(r"Distribution of Component Widths $\sigma_k$")
    ax.legend(fontsize=9)
    ax.grid(**_GRID_KW)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# cholesky heatmap
def plot_cholesky(
    L: np.ndarray,
    pi: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualise the learned Cholesky factors L.

    Parameters
    ----------
    L  : (N, K, D, D)  lower-triangular Cholesky factors from the model.
    pi : (N, K) mixture weights (post-softmax), used to sort components.
    save_path : optional file path.

    Returns
    -------
    fig
    """
    D = L.shape[-1]
    K = L.shape[1]

    L_mean = np.abs(L).mean(axis=(0, 1))  # (D, D)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cholesky L — learned kinematic correlations", fontsize=13)

    # panel 1: mean |L| heatmap
    ax = axes[0]
    im = ax.imshow(L_mean, cmap="plasma", aspect="auto")
    plt.colorbar(im, ax=ax, label="mean |L_ij|")
    ax.set_xticks(range(D))
    ax.set_yticks(range(D))
    ax.set_xticklabels(_TARGET_LABELS[:D], fontsize=7, rotation=15)
    ax.set_yticklabels(_TARGET_LABELS[:D], fontsize=7)
    ax.set_title("Mean |L| across all events & components")
    for i in range(D):
        for j in range(D):
            ax.text(
                j,
                i,
                f"{L_mean[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if L_mean[i, j] > L_mean.max() * 0.5 else "black",
            )

    # panel 2: per-component diagonal (top 16 by mean π)
    pi_mean = pi.mean(axis=0)  # (K,)
    order = np.argsort(-pi_mean)
    top_k = min(16, K)
    diag_np = np.abs(L[:, :, range(D), range(D)]).mean(axis=0)  # (K, D)
    diag_sorted = diag_np[order[:top_k]]  # (top_k, D)

    ax2 = axes[1]
    im2 = ax2.imshow(diag_sorted.T, cmap="viridis", aspect="auto")
    plt.colorbar(im2, ax=ax2, label="mean |diag(L)|")
    ax2.set_yticks(range(D))
    ax2.set_yticklabels(_TARGET_LABELS[:D], fontsize=7)
    ax2.set_xlabel(f"Component (sorted by π, top {top_k})")
    ax2.set_title(f"Per-component diagonal |L| (top-{top_k} by π)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# run both figures from a model + conditioner array
def plot_mdn_internals(
    model,
    X: np.ndarray,
    device: str,
    save_dir: str,
    n_subsample: int = 2_000,
) -> dict:
    """
    Generate MDN-specific diagnostic figures.

    Parameters
    ----------
    model   : Multivariate_MDN instance (forward returns pi, mu, L).
    X       : conditioner array, shape (N, 26), float32.
    device  : torch device string.
    save_dir: directory to save PNGs.
    n_subsample : max events to use (expensive on large test sets).

    Returns
    -------
    paths : dict mapping figure key -> file path.
    """
    import torch

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    rng = np.random.default_rng(7)
    N = min(n_subsample, len(X))
    idx = rng.choice(len(X), N, replace=False)
    Xt = torch.from_numpy(X[idx]).float().to(device)

    with torch.no_grad():
        pi_t, mu_t, L_t = model(Xt)

    pi_np = pi_t.cpu().numpy()  # (N, K)
    L_np = L_t.cpu().numpy()  # (N, K, D, D)
    D = L_np.shape[-1]

    # Extract per-component std from diagonal of L @ L^T
    # var_diag = diag(L @ L^T) = sum over j of L_ij^2 (lower triangular)
    var_diag = (L_np**2).sum(axis=-1)  # (N, K, D)
    sigma_np = np.sqrt(np.clip(var_diag, 1e-12, None))  # (N, K, D)

    # mixture internals
    p = str(save_dir / "mixture_internals.png")
    plot_mixture_internals(pi_np, sigma_np, save_path=p)
    paths["mixture_internals"] = p

    # Cholesky heatmap
    p = str(save_dir / "cholesky.png")
    plot_cholesky(L_np, pi_np, save_path=p)
    paths["cholesky"] = p

    return paths
