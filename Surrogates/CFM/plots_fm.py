"""
Flow-matching-specific diagnostic figures for the CFM surrogate.

Figures produced:
  solver_convergence.png   — sample quality vs NFE for Euler and midpoint
  velocity_jacobian.png    — mean |∂v/∂x_t| 4×4 heatmap (kinematic coupling)
  velocity_field.png       — mean ||v(x_t, t, c)|| vs t for a fixed conditioner
  ode_trajectories.png     — per-dimension ODE trajectory spread for N test events
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.features import TARGET_NAMES

# Re-use the project integrators
from inference import euler_integrate, midpoint_integrate

# Style
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


# solver_convergence  (priority 1)
def plot_solver_convergence(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    n_events: int = 4_000,
    step_counts: list[int] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Sample quality (per-dim W1 distance to truth) vs NFE for Euler and midpoint.

    Parameters
    ----------
    model : VelocityMLP (with noise_loc/noise_scale attached by load_checkpoint)
    X_test : (N, 26) conditioner array, float32
    y_test : (N, 4)  ground-truth targets, float32
    device : torch device string
    n_events : events to use (subsampled for speed)
    step_counts : list of n_steps values to sweep (default 5,10,25,50,100,200)
    save_path : optional output path
    """
    import torch
    from scipy.stats import wasserstein_distance

    if step_counts is None:
        step_counts = [5, 10, 25, 50, 100, 200]

    rng = np.random.default_rng(17)
    n = min(n_events, len(X_test))
    idx = rng.choice(len(X_test), n, replace=False)
    X_sub = torch.from_numpy(X_test[idx]).float().to(device)
    y_sub = y_test[idx]  # numpy, used for W1 reference

    D = y_sub.shape[1]

    results = {"euler": {}, "midpoint": {}}
    with torch.no_grad():
        x0_shared = model.noise_loc + model.noise_scale * torch.randn(
            n, D, device=device
        )

    for solver_name, integrate_fn, nfe_per_step in [
        ("euler", euler_integrate, 1),
        ("midpoint", midpoint_integrate, 2),
    ]:
        for n_steps in step_counts:
            nfe = n_steps * nfe_per_step
            with torch.no_grad():
                x1 = integrate_fn(model, x0_shared, X_sub, n_steps)
            y_pred = x1.cpu().numpy()
            w1_per_dim = [
                wasserstein_distance(y_sub[:, d], y_pred[:, d]) for d in range(D)
            ]
            results[solver_name][nfe] = np.mean(w1_per_dim)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Solver Convergence — W1 distance vs NFE", fontsize=13)

    colors = {"euler": _TRUE_C, "midpoint": _PRED_C}
    markers = {"euler": "o-", "midpoint": "s--"}
    labels = {"euler": "Euler (1 NFE/step)", "midpoint": "Midpoint (2 NFE/step)"}

    for solver_name in ["euler", "midpoint"]:
        nfes = sorted(results[solver_name])
        w1s = [results[solver_name][k] for k in nfes]
        ax.plot(
            nfes,
            w1s,
            markers[solver_name],
            color=colors[solver_name],
            lw=1.8,
            markersize=7,
            label=labels[solver_name],
        )

    ax.set_xlabel("Number of Function Evaluations (NFE)")
    ax.set_ylabel(r"Mean W1 distance (averaged over 4 dims)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(**_GRID_KW)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# velocity_jacobian  (priority 2)
def plot_velocity_jacobian(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    n_events: int = 2_000,
    n_t_points: int = 5,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Mean |∂v/∂x_t| 4×4 Jacobian heatmap — the flow's structural analogue of the MDN's Cholesky heatmap.

    Parameters
    ----------
    model : VelocityMLP
    X_test : (N, 26) conditioner array, float32
    device : torch device string
    n_events : events to average over
    n_t_points : t values to average over (uniformly spaced in (0, 1))
    save_path : optional output path
    """
    import torch

    model.eval()  # BN/dropout off; gradients still flow

    rng = np.random.default_rng(23)
    n = min(n_events, len(X_test))
    idx = rng.choice(len(X_test), n, replace=False)
    X_sub = torch.from_numpy(X_test[idx]).float().to(device)
    y_sub = torch.from_numpy(y_test[idx]).float().to(device)
    D = 4

    t_values = np.linspace(0.1, 0.9, n_t_points)
    jac_sum = np.zeros((D, D), dtype=np.float64)
    count = 0

    for t_val in t_values:
        # Sample x0 from noise prior and construct true trajectory x_t
        with torch.no_grad():
            x0 = model.noise_loc + model.noise_scale * torch.randn(n, D, device=device)
            x_t = (1 - t_val) * x0 + t_val * y_sub
        x_t = x_t.requires_grad_(True)

        t = torch.full((n, 1), t_val, device=device)
        v = model(x_t, t, X_sub)  # (n, D)

        # Full Jacobian
        jac_batch = np.zeros((n, D, D), dtype=np.float32)
        for j in range(D):
            e_j = torch.zeros_like(x_t)
            e_j[:, j] = 1.0
            grad_j = torch.autograd.grad(
                v,
                x_t,
                grad_outputs=e_j,
                create_graph=False,
                retain_graph=(j < D - 1),
            )[
                0
            ]  # (n, D)  — row j of the Jacobian
            jac_batch[:, j, :] = grad_j.detach().cpu().numpy()

        jac_sum += np.abs(jac_batch).mean(axis=0)
        count += 1

    jac_mean = jac_sum / max(count, 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(
        r"Mean $|\partial v / \partial x_t|$ — learned kinematic coupling",
        fontsize=12,
    )

    im = ax.imshow(jac_mean, cmap="plasma", aspect="auto")
    plt.colorbar(im, ax=ax, label=r"mean $|\partial v_i / \partial x_{t,j}|$")
    ax.set_xticks(range(D))
    ax.set_yticks(range(D))
    ax.set_xticklabels(_TARGET_LABELS[:D], fontsize=7, rotation=20, ha="right")
    ax.set_yticklabels(_TARGET_LABELS[:D], fontsize=7)
    ax.set_xlabel("Input dimension $x_{t,j}$", fontsize=9)
    ax.set_ylabel("Velocity component $v_i$", fontsize=9)
    ax.set_title(
        f"Averaged over {n} events, {n_t_points} t values",
        fontsize=9,
        pad=4,
    )

    for i in range(D):
        for j in range(D):
            ax.text(
                j,
                i,
                f"{jac_mean[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if jac_mean[i, j] > jac_mean.max() * 0.5 else "black",
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# velocity_field  (priority 3)
def plot_velocity_field(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    n_events: int = 1_000,
    n_t_points: int = 20,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Mean ||v(x_t, t, c)|| vs t, plotted per output dimension.

    Shows how the velocity magnitude evolves along the trajectory
    """
    import torch

    rng = np.random.default_rng(31)
    n = min(n_events, len(X_test))
    idx = rng.choice(len(X_test), n, replace=False)
    X_sub = torch.from_numpy(X_test[idx]).float().to(device)
    y_sub = torch.from_numpy(y_test[idx]).float().to(device)
    D = 4

    t_vals = np.linspace(0.02, 0.98, n_t_points)
    v_norms = np.zeros((n_t_points, D), dtype=np.float64)

    with torch.no_grad():
        for i, t_val in enumerate(t_vals):
            x0 = model.noise_loc + model.noise_scale * torch.randn(n, D, device=device)
            x_t = (1 - t_val) * x0 + t_val * y_sub
            t = torch.full((n, 1), t_val, device=device)
            v = model(x_t, t, X_sub)  # (n, D)
            v_norms[i] = v.abs().mean(dim=0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Mean |velocity field| vs ODE time $t$", fontsize=13)

    dim_colors = [_TRUE_C, _PRED_C, _MEAN_C, "green"]
    for d in range(D):
        ax.plot(
            t_vals, v_norms[:, d], color=dim_colors[d], lw=1.8, label=_TARGET_LABELS[d]
        )

    ax.set_xlabel("ODE time $t$  (0 = noise, 1 = data)")
    ax.set_ylabel(r"Mean $|v_d(x_t, t, c)|$")
    ax.legend(fontsize=9)
    ax.grid(**_GRID_KW)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ode_trajectories  (priority 4)
def plot_ode_trajectories(
    model,
    X_test: np.ndarray,
    device: str,
    n_events: int = 50,
    n_steps: int = 50,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Per-dimension ODE trajectories for a small set of test events.

    Traces x_t from t=0 (noise) to t=1 (data) for each event, one panel per target dimension.  Shows trajectory spread and convergence.
    """
    import torch

    rng = np.random.default_rng(37)
    n = min(n_events, len(X_test))
    idx = rng.choice(len(X_test), n, replace=False)
    X_sub = torch.from_numpy(X_test[idx]).float().to(device)
    D = 4
    dt = 1.0 / n_steps
    t_checkpoints = np.arange(n_steps + 1) * dt

    # Record trajectory using midpoint integrator step-by-step
    with torch.no_grad():
        x = model.noise_loc + model.noise_scale * torch.randn(n, D, device=device)
        trajectories = [x.cpu().numpy()]

        for i in range(n_steps):
            t = torch.full((n, 1), i * dt, device=device)
            t_mid = torch.full((n, 1), (i + 0.5) * dt, device=device)
            v1 = model(x, t, X_sub)
            x_mid = x + 0.5 * dt * v1
            v2 = model(x_mid, t_mid, X_sub)
            x = x + dt * v2
            trajectories.append(x.cpu().numpy())

    traj = np.stack(trajectories, axis=1)  # (n, n_steps+1, D)

    fig, axes = plt.subplots(1, D, figsize=(5 * D, 4), sharey=False)
    fig.suptitle("ODE Trajectories (midpoint, per event)", fontsize=13)

    dim_colors = [_TRUE_C, _PRED_C, _MEAN_C, "green"]
    for d in range(D):
        ax = axes[d]
        for ev in range(n):
            ax.plot(
                t_checkpoints, traj[ev, :, d], color=dim_colors[d], alpha=0.15, lw=0.8
            )
        # Mean trajectory
        ax.plot(
            t_checkpoints,
            traj[:, :, d].mean(axis=0),
            color="black",
            lw=2.0,
            label="Mean",
        )
        ax.set_xlabel("ODE time $t$")
        ax.set_ylabel(_TARGET_LABELS[d])
        ax.set_title(_TARGET_LABELS[d], fontsize=9)
        ax.grid(**_GRID_KW)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# Convenience driver
def plot_fm_internals(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    save_dir: str,
    n_events_convergence: int = 4_000,
    n_events_jacobian: int = 2_000,
    n_events_field: int = 1_000,
    n_events_traj: int = 50,
    step_counts: list[int] | None = None,
) -> dict:
    """
    Generate all FM-specific diagnostic figures.

    Parameters
    ----------
    model : VelocityMLP (with noise_loc/noise_scale attached)
    X_test : (N, 26) conditioner array, float32
    y_test : (N, 4) ground-truth targets, float32
    device : torch device string
    save_dir : directory to write PNGs to
    n_events_* : event counts for each figure
    step_counts : n_steps values for convergence sweep

    Returns
    -------
    paths : dict mapping figure key -> file path
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    print("[fm_plots] Generating solver_convergence.png ...")
    p = str(save_dir / "solver_convergence.png")
    plot_solver_convergence(
        model,
        X_test,
        y_test,
        device,
        n_events=n_events_convergence,
        step_counts=step_counts,
        save_path=p,
    )
    paths["solver_convergence"] = p

    print("[fm_plots] Generating velocity_jacobian.png ...")
    p = str(save_dir / "velocity_jacobian.png")
    plot_velocity_jacobian(
        model, X_test, y_test, device, n_events=n_events_jacobian, save_path=p
    )
    paths["velocity_jacobian"] = p

    print("[fm_plots] Generating velocity_field.png ...")
    p = str(save_dir / "velocity_field.png")
    plot_velocity_field(
        model, X_test, y_test, device, n_events=n_events_field, save_path=p
    )
    paths["velocity_field"] = p

    print("[fm_plots] Generating ode_trajectories.png ...")
    p = str(save_dir / "ode_trajectories.png")
    plot_ode_trajectories(model, X_test, device, n_events=n_events_traj, save_path=p)
    paths["ode_trajectories"] = p

    return paths
