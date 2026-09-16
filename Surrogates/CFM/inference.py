"""
ODE integration and API for the CFM surrogate.

No phi re-wrapping between ODE steps.
nTrueInt is sampled from the training pileup profile.
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch

# Import from common
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.features import (
    eng_conditioner,
    reconstruct_jet,
    reconstruct_jet_torch,
    CONDITIONER_NAMES,
    TARGET_NAMES,
    _FLAVOUR_MAP,
    _N_FLAVOURS,
)
from common.dataset import sample_nTrueInt

from model import VelocityMLP


# ODE solvers  (t: 0 -> 1)
def euler_integrate(model, x0, c, n_steps=100):
    """Euler integrator"""
    dt = 1.0 / n_steps
    x = x0
    for i in range(n_steps):
        t = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
        v = model(x, t, c)
        x = x + dt * v
    return x


def midpoint_integrate(model, x0, c, n_steps=100):
    """Midpoint (RK2) integrator.  2 NFE per step"""
    dt = 1.0 / n_steps
    x = x0
    for i in range(n_steps):
        t = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
        t_mid = torch.full(
            (x.shape[0], 1), (i + 0.5) * dt, device=x.device, dtype=x.dtype
        )
        v1 = model(x, t, c)
        x_mid = x + 0.5 * dt * v1
        v2 = model(x_mid, t_mid, c)
        x = x + dt * v2
    return x


# Checkpoint loading
def load_checkpoint(path, device="cuda"):
    """Load model + stats from a training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = VelocityMLP(
        cond_dim=config.get("cond_dim", 26),
        target_dim=config.get("target_dim", 4),
        width=config.get("width", 512),
        depth=config.get("depth", 6),
        backbone=config.get("backbone", "plain_mlp"),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Attach noise prior to model for convenience
    model.noise_loc = checkpoint["noise_loc"].to(device)
    model.noise_scale = checkpoint["noise_scale"].to(device)

    stats = checkpoint["stats"]
    return model, stats, checkpoint


# Exact log-likelihood via ODE divergence integration
@torch.enable_grad()
def compute_log_likelihood(model, x1, c, noise_loc, noise_scale, n_steps=100):
    """
    Exact conditional log-likelihood via reverse ODE + divergence.

    At d=4, compute exact trace via 4 VJPs per step.

    Integration scheme:
    This uses a pseudo-midpoint Euler scheme:
    velocity is evaluated at the interval's midpoint TIME (t = (i+0.5)*dt) but at the current state x, then x is advanced by a single Euler step.
    This is 1 NFE per step and first-order in x.

    ``midpoint_integrate`` (used for sampling) is true RK2:
    evaluate at t, half-step to a midpoint state, re-evaluate, advance.
    That is 2 NFE per step and second-order.

    Parameters:
    model : VelocityMLP
    x1 : (B, 4)  target data points
    c : (B, 26)  conditioners
    noise_loc, noise_scale : (4,) noise prior parameters
    n_steps : int

    Returns:
    log_p : (B,) log-likelihoods
    """
    device = x1.device
    B, D = x1.shape
    dt = 1.0 / n_steps

    x = x1.clone().detach().requires_grad_(True)
    log_det = torch.zeros(B, device=device)

    # Reverse ODE: integrate from t=1 back to t=0
    for i in reversed(range(n_steps)):
        t_val = (i + 0.5) * dt  # midpoint of the interval
        t = torch.full((B, 1), t_val, device=device)

        v = model(x, t, c)

        div = torch.zeros(B, device=device)
        for j in range(D):
            e_j = torch.zeros_like(x)
            e_j[:, j] = 1.0
            v_j_grad = torch.autograd.grad(
                v,
                x,
                grad_outputs=e_j,
                create_graph=False,
                retain_graph=(j < D - 1),
            )[0]
            div = div + v_j_grad[:, j]

        with torch.no_grad():
            x = x - dt * v
            log_det = log_det - dt * div
            x = x.detach().requires_grad_(True)

    # Base distribution log-density
    var = noise_scale**2
    log_p0 = (
        -0.5 * D * math.log(2 * math.pi)
        - 0.5 * torch.log(var).sum()
        - 0.5 * ((x.detach() - noise_loc) ** 2 / var).sum(dim=1)
    )

    return log_p0 + log_det


# Joint HDR-rank calibration
def compute_hdr_logp(
    model,
    x1,
    c,
    noise_loc,
    noise_scale,
    n_samples=20,
    n_steps=50,
    solver="midpoint",
):
    """
    Log-densities for exact joint HDR-rank coverage, flow-matching version.

    Returns
    -------
    log_p_truth : (B,)
    log_p_samples : (B, M)
    """
    B = x1.shape[0]
    integrate_fn = midpoint_integrate if solver == "midpoint" else euler_integrate

    log_p_truth = compute_log_likelihood(
        model,
        x1,
        c,
        noise_loc,
        noise_scale,
        n_steps=n_steps,
    ).detach()

    log_p_samples = torch.empty(B, n_samples, device=x1.device)
    for m in range(n_samples):
        with torch.no_grad():
            x0 = noise_loc + noise_scale * torch.randn(B, x1.shape[1], device=x1.device)
            xs = integrate_fn(model, x0, c, n_steps)
        log_p_samples[:, m] = compute_log_likelihood(
            model,
            xs,
            c,
            noise_loc,
            noise_scale,
            n_steps=n_steps,
        ).detach()

    return log_p_truth, log_p_samples


# Conditioner construction from raw kinematics
def _build_conditioner_from_kinematics(
    parton_pt,
    parton_eta,
    parton_phi,
    pdg_id,
    nTrueInt,
    stats,
    device="cuda",
):
    """Build the 26-dim conditioner tensor from raw parton kinematics."""
    import polars as pl

    df = pl.DataFrame(
        {
            "parton_pt": np.asarray(parton_pt, dtype=np.float64),
            "parton_eta": np.asarray(parton_eta, dtype=np.float64),
            "parton_phi": np.asarray(parton_phi, dtype=np.float64),
            "parton_pdgId": np.asarray(pdg_id, dtype=np.int32),
            "pileup_nTrueInt": np.asarray(nTrueInt, dtype=np.float64),
        }
    )

    X, _, _ = eng_conditioner(df, source_prefix="parton", stats=stats)
    return torch.from_numpy(X).float().to(device)


# API:  generate jets
@torch.no_grad()
def generate_jets(
    model,
    parton_pt,
    parton_eta,
    parton_phi,
    parton_mass,
    pdg_id,
    stats,
    n_steps=100,
    solver="midpoint",
    nTrueInt_override=None,
    device="cuda",
    n_samples=1,
):
    """
    Public API:  (pt, eta, phi, mass, pdgId) -> (pt, eta, phi, mass).

    Parameters
    parton_pt, parton_eta, parton_phi, parton_mass : array-like (N,)
    pdg_id : array-like (N,) PDG IDs
    stats : dict from training checkpoint
    n_steps : ODE integration steps
    solver : 'euler' or 'midpoint'
    nTrueInt_override : float or None.  If None, sampled from training profile.
    n_samples : number of samples per event.  If > 1, returns stacked results.

    Returns
    dict with keys: jet_pt, jet_eta, jet_phi, jet_mass
    """
    parton_pt = np.asarray(parton_pt, dtype=np.float64)
    parton_eta = np.asarray(parton_eta, dtype=np.float64)
    parton_phi = np.asarray(parton_phi, dtype=np.float64)
    pdg_id = np.asarray(pdg_id, dtype=np.int32)
    B = len(parton_pt)

    # Sample pileup
    if nTrueInt_override is not None:
        nTrueInt = np.full(B, nTrueInt_override, dtype=np.float64)
    else:
        nTrueInt = sample_nTrueInt(stats, B)

    # Build conditioner
    c = _build_conditioner_from_kinematics(
        parton_pt,
        parton_eta,
        parton_phi,
        pdg_id,
        nTrueInt,
        stats,
        device,
    )

    # Choose solver
    integrate_fn = midpoint_integrate if solver == "midpoint" else euler_integrate

    all_residuals = []
    for _ in range(n_samples):
        # Sample x0 from noise prior
        x0 = model.noise_loc + model.noise_scale * torch.randn(B, 4, device=device)
        # Integrate ODE
        x1 = integrate_fn(model, x0, c, n_steps)
        all_residuals.append(x1.cpu().numpy())

    if n_samples == 1:
        residuals = all_residuals[0]
    else:
        residuals = np.stack(all_residuals, axis=1)  # (N, M, 4)

    # Reconstruct jets
    if n_samples == 1:
        jets = reconstruct_jet(parton_pt, parton_eta, parton_phi, residuals)
    else:
        # Reconstruct each sample
        jets = {k: [] for k in ["jet_pt", "jet_eta", "jet_phi", "jet_mass"]}
        for s in range(n_samples):
            r = reconstruct_jet(parton_pt, parton_eta, parton_phi, residuals[:, s, :])
            for k in jets:
                jets[k].append(r[k])
        jets = {k: np.stack(v, axis=1) for k, v in jets.items()}  # (N, M)

    return jets


# Batch generate from a DataFrame
@torch.no_grad()
def batch_generate(
    model,
    df,
    stats,
    n_steps=100,
    solver="midpoint",
    device="cuda",
    batch_size=4096,
    n_samples=1,
):
    """Batch inference on a polars DataFrame."""
    import polars as pl

    parton_pt = np.asarray(df["parton_pt"], dtype=np.float64)
    parton_eta = np.asarray(df["parton_eta"], dtype=np.float64)
    parton_phi = np.asarray(df["parton_phi"], dtype=np.float64)
    parton_mass = np.asarray(df["parton_mass"], dtype=np.float64)
    pdg_id = np.asarray(df["parton_pdgId"], dtype=np.int32)

    N = len(parton_pt)
    all_jets = {k: [] for k in ["jet_pt", "jet_eta", "jet_phi", "jet_mass"]}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        jets = generate_jets(
            model,
            parton_pt[start:end],
            parton_eta[start:end],
            parton_phi[start:end],
            parton_mass[start:end],
            pdg_id[start:end],
            stats,
            n_steps=n_steps,
            solver=solver,
            device=device,
            n_samples=n_samples,
        )
        for k in all_jets:
            all_jets[k].append(jets[k])

    return {k: np.concatenate(v, axis=0) for k, v in all_jets.items()}
