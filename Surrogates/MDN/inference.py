"""
Inference and evaluation for the MDN surrogate.

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
    CONDITIONER_NAMES,
    TARGET_NAMES,
)
from common.dataset import sample_nTrueInt

from model import Multivariate_MDN, mixture_log_prob


# Checkpoint loading
def load_checkpoint(path, device="cuda"):
    """Load model + stats from a training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = Multivariate_MDN(
        input_dim=config.get("input_dim", 26),
        n_components=config.get("n_components", 32),
        depth=config.get("depth", 6),
        hidden_dim=config.get("hidden_dim", 512),
        dropout=config.get("dropout", 0.0),
        backbone=config.get("backbone", "plain_mlp"),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = checkpoint["stats"]
    return model, stats, checkpoint


# Exact conditional log-likelihood (closed form)
@torch.no_grad()
def compute_log_likelihood(model, X, y, pi_temperature=1.0):
    """
    Exact conditional log-likelihood from the MDN mixture.

    Parameters
    ----------
    model : Multivariate_MDN
    X : (B, 26) conditioner
    y : (B, 4) targets

    Returns
    -------
    log_p : (B,) per-event log-likelihoods
    """
    pi, mu, L = model(X, pi_temperature=pi_temperature)
    return mixture_log_prob(pi, mu, L, y)


# PIT values (per-dimension, via sampling)
@torch.no_grad()
def compute_pit(model, X, y):
    """
    EXACT per-dimension PIT via the mixture's marginal CDF — no sampling noise.

    Parameters
    ----------
    model : Multivariate_MDN
    X : (B, 26)
    y : (B, 4)

    Returns
    -------
    pit : (B, 4) PIT values in [0, 1]
    """
    pi, mu, L = model(X)
    # diag(L L^T) — per-component marginal variances
    var = L.pow(2).sum(dim=-1).clamp(min=1e-12)  # (B, K, D)
    sd = var.sqrt()

    # Standard normal CDF, broadcast over components
    z = (y.unsqueeze(1) - mu) / sd  # (B, K, D)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))  # (B, K, D)
    return (pi.unsqueeze(-1) * cdf).sum(dim=1)  # (B, D)


@torch.no_grad()
def compute_hdr_logp(model, X, y, n_samples=50):
    """
    Log-densities needed for exact joint HDR-rank coverage.

    Returns
    -------
    log_p_truth : (B,) mixture log-density at the observed target
    log_p_samples : (B, M) mixture log-density at M draws from the same
                    conditional

    Feed both to `common.metrics.compute_hdr_coverage`. Exact and cheap here
    because the MDN density is closed form — one forward pass, reused for
    both the truth and the draws.
    """
    pi, mu, L = model(X)
    samples = model.sample(X, n_samples=n_samples)  # (B, M, D)
    log_p_truth = mixture_log_prob(pi, mu, L, y)  # (B,)
    log_p_samples = mixture_log_prob(pi, mu, L, samples)  # (B, M)
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


# API: generate jets
@torch.no_grad()
def generate_jets(
    model,
    parton_pt,
    parton_eta,
    parton_phi,
    parton_mass,
    pdg_id,
    stats,
    nTrueInt_override=None,
    device="cuda",
    n_samples=1,
):
    """
    API:  (pt, eta, phi, mass, pdgId) → (pt, eta, phi, mass).

    Parameters
    ----------
    model : Multivariate_MDN
    parton_pt, parton_eta, parton_phi, parton_mass : array-like (N,)
    pdg_id : array-like (N,)
    stats : dict from training checkpoint
    nTrueInt_override : float or None
    n_samples : samples per event

    Returns
    -------
    dict with keys: jet_pt, jet_eta, jet_phi, jet_mass
    """
    parton_pt = np.asarray(parton_pt, dtype=np.float64)
    parton_eta = np.asarray(parton_eta, dtype=np.float64)
    parton_phi = np.asarray(parton_phi, dtype=np.float64)
    pdg_id = np.asarray(pdg_id, dtype=np.int32)
    B = len(parton_pt)

    # Sample pileup (never mean-impute — see sample_nTrueInt docstring)
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

    # Sample from MDN
    samples = model.sample(c, n_samples=n_samples)  # (B, D) or (B, M, D)

    if n_samples == 1:
        residuals = samples.cpu().numpy()
        jets = reconstruct_jet(parton_pt, parton_eta, parton_phi, residuals)
    else:
        residuals = samples.cpu().numpy()  # (B, M, D)
        jets = {k: [] for k in ["jet_pt", "jet_eta", "jet_phi", "jet_mass"]}
        for s in range(n_samples):
            r = reconstruct_jet(parton_pt, parton_eta, parton_phi, residuals[:, s, :])
            for k in jets:
                jets[k].append(r[k])
        jets = {k: np.stack(v, axis=1) for k, v in jets.items()}

    return jets


# Batch generate from DataFrame
@torch.no_grad()
def batch_generate(
    model,
    df,
    stats,
    device="cuda",
    batch_size=4096,
    n_samples=1,
):
    """Batch inference on a polars DataFrame."""
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
            device=device,
            n_samples=n_samples,
        )
        for k in all_jets:
            all_jets[k].append(jets[k])

    return {k: np.concatenate(v, axis=0) for k, v in all_jets.items()}
