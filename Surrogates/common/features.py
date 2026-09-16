"""
Feature engineering for parton -> jet surrogates (V2).

Shared by both the CFM and MDN surrogates to guarantee identical conditioner/target definitions across models.
"""

import numpy as np

# Physical constants & safe-clipping bounds
_LHC_SQRT_S = 13_000.0  # GeV
_LOG_PT_RESP_MIN = np.log(1e-3)
_LOG_PT_RESP_MAX = np.log(_LHC_SQRT_S)
_LOG_MASS_FRAC_MIN = np.log(1e-4)
_LOG_MASS_FRAC_MAX = np.log(_LHC_SQRT_S)
_DELTA_ETA_MAX = 10.0  # generous; real bound is ~0.2 from the cone
_DELTA_PHI_MAX = np.pi
_DR_CONE = 0.2  # matching-cone radius


def residual_cone_radius(residuals):
    """
    Compute the cone radius of the residuals.
    """
    res = np.asarray(residuals)
    return np.sqrt(res[..., 1] ** 2 + res[..., 2] ** 2)


def residual_out_of_cone(residuals):
    """Boolean mask of residual rows the disk clamp would rescale."""
    return residual_cone_radius(residuals) > _DR_CONE

# Flavour mapping:  abs(pdgId)  ->  one-hot position
#   d=1, u=2, s=3, c=4, b=5, g=21

_FLAVOUR_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 21: 5}
_N_FLAVOURS = 6
FLAVOUR_NAMES = ["d", "u", "s", "c", "b", "g"]

STAGE_CONFIGS = {
    "parton2reco": dict(
        source_prefix="parton", target_prefix="jet",
        flavour_col=None, dr_cone_fixed=0.2,
    ),
    "parton2genjet": dict(
        source_prefix="parton", target_prefix="genjet",
        flavour_col=None, dr_cone_fixed=None,
    ),
    "genjet2reco": dict(
        source_prefix="genjet", target_prefix="jet",
        flavour_col="genjet_partonFlavour", dr_cone_fixed=None,
    ),
}

# Conditioner feature names (26 dims)
CONDITIONER_NAMES = (
    ["log_pt", "abs_eta", "log_cosh_eta", "nTrueInt_std"]
    + [f"flav_{f}" for f in FLAVOUR_NAMES]             # 6
    + [f"{'sin' if i % 2 == 0 else 'cos'}_{(i // 2) + 1}_phi"
       for i in range(16)]                               # 16
)
assert len(CONDITIONER_NAMES) == 26

# Target feature names (4 dims)
TARGET_NAMES = ["log_pT_resp", "delta_eta", "delta_phi", "log_mass_frac"]


# Conditioner  (26 dims)
# Continuous conditioner features that get z-score standardised.
# (The one-hot and Fourier blocks are already O(1)-scaled and are left as is.)
_STANDARDIZED_FEATURES = ["log_pt", "abs_eta", "log_cosh_eta", "nTrueInt"]

# Conditioner feature blocks (ablation dimension): which groups of features
# eng_conditioner builds.  The fine-grained split is:
#   "kinematics"  (3 dims) -- log_pt, abs_eta, log_cosh_eta.  MANDATORY.
#   "pileup"      (1 dim)  -- nTrueInt (z-scored).  Separated from kinematics
#                             so the pileup -> mass information channel can be
#                             ablated independently.
#   "flavour"     (6 dims) -- one-hot PDG-id.
#   "fourier_phi" (16 dims) -- sin/cos(k*phi), k=1..8.
#
_CONDITIONER_BLOCK_NAMES = {
    "kinematics": ["log_pt", "abs_eta", "log_cosh_eta"],
    "pileup": ["nTrueInt_std"],
    "flavour": [f"flav_{f}" for f in FLAVOUR_NAMES],
    "fourier_phi": [f"{'sin' if i % 2 == 0 else 'cos'}_{(i // 2) + 1}_phi"
                     for i in range(16)],
}
ALL_CONDITIONER_BLOCKS = ["kinematics", "pileup", "flavour", "fourier_phi"]

# Legacy alias: "continuous" -> ["kinematics", "pileup"]
_CONTINUOUS_EXPANSION = ["kinematics", "pileup"]


def _expand_blocks(blocks: list[str]) -> list[str]:
    """Expand the legacy ``"continuous"`` alias and deduplicate."""
    out = []
    for b in blocks:
        if b == "continuous":
            for sub in _CONTINUOUS_EXPANSION:
                if sub not in out:
                    out.append(sub)
        else:
            if b not in out:
                out.append(b)
    return out


def eng_conditioner(
    df,
    source_prefix: str = "parton",
    stats: dict | None = None,
    flavour_col: str | None = None,
    blocks: list[str] | None = None,
):
    """
    Build the conditioner from a polars/pandas-like DataFrame.

    Parameters
    ----------
    df : DataFrame with columns ``{source_prefix}_pt``, ``_eta``, ``_phi``,
         a flavour column, and ``pileup_nTrueInt``.
    source_prefix : column prefix (``"parton"`` for single-stage,
                    ``"genjet"`` for the second stage).
    stats : optional dict with ``"{feature}_mean"`` / ``"{feature}_std"``
        for ``feature`` in ``log_pt, abs_eta, log_cosh_eta, nTrueInt``.
        Any entries not present are computed from ``df`` (only safe if
        ``df`` is the train split) and written into the returned stats
        dict. A copy is made; the caller's dict is never mutated.
    flavour_col : column holding the flavour PDG id, absolute-valued before
        lookup in ``_FLAVOUR_MAP``. Defaults to ``f"{source_prefix}_pdgId"``.
        Needed because the genjet2reco stage's flavour lives in
        ``genjet_partonFlavour``, not ``genjet_pdgId`` (which doesn't exist).
    blocks : subset of ``ALL_CONDITIONER_BLOCKS`` (``["kinematics",
        "pileup", "flavour", "fourier_phi"]``), default all four (the
        production conditioner, 26 dims).  The legacy name ``"continuous"``
        is accepted and expands to ``["kinematics", "pileup"]``.
        ``"kinematics"`` cannot be dropped.

    Returns
    -------
    X : np.ndarray, shape (N, D), float32, D depending on ``blocks``
        (3 kinematics-only .. 26 all four) — the continuous features are
        z-score standardised, one-hot/Fourier blocks are left as is.
    names : list[str]
    stats : dict with the mean/std used for each standardised feature.
    """
    blocks = _expand_blocks(list(blocks)) if blocks is not None else list(ALL_CONDITIONER_BLOCKS)
    unknown = set(blocks) - set(ALL_CONDITIONER_BLOCKS)
    if unknown:
        raise ValueError(
            f"eng_conditioner: unknown conditioner block(s) {sorted(unknown)}; "
            f"expected subset of {ALL_CONDITIONER_BLOCKS}"
        )
    if "kinematics" not in blocks:
        raise ValueError(
            "eng_conditioner: 'kinematics' (log_pt/abs_eta/log_cosh_eta) "
            "cannot be excluded -- a block set without it conditions on "
            "nothing anchored to the hard-process kinematics, which isn't a "
            "meaningful ablation of this project's design."
        )

    flavour_col = flavour_col or f"{source_prefix}_pdgId"
    pt = np.asarray(df[f"{source_prefix}_pt"], dtype=np.float64)
    eta = np.asarray(df[f"{source_prefix}_eta"], dtype=np.float64)
    nTrueInt = np.asarray(df["pileup_nTrueInt"], dtype=np.float64)

    stats = dict(stats) if stats is not None else {}

    def _standardize(x, key):
        mean = stats.get(f"{key}_mean")
        std = stats.get(f"{key}_std")
        if mean is None or std is None:
            mean = float(np.mean(x))
            std = float(np.std(x))
        stats[f"{key}_mean"] = mean
        stats[f"{key}_std"] = std
        return (x - mean) / (std + 1e-8)

    # kinematics features (raw, pre-standardisation)
    log_pt = np.log(pt)
    abs_eta = np.abs(eta)
    log_cosh_eta = np.log(np.cosh(eta))

    log_pt_s = _standardize(log_pt, "log_pt")
    abs_eta_s = _standardize(abs_eta, "abs_eta")
    log_cosh_eta_s = _standardize(log_cosh_eta, "log_cosh_eta")

    parts = [log_pt_s, abs_eta_s, log_cosh_eta_s]
    names = list(_CONDITIONER_BLOCK_NAMES["kinematics"])

    if "pileup" in blocks:
        nTrueInt_s = _standardize(nTrueInt, "nTrueInt")
        parts.append(nTrueInt_s)
        names += _CONDITIONER_BLOCK_NAMES["pileup"]

    if "flavour" in blocks:
        pdg = np.abs(np.asarray(df[flavour_col], dtype=np.int32))
        # flavour one-hot (6 classes: d, u, s, c, b, g)
        onehot = np.zeros((len(pdg), _N_FLAVOURS), dtype=np.float32)
        for abs_id, col_idx in _FLAVOUR_MAP.items():
            onehot[pdg == abs_id, col_idx] = 1.0


        n_hit = onehot.sum(axis=1)
        if not np.all(n_hit > 0):
            bad = np.asarray(df[flavour_col])[n_hit == 0]
            raise ValueError(
                f"eng_conditioner: {len(bad):,} rows in '{flavour_col}' do "
                f"not map to a known flavour (expected abs(id) in "
                f"{sorted(_FLAVOUR_MAP)}), e.g. {np.unique(bad)[:10].tolist()}. "
                "Apply the common event-set filter (flavour != 0) before "
                "this call, or the one-hot block is silently all-zero for "
                "these rows."
            )
        parts.append(onehot)
        names += _CONDITIONER_BLOCK_NAMES["flavour"]

    if "fourier_phi" in blocks:
        phi = np.asarray(df[f"{source_prefix}_phi"], dtype=np.float64)
        # Fourier features of phi: sin(k*phi), cos(k*phi), k=1..8
        fourier = np.empty((len(phi), 16), dtype=np.float64)
        for k in range(1, 9):
            fourier[:, 2 * (k - 1)] = np.sin(k * phi)
            fourier[:, 2 * (k - 1) + 1] = np.cos(k * phi)
        parts.append(fourier)
        names += _CONDITIONER_BLOCK_NAMES["fourier_phi"]

    X = np.column_stack(parts).astype(np.float32)

    return X, names, stats


# Targets  (4 dims)
def eng_targets(
    df,
    source_prefix: str = "parton",
    target_prefix: str = "jet",
):
    """
    Build 4-dim target residuals.

    y0 = log(jet_pt / parton_pt)
    y1 = sign(parton_eta) * (jet_eta - parton_eta) # parity-folded
    y2 = wrap(jet_phi - parton_phi)
    y3 = log(jet_mass / jet_pt)  mass fraction

    Parameters
    ----------
    df : DataFrame with columns  ``{target_prefix}_pt``, ``_eta``, ``_phi``,
         ``_mass``, ``{source_prefix}_pt``, ``_eta``, ``_phi``.

    Returns
    -------
    y : np.ndarray, shape (N, 4), float32
    names : list[str]
    """
    src_pt = np.asarray(df[f"{source_prefix}_pt"], dtype=np.float64)
    src_eta = np.asarray(df[f"{source_prefix}_eta"], dtype=np.float64)
    src_phi = np.asarray(df[f"{source_prefix}_phi"], dtype=np.float64)

    tgt_pt = np.asarray(df[f"{target_prefix}_pt"], dtype=np.float64)
    tgt_eta = np.asarray(df[f"{target_prefix}_eta"], dtype=np.float64)
    tgt_phi = np.asarray(df[f"{target_prefix}_phi"], dtype=np.float64)
    tgt_mass = np.asarray(df[f"{target_prefix}_mass"], dtype=np.float64)

    # y0: log pT response
    y0 = np.log(np.clip(tgt_pt / src_pt, 1e-6, None))
    y0 = np.clip(y0, _LOG_PT_RESP_MIN, _LOG_PT_RESP_MAX)

    # y1: parity-folded delta-eta
    delta_eta = tgt_eta - src_eta
    eta_sign = np.sign(src_eta)
    # Handle eta=0 edge case: sign(0)=0 -> treat as +1
    eta_sign[eta_sign == 0] = 1.0
    y1 = eta_sign * delta_eta

    # y2: wrapped delta-phi
    dphi = tgt_phi - src_phi
    y2 = np.arctan2(np.sin(dphi), np.cos(dphi))

    # y3: log mass fraction  (m / pt_jet)
    mass_safe = np.clip(tgt_mass, 1e-4, None)
    y3 = np.log(mass_safe / tgt_pt)
    y3 = np.clip(y3, _LOG_MASS_FRAC_MIN, _LOG_MASS_FRAC_MAX)

    y = np.column_stack([y0, y1, y2, y3]).astype(np.float32)
    return y, TARGET_NAMES


# Reconstruction:  (parton kinematics, residuals) -> jet 4-vector
def reconstruct_jet(parton_pt, parton_eta, parton_phi, residuals, dr_cone: float = _DR_CONE):
    """
    Reconstruct jet kinematics from parton 4-vector and model residuals.

    Parameters
    ----------
    parton_pt, parton_eta, parton_phi : array-like, shape (N,)
    residuals : array-like, shape (N, 4)
        [log_pT_resp, delta_eta_folded, delta_phi, log_mass_frac]
    dr_cone : disk-clamp radius.

    Returns
    -------
    dict with keys: jet_pt, jet_eta, jet_phi, jet_mass, jet_px, jet_py,
                    jet_pz, jet_energy
    """
    parton_pt = np.asarray(parton_pt, dtype=np.float64)
    parton_eta = np.asarray(parton_eta, dtype=np.float64)
    parton_phi = np.asarray(parton_phi, dtype=np.float64)
    res = np.asarray(residuals, dtype=np.float64)

    y0, y1, y2, y3 = res[:, 0], res[:, 1], res[:, 2], res[:, 3]

    r = np.sqrt(y1**2 + y2**2)
    scale = np.minimum(1.0, dr_cone / np.clip(r, 1e-12, None))
    y1 = y1 * scale
    y2 = y2 * scale

    # jet pT
    jet_pt = parton_pt * np.exp(y0)
    jet_pt = np.clip(jet_pt, 0.0, _LHC_SQRT_S)

    # jet eta — undo parity folding
    eta_sign = np.sign(parton_eta)
    eta_sign[eta_sign == 0] = 1.0
    jet_eta = parton_eta + eta_sign * y1

    # jet phi — wrap to [-pi, pi]
    jet_phi = np.arctan2(
        np.sin(parton_phi + y2),
        np.cos(parton_phi + y2),
    )

    # jet mass
    jet_mass = jet_pt * np.exp(y3)
    jet_mass = np.clip(jet_mass, 0.0, _LHC_SQRT_S)

    # 4-vector components
    jet_px = jet_pt * np.cos(jet_phi)
    jet_py = jet_pt * np.sin(jet_phi)
    jet_pz = jet_pt * np.sinh(np.clip(jet_eta, -10.0, 10.0))
    jet_energy = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2 + jet_mass**2)

    return {
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "jet_phi": jet_phi,
        "jet_mass": jet_mass,
        "jet_px": jet_px,
        "jet_py": jet_py,
        "jet_pz": jet_pz,
        "jet_energy": jet_energy,
    }


def reconstruct_jet_torch(parton_pt, parton_eta, parton_phi, residuals, dr_cone: float = _DR_CONE):
    """
    Torch version of reconstruct_jet for GPU inference.

    Parameters
    ----------
    parton_pt, parton_eta, parton_phi : Tensor, shape (N,)
    residuals : Tensor, shape (N, 4)
    dr_cone : disk-clamp radius -- see `reconstruct_jet`..

    Returns
    -------
    dict of Tensors
    """
    import torch

    y0 = residuals[:, 0]
    y1 = residuals[:, 1]
    y2 = residuals[:, 2]
    y3 = residuals[:, 3]

    # Enforce the matching-cone bound.
    r = torch.sqrt(y1**2 + y2**2)
    scale = torch.clamp(dr_cone / r.clamp(min=1e-12), max=1.0)
    y1 = y1 * scale
    y2 = y2 * scale

    jet_pt = parton_pt * torch.exp(y0)
    jet_pt = jet_pt.clamp(0.0, _LHC_SQRT_S)

    eta_sign = torch.sign(parton_eta)
    eta_sign[eta_sign == 0] = 1.0
    jet_eta = parton_eta + eta_sign * y1

    jet_phi = torch.atan2(
        torch.sin(parton_phi + y2),
        torch.cos(parton_phi + y2),
    )

    jet_mass = jet_pt * torch.exp(y3)
    jet_mass = jet_mass.clamp(0.0, _LHC_SQRT_S)

    jet_px = jet_pt * torch.cos(jet_phi)
    jet_py = jet_pt * torch.sin(jet_phi)
    jet_pz = jet_pt * torch.sinh(jet_eta.clamp(-10.0, 10.0))
    jet_energy = torch.sqrt(jet_px**2 + jet_py**2 + jet_pz**2 + jet_mass**2)

    return {
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "jet_phi": jet_phi,
        "jet_mass": jet_mass,
        "jet_px": jet_px,
        "jet_py": jet_py,
        "jet_pz": jet_pz,
        "jet_energy": jet_energy,
    }
