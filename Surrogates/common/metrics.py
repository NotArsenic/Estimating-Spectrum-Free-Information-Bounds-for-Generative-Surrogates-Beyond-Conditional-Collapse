"""
Evaluation metrics for V2 surrogates.

Priority ordering:
  1. Exact conditional log-likelihood on held-out data.
  2. Per-flavour breakdown of every metric.
  3. Conditional-independence surrogate test.
  4. PIT histograms + coverage.
  5. Physics closure checks.
"""

import numpy as np
from scipy import stats as sp_stats
from scipy.special import rel_entr

from .features import (
    FLAVOUR_NAMES,
    _FLAVOUR_MAP,
    TARGET_NAMES,
    _DR_CONE,
)
from .settings import FLOOR_ESTIMATOR


# 1-D marginal metrics  (secondary)
def compute_kl_divergence(
    truth: np.ndarray,
    sampled: np.ndarray,
    n_bins: int = 200,
    epsilon: float = 1e-10,
    report_floor: bool = True,
    floor_seed: int = 0,
) -> dict:
    """
    Per-dimension KL divergence  KL(truth || sampled)  via histogram binning.

    Returns dict mapping target name → KL value (plus floor keys).
    """
    results = {}
    rng = np.random.default_rng(floor_seed)
    for dim, name in enumerate(TARGET_NAMES):
        t = truth[:, dim]
        s = sampled[:, dim]
        edges = np.linspace(t.min(), t.max(), n_bins + 1)

        p, _ = np.histogram(t, bins=edges, density=True)
        q, _ = np.histogram(s, bins=edges, density=True)
        p = p + epsilon
        q = q + epsilon
        p = p / p.sum()
        q = q / q.sum()

        results[f"kl_{name}"] = float(np.sum(rel_entr(p, q)))

        if report_floor:
            perm = rng.permutation(len(t))
            half = len(t) // 2
            p1, _ = np.histogram(t[perm[:half]], bins=edges, density=True)
            p2, _ = np.histogram(t[perm[half:]], bins=edges, density=True)
            p1 = p1 + epsilon
            p2 = p2 + epsilon
            p1 = p1 / p1.sum()
            p2 = p2 / p2.sum()
            results[f"kl_floor_{name}"] = float(np.sum(rel_entr(p1, p2)))
    return results


def compute_wasserstein(
    truth: np.ndarray,
    sampled: np.ndarray,
) -> dict:
    """Per-dimension 1-Wasserstein distance."""
    results = {}
    for dim, name in enumerate(TARGET_NAMES):
        w = sp_stats.wasserstein_distance(truth[:, dim], sampled[:, dim])
        results[f"w1_{name}"] = float(w)
    return results


# Tail Fidelity Index (TFI)
def compute_tfi(
    truth: np.ndarray,
    sampled: np.ndarray,
    sigma_threshold: float = 2.0,
) -> dict:
    """
    Tail Fidelity Index: fraction of events beyond ``sigma_threshold`` standard deviations, compared between truth and model.

    For delta_eta and delta_phi, the threshold is capped at 2.3σ
    Using 2.0σ by default.
    """
    results = {}
    for dim, name in enumerate(TARGET_NAMES):
        t = truth[:, dim]
        s = sampled[:, dim]
        mu, sig = t.mean(), t.std()
        if sig < 1e-10:
            results[f"tfi_{name}"] = 0.0
            continue
        frac_truth = np.mean(np.abs(t - mu) > sigma_threshold * sig)
        frac_sampled = np.mean(np.abs(s - mu) > sigma_threshold * sig)
        if frac_truth > 1e-10:
            results[f"tfi_{name}"] = float(abs(frac_sampled - frac_truth) / frac_truth)
        else:
            results[f"tfi_{name}"] = 0.0
    return results


# CRPS  (Continuous Ranked Probability Score)
def crps_per_event(
    truth: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """
    Per-event, per-dimension CRPS via order statistics.

    This is the core :func:`compute_crps` averages over. It exists separately so the per-event array can be dumped once and every downstream study (differential-in-pT skill, per-flavour slices, spectrum reweighting) becomes a filter or a weighted mean over it.

    Uses the *fair/unbiased* estimator (denominator ``m(m-1)``); the matching unconditional floor is :func:`crps_unconditional_floor`.

    Parameters
    ----------
    truth : (N, D)
    samples : (N, M, D) - M samples per event, D target dims

    Returns
    -------
    (N, D) array of per-event CRPS values.
    """
    N, M, D = samples.shape
    n_dims = min(D, len(TARGET_NAMES))
    out = np.empty((N, n_dims), dtype=np.float64)
    for dim in range(n_dims):
        y = truth[:, dim]  # (N,)
        s = np.sort(samples[:, :, dim], axis=1)  # (N, M)
        # |y - s_i| averaged
        term1 = np.mean(np.abs(s - y[:, None]), axis=1)
        # |s_i - s_j| averaged (via order-statistic formula)
        ranks = np.arange(1, M + 1)
        term2 = np.sum(s * (2 * ranks - M - 1), axis=1) / (M * (M - 1) + 1e-10)
        out[:, dim] = term1 - term2
    return out


def compute_crps(
    truth: np.ndarray,
    samples: np.ndarray,
) -> dict:
    """
    Exact CRPS via order statistics — the mean of :func:`crps_per_event`.

    Parameters
    ----------
    truth : (N, D)
    samples : (N, M, D) - M samples per event, D target dims

    Returns
    -------
    dict with per-dim CRPS values.
    """
    per_event = crps_per_event(truth, samples)
    return {
        f"crps_{TARGET_NAMES[dim]}": float(np.mean(per_event[:, dim]))
        for dim in range(per_event.shape[1])
    }


def crps_unconditional_floor(
    truth: np.ndarray,
    n_samples: int = 50,
    n_pairs: int | None = None,
) -> dict:
    """
    CRPS floor achievable by a marginal-only (unconditional) sampler.
    """
    results = {}
    N = len(truth)

    w = (2 * np.arange(1, N + 1) - N - 1).astype(np.float64)
    for dim, name in enumerate(TARGET_NAMES):
        y = np.sort(np.asarray(truth[:, dim], dtype=np.float64))
        results[f"crps_floor_{name}"] = float((y @ w) / (N * N))
    results["crps_floor_n_samples"] = int(n_samples)
    results["crps_floor_estimator"] = FLOOR_ESTIMATOR
    results["crps_floor_n_events"] = int(N)
    if n_pairs is not None:
        results["crps_floor_n_pairs_ignored"] = int(n_pairs)
    return results


# 3. Conditional-independence surrogate test
def crps_floor_within(y: np.ndarray, n_pairs: int | None = None, seed=None):
    """
    Exact Gini floor ``0.5 * E|Y - Y'|`` on one subset (for example one pT bin).

    ``n_pairs`` and ``seed`` are accepted for old call sites and ignored.
    Returns ``(D,)``; NaN where the subset has fewer than two rows.
    """
    y = np.asarray(y)
    n, D = y.shape
    out = np.full(D, np.nan)
    if n < 2:
        return out
    w = (2 * np.arange(1, n + 1) - n - 1).astype(np.float64)
    for d in range(D):
        out[d] = float((np.sort(np.asarray(y[:, d], dtype=np.float64)) @ w) / (n * n))
    return out


def conditional_independence_test(
    truth: np.ndarray,
    sampled: np.ndarray,
    source_pt: np.ndarray,
    source_eta: np.ndarray,
    source_phi: np.ndarray,
    target_dim: int = 3,
    n_bins: int = 200,
    target_prefix: str = "jet",
    dr_cone: float | None = None,
) -> dict:
    """
    Conditional-collapse detector, measured in *reconstructed observable*
    space rather than residual space.
    """
    from .features import reconstruct_jet, _DR_CONE

    rng = np.random.default_rng(42)
    name = TARGET_NAMES[target_dim]
    cone = _DR_CONE if dr_cone is None else dr_cone

    truth_shuffled = truth.copy()
    truth_shuffled[:, target_dim] = rng.permutation(truth[:, target_dim])

    arms = {
        "truth": truth,
        "shuffled": truth_shuffled,
        "model": sampled,
    }
    jets = {
        label: reconstruct_jet(source_pt, source_eta, source_phi, res, dr_cone=cone)
        for label, res in arms.items()
    }

    results: dict = {}
    for obs_suffix in ("pt", "mass"):

        obs = f"{target_prefix}_{obs_suffix}"
        stds = {}
        for label, jet in jets.items():
            v = jet[f"jet_{obs_suffix}"]
            stds[label] = float(np.std(v))
            results[f"ci_test_{name}_{obs}_std_{label}"] = stds[label]

        inflation = stds["shuffled"] - stds["truth"]
        results[f"ci_test_{name}_{obs}_decorr_inflation"] = inflation
        if stds["truth"] > 1e-12:
            results[f"ci_test_{name}_{obs}_decorr_inflation_frac"] = (
                inflation / stds["truth"]
            )

        if inflation > 0.01 * stds["truth"]:
            results[f"ci_test_{name}_{obs}_collapse"] = (
                stds["model"] - stds["truth"]
            ) / inflation
        else:
            results[f"ci_test_{name}_{obs}_collapse"] = float("nan")

    lo = min(truth[:, target_dim].min(), sampled[:, target_dim].min())
    hi = max(truth[:, target_dim].max(), sampled[:, target_dim].max())
    edges = np.linspace(lo, hi, n_bins + 1)
    eps = 1e-10
    p_truth, _ = np.histogram(truth[:, target_dim], bins=edges, density=True)
    p_model, _ = np.histogram(sampled[:, target_dim], bins=edges, density=True)
    p_truth = (p_truth + eps) / (p_truth + eps).sum()
    p_model = (p_model + eps) / (p_model + eps).sum()
    results[f"ci_test_{name}_kl_model_vs_truth"] = float(
        np.sum(rel_entr(p_truth, p_model))
    )

    return results


# 4. PIT (Probability Integral Transform) + Coverage
def compute_pit_ks(
    pit_values: np.ndarray,
) -> dict:
    """
    KS-test of PIT values against Uniform(0,1).

    Parameters
    ----------
    pit_values : (N, D) PIT values per dimension

    Returns
    -------
    dict with KS statistic and p-value per dimension.
    """
    results = {}
    D = pit_values.shape[1]
    for dim in range(min(D, len(TARGET_NAMES))):
        name = TARGET_NAMES[dim]
        ks_stat, ks_pval = sp_stats.kstest(pit_values[:, dim], "uniform")
        results[f"pit_ks_{name}"] = float(ks_stat)
        results[f"pit_ks_pval_{name}"] = float(ks_pval)
    return results


def compute_pit_sampled(
    truth: np.ndarray,
    samples: np.ndarray,
    seed: int = 7,
) -> np.ndarray:
    """
    Model-agnostic per-dimension PIT from an ensemble, via *randomized* rank.

    Parameters
    ----------
    truth : (N, D)
    samples : (N, M, D)

    Returns
    -------
    pit : (N, D) values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    N, M, D = samples.shape
    below = (samples < truth[:, None, :]).sum(axis=1)  # (N, D)
    u = rng.random((N, D))
    return (below + u) / (M + 1)


def hdr_rank_per_event(
    log_p_truth: np.ndarray,
    log_p_samples: np.ndarray,
    seed: int = 11,
) -> np.ndarray:
    """
    Per-event randomized HDR rank — the core :func:`compute_hdr_coverage`
    reduces to coverage levels and a KS statistic.

    Parameters
    ----------
    log_p_truth : (N,) log density at the observed target
    log_p_samples : (N, M) log density at M draws from the same conditional

    Returns
    -------
    (N,) ranks in [0, 1], Uniform(0,1) under calibration.
    """
    rng = np.random.default_rng(seed)
    N, M = log_p_samples.shape
    below = (log_p_samples < log_p_truth[:, None]).sum(axis=1)  # (N,)
    return (below + rng.random(N)) / (M + 1)


def compute_hdr_coverage(
    log_p_truth: np.ndarray,
    log_p_samples: np.ndarray,
    levels: np.ndarray | None = None,
    seed: int = 11,
) -> dict:
    """
    Exact joint (D-dimensional) coverage via highest-density-region rank.

    Parameters
    ----------
    log_p_truth : (N,) log density at the observed target
    log_p_samples : (N, M) log density at M draws from the same conditional
    levels : nominal levels (default 0.05 ... 0.95 in 19 steps)

    Returns
    -------
    dict with per-level coverage, plus the KS statistic of the rank against Uniform(0,1) and the maximum deviation from y=x.
    """
    if levels is None:
        levels = np.linspace(0.05, 0.95, 19)

    rank = hdr_rank_per_event(log_p_truth, log_p_samples, seed=seed)

    results = {}
    devs = []
    for cl in levels:
        cov = float(np.mean(rank <= cl))
        results[f"hdr_coverage_{int(round(cl * 100))}"] = cov
        devs.append(abs(cov - float(cl)))

    ks_stat, ks_pval = sp_stats.kstest(rank, "uniform")
    results["hdr_rank_ks"] = float(ks_stat)
    results["hdr_rank_ks_pval"] = float(ks_pval)
    results["hdr_max_deviation"] = float(max(devs))
    results["hdr_n_events"] = int(len(rank))

    results["hdr_n_samples"] = int(log_p_samples.shape[1])
    return results


def _finite_M_coverage_reference(
    M: int,
    levels: list[float],
    n_sim: int = 50_000,
    seed: int = 12345,
) -> dict:
    """
    Monte Carlo finite-sample coverage ideal for the empirical-quantile intervals ``compute_coverage`` builds from M samples.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((n_sim, M))
    truth = rng.standard_normal(n_sim)
    ref = {}
    for level in levels:
        alpha = (1 - level) / 2
        lo = np.quantile(samples, alpha, axis=1)
        hi = np.quantile(samples, 1 - alpha, axis=1)
        ref[level] = float(np.mean((truth >= lo) & (truth <= hi)))
    return ref


def compute_coverage(
    truth: np.ndarray,
    samples: np.ndarray,
    levels: list[float] | None = None,
) -> dict:
    """
    Empirical *marginal* (per-dimension) coverage at specified levels.

    Parameters
    ----------
    truth : (N, D)
    samples : (N, M, D)
    levels : nominal coverage levels to check (default: [0.5, 0.8, 0.9, 0.95])

    Returns
    -------
    dict with actual coverage at each nominal level, per dimension, plus the finite-M ideal per level.
    """
    if levels is None:
        levels = [0.5, 0.8, 0.9, 0.95]
    results = {}
    N, M, D = samples.shape

    for dim in range(min(D, len(TARGET_NAMES))):
        name = TARGET_NAMES[dim]
        y = truth[:, dim]
        s = samples[:, :, dim]

        for level in levels:
            alpha = (1 - level) / 2
            lo = np.quantile(s, alpha, axis=1)
            hi = np.quantile(s, 1 - alpha, axis=1)
            covered = np.mean((y >= lo) & (y <= hi))
            results[f"coverage_{name}_{int(level*100)}"] = float(covered)

    finite_M_ref = _finite_M_coverage_reference(M, levels)
    for level in levels:
        results[f"coverage_ideal_M{M}_{int(level*100)}"] = finite_M_ref[level]

    return results


# 2. Per-flavour breakdown
def per_flavour_metrics(
    truth: np.ndarray,
    sampled: np.ndarray,
    pdg_ids: np.ndarray,
    metric_fn,
    **metric_kwargs,
) -> dict:
    """
    Run a metric function separately for each flavour class.

    Parameters
    ----------
    truth, sampled : (N, D) arrays
    pdg_ids : (N,) absolute pdgId values
    metric_fn : callable(truth, sampled, **kwargs) -> dict

    Returns
    -------
    dict with keys prefixed by flavour name.
    """
    results = {}
    abs_pdg = np.abs(pdg_ids)

    for abs_id, col_idx in _FLAVOUR_MAP.items():
        fname = FLAVOUR_NAMES[col_idx]
        mask = abs_pdg == abs_id
        n = mask.sum()
        if n < 100:
            continue
        sub_truth = truth[mask]
        sub_sampled = sampled[mask]
        sub_results = metric_fn(sub_truth, sub_sampled, **metric_kwargs)
        for k, v in sub_results.items():
            results[f"{fname}/{k}"] = v
        results[f"{fname}/n_events"] = int(n)

    return results


def per_flavour_grouped_metrics(
    labels: list[str],
    pdg_ids: np.ndarray,
) -> dict:
    """Return flavour-grouped label membership: {flav: [indices]}."""
    abs_pdg = np.abs(pdg_ids)
    groups = {}
    for flav_name, abs_id in zip(
        FLAVOUR_NAMES,
        [1, 2, 3, 4, 5, 21],
    ):
        mask = abs_pdg == abs_id
        if mask.sum() > 0:
            groups[flav_name] = np.where(mask)[0]
    # 'uds' group
    uds_mask = np.isin(abs_pdg, [1, 2, 3])
    if uds_mask.sum() > 0:
        groups["uds"] = np.where(uds_mask)[0]
    return groups


# 5. Physics closure checks
def physics_closure_checks(
    truth_targets: np.ndarray,
    sampled_targets: np.ndarray,
    parton_pt: np.ndarray,
    parton_eta: np.ndarray,
    parton_phi: np.ndarray,
    pdg_ids: np.ndarray,
    pt_hat: np.ndarray | None = None,
) -> dict:
    """
    Physics-motivated closure checks with known reference values.

    Performed on a fiducial sub-sample:
    ``parton_pt ∈ [90, 110]``, ``|parton_eta| < 2.4``, and (if available) ``pt_hat ∈ [80, 170]``.

    """
    results = {}
    abs_pdg = np.abs(pdg_ids)

    # Fiducial cut
    mask = (parton_pt >= 90) & (parton_pt <= 110) & (np.abs(parton_eta) < 2.4)
    if pt_hat is not None:
        mask = mask & (pt_hat >= 80) & (pt_hat <= 170)

    if mask.sum() < 100:
        results["closure_n_fiducial"] = int(mask.sum())
        return results

    fid_truth = truth_targets[mask]
    fid_sampled = sampled_targets[mask]
    fid_pdg = abs_pdg[mask]
    results["closure_n_fiducial"] = int(mask.sum())

    # Group masks
    is_gluon = fid_pdg == 21
    is_uds = np.isin(fid_pdg, [1, 2, 3])
    is_b = fid_pdg == 5

    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 10 else float("nan")

    # <log pT_resp> differences
    for label, data in [("truth", fid_truth), ("model", fid_sampled)]:
        g_mean = safe_mean(data[is_gluon, 0])
        uds_mean = safe_mean(data[is_uds, 0])
        b_mean = safe_mean(data[is_b, 0])

        results[f"closure_{label}_logpt_resp_g_minus_uds"] = g_mean - uds_mean
        results[f"closure_{label}_logpt_resp_b_minus_uds"] = b_mean - uds_mean

    # Mass fraction ratio: <exp(y3)>_gluon / <exp(y3)>_uds
    for label, data in [("truth", fid_truth), ("model", fid_sampled)]:
        g_mf = safe_mean(np.exp(data[is_gluon, 3]))
        uds_mf = safe_mean(np.exp(data[is_uds, 3]))
        if uds_mf > 1e-10:
            results[f"closure_{label}_mass_frac_ratio_g_uds"] = g_mf / uds_mf
        else:
            results[f"closure_{label}_mass_frac_ratio_g_uds"] = float("nan")

    return results


def phi_profile(
    sampled_phi: np.ndarray,
    sampled_pt_response: np.ndarray,
    n_bins: int = 36,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the φ-binned jet energy response profile.

    Parameters
    ----------
    sampled_phi : reconstructed jet φ
    sampled_pt_response : log(pT_jet / pT_parton) from the model

    Returns
    -------
    bin_centers : (n_bins,)
    mean_response : (n_bins,) mean log-response per φ bin
    std_response : (n_bins,) standard deviation per φ bin
    counts : (n_bins,) number of events per φ bin
    """
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(sampled_phi, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    means = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        mask = bin_idx == i
        n = mask.sum()
        counts[i] = n
        if n > 0:
            vals = sampled_pt_response[mask]
            means[i] = np.mean(vals)
            stds[i] = np.std(vals, ddof=1) if n > 1 else 0.0

    return centers, means, stds, counts


# Reporting helpers
def sanitize_for_json(metrics: dict) -> dict:
    """
    Return a copy with non-finite floats replaced by None.
    """
    out = {}
    for k, v in metrics.items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


def print_ci_test_summary(metrics: dict, target_prefix: str = "jet") -> None:
    """Print the conditional-collapse table from an `evaluate_all` result."""
    if metrics.get("ci_test_skipped_no_parton_kinematics"):
        print("  CI test: SKIPPED (no parton kinematics supplied)")
        return

    print("  Conditional-collapse test (0 = perfect, 1 = fully decorrelated):")
    for dim_name in ("log_mass_frac", "log_pT_resp"):
        for obs in (f"{target_prefix}_pt", f"{target_prefix}_mass"):
            collapse = metrics.get(f"ci_test_{dim_name}_{obs}_collapse")
            if collapse is None:
                continue
            infl = metrics.get(f"ci_test_{dim_name}_{obs}_decorr_inflation_frac", 0.0)
            if abs(infl) < 0.01 or not np.isfinite(collapse):
                # e.g. shuffling log_mass_frac cannot move jet_pt at all, so
                # there is no signal to measure and no score to report.
                print(
                    f"    shuffle {dim_name:14s} -> std({obs:8s}) "
                    f"n/a  (decorrelation moves this by only {infl:+.1%})"
                )
            else:
                print(
                    f"    shuffle {dim_name:14s} -> std({obs:8s}) "
                    f"collapse={collapse:+.4f}  (decorr inflates width by {infl:+.1%})"
                )


# Aggregate all metrics
def evaluate_all(
    truth: np.ndarray,
    sampled_single: np.ndarray,
    sampled_multi: np.ndarray | None = None,
    pdg_ids: np.ndarray | None = None,
    parton_pt: np.ndarray | None = None,
    parton_eta: np.ndarray | None = None,
    parton_phi: np.ndarray | None = None,
    pt_hat: np.ndarray | None = None,
    n_samples_for_crps: int = 50,
    target_prefix: str = "jet",
    dr_cone: float | None = None,
) -> dict:
    """
    Run the full evaluation suite.

    Returns
    -------
    dict of all metrics.
    """
    results = {}

    # Marginal metrics
    results.update(compute_kl_divergence(truth, sampled_single))
    results.update(compute_wasserstein(truth, sampled_single))
    results.update(compute_tfi(truth, sampled_single, sigma_threshold=2.0))

    # Conditional independence surrogate
    if parton_pt is not None and parton_eta is not None and parton_phi is not None:
        for dim in (3, 0):
            results.update(
                conditional_independence_test(
                    truth,
                    sampled_single,
                    parton_pt,
                    parton_eta,
                    parton_phi,
                    target_dim=dim,
                    target_prefix=target_prefix,
                    dr_cone=dr_cone,
                )
            )
    else:
        results["ci_test_skipped_no_parton_kinematics"] = True

    # CRPS
    if sampled_multi is not None:

        n_m = len(sampled_multi)
        truth_m = truth[:n_m]

        results.update(compute_crps(truth_m, sampled_multi))
        results.update(
            crps_unconditional_floor(truth_m, n_samples=sampled_multi.shape[1])
        )
        results.update(compute_coverage(truth_m, sampled_multi))
        results["n_multi_sample_events"] = int(n_m)

        for name in TARGET_NAMES:
            crps = results.get(f"crps_{name}")
            floor = results.get(f"crps_floor_{name}")
            if crps is not None and floor is not None and floor > 1e-12:
                results[f"crps_ratio_{name}"] = float(crps / floor)

        # Model-agnostic ensemble PIT (comparable across MDN and flow)
        pit_sampled = compute_pit_sampled(truth_m, sampled_multi)
        for k, v in compute_pit_ks(pit_sampled).items():
            results[f"sampled_{k}"] = v

    # Per-flavour breakdown
    if pdg_ids is not None:
        results.update(
            per_flavour_metrics(
                truth,
                sampled_single,
                pdg_ids,
                compute_kl_divergence,
            )
        )
        results.update(
            per_flavour_metrics(
                truth,
                sampled_single,
                pdg_ids,
                compute_wasserstein,
            )
        )

    # Physics closure
    if parton_pt is not None and parton_eta is not None and parton_phi is not None:
        results.update(
            physics_closure_checks(
                truth,
                sampled_single,
                parton_pt,
                parton_eta,
                parton_phi,
                (
                    pdg_ids
                    if pdg_ids is not None
                    else np.zeros(len(truth), dtype=np.int32)
                ),
                pt_hat=pt_hat,
            )
        )

    return results
