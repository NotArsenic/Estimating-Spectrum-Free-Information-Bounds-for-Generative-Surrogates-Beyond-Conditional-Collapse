"""
CLI entry point for CFM training and evaluation.

Usage:
    python main.py --data ../../Data/dataset/PJ_dataset_qcd_flat_15to7000.parquet --seeds 42 --epochs 100
"""

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch

# Ensure common/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import train
from inference import (
    load_checkpoint,
    compute_log_likelihood,
    compute_hdr_logp,
    midpoint_integrate,
    euler_integrate,
)
from model import VelocityMLP

from common.dataset import (
    prepare_datasets,
    DEFAULT_PT_MIN,
    DEFAULT_PT_MAX,
)
from common.features import TARGET_NAMES, STAGE_CONFIGS
from common import paths as _paths
from common.metrics import (
    evaluate_all,
    compute_pit_sampled,
    compute_pit_ks,
    compute_hdr_coverage,
    print_ci_test_summary,
    sanitize_for_json,
)
from common.plots import plot_all
from common.metrics import compute_kl_divergence
from common.dump import write_dump
from common.provenance import checkpoint_sha256, event_set_sha256, write_json
from common.seeding import DUMP, EVAL_HDR, EVAL_MULTI, EVAL_SINGLE, seed_torch
from common.settings import CRPS_EVAL_EVENTS, DATA_PATH
from plots_fm import plot_fm_internals


def _assert_stage(stats: dict, requested_stage: str) -> None:
    ckpt_stage = stats.get("stage", "parton2reco")
    if ckpt_stage != requested_stage:
        raise ValueError(
            f"Checkpoint was trained under stage={ckpt_stage!r} but "
            f"stage={requested_stage!r} was requested. These are different "
            "conditioner/target definitions -- evaluating one under the "
            "other's labels is not an error you want to discover downstream."
        )


def generate_dump(
    checkpoint_path: str,
    data_path: str,
    save_dir: str,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    m: int = 200,
    subset_size: int = 100_000,
    n_steps: int = 100,
    solver: str = "midpoint",
    n_hdr_samples: int = 20,
    n_hdr_steps: int = 50,
    n_jac_t: int = 5,
    limit: int | None = None,
    device: str = "cuda",
    pt_min: float | None = DEFAULT_PT_MIN,
    pt_max: float | None = DEFAULT_PT_MAX,
    stage: str = "parton2reco",
):
    """Write the per-event evaluation dump (HANDOFF_V3 B3)."""
    model, stats, _ = load_checkpoint(checkpoint_path, device)
    _assert_stage(stats, stage)

    ckpt_pt_min = stats.get("pt_min", "__absent__")
    if ckpt_pt_min != "__absent__":
        pt_min, pt_max = ckpt_pt_min, stats.get("pt_max")

    datasets, _, raw_splits = prepare_datasets(
        data_path,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        pt_min=pt_min,
        pt_max=pt_max,
        stage=stage,
        conditioner_blocks=stats.get("conditioner_blocks"),
    )
    X_test = datasets["test"].X.numpy()
    y_test = datasets["test"].y.numpy()
    batch_size = 4096
    integrate_fn = midpoint_integrate if solver == "midpoint" else euler_integrate

    def sample_fn(Xb, m):
        B = len(Xb)
        c = torch.from_numpy(Xb).float().to(device)
        out = np.empty((B, m, 4), dtype=np.float32)
        with torch.no_grad():
            for j in range(m):
                x0 = model.noise_loc + model.noise_scale * torch.randn(
                    B, 4, device=device
                )
                out[:, j, :] = integrate_fn(model, x0, c, n_steps).cpu().numpy()
        return out

    def ll_fn(Xs, ys):
        out = []
        for s in range(0, len(Xs), batch_size):
            e = min(s + batch_size, len(Xs))
            out.append(
                compute_log_likelihood(
                    model,
                    torch.from_numpy(ys[s:e]).float().to(device),
                    torch.from_numpy(Xs[s:e]).float().to(device),
                    model.noise_loc,
                    model.noise_scale,
                    n_steps=n_steps,
                )
                .detach()
                .cpu()
                .numpy()
            )
        return np.concatenate(out)

    def hdr_fn(Xs, ys):
        t_all, s_all = [], []
        hdr_bs = 1024
        for s in range(0, len(Xs), hdr_bs):
            e = min(s + hdr_bs, len(Xs))
            t, sm = compute_hdr_logp(
                model,
                torch.from_numpy(ys[s:e]).float().to(device),
                torch.from_numpy(Xs[s:e]).float().to(device),
                model.noise_loc,
                model.noise_scale,
                n_samples=n_hdr_samples,
                n_steps=n_hdr_steps,
                solver=solver,
            )
            t_all.append(t.cpu().numpy())
            s_all.append(sm.cpu().numpy())
        return np.concatenate(t_all), np.concatenate(s_all)

    def internals_fn(Xs):
        # Mean |d v / d x_t| Jacobian on a grid of ODE times
        t_grid = np.linspace(0.0, 1.0, n_jac_t, dtype=np.float32)
        jac = np.zeros((len(Xs), n_jac_t, 4, 4), dtype=np.float32)
        jac_bs = 512
        for s in range(0, len(Xs), jac_bs):
            e = min(s + jac_bs, len(Xs))
            c = torch.from_numpy(Xs[s:e]).float().to(device)
            B = e - s
            for ti, tv in enumerate(t_grid):
                x = (
                    model.noise_loc
                    + model.noise_scale * torch.randn(B, 4, device=device)
                ).requires_grad_(True)
                t = torch.full((B, 1), float(tv), device=device)
                v = model(x, t, c)
                for d in range(4):
                    g = torch.autograd.grad(v[:, d].sum(), x, retain_graph=(d < 3))[0]
                    jac[s:e, ti, d, :] = g.detach().cpu().numpy()
        return {"jacobian": jac, "t_grid": t_grid}

    seed_torch(seed + DUMP)
    return write_dump(
        os.path.join(save_dir, "dump"),
        model_name="FlowMatching_SingleStage_V2",
        seed=seed,
        df_test=raw_splits["test"],
        X_test=X_test,
        y_test=y_test,
        sample_fn=sample_fn,
        m=m,
        subset_size=subset_size,
        batch_size=batch_size,
        ll_fn=ll_fn,
        hdr_fn=hdr_fn,
        internals_fn=internals_fn,
        limit=limit,
        extra_meta={
            "pt_min": pt_min,
            "pt_max": pt_max,
            "n_steps": n_steps,
            "solver": solver,
            "n_hdr_samples": n_hdr_samples,
            "n_hdr_steps": n_hdr_steps,
            "stage": stage,
            "conditioner_blocks": stats.get("conditioner_blocks"),
            "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        },
    )


def evaluate_model(
    checkpoint_path: str,
    data_path: str,
    save_dir: str,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    n_steps: int = 100,
    solver: str = "midpoint",
    n_eval_samples: int = 50,
    n_hdr_events: int = 5_000,
    n_hdr_samples: int = 20,
    n_hdr_steps: int = 50,
    device: str = "cuda",
    pt_min: float | None = DEFAULT_PT_MIN,
    pt_max: float | None = DEFAULT_PT_MAX,
    stage: str = "parton2reco",
    require_common_event_set: bool | None = None,
    metrics_filename: str = "metrics.json",
):
    """Run full evaluation on test set after training.

    ``metrics_filename`` (HANDOFF_E2_FIX F1): override so a re-evaluation
    under a different ``require_common_event_set`` doesn't silently
    overwrite an existing metrics.json for the same checkpoint -- e.g. the
    parton2reco common-event-set rerun needed to fix the E2 third-term
    contamination must land next to, not on top of, the original.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating FM V2 checkpoint  (stage={stage})")
    print(f"{'='*60}")

    model, stats, checkpoint = load_checkpoint(checkpoint_path, device)
    _assert_stage(stats, stage)

    # Re-load data for the test split.
    ckpt_pt_min = stats.get("pt_min", "__absent__")
    ckpt_pt_max = stats.get("pt_max", "__absent__")
    if ckpt_pt_min == "__absent__":
        print(
            "[eval] WARNING: checkpoint predates pT-window recording; "
            f"using requested window [{pt_min}, {pt_max})."
        )
    else:
        if (ckpt_pt_min, ckpt_pt_max) != (pt_min, pt_max):
            print(
                f"[eval] pT window from checkpoint "
                f"[{ckpt_pt_min}, {ckpt_pt_max}) overrides requested "
                f"[{pt_min}, {pt_max})."
            )
        pt_min, pt_max = ckpt_pt_min, ckpt_pt_max

    datasets, eval_stats, raw_splits = prepare_datasets(
        data_path,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        pt_min=pt_min,
        pt_max=pt_max,
        stage=stage,
        require_common_event_set=require_common_event_set,
        conditioner_blocks=stats.get("conditioner_blocks"),
    )

    test_ds = datasets["test"]
    df_test = raw_splits["test"]

    X_test = test_ds.X.numpy()
    y_test = test_ds.y.numpy()

    N_test = len(y_test)
    print(f"[eval] Test set: {N_test:,} events")

    source_prefix = stats.get("source_prefix", "parton")
    target_prefix = stats.get("target_prefix", "jet")
    dr_cone = stats.get("dr_cone", 0.2)

    integrate_fn = midpoint_integrate if solver == "midpoint" else euler_integrate
    batch_size = 4096

    # 1. Generate single samples
    seed_torch(seed + EVAL_SINGLE)
    print("[eval] Generating single samples...")
    all_samples = []
    for start in range(0, N_test, batch_size):
        end = min(start + batch_size, N_test)
        c = torch.from_numpy(X_test[start:end]).float().to(device)
        x0 = model.noise_loc + model.noise_scale * torch.randn(
            end - start, 4, device=device
        )
        with torch.no_grad():
            x1 = integrate_fn(model, x0, c, n_steps)
        all_samples.append(x1.cpu().numpy())

    sampled = np.concatenate(all_samples, axis=0)

    # 2. Generate multiple samples (for CRPS / coverage)
    seed_torch(seed + EVAL_MULTI)
    n_multi = min(N_test, CRPS_EVAL_EVENTS)
    print(f"[eval] Generating {n_eval_samples} samples per event (n={n_multi:,})...")
    multi_samples = []
    for s in range(n_eval_samples):
        batch_samples = []
        for start in range(0, n_multi, batch_size):
            end = min(start + batch_size, n_multi)
            c = torch.from_numpy(X_test[start:end]).float().to(device)
            x0 = model.noise_loc + model.noise_scale * torch.randn(
                end - start, 4, device=device
            )
            with torch.no_grad():
                x1 = integrate_fn(model, x0, c, n_steps)
            batch_samples.append(x1.cpu().numpy())
        multi_samples.append(np.concatenate(batch_samples, axis=0))

    sampled_multi = np.stack(multi_samples, axis=1)  # (n_multi, n_eval_samples, 4)

    # 3. Compute log-likelihood (tiered subset, Richardson-extrapolated)
    n_ll = min(N_test, 100_000)
    print(
        f"[eval] Computing log-likelihood at n_steps={n_steps} and {2 * n_steps} (n={n_ll:,})..."
    )
    ll_values_n, ll_values_2n = [], []
    for start in range(0, n_ll, batch_size):
        end = min(start + batch_size, n_ll)
        x1_batch = torch.from_numpy(y_test[start:end]).float().to(device)
        c_batch = torch.from_numpy(X_test[start:end]).float().to(device)
        ll_n = compute_log_likelihood(
            model,
            x1_batch,
            c_batch,
            model.noise_loc,
            model.noise_scale,
            n_steps=n_steps,
        )
        ll_2n = compute_log_likelihood(
            model,
            x1_batch,
            c_batch,
            model.noise_loc,
            model.noise_scale,
            n_steps=2 * n_steps,
        )
        ll_values_n.append(ll_n.detach().cpu().numpy())
        ll_values_2n.append(ll_2n.detach().cpu().numpy())

    ll_all_n = np.concatenate(ll_values_n, axis=0)
    ll_all_2n = np.concatenate(ll_values_2n, axis=0)
    mean_ll_n = float(np.mean(ll_all_n))
    mean_ll_2n = float(np.mean(ll_all_2n))
    mean_ll_richardson = mean_ll_2n + (mean_ll_2n - mean_ll_n)
    mean_ll = mean_ll_richardson
    print(
        f"[eval] Mean log-likelihood: n_steps={n_steps} -> {mean_ll_n:.5f}, "
        f"{2*n_steps} -> {mean_ll_2n:.5f}, Richardson -> {mean_ll_richardson:.5f}"
    )

    # 3b. Joint HDR-rank coverage (subset: each density eval is an ODE)
    seed_torch(seed + EVAL_HDR)
    n_hdr = min(N_test, n_hdr_events)
    print(
        f"[eval] Computing joint HDR-rank coverage (n={n_hdr:,}, M={n_hdr_samples})..."
    )
    lp_truth, lp_samp = [], []
    hdr_bs = 1024
    for start in range(0, n_hdr, hdr_bs):
        end = min(start + hdr_bs, n_hdr)
        x1_batch = torch.from_numpy(y_test[start:end]).float().to(device)
        c_batch = torch.from_numpy(X_test[start:end]).float().to(device)
        t, s = compute_hdr_logp(
            model,
            x1_batch,
            c_batch,
            model.noise_loc,
            model.noise_scale,
            n_samples=n_hdr_samples,
            n_steps=n_hdr_steps,
            solver=solver,
        )
        lp_truth.append(t.cpu().numpy())
        lp_samp.append(s.cpu().numpy())

    hdr_metrics = compute_hdr_coverage(
        np.concatenate(lp_truth, axis=0),
        np.concatenate(lp_samp, axis=0),
    )
    hdr_metrics["hdr_n_ode_samples"] = n_hdr_samples
    hdr_metrics["hdr_n_ode_steps"] = n_hdr_steps

    # 4. Collect source (this stage's conditioner) kinematics
    flavour_col = stats.get("flavour_col", f"{source_prefix}_pdgId")
    pdg_ids = np.abs(np.asarray(df_test[flavour_col], dtype=np.int32))
    parton_pt = np.asarray(df_test[f"{source_prefix}_pt"], dtype=np.float64)
    parton_eta = np.asarray(df_test[f"{source_prefix}_eta"], dtype=np.float64)
    parton_phi = np.asarray(df_test[f"{source_prefix}_phi"], dtype=np.float64)
    common_parton_pt = np.asarray(df_test["parton_pt"], dtype=np.float64)

    parton_energy = (
        np.asarray(df_test[f"{source_prefix}_energy"], dtype=np.float64)
        if f"{source_prefix}_energy" in df_test.columns
        else None
    )
    pt_hat = (
        np.asarray(df_test["pt_hat"], dtype=np.float64)
        if "pt_hat" in df_test.columns
        else None
    )

    # 5. Run metrics suite
    print("[eval] Computing metrics...")
    metrics = evaluate_all(
        truth=y_test,
        sampled_single=sampled,
        sampled_multi=sampled_multi,
        pdg_ids=pdg_ids,
        parton_pt=parton_pt,
        parton_eta=parton_eta,
        parton_phi=parton_phi,
        pt_hat=pt_hat,
        target_prefix=target_prefix,
        dr_cone=dr_cone,
    )
    metrics["mean_log_likelihood"] = mean_ll
    metrics["mean_log_likelihood_at_n_steps"] = mean_ll_n
    metrics["mean_log_likelihood_at_2n_steps"] = mean_ll_2n
    metrics["mean_log_likelihood_richardson"] = mean_ll_richardson
    metrics["n_test"] = N_test
    metrics["n_ll"] = n_ll
    metrics["config_hash"] = stats.get("config_hash")
    metrics.update(hdr_metrics)
    metrics["hdr_scheme_note"] = (
        "HDR samples drawn with 2nd-order RK2 (midpoint_integrate); "
        "HDR densities (and mean_log_likelihood) scored with a 1st-order "
        "pseudo-midpoint Euler scheme at the same n_steps -- schemes differ, "
        "accuracies are not matched at equal n_steps."
    )

    # KL divergences (for kl_summary figure)
    kl_divergences = compute_kl_divergence(y_test, sampled)
    metrics.update(kl_divergences)

    # Rebuild HDR rank array for the joint coverage plot.
    _lp_truth_arr = np.concatenate(lp_truth, axis=0)
    _lp_samp_arr = np.concatenate(lp_samp, axis=0)
    _rng_rank = np.random.default_rng(11)
    _below = (_lp_samp_arr < _lp_truth_arr[:, None]).sum(axis=1)
    hdr_rank_arr = (_below + _rng_rank.random(len(_below))) / (
        _lp_samp_arr.shape[1] + 1
    )

    # Inference speed benchmark
    import time as _time

    _n_warmup, _n_iters = 10, 30
    speed_results = {}
    for _ns in (10, 50, 100):
        for _bs in (256, 1024, 4096):
            _bs_eff = min(_bs, N_test)
            _c = torch.from_numpy(X_test[:_bs_eff]).float().to(device)

            def _run():
                _x0 = model.noise_loc + model.noise_scale * torch.randn(
                    _bs_eff, 4, device=device
                )
                return integrate_fn(model, _x0, _c, _ns)

            with torch.no_grad():
                for _ in range(_n_warmup):
                    _run()
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                _t0 = _time.perf_counter()
                for _ in range(_n_iters):
                    _run()
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                _t1 = _time.perf_counter()
            speed_results[f"FM {_ns} steps bs={_bs_eff}"] = (
                (_t1 - _t0) * 1e6 / (_n_iters * _bs_eff)
            )

    # Ensemble PIT
    pit_all = compute_pit_sampled(y_test[: len(sampled_multi)], sampled_multi)

    # 6. Save metrics
    metrics_out = sanitize_for_json(metrics)
    metrics_out.update(
        {
            "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
            "seed": seed,
            "stage": stage,
            "conditioner_blocks": stats.get("conditioner_blocks"),
            "require_common_event_set": bool(eval_stats["require_common_event_set"]),
            "test_ids_sha256": event_set_sha256(
                raw_splits["test"]["global_match_id"].to_numpy()
            ),
            "solver": solver,
            "n_steps": n_steps,
        }
    )
    metrics_path = os.path.join(save_dir, metrics_filename)
    write_json(metrics_path, metrics_out)
    print(f"[eval] Metrics saved to {_paths.rel(metrics_path)}")

    # 7. Generate plots
    print("[eval] Generating plots...")
    fig_dir = os.path.join(save_dir, "figures")

    # Compute Physics Observables (Dijet, Casimir)
    from common.features import reconstruct_jet
    from common.plots import plot_dijet_mass, plot_casimir_scaling
    import polars as pl

    true_jet = reconstruct_jet(
        parton_pt, parton_eta, parton_phi, y_test, dr_cone=dr_cone
    )
    sampled_jet = reconstruct_jet(
        parton_pt, parton_eta, parton_phi, sampled, dr_cone=dr_cone
    )

    # Build integer event keys and find 2-jet events via numpy sort.
    event_ids = (
        df_test["run"].cast(pl.Utf8)
        + "_"
        + df_test["luminosityBlock"].cast(pl.Utf8)
        + "_"
        + df_test["event"].cast(pl.Utf8)
    ).to_numpy()

    _, inv, counts = np.unique(event_ids, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    is_dijet = counts == 2
    dijet_mask = is_dijet[inv]
    dijet_order = order[dijet_mask[order]]
    # dijet_order is sorted by group
    idxA = dijet_order[0::2]
    idxB = dijet_order[1::2]

    def _dijet_mass(jet, iA, iB):
        px = jet["jet_px"][iA] + jet["jet_px"][iB]
        py = jet["jet_py"][iA] + jet["jet_py"][iB]
        pz = jet["jet_pz"][iA] + jet["jet_pz"][iB]
        E = jet["jet_energy"][iA] + jet["jet_energy"][iB]
        m2 = E**2 - px**2 - py**2 - pz**2
        return np.sqrt(np.maximum(m2, 0.0))

    if len(idxA) > 0:
        m_true = _dijet_mass(true_jet, idxA, idxB)
        m_model = _dijet_mass(sampled_jet, idxA, idxB)
        os.makedirs(fig_dir, exist_ok=True)
        plot_dijet_mass(
            m_true, m_model, save_path=os.path.join(fig_dir, "dijet_mass.png")
        )
        print(f"[eval] Dijet mass: {len(idxA):,} 2-jet events")

    # Casimir scaling
    abs_pdg = np.abs(pdg_ids)
    uds_mask = np.isin(abs_pdg, [1, 2, 3])
    g_mask = abs_pdg == 21
    pt_bins = [50, 60, 70, 80, 95, 110, 130, 170]
    bin_centers = np.array(
        [0.5 * (pt_bins[i] + pt_bins[i + 1]) for i in range(len(pt_bins) - 1)]
    )

    def _compute_ratios(jet_dict, mask_uds, mask_g, pt):
        m_over_pt_sq = (jet_dict["jet_mass"] / jet_dict["jet_pt"]) ** 2
        ratios = []
        for i in range(len(pt_bins) - 1):
            lo, hi = pt_bins[i], pt_bins[i + 1]
            pt_mask = (pt >= lo) & (pt < hi)
            uds_vals = m_over_pt_sq[mask_uds & pt_mask]
            g_vals = m_over_pt_sq[mask_g & pt_mask]
            if len(uds_vals) > 100 and len(g_vals) > 100:
                ratios.append(np.mean(g_vals) / np.mean(uds_vals))
            else:
                ratios.append(np.nan)
        return np.array(ratios)

    truth_ratios = _compute_ratios(true_jet, uds_mask, g_mask, common_parton_pt)
    model_ratios = _compute_ratios(sampled_jet, uds_mask, g_mask, common_parton_pt)

    nTrueInt = np.asarray(df_test["pileup_nTrueInt"], dtype=np.float64)
    low_pu_mask = nTrueInt < np.median(nTrueInt)

    truth_ratios_lowPU = _compute_ratios(
        true_jet, uds_mask & low_pu_mask, g_mask & low_pu_mask, common_parton_pt
    )
    model_ratios_lowPU = _compute_ratios(
        sampled_jet, uds_mask & low_pu_mask, g_mask & low_pu_mask, common_parton_pt
    )

    plot_casimir_scaling(
        bin_centers,
        truth_ratios,
        model_ratios,
        truth_ratios_lowPU,
        model_ratios_lowPU,
        save_path=os.path.join(fig_dir, "casimir_scaling.png"),
    )

    # losses.json holds the FULL per-epoch history; the checkpoint's arrays are frozen at the best epoch and hide everything after early stopping.
    _losses_path = os.path.join(save_dir, "losses.json")
    _losses = {}
    if os.path.exists(_losses_path):
        with open(_losses_path) as f:
            _losses = json.load(f)
    _pt_label = (
        r"Parton $p_T$ [GeV]"
        if source_prefix == "parton"
        else f"{source_prefix.capitalize()} jet $p_T$ [GeV]"
    )
    plot_all(
        truth=y_test,
        sampled=sampled,
        pdg_ids=pdg_ids,
        parton_pt=parton_pt,
        parton_eta=parton_eta,
        parton_phi=parton_phi,
        parton_energy=parton_energy,
        pt_label=_pt_label,
        train_losses=_losses.get("train_losses", checkpoint.get("train_losses")),
        val_losses=_losses.get("val_losses", checkpoint.get("val_losses")),
        lr_history=_losses.get("lr_history", checkpoint.get("lr_history")),
        grad_norm_history=_losses.get(
            "grad_norm_history", checkpoint.get("grad_norm_history")
        ),
        pit_values=pit_all,
        hdr_rank=hdr_rank_arr,
        kl_divergences=kl_divergences,
        speed_results=speed_results,
        save_dir=fig_dir,
    )
    # FM-specific figures (solver convergence, velocity Jacobian, field, trajectories)
    plot_fm_internals(model, X_test, y_test, device, fig_dir)
    print(f"[eval] Figures saved to {fig_dir}")

    # 8. Print summary
    print(f"\n{'='*60}")
    print(f"  Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Mean log-likelihood: {mean_ll:.4f}")
    for name in TARGET_NAMES:
        kl = metrics.get(f"kl_{name}", float("nan"))
        w1 = metrics.get(f"w1_{name}", float("nan"))
        ks = metrics.get(f"sampled_pit_ks_{name}", float("nan"))
        ratio = metrics.get(f"crps_ratio_{name}", float("nan"))
        print(
            f"  {name:20s}  KL={kl:.6f}  W1={w1:.6f}  "
            f"PIT_KS={ks:.6f}  CRPS/floor={ratio:.4f}"
        )

    print(
        f"  Joint HDR coverage: max dev={metrics.get('hdr_max_deviation', float('nan')):.4f}"
        f"  rank KS={metrics.get('hdr_rank_ks', float('nan')):.4f}"
    )

    print_ci_test_summary(metrics, target_prefix=target_prefix)

    # Closure checks
    for key in sorted(k for k in metrics if k.startswith("closure_")):
        val = metrics[key]
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    return metrics


def main():
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=str, default=None)
    _pre_args, _ = _pre.parse_known_args()

    parser = argparse.ArgumentParser(
        description="CFM Training & Evaluation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON file of argparse defaults (base + delta, merged by "
        "run_experiment.py). Explicit CLI flags still override it.",
    )
    parser.add_argument(
        "--config_hash",
        type=str,
        default=None,
        help="Provenance tag written into stats/meta.json; set by "
        "run_experiment.py, not meant to be hand-typed.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="parton2reco",
        choices=sorted(STAGE_CONFIGS),
        help="Which stage of the two-stage decomposition this run is for "
        "(HANDOFF_E2 §3.4). Drives source/target prefix, flavour "
        "column and the disk-clamp cone.",
    )
    parser.add_argument(
        "--force_common_event_set",
        action="store_true",
        help="Force the common-event-set filter on even for parton2reco "
        "(HANDOFF_E2 §3.3/gate 6: re-evaluate an existing parton2reco "
        "checkpoint on the restricted set without retraining).",
    )
    parser.add_argument(
        "--metrics_filename",
        type=str,
        default="metrics.json",
        help="Output filename for metrics.json under save_dir "
        "(HANDOFF_E2_FIX F1: use a distinct name for a "
        "--force_common_event_set rerun of an existing parton2reco "
        "checkpoint so it does not overwrite the original evaluation).",
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
        help="Repo-relative parquet path; run from the repository root.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help="Where run directories go. Default: "
        "Output/fm/ for parton2reco, "
        "Output/fm/Experiments/<stage>/ otherwise. "
        "A bare name (e.g. 'ablation') is placed under "
        "Output/fm/Experiments/.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Appended to the per-seed directory name: 'seed_<seed>' becomes "
        "'seed_<seed>_<run_tag>'. For distinguishing runs that share a "
        "seed and save_root -- e.g. an ablation or a smoke test -- "
        "without them overwriting each other's checkpoint.",
    )

    # Seeds
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)

    # Architecture
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--backbone", type=str, default="plain_mlp")

    # Inference / eval
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument(
        "--solver", type=str, default="midpoint", choices=["euler", "midpoint"]
    )
    parser.add_argument("--n_eval_samples", type=int, default=50)

    # Joint HDR coverage.
    parser.add_argument("--n_hdr_events", type=int, default=5_000)
    parser.add_argument("--n_hdr_samples", type=int, default=20)
    parser.add_argument("--n_hdr_steps", type=int, default=50)
    parser.add_argument(
        "--val_crps_events",
        type=int,
        default=20_000,
        help="Events in the fixed per-epoch validation-CRPS probe (B1). "
        "Instrumentation only -- model selection stays on val loss.",
    )
    parser.add_argument("--val_crps_samples", type=int, default=20)
    parser.add_argument(
        "--val_crps_steps",
        type=int,
        default=20,
        help="Midpoint ODE steps for the val-CRPS probe. Kept low: this runs "
        "every epoch and midpoint sits at the W1 floor well below the "
        "sampling n_steps default.",
    )

    # Data splits
    parser.add_argument("--pt_min", type=float, default=DEFAULT_PT_MIN)
    parser.add_argument("--pt_max", type=float, default=DEFAULT_PT_MAX)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)

    # Per-event dump
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Write the per-event evaluation dump to <save_dir>/dump/.",
    )
    parser.add_argument(
        "--dump_only",
        action="store_true",
        help="Write the dump from an existing checkpoint; skip train+eval.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Evaluate an existing checkpoint (metrics only); no training, no dump.",
    )
    parser.add_argument("--dump_m", type=int, default=200)
    parser.add_argument("--dump_subset", type=int, default=100_000)
    parser.add_argument(
        "--dump_limit",
        type=int,
        default=None,
        help="Truncate the dump to the first N events (verification runs).",
    )
    parser.add_argument(
        "--dump_n_steps",
        type=int,
        default=None,
        help="ODE steps for dump sampling. Defaults to --n_steps; pick from "
        "solver_convergence.png rather than assuming.",
    )

    # AMP
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    parser.add_argument(
        "--conditioner_blocks",
        type=str,
        nargs="+",
        default=["kinematics", "pileup", "flavour", "fourier_phi"],
    )

    if _pre_args.config:
        with open(_pre_args.config) as f:
            _cfg = json.load(f)
        _known = {a.dest for a in parser._actions}
        _defaults = {k: v for k, v in _cfg.items() if k in _known}
        _ignored = sorted(set(_cfg) - _known)
        if _ignored:
            print(
                f"[main] --config: ignoring keys not recognised by this "
                f"CLI: {_ignored}"
            )
        parser.set_defaults(**_defaults)

    args = parser.parse_args()
    if args.force_common_event_set and not args.eval_only:
        parser.error(
            "--force_common_event_set requires --eval_only "
            "(re-evaluating on the common event set must never retrain)"
        )
    if args.eval_only and args.dump_only:
        parser.error("--eval_only and --dump_only are mutually exclusive")
    if args.save_root and Path(args.save_root).is_absolute():
        parser.error("--save_root must be a bare directory name, not an absolute path")

    if args.save_root:
        save_root = str(
            _paths.model_output_root("fm") / "Experiments" / Path(args.save_root).name
        )
    else:
        save_root = str(_paths.save_root("fm", args.stage))
    require_common_event_set = True if args.force_common_event_set else None

    for seed in args.seeds:
        seed_dir_name = f"seed_{seed}" + (f"_{args.run_tag}" if args.run_tag else "")
        save_dir = os.path.join(save_root, seed_dir_name)

        if args.eval_only:
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
            if not os.path.exists(checkpoint_path):
                raise SystemExit(
                    f"--eval_only: no checkpoint at {_paths.rel(checkpoint_path)}"
                )
            evaluate_model(
                checkpoint_path=checkpoint_path,
                data_path=args.data,
                save_dir=save_dir,
                seed=seed,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                n_steps=args.n_steps,
                solver=args.solver,
                pt_min=args.pt_min,
                pt_max=args.pt_max,
                n_eval_samples=args.n_eval_samples,
                n_hdr_events=args.n_hdr_events,
                n_hdr_samples=args.n_hdr_samples,
                n_hdr_steps=args.n_hdr_steps,
                device="cuda" if torch.cuda.is_available() else "cpu",
                stage=args.stage,
                require_common_event_set=require_common_event_set,
                metrics_filename=args.metrics_filename,
            )
            continue

        if args.dump_only:
            generate_dump(
                checkpoint_path=os.path.join(save_dir, "best_model.pt"),
                data_path=args.data,
                save_dir=save_dir,
                seed=seed,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                m=args.dump_m,
                subset_size=args.dump_subset,
                n_steps=args.dump_n_steps or args.n_steps,
                solver=args.solver,
                n_hdr_samples=args.n_hdr_samples,
                n_hdr_steps=args.n_hdr_steps,
                limit=args.dump_limit,
                device="cuda" if torch.cuda.is_available() else "cpu",
                pt_min=args.pt_min,
                pt_max=args.pt_max,
                stage=args.stage,
            )
            continue

        # Train
        best_val_loss = train(
            data_path=args.data,
            save_dir=save_dir,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            width=args.width,
            depth=args.depth,
            backbone=args.backbone,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            num_workers=args.num_workers,
            pt_min=args.pt_min,
            pt_max=args.pt_max,
            amp=args.amp,
            bf16=args.bf16,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            n_steps_eval=args.n_steps,
            solver_eval=args.solver,
            val_crps_events=args.val_crps_events,
            val_crps_samples=args.val_crps_samples,
            val_crps_steps=args.val_crps_steps,
            stage=args.stage,
            config_hash=args.config_hash,
            conditioner_blocks=args.conditioner_blocks,
        )

        # Evaluate
        checkpoint_path = os.path.join(save_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            evaluate_model(
                checkpoint_path=checkpoint_path,
                data_path=args.data,
                save_dir=save_dir,
                seed=seed,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                n_steps=args.n_steps,
                solver=args.solver,
                pt_min=args.pt_min,
                pt_max=args.pt_max,
                n_eval_samples=args.n_eval_samples,
                n_hdr_events=args.n_hdr_events,
                n_hdr_samples=args.n_hdr_samples,
                n_hdr_steps=args.n_hdr_steps,
                device="cuda" if torch.cuda.is_available() else "cpu",
                stage=args.stage,
                require_common_event_set=require_common_event_set,
                metrics_filename=args.metrics_filename,
            )

            if args.dump:
                generate_dump(
                    checkpoint_path=checkpoint_path,
                    data_path=args.data,
                    save_dir=save_dir,
                    seed=seed,
                    val_frac=args.val_frac,
                    test_frac=args.test_frac,
                    m=args.dump_m,
                    subset_size=args.dump_subset,
                    n_steps=args.dump_n_steps or args.n_steps,
                    solver=args.solver,
                    n_hdr_samples=args.n_hdr_samples,
                    n_hdr_steps=args.n_hdr_steps,
                    limit=args.dump_limit,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    pt_min=args.pt_min,
                    pt_max=args.pt_max,
                    stage=args.stage,
                )


if __name__ == "__main__":
    main()
