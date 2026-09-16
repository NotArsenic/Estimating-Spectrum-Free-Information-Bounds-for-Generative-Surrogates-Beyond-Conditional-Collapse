"""
Training loop for the MDN surrogate.

Loss = NLL + lambda_calib * calibration penalty.

The calibration term (`model.mixture_calibration_penalty`) drives the batch mean of the pi-weighted normalized squared residual to D. It is ON by default at 0.1, matching MDN_V1, where it is the likely source of that model's clean calibration (PIT KS 0.010-0.024, HDR tracking y=x)
"""

import os
import sys
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Import from common
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dataset import (
    prepare_datasets,
    save_stats,
    make_loader,
    DEFAULT_PT_MIN,
    DEFAULT_PT_MAX,
)
from common.features import CONDITIONER_NAMES, TARGET_NAMES
from common.metrics import evaluate_all, compute_crps
from common.plots import plot_all
from common.provenance import write_json

from model import (
    Multivariate_MDN,
    mixture_log_prob,
    mixture_calibration_penalty,
)


# LR schedule: linear warmup + cosine annealing
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# Temperature annealing for π logits
def get_pi_temperature(step, total_steps, tau_start, tau_anneal_frac):
    """Cosine anneal from tau_start → 1.0 over the first anneal_frac of training."""
    anneal_steps = int(total_steps * tau_anneal_frac)
    if step >= anneal_steps:
        return 1.0
    progress = float(step) / float(max(1, anneal_steps))
    return 1.0 + (tau_start - 1.0) * 0.5 * (1.0 + math.cos(math.pi * progress))


# Main training function
def train(
    data_path: str,
    save_dir: str,
    seed: int = 42,
    epochs: int = 100,
    batch_size: int = 4096,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    n_components: int = 8,
    width: int = 512,
    depth: int = 6,
    backbone: str = "plain_mlp",
    dropout: float = 0.0,
    pi_tau_start: float = 1.5,
    pi_tau_anneal_frac: float = 0.5,
    lambda_calib: float = 0.1,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    num_workers: int = 4,
    amp: bool = False,
    bf16: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    pt_min: float | None = DEFAULT_PT_MIN,
    pt_max: float | None = DEFAULT_PT_MAX,
    val_crps_events: int = 20_000,
    val_crps_samples: int = 20,
    stage: str = "parton2reco",
    config_hash: str | None = None,
    conditioner_blocks: list[str] | None = None,
):
    """Train the MDN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # W&B
    if wandb_project:
        import wandb

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"MDN_V2_seed{seed}",
            config={
                "model": "MDN_V2",
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "n_components": n_components,
                "width": width,
                "depth": depth,
                "backbone": backbone,
                "lambda_calib": lambda_calib,
                "stage": stage,
                "config_hash": config_hash,
            },
        )

    # Data
    print(f"\n{'='*60}")
    print(f"  MDN_V2 — seed {seed}  stage={stage}")
    print(f"{'='*60}")

    datasets, stats, raw_splits = prepare_datasets(
        data_path,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        pt_min=pt_min,
        pt_max=pt_max,
        stage=stage,
        conditioner_blocks=conditioner_blocks,
    )

    stats["config_hash"] = config_hash
    print(
        f"[train] stage={stats['stage']}  source={stats['source_prefix']}  "
        f"target={stats['target_prefix']}  dr_cone={stats['dr_cone']:.4f}  "
        f"common_event_set={stats['require_common_event_set']}"
    )
    print(
        f"[train] conditioner_blocks={stats['conditioner_blocks']}  "
        f"input_dim={stats['input_dim']}"
    )

    train_loader = make_loader(
        datasets["train"],
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = make_loader(
        datasets["val"],
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # Model
    model = Multivariate_MDN(
        input_dim=stats["input_dim"],
        n_components=n_components,
        depth=depth,
        hidden_dim=width,
        dropout=dropout,
        backbone=backbone,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train] Model parameters: {n_params:,}")
    print(f"[train] Components: {n_components}")

    # Optimizer with selective weight decay
    # No weight decay on MDN output heads
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if any(h in name for h in ["pi_head", "mu_head", "tril_head"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = AdamW(
        [
            {"params": backbone_params, "weight_decay": weight_decay},
            {"params": head_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = min(2000, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    #  AMP
    if amp or bf16:
        raise ValueError(
            "The MDN does not support --amp/--bf16: mixture_log_prob calls "
            "torch.linalg.solve_triangular, which has no bf16/fp16 kernel. "
            "Run the MDN in fp32 (drop the flag); the model is small enough "
            "that the speedup would be marginal."
        )
    use_amp = False
    dtype = torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    val_crps_n = min(val_crps_events, len(datasets["val"]))
    val_crps_X = datasets["val"].X[:val_crps_n].to(device)
    val_crps_y = datasets["val"].y[:val_crps_n].numpy()
    print(f"[train] Val-CRPS probe: n={val_crps_n:,}  m={val_crps_samples}")

    # Training
    os.makedirs(save_dir, exist_ok=True)
    best_val_nll = float("inf")
    train_losses = []
    val_losses = []
    val_crps_history = []
    lr_history = []
    grad_norm_history = []
    step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # Temperature annealing for π
            pi_temp = get_pi_temperature(
                step,
                total_steps,
                pi_tau_start,
                pi_tau_anneal_frac,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                pi, mu, L = model(batch_x, pi_temperature=pi_temp)
                nll = model.neg_log_likelihood(pi, mu, L, batch_y)
                if lambda_calib > 0:
                    calib = mixture_calibration_penalty(pi, mu, L, batch_y)
                    loss = nll + lambda_calib * calib
                else:
                    calib = torch.zeros((), device=nll.device)
                    loss = nll

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            grad_norm_history.append(float(_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += nll.item()
            n_batches += 1
            step += 1
            lr_history.append(scheduler.get_last_lr()[0])

            if wandb_project and step % 100 == 0:
                import wandb

                wandb.log(
                    {
                        "train/nll_step": nll.item(),
                        "train/calib_step": calib.item(),
                        "train/pi_temp": pi_temp,
                        "train/lr": scheduler.get_last_lr()[0],
                        "step": step,
                    }
                )

        epoch_nll = epoch_loss / max(n_batches, 1)
        train_losses.append(epoch_nll)

        # Validation
        model.eval()
        val_nll = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                    pi, mu, L = model(batch_x, pi_temperature=1.0)
                    nll = model.neg_log_likelihood(pi, mu, L, batch_y)

                val_nll += nll.item()
                n_val += 1

        val_nll /= max(n_val, 1)
        val_losses.append(val_nll)

        _cpu_rng_state = torch.get_rng_state()
        _cuda_rng_state = (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        )
        torch.manual_seed(seed + 999_999)
        with torch.no_grad():
            draws = []
            for start in range(0, val_crps_n, batch_size):
                end = min(start + batch_size, val_crps_n)
                s = model.sample(val_crps_X[start:end], n_samples=val_crps_samples)
                draws.append(s.float().cpu().numpy())
            samples = np.concatenate(draws, axis=0)  # (n, m, D)
        torch.set_rng_state(_cpu_rng_state)
        if _cuda_rng_state is not None:
            torch.cuda.set_rng_state(_cuda_rng_state, device)

        val_crps = compute_crps(val_crps_y, samples)
        val_crps_history.append(val_crps)
        val_crps_mean = float(np.mean(list(val_crps.values())))

        elapsed = time.time() - t_start
        print(
            f"  Epoch {epoch + 1:3d}/{epochs} | "
            f"Train NLL {epoch_nll:.4f} | Val NLL {val_nll:.4f} | "
            f"ValCRPS {val_crps_mean:.6f} | "
            f"Time {elapsed:.1f}s"
        )

        if wandb_project:
            import wandb

            wandb.log(
                {
                    "train/nll_epoch": epoch_nll,
                    "val/nll_epoch": val_nll,
                    "val/crps_mean": val_crps_mean,
                    **{f"val/{k}": v for k, v in val_crps.items()},
                    "epoch": epoch,
                }
            )

        # Checkpoint best model
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "input_dim": stats["input_dim"],
                    "n_components": n_components,
                    "depth": depth,
                    "hidden_dim": width,
                    "dropout": dropout,
                    "backbone": backbone,
                    "seed": seed,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "lambda_calib": lambda_calib,
                    "stage": stage,
                    "config_hash": config_hash,
                    "conditioner_blocks": stats["conditioner_blocks"],
                },
                "stats": stats,
                "epoch": epoch,
                "step": step,
                "best_val_nll": best_val_nll,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "lr_history": lr_history,
                "grad_norm_history": grad_norm_history,
            }
            ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"    → Saved best model (val_nll={best_val_nll:.4f})")

        write_json(
            os.path.join(save_dir, "losses.json"),
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_crps_history": val_crps_history,
                "val_crps_config": {
                    "n_events": val_crps_n,
                    "n_samples": val_crps_samples,
                },
                "lr_history": lr_history,
                "grad_norm_history": grad_norm_history,
                "best_val_nll": best_val_nll,
            },
            allow_nan=True,
        )

    # Save stats
    save_stats(stats, os.path.join(save_dir, "stats.json"))

    if wandb_project:
        import wandb

        wandb.finish()

    print(f"\n[train] Done. Best val NLL: {best_val_nll:.4f}")
    return best_val_nll
