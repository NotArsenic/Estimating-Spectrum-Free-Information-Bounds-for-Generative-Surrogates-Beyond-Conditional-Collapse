"""
Training loop for the CFM surrogate.
"""

import os
import sys
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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

from model import VelocityMLP
from inference import midpoint_integrate


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


# Main training function
def train(
    data_path: str,
    save_dir: str,
    seed: int = 42,
    epochs: int = 100,
    batch_size: int = 4096,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    width: int = 512,
    depth: int = 6,
    backbone: str = "plain_mlp",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    num_workers: int = 4,
    amp: bool = False,
    bf16: bool = False,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    n_steps_eval: int = 100,
    solver_eval: str = "midpoint",
    pt_min: float | None = DEFAULT_PT_MIN,
    pt_max: float | None = DEFAULT_PT_MAX,
    val_crps_events: int = 20_000,
    val_crps_samples: int = 20,
    val_crps_steps: int = 20,
    stage: str = "parton2reco",
    config_hash: str | None = None,
    conditioner_blocks: list[str] | None = None,
):
    """Train the flow matching velocity field."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- W&B ---
    if wandb_project:
        import wandb

        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=f"FM_V2_seed{seed}",
            config={
                "model": "FlowMatching_SingleStage_V2",
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "width": width,
                "depth": depth,
                "backbone": backbone,
                "stage": stage,
                "config_hash": config_hash,
            },
        )

    # Data
    print(f"\n{'='*60}")
    print(f"  FlowMatching_SingleStage_V2 — seed {seed}  stage={stage}")
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

    # Noise prior (moment-matched to training targets)
    noise_loc = datasets["train"].y.mean(dim=0).to(device)
    noise_scale = datasets["train"].y.std(dim=0).to(device)
    print(f"\n[train] Noise prior:")
    for i, name in enumerate(TARGET_NAMES):
        print(f"  {name:20s}  loc={noise_loc[i]:.4f}  scale={noise_scale[i]:.4f}")

    # Model
    model = VelocityMLP(
        cond_dim=stats["input_dim"],
        target_dim=4,
        width=width,
        depth=depth,
        backbone=backbone,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train] Model parameters: {n_params:,}")

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = min(2000, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # AMP
    use_amp = amp or bf16
    dtype = torch.bfloat16 if bf16 else (torch.float16 if amp else torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and not bf16)

    val_crps_n = min(val_crps_events, len(datasets["val"]))
    val_crps_X = datasets["val"].X[:val_crps_n].to(device)
    val_crps_y = datasets["val"].y[:val_crps_n].numpy()
    print(
        f"[train] Val-CRPS probe: n={val_crps_n:,}  m={val_crps_samples}  "
        f"n_steps={val_crps_steps} (midpoint)"
    )

    # Training
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
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
            B = batch_y.shape[0]

            # Coupling
            # x1 = target for THIS row
            # x0 ~ N(noise_loc, noise_scale^2), independent of c
            x1 = batch_y
            x0 = noise_loc + noise_scale * torch.randn_like(x1)
            t = torch.rand(B, 1, device=device)
            xt = (1 - t) * x0 + t * x1
            v_target = x1 - x0

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                v_pred = model(xt, t, batch_x)
                loss = F.mse_loss(v_pred, v_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            grad_norm_history.append(float(_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1
            lr_history.append(scheduler.get_last_lr()[0])

            if wandb_project and step % 100 == 0:
                import wandb

                wandb.log(
                    {
                        "train/loss_step": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "step": step,
                    }
                )

        epoch_loss /= max(n_batches, 1)
        train_losses.append(epoch_loss)

        # Validation
        val_gen = torch.Generator(device=device)
        val_gen.manual_seed(seed + 999_999)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                B = batch_y.shape[0]

                x1 = batch_y
                x0 = noise_loc + noise_scale * torch.randn_like(x1, generator=val_gen)
                t = torch.rand(B, 1, device=device, generator=val_gen)
                xt = (1 - t) * x0 + t * x1
                v_target = x1 - x0

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                    v_pred = model(xt, t, batch_x)
                    loss = F.mse_loss(v_pred, v_target)

                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)
        val_losses.append(val_loss)

        val_gen.manual_seed(seed + 999_999)
        with torch.no_grad():
            samples = np.empty(
                (val_crps_n, val_crps_samples, len(TARGET_NAMES)), dtype=np.float32
            )
            for m in range(val_crps_samples):
                draws = []
                for start in range(0, val_crps_n, batch_size):
                    end = min(start + batch_size, val_crps_n)
                    c = val_crps_X[start:end]
                    x0 = noise_loc + noise_scale * torch.randn(
                        end - start,
                        len(TARGET_NAMES),
                        device=device,
                        generator=val_gen,
                    )
                    draws.append(
                        midpoint_integrate(model, x0, c, val_crps_steps).cpu().numpy()
                    )
                samples[:, m, :] = np.concatenate(draws, axis=0)
        val_crps = compute_crps(val_crps_y, samples)
        val_crps_history.append(val_crps)
        val_crps_mean = float(np.mean(list(val_crps.values())))

        elapsed = time.time() - t_start
        print(
            f"  Epoch {epoch + 1:3d}/{epochs} | "
            f"Train {epoch_loss:.6f} | Val {val_loss:.6f} | "
            f"ValCRPS {val_crps_mean:.6f} | "
            f"Time {elapsed:.1f}s"
        )

        if wandb_project:
            import wandb

            wandb.log(
                {
                    "train/loss_epoch": epoch_loss,
                    "val/loss_epoch": val_loss,
                    "val/crps_mean": val_crps_mean,
                    **{f"val/{k}": v for k, v in val_crps.items()},
                    "epoch": epoch,
                }
            )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "cond_dim": stats["input_dim"],
                    "target_dim": 4,
                    "width": width,
                    "depth": depth,
                    "backbone": backbone,
                    "seed": seed,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "stage": stage,
                    "config_hash": config_hash,
                    "conditioner_blocks": stats["conditioner_blocks"],
                },
                "noise_loc": noise_loc.cpu(),
                "noise_scale": noise_scale.cpu(),
                "stats": stats,
                "epoch": epoch,
                "step": step,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "lr_history": lr_history,
                "grad_norm_history": grad_norm_history,
            }
            ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"    → Saved best model (val_loss={best_val_loss:.6f})")

        write_json(
            os.path.join(save_dir, "losses.json"),
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_crps_history": val_crps_history,
                "val_crps_config": {
                    "n_events": val_crps_n,
                    "n_samples": val_crps_samples,
                    "n_steps": val_crps_steps,
                    "solver": "midpoint",
                },
                "lr_history": lr_history,
                "grad_norm_history": grad_norm_history,
                "best_val_loss": best_val_loss,
            },
            allow_nan=True,
        )

    # Save stats
    save_stats(stats, os.path.join(save_dir, "stats.json"))

    if wandb_project:
        import wandb

        wandb.finish()

    print(f"\n[train] Done. Best val loss: {best_val_loss:.6f}")
    return best_val_loss
