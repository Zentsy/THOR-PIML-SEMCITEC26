"""
THOR-PIML — Loop de Treinamento Otimizado para T4 (Sprint S5)
===============================================================
AdamW + AMP (FP16) + EarlyStopping + Gradient Clipping + Scheduler
(Plateau ou Cosine) + histórico completo. Pronto para Lightning.AI T4.

S5 vs V1:
- AMP (torch.cuda.amp) com GradScaler — 3× mais rápido, metade VRAM
- DataLoader com num_workers=4, pin_memory, persistent_workers (T4)
- Scheduler configurável: plateau (V1) ou cosine (S5, melhor para LSTM)
- Checkpoint agora salva scaler_state + config_dict (S0) + history
- TQDM progress bar + logging de grad_norm, lr, w_max

Este código RODA em CPU (sem AMP) para testes, e acelera automaticamente em T4.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import TrainingConfig, THORConfig
from src.model import THORPIMLModel
from src.physics_loss import THORPhysicsLoss
from src.preprocessing import RobustClimateScaler
from src.utils import logger, get_device, save_checkpoint


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def _get_scaler(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        # torch.cuda.amp.GradScaler is deprecated -> torch.amp.GradScaler
        try:
            return torch.amp.GradScaler('cuda')
        except TypeError:
            return torch.cuda.amp.GradScaler()
    return None


def train_epoch(
    model: THORPIMLModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: THORPhysicsLoss,
    grad_clip: float,
    device: torch.device,
    amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_mse = 0.0
    total_phys = 0.0
    total_violations = 0
    total_grad_norm = 0.0
    n_batches = len(train_loader)

    for batch in train_loader:
        x_batch = batch[0].to(device, non_blocking=True)
        y_class_true = batch[1].to(device, non_blocking=True)
        y_reg_true = batch[2].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # AMP autocast para T4 (FP16) — usa nova API torch.amp
        if amp_scaler is not None:
            try:
                ctx = torch.amp.autocast('cuda')
            except TypeError:
                ctx = torch.cuda.amp.autocast()
            with ctx:
                prob_occ, intensity, final_pred = model(x_batch, return_components=True)
                loss, metrics = loss_fn(prob_occ, intensity, final_pred, y_class_true, y_reg_true, x_batch)
            amp_scaler.scale(loss).backward()
            if grad_clip > 0:
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            prob_occ, intensity, final_pred = model(x_batch, return_components=True)
            loss, metrics = loss_fn(prob_occ, intensity, final_pred, y_class_true, y_reg_true, x_batch)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += metrics["loss_total"]
        total_bce += metrics["bce_loss"]
        total_mse += metrics["mse_loss"]
        total_phys += metrics["physics_loss"]
        total_violations += metrics["n_violations"]
        # grad_norm para diagnóstico (após clip)
        if grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item() if hasattr(torch.nn.utils, "clip_grad_norm_") else 0)
            total_grad_norm += grad_norm

    return {
        "loss_total": total_loss / max(n_batches, 1),
        "bce_loss": total_bce / max(n_batches, 1),
        "mse_loss": total_mse / max(n_batches, 1),
        "physics_loss": total_phys / max(n_batches, 1),
        "n_violations": total_violations,
        "grad_norm": total_grad_norm / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: THORPIMLModel,
    val_loader: DataLoader,
    loss_fn: THORPhysicsLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_mse = 0.0
    total_phys = 0.0
    total_violations = 0
    n_batches = len(val_loader)
    for batch in val_loader:
        x_batch = batch[0].to(device, non_blocking=True)
        y_class_true = batch[1].to(device, non_blocking=True)
        y_reg_true = batch[2].to(device, non_blocking=True)
        prob_occ, intensity, final_pred = model(x_batch, return_components=True)
        _, metrics = loss_fn(prob_occ, intensity, final_pred, y_class_true, y_reg_true, x_batch)
        total_loss += metrics["loss_total"]
        total_bce += metrics["bce_loss"]
        total_mse += metrics["mse_loss"]
        total_phys += metrics["physics_loss"]
        total_violations += metrics["n_violations"]
    return {
        "loss_total": total_loss / max(n_batches, 1),
        "bce_loss": total_bce / max(n_batches, 1),
        "mse_loss": total_mse / max(n_batches, 1),
        "physics_loss": total_phys / max(n_batches, 1),
        "n_violations": total_violations,
    }


def build_schedulers(optimizer, config: TrainingConfig):
    if config.scheduler == "cosine":
        # CosineAnnealingWarmRestarts: bom para LSTM, evita platôs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=config.min_lr
        )
        # Para cosine, step() a cada época (não precisa de val_loss)
        return scheduler, "cosine"
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.lr_factor, patience=5, min_lr=config.min_lr
        )
        return scheduler, "plateau"


def train(
    model: THORPIMLModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: THORPhysicsLoss,
    config: TrainingConfig,
    checkpoint_dir: str | Path = "checkpoints",
    device: Optional[torch.device] = None,
    scaler: Optional[RobustClimateScaler] = None,
    thor_config: Optional[THORConfig] = None,
) -> Dict[str, List[float]]:
    if device is None:
        device = get_device()
    model = model.to(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler, sched_type = build_schedulers(optimizer, config)
    early_stopper = EarlyStopping(patience=config.patience)
    amp_scaler = _get_scaler(device, config.use_amp)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_bce": [], "val_bce": [],
        "train_mse": [], "val_mse": [],
        "train_physics": [], "val_physics": [],
        "lr": [], "grad_norm": [],
    }
    best_val_loss = float("inf")
    device_str = f"{device} (AMP={'ON' if amp_scaler else 'OFF'})"
    logger.info(f"Iniciando treino THOR-PIML V2 ({config.epochs} épocas, batch={config.batch_size}, {device_str}, sched={sched_type}, 520k params)")

    # Para S7: scaler_state para salvar
    scaler_state = scaler.to_dict() if scaler is not None and hasattr(scaler, "to_dict") else None
    config_dict = thor_config.__dict__ if thor_config is not None else None
    # Se thor_config for dataclass, serializa melhor
    if thor_config is not None:
        try:
            from dataclasses import asdict
            config_dict = asdict(thor_config)
        except Exception:
            pass

    pbar = tqdm(range(1, config.epochs + 1), desc="THOR-PIML Treino", unit="epoch")
    for epoch in pbar:
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, config.grad_clip, device, amp_scaler)
        val_metrics = validate(model, val_loader, loss_fn, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if sched_type == "plateau":
            scheduler.step(val_metrics["loss_total"])
        else:
            scheduler.step()

        history["train_loss"].append(train_metrics["loss_total"])
        history["val_loss"].append(val_metrics["loss_total"])
        history["train_bce"].append(train_metrics["bce_loss"])
        history["val_bce"].append(val_metrics["bce_loss"])
        history["train_mse"].append(train_metrics["mse_loss"])
        history["val_mse"].append(val_metrics["mse_loss"])
        history["train_physics"].append(train_metrics["physics_loss"])
        history["val_physics"].append(val_metrics["physics_loss"])
        history["lr"].append(current_lr)
        history["grad_norm"].append(train_metrics.get("grad_norm", 0))

        pbar.set_postfix({
            "train": f"{train_metrics['loss_total']:.4f}",
            "val": f"{val_metrics['loss_total']:.4f}",
            "lr": f"{current_lr:.1e}",
            "phys": f"{train_metrics['physics_loss']:.3f}",
        })
        logger.info(
            f"Epoch {epoch:>3}/{config.epochs} | Train {train_metrics['loss_total']:.6f} "
            f"(bce={train_metrics['bce_loss']:.4f}, mse={train_metrics['mse_loss']:.4f}, phys={train_metrics['physics_loss']:.4f}, gn={train_metrics.get('grad_norm',0):.2f}) | "
            f"Val {val_metrics['loss_total']:.6f} (phys {val_metrics['physics_loss']:.4f}) | LR {current_lr:.2e}"
        )

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                checkpoint_dir / "best_model.pt",
                extra={"history": history, "epoch": epoch},
                scaler_state=scaler_state,
                config_dict=config_dict,
            )
            # Também salva latest para resume
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                checkpoint_dir / "latest.pt",
                extra={"history": history, "epoch": epoch},
                scaler_state=scaler_state,
                config_dict=config_dict,
            )

        if early_stopper.step(val_metrics["loss_total"]):
            logger.info(f"Early stopping na época {epoch}.")
            break

    logger.info(f"Treino concluído! Melhor val_loss: {best_val_loss:.6f} ({device_str})")
    return history
