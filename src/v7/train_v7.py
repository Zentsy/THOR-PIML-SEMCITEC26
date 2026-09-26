"""
THOR-PIML V7 — Loop de Treinamento
===================================
- AdamW + OneCycleLR (3e-4 → max_lr 1e-3 → 3e-5, docs §9) com step por batch
- AMP FP16 (T4) + grad clip (unscale-then-clip)
- Early stopping na val loss; SELEÇÃO de checkpoint por val KGE (a config V6
  prometia 'kge' mas o train.py ignorava — aqui é real)
- Checkpoint auto-contido: thor_version='v7', variant, feature_cols, lags,
  config asdict, scaler_state, history, seed, git hash
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils import logger, save_checkpoint
from src.v7.physics_loss_v7 import THORLossV7


class EarlyStoppingV7:
    def __init__(self, patience: int = 50, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = -float("inf") if mode == "max" else float("inf")
        self.stop = False

    def step(self, score: float) -> bool:
        if np.isnan(score) or self.patience <= 0:
            return False
        improved = (score > self.best + self.min_delta) if self.mode == "max" else (score < self.best - self.min_delta)
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def _kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.std() == 0 or y_pred.std() == 0 or len(y_true) < 2:
        return float("nan")
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = float(y_pred.std() / y_true.std())
    beta = float(y_pred.mean() / y_true.mean()) if y_true.mean() != 0 else float("nan")
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


@torch.no_grad()
def validate_v7(model, val_loader, loss_fn, device) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    n = 0
    preds, trues = [], []
    for batch in val_loader:
        x, yc, yr = (t.to(device, non_blocking=True) for t in batch)
        prob, inten, final = model(x, return_components=True)
        _, metrics = loss_fn(prob, inten, final, yc, yr, x)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        preds.append(final.squeeze(-1).float().cpu().numpy())
        trues.append(yr.squeeze(-1).float().cpu().numpy())
        n += 1
    out = {k: v / max(n, 1) for k, v in totals.items()}
    out["kge"] = _kge(np.concatenate(trues), np.concatenate(preds))
    return out


def _autocast_ctx(device: torch.device):
    try:
        return torch.amp.autocast("cuda" if device.type == "cuda" else "cpu")
    except TypeError:
        return torch.cuda.amp.autocast()


def _grad_scaler(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            return torch.cuda.amp.GradScaler()
    return None


def train_v7(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: THORLossV7,
    config,  # THORConfigV7
    ckpt_path: Path,
    device: Optional[torch.device] = None,
    scaler=None,
    variant: str = "hybrid",
    feature_cols: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[float]]:
    from src.utils import get_device
    if device is None:
        device = get_device()
    model = model.to(device)
    tcfg = config.training

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    if tcfg.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=tcfg.max_lr,
            total_steps=tcfg.epochs * max(len(train_loader), 1),
            pct_start=0.3,
            div_factor=max(tcfg.max_lr / tcfg.lr, 1.0),
            final_div_factor=max(tcfg.max_lr / tcfg.min_lr, 1.0),
        )
        sched_per_step = True
    elif tcfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=tcfg.min_lr
        )
        sched_per_step = False
    else:
        raise ValueError(f"Scheduler desconhecido: {tcfg.scheduler}")

    early = EarlyStoppingV7(patience=tcfg.patience)
    amp_scaler = _grad_scaler(device, tcfg.use_amp)
    metric = tcfg.checkpoint_metric  # "kge" (max) ou "loss" (min)
    best_score = -float("inf") if metric == "kge" else float("inf")

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_kge": [], "lr": [],
        "val_bce": [], "val_int": [], "val_extreme_fn": [],
    }

    pbar = tqdm(range(1, tcfg.epochs + 1), desc=f"THOR-V7 [{variant}]", unit="epoch")
    for epoch in pbar:
        model.train()
        running = 0.0
        nb = 0
        for batch in train_loader:
            x, yc, yr = (t.to(device, non_blocking=True) for t in batch)
            optimizer.zero_grad(set_to_none=True)
            stepped = True
            if amp_scaler is not None:
                with _autocast_ctx(device):
                    prob, inten, final = model(x, return_components=True)
                    loss, m = loss_fn(prob, inten, final, yc, yr, x)
                amp_scaler.scale(loss).backward()
                if tcfg.grad_clip > 0:
                    amp_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                scale_before = amp_scaler.get_scale()
                amp_scaler.step(optimizer)
                amp_scaler.update()
                # GradScaler pula optimizer.step() quando acha inf/nan (escala cai pela
                # metade). OneCycleLR só pode avançar se um passo REAL aconteceu — senão
                # o PyTorch warn "scheduler.step() before optimizer.step()" e o ciclo
                # de LR dessincroniza.
                stepped = amp_scaler.get_scale() >= scale_before
            else:
                prob, inten, final = model(x, return_components=True)
                loss, m = loss_fn(prob, inten, final, yc, yr, x)
                loss.backward()
                if tcfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                optimizer.step()
            if sched_per_step and stepped:
                scheduler.step()
            running += m["loss_total"]
            nb += 1
        if not sched_per_step:
            scheduler.step()

        val_metrics = validate_v7(model, val_loader, loss_fn, device)
        history["train_loss"].append(running / max(nb, 1))
        history["val_loss"].append(val_metrics["loss_total"])
        history["val_kge"].append(val_metrics.get("kge", float("nan")))
        history["val_bce"].append(val_metrics.get("bce_loss", 0.0))
        history["val_int"].append(val_metrics.get("mse_loss", 0.0))
        history["val_extreme_fn"].append(val_metrics.get("extreme_fn_loss", 0.0))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        score = val_metrics.get("kge", float("nan")) if metric == "kge" else val_metrics["loss_total"]
        improved = (
            not np.isnan(score)
            and ((score > best_score) if metric == "kge" else (score < best_score))
        )
        if improved:
            best_score = score
            save_checkpoint(
                model, optimizer, epoch, val_metrics["loss_total"],
                ckpt_path,
                extra={
                    "thor_version": "v7",
                    "model_variant": variant,
                    "feature_cols": feature_cols,
                    "lags": lags,
                    "history": history,
                    "seed": seed,
                    "checkpoint_metric": f"{metric}={score:.4f}",
                },
                scaler_state=scaler.to_dict() if scaler is not None and hasattr(scaler, "to_dict") else None,
                config_dict=asdict(config),
            )
        pbar.set_postfix(
            train=f"{history['train_loss'][-1]:.4f}",
            val=f"{val_metrics['loss_total']:.4f}",
            kge=f"{val_metrics.get('kge', float('nan')):.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )
        if early.step(val_metrics["loss_total"]):
            logger.info(f"Early stopping na época {epoch} (best {metric}={best_score:.4f})")
            break

    logger.info(f"Treino V7 [{variant}] concluído — melhor val {metric}: {best_score:.4f}")
    return history
