"""
THOR-PIML — Utilitários do Sistema (Sem Restrição de Parâmetros)
===============================================================
Seeding determinístico, detecção de hardware (GPU/CPU), salvamento/carregamento
de checkpoints e sumário detalhado de parâmetros da rede neural.
"""
from __future__ import annotations
import os
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn


def get_logger(name: str = "THOR-PIML") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s | %(name)s | %(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger()


def set_seed(seed: Optional[int] = None, *, eval_mode: bool = False) -> int:
    """Fixa as seeds para reprodutibilidade.

    Se seed=None, gera aleatória (para busca de sorte) e loga.
    Retorna a seed usada (para salvar no checkpoint).

    - seed=None → treino exploratório: deterministic=False, benchmark=True (rápido T4)
    - seed=int → reprodutível: deterministic=True, benchmark=False
    - eval_mode=True → loga como 'Seed avaliação reprodutível' para não confundir com treino
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"Seed aleatória gerada: {seed} (treino exploratório, não-determinístico)")
        deterministic = False
        benchmark = True
    else:
        if eval_mode:
            logger.info(f"Seed avaliação reprodutível: {seed} (determinístico, para pipeline zero-leakage)")
        else:
            logger.info(f"Seed fixada: {seed} (reprodutível)")
        deterministic = True
        benchmark = False

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return seed


def get_device() -> torch.device:
    """Detecta automaticamente GPU (CUDA) ou CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        device = torch.device("cpu")
        logger.info("GPU não disponível, usando CPU.")
    return device


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Retorna (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model: nn.Module) -> str:
    """Gera um resumo formatado de cada camada e contagem total de parâmetros."""
    lines = [
        "=" * 70,
        f"{'Layer':<45} {'Params':>12} {'Trainable':>10}",
        "=" * 70,
    ]
    for name, param in model.named_parameters():
        lines.append(f"{name:<45} {param.numel():>12,} {'[Y]' if param.requires_grad else '[N]':>10}")

    total, trainable = count_parameters(model)
    lines.extend([
        "=" * 70,
        f"{'TOTAL DE PARÂMETROS':<45} {total:>12,} {trainable:>10,}",
        f"Capacidade: Arquitetura de Alta Capacidade (Sem Restrição de Tamanho)",
        "=" * 70,
    ])

    return "\n".join(lines)


def _get_git_hash() -> str:
    """Retorna hash do commit atual (ou 'unknown' se não for repo git)."""
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    filepath: Path | str,
    extra: Optional[Dict[str, Any]] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """Salva checkpoint do modelo com reprodutibilidade (S0).

    Inclui automaticamente git_hash, scaler_state e config_dict para
    que qualquer checkpoint seja auto-contido e auditável (essencial para CORDEX).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "git_hash": _get_git_hash(),
        "torch_version": torch.__version__,
    }
    if scaler_state is not None:
        state["scaler_state"] = scaler_state
    if config_dict is not None:
        state["config"] = config_dict
    if extra:
        # extra nunca sobrescreve campos canônicos se houver conflito
        for k, v in extra.items():
            if k not in state:
                state[k] = v

    torch.save(state, filepath)
    logger.info(f"Checkpoint salvo em: {filepath} (git={state['git_hash']})")

    # Também salva symlink/copy como latest.pt para resume fácil
    try:
        latest = filepath.parent / "latest.pt"
        import shutil

        shutil.copy2(filepath, latest)
    except Exception:
        pass
