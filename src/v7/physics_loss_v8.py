"""
THOR-PIML V8 — Loss com barreira física Clausius-Clapeyron via TCWV REAL
========================================================================
Herda THORLossV7 e reativa a física (λ_physics > 0) — agora defensável, pois o
GT V3 tem TCWV real (V2 tinha constante 20mm, barrier era ruído).

Barreira: precipitação diária não pode exceder a água precipitável disponível
vezes um fator de reciclagem k (convergência de umidade permite chuva > TCWV):

    L_phys = softplus(ŷ − k·TCWV)²   com k default 4.0

TCWV vem denormalizado do scaler (feature 'tcwv' presente no pipeline V8/V3).
"""
from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from src.preprocessing import RobustClimateScaler
from src.v7.physics_loss_v7 import THORLossV7


class THORLossV8(THORLossV7):
    def __init__(
        self,
        scaler: RobustClimateScaler,
        training_config,
        occurrence_threshold: float = 1.0,
        tcwv_recycle_factor: float = 4.0,
    ):
        super().__init__(scaler, training_config, occurrence_threshold)
        self.k_recycle = tcwv_recycle_factor

    def forward(
        self,
        prob_occ: Tensor,
        intensity: Tensor,
        final_pred: Tensor,
        y_class_true: Tensor,
        y_reg_true: Tensor,
        x_batch: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        loss, metrics = super().forward(
            prob_occ, intensity, final_pred, y_class_true, y_reg_true, x_batch
        )
        if self.cfg.lambda_physics > 0:
            try:
                tcwv_mm = self._denorm(x_batch, "tcwv", fallback_idx=0).squeeze(-1)
                w_max = self.k_recycle * tcwv_mm
                viol = final_pred.squeeze(-1) > w_max
                phys = F.softplus(final_pred.squeeze(-1) - w_max) ** 2
                loss_physics = self.cfg.lambda_physics * phys.mean()
                loss = loss + loss_physics
                metrics["physics_loss"] = float(loss_physics.item())
                metrics["n_violations"] = int(viol.sum().item())
            except Exception:
                # sem 'tcwv' no scaler → física permanece off (não derruba o treino)
                metrics["physics_loss"] = 0.0
                metrics["n_violations"] = 0
        return loss, metrics
