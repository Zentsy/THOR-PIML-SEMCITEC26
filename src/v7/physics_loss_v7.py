"""
THOR-PIML V7 — Loss (base V6d balanceada + extremes-recall + variance matching)
================================================================================
Mesma assinatura do V6: forward(prob, intensity, final, y_class, y_reg, x) →
(loss, metrics). Todos os pesos vêm de TrainingConfigV7 (auditáveis no ckpt).

Componentes:
1. Focal occurrence (γ1.5 α0.5, threshold 1.0mm WMO) — igual V6d
2. Quantile-Huber intensity (q=0.65) com storm weights R10 1.3× / R20 2.0×
   e underestimation boost 1.3×/1.5× — igual V6d
3. Dry BCE 0.3 — igual V6d
4. Final-product quantile 0.2 — igual V6d
5. Sharpness one-sided 0.05 — igual V6d
6. NOVO extremes-recall: FN em R10/R20 (chuva forte predita fraca) —
   relu(y - final) nos dias R10+/R20+ com boost 1.5×/2.5× e peso 0.5.
   Ataca o modo de falha dominante do V6d: R20 recall 13% (13 vs 100).
7. NOVO variance matching simétrico |std_pred - std_true| peso 0.02 —
   V6d só tinha relu one-sided; o simétrico evita inflar bias no outro lado.
8. Physics barrier (softplus CC): λ=0 no V7 (tcwv V2 é constante 20mm —
   barrier seria ruído); reativa no V8 com TCWV real.

Lição V6c respeitada: nada de multiplicadores brutos (6× virou R10 1252);
extremos entram com peso moderado e configurável.
"""
from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.preprocessing import RobustClimateScaler


class THORLossV7(nn.Module):
    def __init__(
        self,
        scaler: RobustClimateScaler,
        training_config,
        occurrence_threshold: float = 1.0,
    ):
        super().__init__()
        self.scaler = scaler
        self.cfg = training_config
        self.occurrence_thresh = occurrence_threshold

    def _denorm(self, x_batch: Tensor, feature: str, fallback_idx: int) -> Tensor:
        last = x_batch[:, -1, :]
        if getattr(self.scaler, "feature_names", None) is not None and feature in self.scaler.feature_names:
            idx = self.scaler.feature_names.index(feature)
            col_slice = last[:, idx : idx + 1]
            return self.scaler.denormalize_column(col_slice, idx)
        col_slice = last[:, fallback_idx : fallback_idx + 1]
        if hasattr(self.scaler, "denormalize_column") and getattr(self.scaler, "fitted", False):
            return self.scaler.denormalize_column(col_slice, fallback_idx)
        return col_slice

    def denormalize_feature(self, x_batch: Tensor, feature_idx: int) -> Tensor:
        mapping = {0: "tmean", 3: "rh", 4: "psfc"}
        feature = mapping.get(feature_idx, f"idx_{feature_idx}")
        return self._denorm(x_batch, feature, feature_idx)

    def forward(
        self,
        prob_occ: Tensor,
        intensity: Tensor,
        final_pred: Tensor,
        y_class_true: Tensor,
        y_reg_true: Tensor,
        x_batch: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:

        cfg = self.cfg
        y_class_1mm = (y_reg_true >= self.occurrence_thresh).float()
        wet_mask = y_class_1mm >= 0.5
        dry_mask = ~wet_mask

        # 1. Focal occurrence
        prob_safe = torch.clamp(prob_occ, min=1e-7, max=1 - 1e-7)
        logits = torch.logit(prob_safe)
        bce_raw = F.binary_cross_entropy_with_logits(logits, y_class_1mm, reduction="none")
        p_t = torch.where(y_class_1mm >= 0.5, prob_safe, 1 - prob_safe)
        focal_weight = (1.0 - p_t) ** cfg.gamma_focal
        alpha_t = torch.where(y_class_1mm >= 0.5, cfg.alpha_focal, 1.0 - cfg.alpha_focal)
        loss_occ = torch.mean(alpha_t * focal_weight * bce_raw)

        # 2. Quantile-Huber intensity estratificado (q=0.500 para dias normais, q_r20=0.68 para tempestades severas)
        q_norm = cfg.quantile
        q_r20 = getattr(cfg, "quantile_r20", 0.68)
        r10_mask = y_reg_true >= 10.0
        r20_mask = y_reg_true >= 20.0
        q_tensor = torch.where(r20_mask, torch.full_like(y_reg_true, q_r20), torch.full_like(y_reg_true, q_norm))

        error_int = y_reg_true - intensity
        huber_int = F.huber_loss(intensity, y_reg_true, delta=cfg.huber_delta, reduction="none")
        quantile_int = torch.where(error_int >= 0, q_tensor * huber_int, (1 - q_tensor) * huber_int)

        weight = torch.ones_like(y_reg_true)
        weight = torch.where(r10_mask, torch.full_like(weight, cfg.storm_weight_r10), weight)
        weight = torch.where(r20_mask, torch.full_like(weight, cfg.storm_weight_r20), weight)
        underest_r10 = (intensity < y_reg_true) & r10_mask
        underest_r20 = (intensity < y_reg_true) & r20_mask
        weight = torch.where(underest_r10, weight * cfg.underest_boost_r10, weight)
        weight = torch.where(underest_r20, weight * cfg.underest_boost_r20, weight)

        if wet_mask.any():
            loss_int = torch.mean(weight[wet_mask] * quantile_int[wet_mask])
        else:
            loss_int = torch.zeros((), device=intensity.device)

        # 3. Dry BCE
        if dry_mask.any() and cfg.lambda_dry > 0:
            prob_dry = prob_occ[dry_mask]
            prob_dry_safe = torch.clamp(prob_dry, min=1e-7, max=1 - 1e-7)
            logits_dry = torch.logit(prob_dry_safe)
            loss_dry = F.binary_cross_entropy_with_logits(
                logits_dry, torch.zeros_like(logits_dry)
            )
        else:
            loss_dry = torch.zeros((), device=prob_occ.device)

        # 4. Final-product quantile
        error_final = y_reg_true - final_pred
        huber_final = F.huber_loss(final_pred, y_reg_true, delta=cfg.huber_delta, reduction="none")
        quantile_final = torch.where(error_final >= 0, q_tensor * huber_final, (1 - q_tensor) * huber_final)
        loss_final = cfg.lambda_final * torch.mean(weight * quantile_final)

        # 5. Sharpness one-sided
        if y_reg_true.numel() > 1:
            std_true = torch.std(y_reg_true)
            std_pred = torch.std(final_pred)
            sharpness_loss = cfg.lambda_sharpness * F.relu(0.5 * std_true - std_pred)
            var_match_loss = cfg.lambda_var_match * torch.abs(std_pred - std_true)
        else:
            sharpness_loss = torch.zeros((), device=final_pred.device)
            var_match_loss = torch.zeros((), device=final_pred.device)

        # 6. Extremes-recall focado em tempestades
        fn_r10 = r10_mask & (final_pred < 10.0)
        fn_r20 = r20_mask & (final_pred < 20.0)
        miss = F.relu(y_reg_true - final_pred)
        if y_reg_true.numel() > 0:
            loss_extreme_fn = cfg.lambda_extreme_fn * torch.mean(
                fn_r10.float() * miss * cfg.extreme_fn_boost_r10
                + fn_r20.float() * miss * cfg.extreme_fn_boost_r20
            )
        else:
            loss_extreme_fn = torch.zeros((), device=final_pred.device)

        # 8. Physics (off no V7 — mantém assinatura/compat)
        loss_physics = torch.zeros((), device=final_pred.device)

        loss_total = (
            cfg.alpha_bce * loss_occ
            + cfg.beta_mse * loss_int
            + cfg.lambda_dry * loss_dry
            + loss_final
            + sharpness_loss
            + var_match_loss
            + loss_extreme_fn
            + cfg.lambda_physics * loss_physics
        )

        metrics = {
            "loss_total": float(loss_total.item()),
            "bce_loss": float(loss_occ.item()),
            "mse_loss": float(loss_int.item()),
            "physics_loss": 0.0,
            "dry_loss": float(loss_dry.item()),
            "final_loss": float(loss_final.item()),
            "sharpness_loss": float(sharpness_loss.item()),
            "var_match_loss": float(var_match_loss.item()),
            "extreme_fn_loss": float(loss_extreme_fn.item()),
            "n_violations": 0,
            "n_storms": int(r10_mask.sum().item()),
            "wet_ratio": float(wet_mask.float().mean().item()),
        }
        return loss_total, metrics
