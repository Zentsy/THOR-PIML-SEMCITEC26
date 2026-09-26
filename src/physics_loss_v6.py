"""
THOR-PIML V6 FINAL ALL-IN — Loss corajosa (risk-seeking)
Resolve o problema do "profissional medíocre com medo de arriscar" (regression to the mean, Flaw of Averages, loss aversion).

Filosofia:
- MSE simétrico pune superestimar tempestade tanto quanto subestimar → modelo aprende média segura 3-4mm, nunca R10/R20
- Solução: asymmetric + quantile 0.85 + storm boost que pune MAIS quem subestima tempestade

Referências filosofia:
- Kahneman & Tversky Prospect Theory (loss aversion)
- Savage Flaw of Averages
- Goodhart/Campbell (NSE baseado em MSE vira alvo ruim)
"""

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config_v6 import TrainingConfigV6
from src.preprocessing import RobustClimateScaler
from src.config import PhysicsConstants


class THORLossV6(nn.Module):
    def __init__(
        self,
        scaler: RobustClimateScaler,
        physics_config: PhysicsConstants = PhysicsConstants(),
        training_config: TrainingConfigV6 = TrainingConfigV6(),
        occurrence_threshold: float = 1.0,
        temp_idx: int = 0,
        rh_idx: int = 3,
        psfc_idx: int = 4,
    ):
        super().__init__()
        self.scaler = scaler
        self.phys = physics_config
        self.cfg = training_config
        self.occurrence_thresh = occurrence_threshold
        self.temp_idx = temp_idx
        self.rh_idx = rh_idx
        self.psfc_idx = psfc_idx

    # Compatibilidade com run_thor_training.py
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
        mapping = {self.temp_idx: "tmean", self.rh_idx: "rh", self.psfc_idx: "psfc"}
        feature = mapping.get(feature_idx, f"idx_{feature_idx}")
        return self._denorm(x_batch, feature, feature_idx)

    def compute_thermodynamic_limit_wmax(self, tmean_celsius: Tensor, rh_pct: Tensor, psfc_hpa: Tensor | None = None) -> Tensor:
        e_s = self.phys.tetens_e0 * torch.exp(
            (self.phys.tetens_a * tmean_celsius) / (self.phys.tetens_b + tmean_celsius)
        )
        rh_frac = torch.clamp(rh_pct / 100.0, 0.0, 1.0)
        e_act = e_s * rh_frac
        if psfc_hpa is not None:
            psfc_safe = torch.clamp(psfc_hpa, min=800.0, max=1050.0)
            denom = torch.clamp(psfc_safe - 0.378 * e_act, min=1.0)
            q_sat = (0.622 * e_act) / denom
            w_max = (q_sat * psfc_safe * 100.0) / self.phys.g
        else:
            w_max = (self.phys.Mw_over_Rstar * e_s * rh_frac * self.phys.H_eff / self.phys.rho_w) * 10.0
        return torch.clamp(w_max, min=0.0, max=500.0)

    def forward(
        self,
        prob_occ: Tensor,
        intensity: Tensor,
        final_pred: Tensor,
        y_class_true: Tensor,
        y_reg_true: Tensor,
        x_batch: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:

        # 1. y_class com threshold 1.0mm WMO
        y_class_1mm = (y_reg_true >= self.occurrence_thresh).float()
        wet_mask = y_class_1mm >= 0.5
        dry_mask = ~wet_mask

        # 2. Focal Loss ocorrência (threshold 1.0)
        prob_safe = torch.clamp(prob_occ, min=1e-7, max=1 - 1e-7)
        logits = torch.logit(prob_safe)
        bce_raw = F.binary_cross_entropy_with_logits(logits, y_class_1mm, reduction="none")
        p_t = torch.where(y_class_1mm >= 0.5, prob_safe, 1 - prob_safe)
        focal_weight = (1.0 - p_t) ** self.cfg.gamma_focal
        alpha_t = torch.where(y_class_1mm >= 0.5, self.cfg.alpha_focal, 1.0 - self.cfg.alpha_focal)
        loss_occ = torch.mean(alpha_t * focal_weight * bce_raw)

        # 3. Intensity — LOSS BALANCEADA V6d: quantile 0.65 + storm boost moderado
        # V6c era q=0.85 (muito corajosa) -> R10 1252 vs 287, bias +9, NSE -3.7
        # V6d usa q=0.65 (equilibrada) -> pune subestimação 1.8× mais que superestimação, não 5.6×
        q = 0.65
        error_int = y_reg_true - intensity
        huber_int = F.huber_loss(intensity, y_reg_true, delta=5.0, reduction='none')
        quantile_int = torch.where(error_int >= 0, q * huber_int, (1 - q) * huber_int)

        weight = torch.ones_like(y_reg_true)
        r10_mask = y_reg_true >= 10.0
        r20_mask = y_reg_true >= 20.0
        storm_mask = y_reg_true >= self.cfg.storm_threshold

        # Peso moderado: R10 1.3×, R20 2.0× (antes 1.5× e 3.0×)
        weight = torch.where(r10_mask, torch.tensor(1.3, device=weight.device), weight)
        weight = torch.where(r20_mask, torch.tensor(self.cfg.storm_weight, device=weight.device), weight)

        # Underestimation boost moderado: R10 under 1.3×, R20 under 1.5× (antes 2.0× e 1.5× = 6× total)
        underest_mask = (intensity < y_reg_true) & (y_reg_true >= 10.0)
        underest_r20 = (intensity < y_reg_true) & (y_reg_true >= 20.0)
        weight = torch.where(underest_mask, weight * 1.3, weight)  # 1.3→1.69 R10
        weight = torch.where(underest_r20, weight * 1.5, weight)  # 2.0→3.0 R20

        if wet_mask.any():
            loss_int = torch.mean(weight[wet_mask] * quantile_int[wet_mask])
        else:
            loss_int = torch.tensor(0.0, device=intensity.device)

        # 4. Dry loss — BCEWithLogits com peso maior (0.3) para evitar colapso F1=0
        if dry_mask.any() and self.cfg.lambda_dry > 0:
            prob_dry = prob_occ[dry_mask]
            prob_dry_safe = torch.clamp(prob_dry, min=1e-7, max=1-1e-7)
            logits_dry = torch.logit(prob_dry_safe)
            loss_dry = F.binary_cross_entropy_with_logits(logits_dry, torch.zeros_like(logits_dry))
        else:
            loss_dry = torch.tensor(0.0, device=prob_occ.device)

        # 5. Final loss — também quantile balanceada + storm boost, peso 0.2 (era 0.3)
        error_final = y_reg_true - final_pred
        huber_final = F.huber_loss(final_pred, y_reg_true, delta=5.0, reduction='none')
        quantile_final = torch.where(error_final >= 0, q * huber_final, (1 - q) * huber_final)
        loss_final = torch.mean(weight * quantile_final * 0.2)

        # 6. Sharpness loss — encoraja variância, peso 0.05 (era 0.1) para não explodir R10
        if y_reg_true.numel() > 1:
            std_true = torch.std(y_reg_true)
            std_pred = torch.std(final_pred)
            sharpness_loss = F.relu(0.5 * std_true - std_pred)
            sharpness_loss = sharpness_loss * 0.05
        else:
            sharpness_loss = torch.tensor(0.0, device=final_pred.device)

        loss_physics = torch.tensor(0.0, device=final_pred.device)

        loss_total = (
            self.cfg.alpha_bce * loss_occ
            + self.cfg.beta_mse * loss_int
            + self.cfg.lambda_dry * loss_dry
            + loss_final
            + sharpness_loss
            + self.cfg.lambda_physics * loss_physics
        )

        metrics = {
            "loss_total": loss_total.item(),
            "bce_loss": loss_occ.item(),
            "focal_loss": loss_occ.item(),
            "mse_loss": loss_int.item(),
            "physics_loss": loss_physics.item(),
            "dry_loss": loss_dry.item(),
            "final_loss": loss_final.item(),
            "sharpness_loss": float(sharpness_loss.item()) if isinstance(sharpness_loss, Tensor) else float(sharpness_loss),
            "n_violations": 0,
            "n_storms": int(storm_mask.sum().item()),
            "wet_ratio": float(wet_mask.float().mean().item()),
        }
        return loss_total, metrics
