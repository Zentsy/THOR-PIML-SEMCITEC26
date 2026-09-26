"""
THOR-PIML — Physics-Informed Loss V2 (Sprint S3)
==================================================
Função de perda composta com física honesta:

  L_total = α·L_focal(γ=2.0) + β·L_mse·w_storm + λ·L_physics

Baseado em:
- PIML com Clausius-Clapeyron (Beucler et al. 2021; Kashinath et al. 2021)
  → q_sat via Tetens, com psfc real (não H_eff fixo)
- Focal Loss (Lin et al. 2017) para desbalanceamento seco/chuvoso
- Softplus barrier para evitar dead zones (ReLU → Softplus)

Mudanças S3 vs V1:
- W_max agora usa psfc real (ERA5-Land 9km), não H_eff=2500m fixo
- Hiperparâmetros lidos de TrainingConfig (gamma_focal=2.0, storm_weight=3.0)
- denormalize por nome (tmean/rh/psfc), não índice mágico 0/3
- beta_softplus 0.5→1.0 (menos suave), lambda 0.1→0.5 (relevante)
"""
from __future__ import annotations
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import PhysicsConstants, TrainingConfig
from src.preprocessing import RobustClimateScaler


class THORPhysicsLoss(nn.Module):
    def __init__(
        self,
        scaler: RobustClimateScaler,
        physics_config: PhysicsConstants = PhysicsConstants(),
        training_config: TrainingConfig = TrainingConfig(),
        # Overrides opcionais (se None, lê de training_config)
        gamma_focal: Optional[float] = None,
        alpha_focal: Optional[float] = None,
        beta_softplus: Optional[float] = None,
        storm_threshold: Optional[float] = None,
        storm_weight: Optional[float] = None,
        # Índices legados (mantidos por compatibilidade, mas preferir nomes)
        temp_idx: int = 0,
        rh_idx: int = 3,
        psfc_idx: int = 4,
    ):
        super().__init__()
        self.scaler = scaler
        self.phys = physics_config
        self.cfg = training_config

        # Lê de TrainingConfig se não houver override explícito (S3)
        self.gamma_focal = gamma_focal if gamma_focal is not None else training_config.gamma_focal
        self.alpha_focal = alpha_focal if alpha_focal is not None else training_config.alpha_focal
        self.beta_softplus = beta_softplus if beta_softplus is not None else training_config.beta_softplus
        self.storm_threshold = storm_threshold if storm_threshold is not None else training_config.storm_threshold
        self.storm_weight = storm_weight if storm_weight is not None else training_config.storm_weight
        self.temp_idx = temp_idx
        self.rh_idx = rh_idx
        self.psfc_idx = psfc_idx

        self.mse_loss = nn.MSELoss(reduction="none")

    # ---- Física: W_max honesto com psfc ----

    def compute_thermodynamic_limit_wmax(
        self,
        tmean_celsius: Tensor,
        rh_pct: Tensor,
        psfc_hpa: Optional[Tensor] = None,
    ) -> Tensor:
        """Teto precipitable water W_max (mm) — física honesta S3.

        V1: W_max = Mw/R* * e_s(T) * RH * H_eff / rho *10  (H_eff=2500m fixo, sem psfc)
        V2: q_sat = 0.622*e / (psfc -0.378*e),  W_max ≈ q_sat * psfc/g *10  (usa psfc real)

        Onde:
          e_s(T) = e0 * exp(a*T/(b+T))  [hPa, Tetens]
          e = e_s * RH  [hPa, pressão vapor real]
          q_sat em kg/kg, W_max em mm (kg/m²).

        Se psfc não fornecido, fallback para H_eff (compatível V1) com warning implícito.
        """
        e_s = self.phys.tetens_e0 * torch.exp(
            (self.phys.tetens_a * tmean_celsius) / (self.phys.tetens_b + tmean_celsius)
        )
        rh_frac = torch.clamp(rh_pct / 100.0, 0.0, 1.0)
        e_act = e_s * rh_frac  # hPa

        if psfc_hpa is not None:
            # Fórmula com psfc real (ERA5-Land 9km) — S3
            psfc_safe = torch.clamp(psfc_hpa, min=800.0, max=1050.0)
            # q_sat (kg/kg)
            denom = torch.clamp(psfc_safe - 0.378 * e_act, min=1.0)
            q_sat = (0.622 * e_act) / denom  # kg/kg
            # Coluna de água: W = q * psfc / g  [kg/m² = mm]
            w_max = (q_sat * psfc_safe * 100.0) / self.phys.g  # psfc hPa→Pa (×100)
        else:
            # Fallback V1 (H_eff fixo) — mantido para compatibilidade
            w_max = (
                self.phys.Mw_over_Rstar * e_s * rh_frac * self.phys.H_eff / self.phys.rho_w
            ) * 10.0

        return torch.clamp(w_max, min=0.0, max=500.0)

    # ---- Denormalização por nome (S2) ----

    def _denorm(self, x_batch: Tensor, feature: str, fallback_idx: int) -> Tensor:
        last = x_batch[:, -1, :]
        # Tenta por nome primeiro (S2), fallback por índice (V1)
        if getattr(self.scaler, "feature_names", None) is not None and feature in self.scaler.feature_names:
            idx = self.scaler.feature_names.index(feature)
            col_slice = last[:, idx : idx + 1]
            return self.scaler.denormalize_column(col_slice, idx)
        # Fallback índice
        col_slice = last[:, fallback_idx : fallback_idx + 1]
        if hasattr(self.scaler, "denormalize_column") and getattr(self.scaler, "fitted", False):
            return self.scaler.denormalize_column(col_slice, fallback_idx)
        elif getattr(self.scaler, "data_min_", None) is not None:
            v_min = torch.tensor(self.scaler.data_min_[fallback_idx], device=x_batch.device, dtype=x_batch.dtype)
            v_max = torch.tensor(self.scaler.data_max_[fallback_idx], device=x_batch.device, dtype=x_batch.dtype)
            return col_slice * (v_max - v_min) + v_min
        else:
            # Último fallback físico — nunca deveria chegar aqui em V2
            if feature == "tmean":
                return col_slice * 35.0
            elif feature == "rh":
                return col_slice * 100.0
            elif feature == "psfc":
                return col_slice * 200.0 + 850.0
            return col_slice

    def denormalize_feature(self, x_batch: Tensor, feature_idx: int) -> Tensor:
        """Compatibilidade V1: denormalize por índice (usado em testes legados)."""
        mapping = {self.temp_idx: "tmean", self.rh_idx: "rh", self.psfc_idx: "psfc"}
        feature = mapping.get(feature_idx, f"idx_{feature_idx}")
        return self._denorm(x_batch, feature, feature_idx)

    # ---- Forward ----

    def forward(
        self,
        prob_occ: Tensor,
        intensity: Tensor,
        final_pred: Tensor,
        y_class_true: Tensor,
        y_reg_true: Tensor,
        x_batch: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        # 1. Focal Loss (γ=2.0, α=0.25) — agora ativa por padrão via TrainingConfig
        prob_occ_safe = torch.clamp(prob_occ, min=1e-7, max=1.0 - 1e-7)
        logits = torch.logit(prob_occ_safe)
        bce_raw = F.binary_cross_entropy_with_logits(logits, y_class_true, reduction="none")
        p_t = torch.where(y_class_true >= 0.5, prob_occ_safe, 1.0 - prob_occ_safe)
        focal_weight = (1.0 - p_t) ** self.gamma_focal
        if self.alpha_focal is not None:
            alpha_t = torch.where(y_class_true >= 0.5, self.alpha_focal, 1.0 - self.alpha_focal)
            loss_bce = torch.mean(alpha_t * focal_weight * bce_raw)
        else:
            loss_bce = torch.mean(focal_weight * bce_raw)

        # 2. Heavy Storm Weighting (>20 mm, peso 3.0)
        sq_err = (final_pred - y_reg_true) ** 2
        storm_mask = y_reg_true > self.storm_threshold
        weight_tensor = torch.ones_like(y_reg_true)
        if storm_mask.any():
            w_val = torch.tensor(self.storm_weight, device=y_reg_true.device, dtype=y_reg_true.dtype)
            weight_tensor = torch.where(storm_mask, w_val, weight_tensor)
        loss_mse = torch.mean(weight_tensor * sq_err)

        # 3. Física PIML: W_max com psfc real + Softplus β=1.0
        tmean_c = self._denorm(x_batch, "tmean", self.temp_idx)
        rh_pct = self._denorm(x_batch, "rh", self.rh_idx)
        psfc_hpa = self._denorm(x_batch, "psfc", self.psfc_idx)
        # Usa psfc se scaler tem psfc (V2), senão None (fallback H_eff)
        use_psfc = psfc_hpa is not None and not torch.isnan(psfc_hpa).all()
        w_max = self.compute_thermodynamic_limit_wmax(tmean_c, rh_pct, psfc_hpa if use_psfc else None)
        smooth_violation = F.softplus(final_pred - w_max, beta=self.beta_softplus)
        loss_physics = torch.mean(smooth_violation ** 2)

        # 4. Anti-garoa: penaliza chuva em dias secos (y_true==0) — S6 fix re-treino
        # Sem isso, hurdle contínuo (prob*intensity) com intensity Softplus>0 gera CWD 237
        lambda_dry = getattr(self.cfg, "lambda_dry", 0.0)
        if lambda_dry > 0:
            dry_mask = (y_reg_true == 0)  # ou <0.1 mm
            if dry_mask.any():
                # Penaliza final_pred>0 em dias secos (L2)
                loss_dry = torch.mean(torch.where(dry_mask, final_pred ** 2, torch.zeros_like(final_pred)))
            else:
                loss_dry = torch.tensor(0.0, device=final_pred.device)
        else:
            loss_dry = torch.tensor(0.0, device=final_pred.device)

        loss_total = (
            self.cfg.alpha_bce * loss_bce + self.cfg.beta_mse * loss_mse + self.cfg.lambda_physics * loss_physics + lambda_dry * loss_dry
        )
        n_violations = int((final_pred > w_max).sum().item())
        n_storms = int(storm_mask.sum().item())

        metrics_dict = {
            "loss_total": loss_total.item(),
            "bce_loss": loss_bce.item(),
            "focal_loss": loss_bce.item(),
            "mse_loss": loss_mse.item(),
            "physics_loss": loss_physics.item(),
            "dry_loss": float(loss_dry.item()) if "loss_dry" in locals() else 0.0,
            "n_violations": n_violations,
            "n_storms": n_storms,
            "w_max_mean": float(w_max.mean().item()),
            "w_max_min": float(w_max.min().item()),
        }
        return loss_total, metrics_dict
