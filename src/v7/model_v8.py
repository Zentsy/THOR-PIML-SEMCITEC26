"""
THOR-PIML V8 — Híbrida espacial: CNN 2D sinótica + tronco híbrido V7
====================================================================
Consome por dia os campos do domínio ERA5 (data/era5pl_domain_daily_1981_2026.nc,
produzido pelo fetch_era5_pl.py): z500, u700, v700, q700, w500 (H×W células).

Fluxo:
  spatial (B, T, H, W, C) → SpatialEncoder (CNN por dia) → emb sinótica (B, T, d_syn)
  concat com superfície (B, T, F) → (B, T, F + d_syn) → tronco V7 (LSTM||TCN →
  gated fusion → SDPA causal → heads hurdle).

A CNN aprende gradientes/vorticidade/dissonância direto do campo (em vez de
features manuais de domínio) — é o passo "espacial + sinótico" do diagnóstico V6d.
Física CC com TCWV real volta via physics_loss_v8 (barreira softplus).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.model_v6 import ResBiLSTMEncoderV6, SDPAAttention
from src.v7.model_v7 import GatedFusion, TCNEncoderV7


class SpatialSynopticEncoder(nn.Module):
    """CNN 2D por dia: (B*T, C, H, W) → (B*T, d_syn). Gradientes → pooling."""

    def __init__(self, in_channels: int, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class THORSpatialHybridModel(nn.Module):
    """V8: forward(x_surface, x_spatial) → prob/intensity/final (contrato hurdle)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # spatial config
        self.spatial_channels = getattr(config, "spatial_channels", 5)
        self.spatial_embed = getattr(config, "spatial_embed_dim", 64)
        self.spatial_cnn = SpatialSynopticEncoder(
            self.spatial_channels, self.spatial_embed,
            dropout=getattr(config, "spatial_dropout", 0.2),
        )
        trunk_input = config.n_features + self.spatial_embed

        # Tronco temporal = mesma híbrida V7 (switches de ablação valem igual)
        use_lstm = getattr(config, "use_lstm_branch", True)
        use_tcn = getattr(config, "use_tcn_branch", True)
        if not use_lstm and not use_tcn:
            raise ValueError("V8: pelo menos um ramo temporal ativo")
        self.lstm = (
            ResBiLSTMEncoderV6(
                input_size=trunk_input,
                hidden_size=config.lstm_hidden,
                num_layers=config.lstm_layers,
                dropout=config.lstm_dropout,
            )
            if use_lstm
            else None
        )
        lstm_dim = self.lstm.output_dim if self.lstm else 0
        self.tcn = (
            TCNEncoderV7(
                in_features=trunk_input,
                channels=config.tcn_channels,
                kernels=config.tcn_kernels,
                dilations=config.tcn_dilations,
                dropout=config.tcn_dropout,
            )
            if use_tcn
            else None
        )
        tcn_dim = self.tcn.output_dim if self.tcn else 0

        fusion_dim = config.fusion_dim
        if use_lstm and use_tcn:
            self.fusion = GatedFusion(lstm_dim, tcn_dim, fusion_dim)
        elif use_lstm:
            self.fusion = nn.Linear(lstm_dim, fusion_dim) if lstm_dim != fusion_dim else nn.Identity()
        else:
            self.fusion = nn.Linear(tcn_dim, fusion_dim) if tcn_dim != fusion_dim else nn.Identity()

        self.attention = (
            SDPAAttention(
                fusion_dim, heads=getattr(config, "attn_heads", 8),
                dropout=getattr(config, "attn_dropout", 0.2),
                causal=getattr(config, "attn_causal", True),
            )
            if getattr(config, "use_attention", True)
            else None
        )
        self.layer_norm = nn.LayerNorm(fusion_dim)

        occ_hidden = getattr(config, "occurrence_hidden", 128)
        int_hidden = getattr(config, "intensity_hidden", 192)
        self.occurrence_head = nn.Sequential(
            nn.Linear(fusion_dim, occ_hidden), nn.LayerNorm(occ_hidden), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(occ_hidden, occ_hidden // 2), nn.SiLU(),
            nn.Linear(occ_hidden // 2, 1), nn.Sigmoid(),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(fusion_dim, int_hidden), nn.LayerNorm(int_hidden), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(int_hidden, int_hidden // 2), nn.SiLU(),
            nn.Linear(int_hidden // 2, 1), nn.Softplus(),
        )

    def encode(self, x_surface: Tensor, x_spatial: Tensor) -> Tensor:
        # x_surface (B,T,F) | x_spatial (B,T,H,W,C)
        B, T = x_surface.shape[:2]
        sp = x_spatial.reshape(B * T, *x_spatial.shape[2:]).permute(0, 3, 1, 2)  # (B*T,C,H,W)
        syn = self.spatial_cnn(sp).reshape(B, T, -1)  # (B,T,d_syn)
        x = torch.cat([x_surface, syn], dim=-1)
        outs = []
        if self.lstm is not None:
            outs.append(self.lstm(x)[0])
        if self.tcn is not None:
            outs.append(self.tcn(x))
        fused = self.fusion(outs[0], outs[1]) if len(outs) == 2 else self.fusion(outs[0])
        if self.attention is not None:
            fused = fused + self.attention(fused)
        return self.layer_norm(fused)

    def forward(self, x_surface: Tensor, x_spatial: Tensor, return_components: bool = False):
        fused = self.encode(x_surface, x_spatial)
        last = fused[:, -1, :]
        prob = self.occurrence_head(last)
        intensity = self.intensity_head(last)
        final = prob * intensity
        if return_components:
            return prob, intensity, final
        return final
