"""
THOR-PIML V7 — Arquitetura Híbrida LSTM + TCN (encoder temporal)
=================================================================
Fusão gated dos dois ramos + SDPA causal sobre a sequência fundida + heads
hurdle (contrato idêntico ao V6: forward(x) → prob, intensity, final=prob*int).

Por que híbrido (docs/ESTADO_ATUAL_FINAL_V6.md §9 + V7_HYBRID_ARCHITECTURE.md):
- LSTM: memória de longo prazo / inércia (14 dias), bom para persistência.
- TCN causal multi-escala: detecta padrões de forma (frente fria 2-4d, onda
  7-14d) com gradiente estável e paralelismo; kernels {3,5,7} paralelos.
- Fusão gated por timestep: a rede aprende QUANDO confiar em cada ramo
  (ex.: TCN em quebra sinótica, LSTM em persistência de garoa).

Ablação sem mudar classe: config.use_lstm_branch / use_tcn_branch.
Causalidade: LSTM unidirecional + convs causais (pad à esquerda, chomp à
direita) + máscara causal na SDPA — dia t nunca vê t+1 (lição V3).

Refs: Bai et al. 2018 (TCN empírico vs LSTM); DEMM KDD'22 (hurdle DL);
ver docs/PAPER_REFERENCES.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.model_v6 import ResBiLSTMEncoderV6, SDPAAttention


class CausalConv1d(nn.Module):
    """Conv1d causal: pad (kernel-1)*dilation à ESQUERDA e corta à direita."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, dilation=dilation, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:  # x: (B, C, T)
        x = torch.nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class MultiScaleTCNBlock(nn.Module):
    """Bloco residual: convs causais k∈{3,5,7} em paralelo → concat → fuse 1x1.

    Camada por timestep com LayerNorm sobre canais; dilação crescente entre
    blocos dá campo receptivo exponencial (Bai et al. 2018).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernels: list[int],
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [CausalConv1d(in_ch, out_ch, k, dilation) for k in kernels]
        )
        total = out_ch * len(kernels)
        self.fuse = nn.Conv1d(total, out_ch, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(out_ch)
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:  # x: (B, T, C)
        residual = self.residual(x.transpose(1, 2))  # (B, C_out, T)
        y = x.transpose(1, 2)                        # (B, C, T)
        y = torch.cat([conv(y) for conv in self.convs], dim=1)
        y = self.fuse(y)
        y = self.act(y)
        y = self.dropout(y)
        y = y + residual
        y = self.norm(y.transpose(1, 2))             # (B, T, C_out)
        return y


class TCNEncoderV7(nn.Module):
    """Stack de blocos multi-escala com dilações crescentes. Output (B, T, C)."""

    def __init__(
        self,
        in_features: int,
        channels: list[int],
        kernels: list[int],
        dilations: list[int],
        dropout: float = 0.2,
    ):
        super().__init__()
        if len(channels) != len(dilations):
            raise ValueError(
                f"TCN: channels ({len(channels)}) e dilations ({len(dilations)}) devem ter mesmo comprimento"
            )
        blocks = []
        in_ch = in_features
        for ch, dil in zip(channels, dilations):
            blocks.append(MultiScaleTCNBlock(in_ch, ch, kernels, dil, dropout))
            in_ch = ch
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = channels[-1]

    def forward(self, x: Tensor) -> Tensor:  # (B, T, F) → (B, T, C)
        y = x
        for block in self.blocks:
            y = block(y)
        return y


class GatedFusion(nn.Module):
    """Fusão gated por timestep: g = σ(W[lstm;tcn]); out = g·lstm + (1-g)·tcn."""

    def __init__(self, lstm_dim: int, tcn_dim: int, out_dim: int):
        super().__init__()
        self.lstm_proj = nn.Linear(lstm_dim, out_dim) if lstm_dim != out_dim else nn.Identity()
        self.tcn_proj = nn.Linear(tcn_dim, out_dim) if tcn_dim != out_dim else nn.Identity()
        self.gate = nn.Linear(2 * out_dim, out_dim)

    def forward(self, lstm_out: Tensor, tcn_out: Tensor) -> Tensor:
        l = self.lstm_proj(lstm_out)
        t = self.tcn_proj(tcn_out)
        g = torch.sigmoid(self.gate(torch.cat([l, t], dim=-1)))
        return g * l + (1.0 - g) * t


class THORHybridModel(nn.Module):
    """Híbride LSTM+TCN com heads hurdle. Contrato V6: (B,T,F) → prob/intensity/final."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        use_lstm = getattr(config, "use_lstm_branch", True)
        use_tcn = getattr(config, "use_tcn_branch", True)
        if not use_lstm and not use_tcn:
            raise ValueError("THORHybridModel: pelo menos um ramo deve estar ativo")
        self.use_lstm = use_lstm
        self.use_tcn = use_tcn

        if use_lstm:
            self.lstm = ResBiLSTMEncoderV6(
                input_size=config.n_features,
                hidden_size=config.lstm_hidden,
                num_layers=config.lstm_layers,
                bidirectional=getattr(config, "lstm_bidirectional", False),
                dropout=config.lstm_dropout,
            )
            lstm_dim = self.lstm.output_dim
        else:
            self.lstm = None
            lstm_dim = 0

        if use_tcn:
            self.tcn = TCNEncoderV7(
                in_features=config.n_features,
                channels=config.tcn_channels,
                kernels=config.tcn_kernels,
                dilations=config.tcn_dilations,
                dropout=config.tcn_dropout,
            )
            tcn_dim = self.tcn.output_dim
        else:
            self.tcn = None
            tcn_dim = 0

        fusion_dim = config.fusion_dim
        if use_lstm and use_tcn:
            self.fusion = GatedFusion(lstm_dim, tcn_dim, fusion_dim)
        elif use_lstm:
            self.fusion = (
                nn.Linear(lstm_dim, fusion_dim) if lstm_dim != fusion_dim else nn.Identity()
            )
        else:
            self.fusion = (
                nn.Linear(tcn_dim, fusion_dim) if tcn_dim != fusion_dim else nn.Identity()
            )

        use_attn = getattr(config, "use_attention", True)
        if use_attn:
            self.attention = SDPAAttention(
                dim=fusion_dim,
                heads=getattr(config, "attn_heads", 8),
                dropout=getattr(config, "attn_dropout", 0.2),
                causal=getattr(config, "attn_causal", True),
            )
        else:
            self.attention = None
        self.layer_norm = nn.LayerNorm(fusion_dim)

        occ_hidden = getattr(config, "occurrence_hidden", 128)
        int_hidden = getattr(config, "intensity_hidden", 192)
        self.occurrence_head = nn.Sequential(
            nn.Linear(fusion_dim, occ_hidden),
            nn.LayerNorm(occ_hidden),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(occ_hidden, occ_hidden // 2),
            nn.SiLU(),
            nn.Linear(occ_hidden // 2, 1),
            nn.Sigmoid(),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(fusion_dim, int_hidden),
            nn.LayerNorm(int_hidden),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(int_hidden, int_hidden // 2),
            nn.SiLU(),
            nn.Linear(int_hidden // 2, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        if self.lstm is not None:
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias_ih" in name:
                    hidden = param.shape[0] // 4
                    nn.init.zeros_(param)
                    with torch.no_grad():
                        param[hidden : 2 * hidden].fill_(1.0)  # forget gate
                elif "bias" in name:
                    nn.init.zeros_(param)
        for head in [self.occurrence_head, self.intensity_head]:
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def encode(self, x: Tensor) -> Tensor:
        """(B, T, F) → representação fundida por timestep (B, T, fusion_dim)."""
        outs = []
        if self.lstm is not None:
            lstm_out, _ = self.lstm(x)
            outs.append(lstm_out)
        if self.tcn is not None:
            outs.append(self.tcn(x))
        if len(outs) == 2:
            fused = self.fusion(outs[0], outs[1])
        else:
            fused = self.fusion(outs[0])
        if self.attention is not None:
            attn_out = self.attention(fused)
            fused = fused + attn_out  # resíduo (SDPAAttention já tem LayerNorm interno)
        return self.layer_norm(fused)

    def forward(self, x: Tensor, return_components: bool = False):
        fused = self.encode(x)
        last = fused[:, -1, :]
        prob = self.occurrence_head(last)
        intensity = self.intensity_head(last)
        final = prob * intensity
        if return_components:
            return prob, intensity, final
        return final
