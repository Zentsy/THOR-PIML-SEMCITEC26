"""
THOR-PIML V6 — Arquitetura leve + SDPA attention opcional

Mudanças vs V5:
- ResBiLSTM com dropout em TODAS as camadas (não só N-1) — fixa overfit última camada
- Forget bias fix correto (só bias_ih, não bias_hh)
- Attention: use_attention=False (Identity) para V6a, ou SDPA MultiheadAttention para V6b
- Heads balanceados 96/128, dropout 0.2
- Taylor scale fix (já aplicado em taylor_attention.py)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

from src.taylor_attention import TaylorLinearAttention


class ResBiLSTMCellV6(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.proj = nn.Linear(input_size, self.output_dim) if input_size != self.output_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, hx=None):
        lstm_out, state = self.lstm(x, hx)
        residual = self.proj(x)
        out = self.layer_norm(lstm_out + residual)
        out = self.dropout(out)
        return out, state


class ResBiLSTMEncoderV6(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, bidirectional: bool = False, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_size
        for i in range(num_layers):
            # V6 fix: dropout em TODAS, não só N-1
            layer = ResBiLSTMCellV6(in_dim, hidden_size, bidirectional, dropout)
            self.layers.append(layer)
            in_dim = layer.output_dim
        self.output_dim = in_dim

    def forward(self, x, hx=None):
        out = x
        states = []
        for layer in self.layers:
            out, s = layer(out, None)
            states.append(s)
        return out, states


class SDPAAttention(nn.Module):
    """MultiheadAttention com SDPA + causal mask, para seq_len=30."""
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        # causal mask: upper triangular True = masked
        if self.causal:
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            # MultiheadAttention espera attn_mask com -inf ou bool, vamos usar bool
            out, _ = self.mha(x, x, x, attn_mask=mask, need_weights=False)
        else:
            out, _ = self.mha(x, x, x, need_weights=False)
        return self.norm(out)


class THORPIMLModelV6(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lstm = ResBiLSTMEncoderV6(
            input_size=config.n_features,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            bidirectional=getattr(config, 'lstm_bidirectional', False),
            dropout=config.lstm_dropout,
        )
        lstm_dim = self.lstm.output_dim

        # Attention
        use_attn = getattr(config, 'use_attention', False)
        attn_type = getattr(config, 'attention_type', 'taylor')  # 'taylor' ou 'sdpa'
        if not use_attn:
            self.attention = nn.Identity()
        else:
            if attn_type == "sdpa":
                self.attention = SDPAAttention(
                    dim=lstm_dim,
                    heads=getattr(config, 'attn_heads', 4),
                    dropout=getattr(config, 'attn_dropout', 0.1),
                    causal=getattr(config, 'attn_causal', True),
                )
            else:
                self.attention = TaylorLinearAttention(
                    dim=lstm_dim,
                    dim_head=getattr(config, 'attn_dim_head', 32),
                    heads=getattr(config, 'attn_heads', 4),
                    causal=getattr(config, 'attn_causal', True),
                    dropout=getattr(config, 'attn_dropout', 0.1),
                    remove_even_power_dups=getattr(config, 'remove_even_power_dups', False),
                    prenorm=True,
                )

        self.layer_norm = nn.LayerNorm(lstm_dim)

        # Heads balanceados
        occ_hidden = getattr(config, 'occurrence_hidden', 96)
        int_hidden = getattr(config, 'intensity_hidden', 128)

        self.occurrence_head = nn.Sequential(
            nn.Linear(lstm_dim, occ_hidden),
            nn.LayerNorm(occ_hidden),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(occ_hidden, occ_hidden // 2),
            nn.SiLU(),
            nn.Linear(occ_hidden // 2, 1),
            nn.Sigmoid(),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(lstm_dim, int_hidden),
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
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                # V6 fix: só bias_ih tem forget=1, bias_hh zero
                if "bias_ih" in name:
                    hidden = param.shape[0] // 4
                    nn.init.zeros_(param)
                    with torch.no_grad():
                        param[hidden:2*hidden].fill_(1.0)
                else:
                    nn.init.zeros_(param)

        for head in [self.occurrence_head, self.intensity_head]:
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, return_components: bool = False):
        lstm_out, _ = self.lstm(x)
        if isinstance(self.attention, nn.Identity):
            attn_out = torch.zeros_like(lstm_out)
        else:
            attn_out = self.attention(lstm_out)
        combined = self.layer_norm(lstm_out + attn_out)
        last = combined[:, -1, :]

        prob = self.occurrence_head(last)
        intensity = self.intensity_head(last)
        final = prob * intensity

        if return_components:
            return prob, intensity, final
        return final
