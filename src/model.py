"""
THOR-PIML — Arquitetura de Alta Capacidade (Modelo Matemático Puro Contínuo)
=============================================================================
Modelo THOR-PIML sem truncamentos artificiais nem gambiarras de pós-processamento:
- Head de Ocorrência (Sigmoid): P(Chuva > 0) in [0, 1]
- Head de Intensidade (Softplus): y_intensity >= 0.0 mm
- Produto de Expectância Hurdle Pura: y_pred = P(Occ) * y_intensity
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

from src.config import ModelConfig
from src.taylor_attention import TaylorLinearAttention


class ResBiLSTMCell(nn.Module):
    """
    Bloco de Camada Recorrente (BiLSTM / LSTM) com Conexão Residual (Skip Connection)
    e Normalização por Camada (LayerNorm).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.output_dim = hidden_size * 2 if bidirectional else hidden_size
        if input_size != self.output_dim:
            self.proj = nn.Linear(input_size, self.output_dim)
        else:
            self.proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        hx: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        lstm_out, state = self.lstm(x, hx)
        residual = self.proj(x)
        out = self.layer_norm(lstm_out + residual)
        out = self.dropout(out)
        return out, state


class ResBiLSTMEncoder(nn.Module):
    """
    Encoder Recorrente Profundo com Conexões Residuais inter-camadas (ResBiLSTM).
    Elimina a atenuação de gradientes em profundidade e preserva estados ocultos limpos.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.layers = nn.ModuleList()
        in_dim = input_size
        for i in range(num_layers):
            layer_dropout = dropout if (i < num_layers - 1 and num_layers > 1) else 0.0
            layer = ResBiLSTMCell(
                input_size=in_dim,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                dropout=layer_dropout,
            )
            self.layers.append(layer)
            in_dim = layer.output_dim

        self.output_dim = in_dim

    def forward(
        self,
        x: Tensor,
        hx: list[tuple[Tensor, Tensor]] | tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        out = x
        states = []
        for i, layer in enumerate(self.layers):
            state_i = None
            if hx is not None:
                if isinstance(hx, list) and i < len(hx):
                    state_i = hx[i]
                elif isinstance(hx, tuple):
                    num_dir = 2 if self.bidirectional else 1
                    h_0, c_0 = hx
                    h_i = h_0[i * num_dir : (i + 1) * num_dir]
                    c_i = c_0[i * num_dir : (i + 1) * num_dir]
                    state_i = (h_i, c_i)

            out, new_state = layer(out, state_i)
            states.append(new_state)

        return out, states


class THORPIMLModel(nn.Module):
    """
    Arquitetura THOR-PIML de Alta Capacidade (ResBiLSTM + Taylorformer + SiLU Hurdle Heads).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 1. Encoder Recorrente Residual (ResBiLSTM)
        self.lstm = ResBiLSTMEncoder(
            input_size=config.n_features,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            bidirectional=config.lstm_bidirectional,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
        )

        lstm_output_dim = self.lstm.output_dim

        # 2. Taylor Series Linear Attention (opcional — S6: pode ser Identity para ablação)
        if getattr(config, "use_attention", True):
            self.attention = TaylorLinearAttention(
                dim=lstm_output_dim,
                dim_head=config.attn_dim_head,
                heads=config.attn_heads,
                causal=config.attn_causal,
                dropout=config.attn_dropout,
                remove_even_power_dups=config.remove_even_power_dups,
                prenorm=True,
            )
        else:
            self.attention = nn.Identity()

        # 3. Layer Normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # 4. Hurdle Heads Profundas com Ativação SiLU
        self.occurrence_head = nn.Sequential(
            nn.Linear(lstm_output_dim, config.occurrence_hidden),
            nn.LayerNorm(config.occurrence_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(config.occurrence_hidden, config.occurrence_hidden // 2),
            nn.SiLU(),
            nn.Linear(config.occurrence_hidden // 2, 1),
            nn.Sigmoid(),
        )

        self.intensity_head = nn.Sequential(
            nn.Linear(lstm_output_dim, config.intensity_hidden),
            nn.LayerNorm(config.intensity_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(config.intensity_hidden, config.intensity_hidden // 2),
            nn.SiLU(),
            nn.Linear(config.intensity_hidden // 2, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "proj.weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                # S6: forget gate bias=1 (mantém memória no início)
                # LSTM bias é [b_ih, b_hh] cada com 4*hidden (i,f,g,o) — f é o 2º quarto
                if "bias_ih" in name or "bias_hh" in name:
                    # hidden*2 se bidirectional já é tratado por shape
                    hidden = param.shape[0] // 4
                    nn.init.zeros_(param)
                    # forget gate = 1.0 (índices hidden:2*hidden)
                    with torch.no_grad():
                        param[hidden:2*hidden].fill_(1.0)
                else:
                    nn.init.zeros_(param)

        for head in [self.occurrence_head, self.intensity_head]:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass THOR-PIML — Hurdle CONTÍNUO e diferenciável (S4).

        y_pred = P(Occ) * Intensity  (puro, sem where/threshold)

        O threshold (ex: 0.65) agora é SÓ para métricas em evaluate.py:
            y_class_pred = (prob_occ >= 0.5).astype(int)
        Nunca para o forward — preserva gradiente para Focal Loss.
        """
        lstm_out, _ = self.lstm(x)
        # S6: attention pode ser Identity (ablation)
        if isinstance(self.attention, nn.Identity):
            attn_out = torch.zeros_like(lstm_out)
        else:
            attn_out = self.attention(lstm_out)
        combined = self.layer_norm(lstm_out + attn_out)

        last_step = combined[:, -1, :]

        prob_occ = self.occurrence_head(last_step)     # P(Chuva > 0) in [0,1]
        intensity = self.intensity_head(last_step)    # mm/dia >=0

        # Hurdle contínuo puro — sem gate hard (S4 fix)
        # V1 usava torch.where(prob>=0.65, prob, 0) que criava dead gradient
        final_prediction = prob_occ * intensity

        if return_components:
            return prob_occ, intensity, final_prediction

        return final_prediction

