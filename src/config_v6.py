"""
THOR-PIML — Configuração V6 FINAL ALL-IN (Última tentativa com esta arquitetura)
=================================================================================
Esta é a config definitiva, com tudo que aprendemos:
- Dados REAIS (CHIRPS REAL + ERA5 REAL + cape/tcwv real)
- 14 primárias + sin/cos sazonalidade
- 84 feats com lags 1,2,3,7,14
- Modelo 128x3 SDPA 8 heads, 600k+ params, dropout 0.3 em todas
- Loss corajosa: focal + quantile 0.85 + asymmetric storm boost + dry BCE 0.3
- Sem year_norm leakage, com sin_doy/cos_doy
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

# Base 7 + Thermo 3 + Extras 4 = 14 primárias
BASE_FEATURE_COLS_V6: List[str] = [
    'tmean', 'tmax', 'tmin', 'rh', 'psfc', 'wind_speed', 'solar_rad'
]

THERMO_FEATURE_COLS_V6: List[str] = [
    'dew_point', 'vpd', 'specific_humidity'
]

EXTRA_FEATURE_COLS_V6: List[str] = [
    'cape', 'tcwv', 'sin_doy', 'cos_doy'
]

PRIMARY_FEATURE_COLS_V6: List[str] = BASE_FEATURE_COLS_V6 + THERMO_FEATURE_COLS_V6 + EXTRA_FEATURE_COLS_V6

# Lags: 1,2,3,7,14 — 14 dias pega MJO e frente fria completa
LAG_DAYS_V6: List[int] = [1, 2, 3, 7, 14]

FEATURE_COLS_V6: List[str] = PRIMARY_FEATURE_COLS_V6 + [
    f"{col}_lag_{lag}" for col in PRIMARY_FEATURE_COLS_V6 for lag in LAG_DAYS_V6
]

# Legacy alias para compatibilidade
LAG_DAYS_V6B: List[int] = LAG_DAYS_V6
FEATURE_COLS_V6B: List[str] = FEATURE_COLS_V6


@dataclass
class ModelConfigV6:
    # ALL-IN FINAL
    n_features: int = len(FEATURE_COLS_V6)  # 84 feats (14 primárias × (1+5 lags))
    seq_len: int = 30  # janela sinótica 30 dias (poderia ir para 45, mas 30 é estável)
    lstm_hidden: int = 128
    lstm_layers: int = 3
    lstm_dropout: float = 0.3  # mais regularização para modelo maior
    lstm_bidirectional: bool = False  # causal puro

    # Atenção SDPA com 8 heads, causal, dropout 0.2 — melhor que Taylor bugado
    use_attention: bool = True
    attention_type: str = "sdpa"  # sdpa > taylor
    attn_heads: int = 8
    attn_dim_head: int = 32
    attn_causal: bool = True
    attn_dropout: float = 0.2
    remove_even_power_dups: bool = False

    # Heads balanceados mas com mais capacidade
    occurrence_hidden: int = 128
    intensity_hidden: int = 192
    prob_threshold: float = 0.5
    occurrence_threshold_mm: float = 1.0  # WMO ETCCDI


@dataclass
class TrainingConfigV6:
    batch_size: int = 128
    epochs: int = 150
    lr: float = 5e-4
    weight_decay: float = 1e-3
    alpha_bce: float = 1.0
    beta_mse: float = 1.0
    lambda_physics: float = 0.0
    prob_threshold: float = 0.5
    gamma_focal: float = 1.5
    alpha_focal: float = 0.5
    beta_softplus: float = 1.0
    storm_threshold: float = 10.0  # 10mm para pegar R10 também
    storm_weight: float = 2.0  # BALANCEADO V6d: era 3.0 (muito corajoso virou 1252 R10), agora 2.0
    scheduler: str = "cosine"
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    lambda_dry: float = 0.3
    seed: int | None = None
    patience: int = 20
    min_lr: float = 1e-6
    lr_factor: float = 0.5
    grad_clip: float = 1.0
    early_stop_metric: str = "kge"


@dataclass
class ModelConfigV6B:
    # Alias para compatibilidade, mesmo que V6 final
    n_features: int = len(FEATURE_COLS_V6)
    seq_len: int = 30
    lstm_hidden: int = 128
    lstm_layers: int = 3
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = False
    use_attention: bool = True
    attention_type: str = "sdpa"
    attn_heads: int = 8
    attn_dim_head: int = 32
    attn_causal: bool = True
    attn_dropout: float = 0.2
    remove_even_power_dups: bool = False
    occurrence_hidden: int = 128
    intensity_hidden: int = 192
    prob_threshold: float = 0.5
    occurrence_threshold_mm: float = 1.0


@dataclass
class THORConfigV6:
    model: ModelConfigV6 = field(default_factory=ModelConfigV6)
    training: TrainingConfigV6 = field(default_factory=TrainingConfigV6)
