"""
THOR-PIML V7 — Configuração da Arquitetura Híbrida LSTM+TCN (temporal)
======================================================================
Nova era (branch hybrid-arch-test). Ver docs/V7_HYBRID_ARCHITECTURE.md.

Diferenças vs V6d:
- Modelo híbrido: ResBiLSTM (memória de longo prazo) || TCN multi-escala causal
  (padrões sinóticos: frente fria ~3d, MJO ~14d) com fusão gated.
- 84 feats DE VERDADE (fix do bug lag-14 — V6 treinava com 70 sem saber).
- Loss V7: base V6d balanceada + extremes-recall (FN em R10/R20) + variance
  matching — os dois modos de falha restantes (R20 recall 13%, std subestimado).
- OneCycleLR + seleção de checkpoint por val KGE (a config V6 dizia 'kge' mas o
  train.py ignorava e usava val loss).
- CV temporal bloqueada (ver pipeline_v7.blocked_temporal_folds) para seleção de
  modelo; o teste cego 2019-09→2026-06 nunca é tocado até o relatório final.

Ablação (mesma classe, switches): --model hybrid | lstm | tcn
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

# 14 primárias V6 (base + termo + extras) — cape/tcwv continuam com esses nomes;
# no GT V3 eles são REAIS (substituem o proxy), no V2 são proxy/constante.
BASE_FEATURE_COLS_V7: List[str] = [
    'tmean', 'tmax', 'tmin', 'rh', 'psfc', 'wind_speed', 'solar_rad'
]
THERMO_FEATURE_COLS_V7: List[str] = ['dew_point', 'vpd', 'specific_humidity']
EXTRA_FEATURE_COLS_V7: List[str] = ['cape', 'tcwv', 'sin_doy', 'cos_doy']

# Colunas opcionais do GT V3 (só entram se existirem no CSV — pipeline detecta):
#   pr_grid_max, pr_grid_std  → variabilidade espacial CHIRPS (30 células)
#   z500, u700, v700, q700, w500, ws700, shear_700 → sinótica ERA5 PL (média do domínio)
V7_OPTIONAL_COLS: List[str] = [
    'pr_grid_max', 'pr_grid_std',
    'z500', 'u700', 'v700', 'q700', 'w500', 'ws700', 'shear_700',
]

PRIMARY_FEATURE_COLS_V7: List[str] = (
    BASE_FEATURE_COLS_V7 + THERMO_FEATURE_COLS_V7 + EXTRA_FEATURE_COLS_V7
)

# Lags 1,2,3,7,14 — agora aplicados DE VERDADE (fix bug V6)
LAG_DAYS_V7: List[int] = [1, 2, 3, 7, 14]


def build_primary_cols_v7(available_cols: List[str]) -> List[str]:
    """Primárias V7 = 14 fixas + opcionais V3 presentes no CSV (ordem estável)."""
    present = [c for c in V7_OPTIONAL_COLS if c in available_cols]
    if present:
        print(f"[V7 feats] Sinóticas/espaciais do GT V3 detectadas: {present}")
    return PRIMARY_FEATURE_COLS_V7 + present


def build_feature_cols_v7(primary_cols: List[str], lags: List[int]) -> List[str]:
    return primary_cols + [f"{c}_lag_{lag}" for c in primary_cols for lag in lags]


@dataclass
class ModelConfigV7:
    # input
    n_features: int = 84  # reconciliado com o pipeline real no run_v7 (log alto, nunca silencioso)
    seq_len: int = 30

    # Ramo LSTM (memória de longo prazo) — igual V6d
    use_lstm_branch: bool = True
    lstm_hidden: int = 128
    lstm_layers: int = 3
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = False  # causal puro (lição V3)

    # Ramo TCN (padrões multi-escala) — dilations 1,2,4,8 com kernels 3/5/7 em
    # paralelo: campo receptivo ≥ 1 + 6*(1+2+4+8) = 91 dias > seq_len 30
    use_tcn_branch: bool = True
    tcn_channels: List[int] = field(default_factory=lambda: [96, 96, 128, 128])
    tcn_kernels: List[int] = field(default_factory=lambda: [3, 5, 7])
    tcn_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    tcn_dropout: float = 0.3

    # Fusão gated + atenção sobre a sequência fundida
    fusion_dim: int = 128
    use_attention: bool = True
    attn_heads: int = 8
    attn_dropout: float = 0.2
    attn_causal: bool = True

    # Heads hurdle (contrato V6: sigmoid × softplus)
    occurrence_hidden: int = 128
    intensity_hidden: int = 192
    prob_threshold: float = 0.5
    occurrence_threshold_mm: float = 1.0  # WMO ETCCDI


@dataclass
class TrainingConfigV7:
    batch_size: int = 256
    epochs: int = 150
    lr: float = 3e-4            # OneCycleLR: start = lr, sobe para max_lr, volta a min_lr
    max_lr: float = 1e-3
    min_lr: float = 3e-5
    weight_decay: float = 1e-3
    scheduler: str = "onecycle"  # OneCycleLR (docs §9); "cosine" disponível para ablação
    use_amp: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    grad_clip: float = 1.0
    patience: int = 20
    seed: int | None = None
    checkpoint_metric: str = "kge"  # seleção de melhor época: "kge" ou "loss"

    # --- Loss V7/V8 Calibrada (Padrão Simétrico NeuralHydrology + Supressão Seca) ---
    alpha_bce: float = 1.0
    gamma_focal: float = 1.5
    alpha_focal: float = 0.5
    quantile: float = 0.500        # Mediana simétrica pura para dias normais (Bias +0.31mm / R10 baixo)
    quantile_r20: float = 0.68     # Quantil convectivo apenas em tempestades severas >=20mm (resgata R20)
    huber_delta: float = 5.0
    beta_mse: float = 1.0
    storm_threshold: float = 10.0
    storm_weight_r10: float = 1.0  # Neutro para 10mm
    storm_weight_r20: float = 1.8  # Foco nas tempestades severas
    underest_boost_r10: float = 1.0  # Neutro para 10mm
    underest_boost_r20: float = 1.6  # Foco nas tempestades severas
    lambda_dry: float = 0.3
    lambda_final: float = 0.2
    lambda_sharpness: float = 0.05
    lambda_physics: float = 0.0    # reativa no V8 com TCWV real (via CLI --physics 0.05)

    # --- Extremes-Recall Calibrado (foco em R20mm) ---
    lambda_extreme_fn: float = 0.5
    extreme_fn_boost_r10: float = 1.0  # Neutro para 10mm (evita puxar 7-9mm para >=10mm)
    extreme_fn_boost_r20: float = 1.8  # Foco nas tempestades severas reais

    # --- Penalidade de Falsos Positivos em R10mm ---
    lambda_fp_r10: float = 0.0

    # --- Variance matching simétrico ---
    lambda_var_match: float = 0.02


@dataclass
class THORConfigV7:
    model: ModelConfigV7 = field(default_factory=ModelConfigV7)
    training: TrainingConfigV7 = field(default_factory=TrainingConfigV7)


@dataclass
class ModelConfigV8(ModelConfigV7):
    """V8 = V7 + ramo espacial (CNN 2D sobre domínio ERA5)."""

    spatial_channels: int = 5          # z500, u700, v700, q700, w500
    spatial_embed_dim: int = 64
    spatial_dropout: float = 0.2


@dataclass
class TrainingConfigV8(TrainingConfigV7):
    """Loss V8-Master: Controle Estrito de Volume e R10 (Quantil 0.450 + Quantil 0.680 em R20mm)."""
    patience: int = 50                 # paciência de 50 épocas sobre o KGE (permite ciclo OneCycleLR completo)
    gamma_focal: float = 1.70          # nitidez máxima na fronteira seco vs chuva
    alpha_focal: float = 0.50          # balanceamento neutro
    lambda_dry: float = 0.56           # supressão afiada de cauda seca (controla CWD e CDD)
    quantile: float = 0.450            # quantil assimétrico: penaliza superestimação (1-q=0.55), zera viés e derruba R10mm
    quantile_r20: float = 0.680        # quantil convectivo exclusivo de tempestades severas (R20 cravado em ~100 dias)
    storm_weight_r10: float = 1.00     # 100% neutro em 10mm (evita overcount de R10)
    storm_weight_r20: float = 1.80     # proteção estável de tempestades severas de 20mm
    extreme_fn_boost_r10: float = 0.00 # ZERO: elimina qualquer empurrão de dias 8-9mm para cima de 10mm
    extreme_fn_boost_r20: float = 1.80 # mantém R20 cravado na faixa dos 95-100 dias
    underest_boost_r20: float = 1.60   # boost de subestimação calibrado
    lambda_final: float = 0.20         # regularização final
    lambda_physics: float = 0.05       # barreira termodinâmica CC com TCWV real
    lambda_var_match: float = 0.02     # preservação de variância


@dataclass
class THORConfigV8:
    model: ModelConfigV8 = field(default_factory=ModelConfigV8)
    training: TrainingConfigV8 = field(default_factory=TrainingConfigV8)
