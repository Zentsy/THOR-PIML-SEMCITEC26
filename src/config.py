"""
THOR-PIML — Configuração Central (Alta Capacidade + Hurdle Estrito)
===================================================================
Hiperparâmetros otimizados para treino do Modelo Hurdle com forte peso no BCE
para classificação precisa de tempo seco e chuva.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

# Definições de Colunas de Features Climáticas e Termodinâmicas
BASE_FEATURE_COLS: List[str] = [
    'tmean', 'tmax', 'tmin', 'rh', 'psfc', 'wind_speed', 'solar_rad'
]

THERMO_FEATURE_COLS: List[str] = [
    'dew_point', 'vpd', 'specific_humidity'
]

# Trend climático (ano normalizado) — S6: resolve não-estacionariedade 5.37→3.23 mm
TREND_FEATURE_COLS: List[str] = [
    'year_norm'  # (year - 1981) / 45.0 ∈ [0,1]
]

PRIMARY_FEATURE_COLS: List[str] = BASE_FEATURE_COLS + THERMO_FEATURE_COLS + TREND_FEATURE_COLS

LAG_DAYS: List[int] = [1, 2, 3, 7]

# Lista completa de features (11 primárias + 44 lags = 55)
FEATURE_COLS: List[str] = PRIMARY_FEATURE_COLS + [
    f"{col}_lag_{lag}" for col in PRIMARY_FEATURE_COLS for lag in LAG_DAYS
]


@dataclass
class PhysicsConstants:
    tetens_a: float = 17.269
    tetens_b: float = 237.3          # °C
    tetens_e0: float = 6.1078        # hPa

    g: float = 9.80665               # m/s²
    Rv: float = 461.5                # J/(kg·K)
    rho_w: float = 1000.0            # kg/m³
    Mw_over_Rstar: float = 2.16679   # g/m³·K
    H_eff: float = 2500.0            # metros


@dataclass
class ScalerConfig:
    method: str = "minmax"           # "minmax", "robust", "standard"
    zero_leakage: bool = True


@dataclass
class ModelConfig:
    n_features: int = len(FEATURE_COLS)  # 55 features (S6: 50→55 com year_norm)
    seq_len: int = 30                    # janela temporal (S4 T4: 30→60 em sub-passo)

    # T4 @ Lightning.AI — 1.3M params (S4.1)
    lstm_hidden: int = 128
    lstm_layers: int = 3
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = False     # S6 fix: False (causal puro, antes True quebrava causalidade)

    attn_heads: int = 8
    attn_dim_head: int = 32
    attn_causal: bool = True
    attn_dropout: float = 0.1
    remove_even_power_dups: bool = True
    use_attention: bool = True           # S6: False para ablação (Identity), True para produção

    occurrence_hidden: int = 64          # S6: 128→64 (binário precisa menos)
    intensity_hidden: int = 192          # S6: 128→192 (regressão precisa mais)
    n_outputs: int = 1
    prob_threshold: float = 0.65
    occurrence_threshold_mm: float = 0.1  # V5 legado 0.1, V6 usa 1.0 (WMO) — define y_class = y>=threshold
    use_year_norm: bool = True           # V6a False remove leakage temporal


@dataclass
class TrainingConfig:
    batch_size: int = 128            # V1 64→128 (T4 16GB, S5)
    epochs: int = 100
    lr: float = 7e-4                 # Hotfix garoa: 1e-3→7e-4 (sqrt scaling p/ batch 128)
    weight_decay: float = 5e-4       # Hotfix: 2e-4→5e-4 (regulariza 520k p/ garoa)

    # Pesos da Loss composta L = α·L_focal + β·L_mse + λ·L_physics
    alpha_bce: float = 2.0           # Peso do BCE/Focal (classificação seco/chuvoso)
    beta_mse: float = 1.0            # Peso da intensidade (mm)
    lambda_physics: float = 0.2      # Hotfix: 0.5→0.2 (física irrelevante p/ chuva <50mm)
    prob_threshold: float = 0.5      # Limiar para métricas (S4: 0.65→0.5 padrão F1, forward é contínuo)

    # Hiperparâmetros PIML (S3 — agora canônicos aqui, não soltos em physics_loss)
    gamma_focal: float = 1.5         # Hotfix: 2.0→1.5 (menos agressivo)
    alpha_focal: float = 0.40        # Hotfix: 0.25→0.40 (balanceia FP/FN, 48×→~4×)
    beta_softplus: float = 1.0       # Softplus β (V2: 0.5→1.0, menos suave)
    storm_threshold: float = 20.0    # mm/dia — limiar de tempestade
    storm_weight: float = 1.8        # Hotfix: 3.0→1.8 (reduz bias +2.19)

    # S5 — Scheduler e AMP para T4
    scheduler: str = "plateau"       # "plateau" (ReduceLROnPlateau) ou "cosine" (CosineAnnealingWarmRestarts)
    use_amp: bool = True             # FP16 Mixed Precision na T4 (3× mais rápido, metade VRAM)
    num_workers: int = 4             # DataLoader workers (T4 Lightning: 4)
    pin_memory: bool = True          # pin_memory para GPU

    # S6 fix — Penalidade anti-garoa (re-treino muda CWD de verdade)
    lambda_dry: float = 0.5          # Peso para y_true==0 → penaliza final>0 em dias secos (corrige CWD 237)

    # Seed: None = aleatória (busca de sorte), int = reprodutível (quando achar boa, fixe)
    seed: int | None = None          # V4: 42→None (aleatória até achar boa, depois padroniza)

    patience: int = 15
    min_lr: float = 1e-5
    lr_factor: float = 0.5
    grad_clip: float = 1.0


@dataclass
class THORConfig:
    physics: PhysicsConstants = field(default_factory=PhysicsConstants)
    scaler: ScalerConfig = field(default_factory=ScalerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


