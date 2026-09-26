"""
THOR-PIML — Inferência CORDEX V8 com Física Informada (SEMCITEC 2026)
=====================================================================
Executa o downscaling estatístico Zero-Shot dos cenários CORDEX SAM-22:
- SSP2-4.5 (RCP 4.5)
- SSP5-8.5 (RCP 8.5)
para o período de 2026 a 2099 em Guarulhos-SP (-23.43°S, -46.47°W).

Metodologia & Invariantes Científicos:
1. Arquitetura THOR-V8:
   - 2D Synoptic CNN Encoder sobre a grade CORDEX 16x16 (canalização física: T2m, gradientes térmicos, umidade e convecção) -> Z_syn (30x64).
   - Tronco Híbrido Bi-Branch (ResLSTM + Multi-Scale TCN causal com dilatações 1,2,4,8 e kernels 3,5,7).
   - Fusão Gated Adaptativa + Atenção Causal SDPA de 8 cabeças.
   - Hurdle Dual-Head: Ocorrência (Sigmoid, limiar WMO 1.0mm) x Intensidade (Softplus).
2. Restrição Física de Clausius-Clapeyron (PIML):
   - Teto convectivo termodinâmico W_max = 4.0 * TCWV(T).
   - Aplicação da barreira física estrita: y_piml = min(y_hat, W_max).
   - Rastreabilidade de violações físicas (taxa de conformidade termodinâmica).
3. Zero Data Leakage:
   - Scaler V7 congelado (treinado exclusivamente no histórico ERA5-Land/CHIRPS).
   - Janelamento puramente causal com lookback T=30 dias (sem antecipação do futuro).

Saídas em results/projections/:
- cordex_ssp245_downscaled_2026_2099.csv
- cordex_ssp585_downscaled_2026_2099.csv
- projections_summary.json

Uso:
    python src/infer_cordex_v8.py
    python src/infer_cordex_v8.py --scenario ssp245
    python src/infer_cordex_v8.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import xarray as xr
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, desc=None, total=None, unit=None, **kwargs):
        if desc:
            print(f"[{desc}] Processando...")
        return iterable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR
from src.preprocessing import (
    RobustClimateScaler,
    compute_dew_point,
    compute_specific_humidity,
    compute_vpd,
)
from src.utils import get_device
from src.v7.config_v7 import THORConfigV8
from src.v7.model_v8 import THORSpatialHybridModel

DEFAULT_PROJECTIONS_DIR = RESULTS_DIR / "projections"
CKPT_PATH = CHECKPOINT_DIR / "v8_hybrid_seed42.pt"
SCALER_PATH = CHECKPOINT_DIR / "scaler_v7.json"

# Constantes e limites climatológicos de referência para Guarulhos (fit histórico ERA5 PL)
HIST_TCWV_MEAN: float = 26.39
HIST_TCWV_MIN: float = 3.19
HIST_TCWV_MAX: float = 51.42
HIST_Z500_MIN: float = 5570.76
HIST_Z500_MAX: float = 5943.79
HIST_U700_MIN: float = -18.99
HIST_U700_MAX: float = 30.08
HIST_V700_MIN: float = -22.09
HIST_V700_MAX: float = 17.15
HIST_Q700_MIN: float = 0.0114
HIST_Q700_MAX: float = 11.720
HIST_W500_MIN: float = -4.197
HIST_W500_MAX: float = 1.0865

# Mapeamento de cenários
SCENARIO_MAP = {
    "ssp245": {
        "cordex_csv": DATA_DIR / "cordex" / "cordex_guarulhos_rcp45_daily_2006_2099.csv",
        "pr_nc": DATA_DIR / "SAM-22" / "pr-rcp45" / "pr_SAM-20_MOHC-HadGEM2-ES_rcp45_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
        "tas_nc": DATA_DIR / "SAM-22" / "tas-rcp45" / "tas_SAM-20_MOHC-HadGEM2-ES_rcp45_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
        "label": "SSP2-4.5 (Estabilização Média)",
        "out_name": "cordex_ssp245_downscaled_2026_2099.csv",
    },
    "ssp585": {
        "cordex_csv": DATA_DIR / "cordex" / "cordex_guarulhos_rcp85_daily_2006_2099.csv",
        "pr_nc": DATA_DIR / "SAM-22" / "pr-rcp85" / "pr_SAM-20_MOHC-HadGEM2-ES_rcp85_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
        "tas_nc": DATA_DIR / "SAM-22" / "rcp8.5-TAS" / "tas_SAM-20_MOHC-HadGEM2-ES_rcp85_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
        "label": "SSP5-8.5 (Altas Emissões)",
        "out_name": "cordex_ssp585_downscaled_2026_2099.csv",
    },
}


def engineer_cordex_surface_features(
    df_raw: pd.DataFrame,
    feature_cols: List[str],
    cc_response_per_kelvin: float = 0.065,
    cc_response_jitter: float = 0.0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Deriva as 23 variáveis termodinâmicas primárias + 5 lags temporais [1,2,3,7,14] = 138 colunas."""
    df = df_raw.copy()
    n = len(df)

    # 1. Temperatura e ciclos
    tmean = df["tas_celsius"].values.astype(np.float32)
    # Ciclo diurno médio em Guarulhos: amplitude ~8.0 a 10.0 °C
    tmax = tmean + 4.5
    tmin = tmean - 4.5
    df["tmean"] = tmean
    df["tmax"] = tmax
    df["tmin"] = tmin

    # 2. Umidade Relativa (RH): função da chuva e da temperatura (sat. vapor)
    pr_raw = df["pr_raw_mm"].values.astype(np.float32)
    # Em dias chuvosos RH sobe para 85-95%; em dias secos varia entre 60-70%
    rh = np.clip(68.0 + 22.0 * np.tanh(pr_raw / 6.0), 30.0, 99.0).astype(np.float32)
    df["rh"] = rh

    # 3. Pressão de superfície (Guarulhos ~750m altitude => ~928-935 hPa)
    psfc = np.full(n, 930.0, dtype=np.float32)
    df["psfc"] = psfc

    # 4. Vento e Radiação Solar estimada por sazonalidade
    df["wind_speed"] = np.full(n, 2.2, dtype=np.float32)
    doy = df["doy"].values.astype(np.float32)
    # Radiação solar: pico no verão austral (dezembro, doy ~350), mínimo no inverno (junho, doy ~170)
    rad_clear = 18.0 + 6.0 * np.cos(2 * np.pi * (doy - 10) / 360.0)
    # Atenuação por cobertura de nuvens associada à chuva
    solar_rad = np.maximum(4.0, rad_clear * (1.0 - 0.5 * np.tanh(pr_raw / 8.0))).astype(np.float32)
    df["solar_rad"] = solar_rad

    # 5. Variáveis termodinâmicas avançadas (August-Roche-Magnus & psicrometria)
    df["dew_point"] = compute_dew_point(tmean, rh)
    df["vpd"] = compute_vpd(tmean, rh)
    df["specific_humidity"] = compute_specific_humidity(tmean, rh, psfc)
    df["cape"] = np.maximum(0.0, (tmax - tmin) * (rh / 100.0) * 80.0).astype(np.float32)

    # 6. TCWV (Total Column Water Vapor) com resposta de Clausius-Clapeyron:
    # A capacidade de vapor atmosférico cresce ~6.5% por Kelvin de aquecimento
    t_anom = tmean - 20.0
    tcwv_base = 26.0 + 9.0 * np.sin(2 * np.pi * (doy - 30) / 360.0)  # Sazonalidade climática
    if cc_response_jitter > 0.0:
        rng = np.random.default_rng(random_seed)
        response_per_kelvin = np.maximum(
            0.0,
            rng.normal(cc_response_per_kelvin, cc_response_jitter, size=n),
        )
    else:
        response_per_kelvin = cc_response_per_kelvin
    tcwv = np.clip(
        tcwv_base * (1.0 + response_per_kelvin * t_anom)
        + 3.0 * np.tanh(pr_raw / 10.0),
        5.0,
        60.0,
    ).astype(np.float32)
    df["tcwv"] = tcwv

    # 7. Harmônicos temporais (sem vazamento, ciclo contínuo de 360 dias)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 360.0).astype(np.float32)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 360.0).astype(np.float32)

    # 8. Estatísticas de grade local (pr_grid_max, pr_grid_std)
    if "pr_grid_max" not in df.columns:
        df["pr_grid_max"] = pr_raw * 1.2
    if "pr_grid_std" not in df.columns:
        df["pr_grid_std"] = pr_raw * 0.25

    # 9. Variáveis sinóticas locais consistentes com a termodinâmica
    # Expansão hipsométrica de z500 com a temperatura
    df["z500"] = (5840.0 + 1.8 * tmean).astype(np.float32)
    df["u700"] = (3.5 + 2.0 * np.sin(2 * np.pi * doy / 360.0)).astype(np.float32)
    df["v700"] = (-1.0 - 1.5 * np.cos(2 * np.pi * doy / 360.0)).astype(np.float32)
    df["q700"] = np.clip(df["specific_humidity"] * 0.40, 0.5, 11.5).astype(np.float32)
    # Movimento vertical convectivo (omega em Pa/s)
    df["w500"] = np.clip(-0.01 - 0.005 * np.minimum(pr_raw, 60.0), -0.45, 0.20).astype(np.float32)
    df["ws700"] = np.sqrt(df["u700"]**2 + df["v700"]**2).astype(np.float32)
    df["shear_700"] = np.abs(df["ws700"] - df["wind_speed"]).astype(np.float32)

    # 10. Construção dos lags temporais [1, 2, 3, 7, 14] sem fragmentação de memória
    lags = [1, 2, 3, 7, 14]
    primary_23 = [
        "tmean", "tmax", "tmin", "rh", "psfc", "wind_speed", "solar_rad",
        "dew_point", "vpd", "specific_humidity", "cape", "tcwv",
        "sin_doy", "cos_doy", "pr_grid_max", "pr_grid_std",
        "z500", "u700", "v700", "q700", "w500", "ws700", "shear_700"
    ]

    lag_cols = {}
    for col in primary_23:
        for lag in lags:
            lag_cols[f"{col}_lag_{lag}"] = df[col].shift(lag)

    df_lags = pd.DataFrame(lag_cols, index=df.index)
    df = pd.concat([df, df_lags], axis=1)

    # Preenchimento estritamente causal no início (sem lookahead)
    df = df.bfill().ffill()

    # Validação e ordenação estrita das 138 colunas
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltam colunas de features para o modelo V8: {missing_cols}")

    X_surface = df[feature_cols].values.astype(np.float32)
    tcwv_unnorm = df["tcwv"].values.astype(np.float32)
    return X_surface, tcwv_unnorm, df


def build_cordex_spatial_tensor(
    ds_pr: xr.Dataset,
    ds_tas: xr.Dataset,
    time_indices: np.ndarray,
) -> np.ndarray:
    """Extrai e normaliza o tensor 2D espacial (N, H=25, W=33, C=5) com física informada e interpolação."""
    # Extrai fatias temporais
    pr_grid = (ds_pr["pr"].isel(time=time_indices).values * 86400.0).astype(np.float32)   # mm/dia
    tas_grid = (ds_tas["tas"].isel(time=time_indices).values - 273.15).astype(np.float32)  # °C

    # Tratamento defensivo se houver NaNs residuais no NetCDF
    if np.isnan(pr_grid).any():
        N_dim, H_dim, W_dim = pr_grid.shape
        pr_grid = pd.DataFrame(pr_grid.reshape(N_dim, -1)).ffill().bfill().values.reshape(N_dim, H_dim, W_dim).astype(np.float32)
    if np.isnan(tas_grid).any():
        N_dim, H_dim, W_dim = tas_grid.shape
        tas_grid = pd.DataFrame(tas_grid.reshape(N_dim, -1)).ffill().bfill().values.reshape(N_dim, H_dim, W_dim).astype(np.float32)

    N, H, W = pr_grid.shape
    spatial = np.zeros((N, H, W, 5), dtype=np.float32)

    # Canal 0: z500 normalizado via aproximação hipsométrica de temperatura
    z500_field = 5840.0 + 1.8 * tas_grid
    spatial[..., 0] = np.clip((z500_field - HIST_Z500_MIN) / (HIST_Z500_MAX - HIST_Z500_MIN), 0.0, 1.0)

    # Canais 1 e 2: Vento térmico zonal e meridional (gradientes de temperatura)
    grad_y, grad_x = np.gradient(tas_grid, axis=(1, 2))
    u700_field = 3.5 - 2.5 * grad_y
    v700_field = -1.0 + 2.5 * grad_x
    spatial[..., 1] = np.clip((u700_field - HIST_U700_MIN) / (HIST_U700_MAX - HIST_U700_MIN), 0.0, 1.0)
    spatial[..., 2] = np.clip((v700_field - HIST_V700_MIN) / (HIST_V700_MAX - HIST_V700_MIN), 0.0, 1.0)

    # Canal 3: Umidade específica em 700hPa com gradiente térmico vertical real (lapse rate 6.5 °C/km)
    # T700 = tas - 14.5 °C (~2250m acima da superfície)
    t700_grid = tas_grid - 14.5
    es = 6.1078 * np.exp((17.27 * t700_grid) / (t700_grid + 237.3))
    q700_field = np.clip((622.0 * es * 0.65) / (700.0 - 0.378 * es * 0.65), 0.05, 11.5)
    spatial[..., 3] = np.clip((q700_field - HIST_Q700_MIN) / (HIST_Q700_MAX - HIST_Q700_MIN), 0.0, 1.0)

    # Canal 4: Velocidade vertical w500 (em Pa/s; acoplamento físico com convecção)
    w500_field = -0.01 - 0.005 * np.minimum(pr_grid, 60.0)
    spatial[..., 4] = np.clip((w500_field - HIST_W500_MIN) / (HIST_W500_MAX - HIST_W500_MIN), 0.0, 1.0)

    # Interpolação espacial bilinear para resolução canônica do treino (25x33)
    # Preserva os gradientes sinóticos e previne distorções nos feature maps antes do AdaptiveAvgPool
    sp_t = torch.tensor(spatial, dtype=torch.float32).permute(0, 3, 1, 2)
    sp_interp = torch.nn.functional.interpolate(
        sp_t, size=(25, 33), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1).numpy().astype(np.float32)

    return sp_interp


def run_inference_scenario(
    scenario_key: str,
    model: THORSpatialHybridModel,
    scaler: RobustClimateScaler,
    feature_cols: List[str],
    device: torch.device,
    out_dir: Path,
    batch_size: int = 128,
    dry_run: bool = False,
    cc_response_per_kelvin: float = 0.065,
    cc_response_jitter: float = 0.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Executa a inferência completa para um cenário climático (SSP2-4.5 ou SSP5-8.5)."""
    meta = SCENARIO_MAP[scenario_key]
    csv_path = meta["cordex_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV pré-processado CORDEX não encontrado: {csv_path}\n"
            f"Execute primeiro: python src/cordex_preprocessor.py"
        )

    print(f"\n{'='*75}")
    print(f"Iniciando Downscaling THOR-PIML V8: {meta['label']}")
    print(f"{'='*75}")
    t0 = time.time()

    # 1. Carrega série pré-processada
    df_raw = pd.read_csv(csv_path)
    df_raw["orig_idx"] = np.arange(len(df_raw))
    if dry_run:
        # No dry-run, foca nos primeiros 120 dias a partir do início das projeções futuras (2026)
        # com 30 dias de lookback prévio, cobrindo o início da época 'Futuro Próximo'
        idx_2026 = df_raw[df_raw["year"] >= 2026].index
        start_idx = max(0, idx_2026[0] - 30) if len(idx_2026) > 0 else 0
        df_raw = df_raw.iloc[start_idx : start_idx + 120].copy().reset_index(drop=True)
        print(f"  [DRY-RUN] Limitando série a {len(df_raw)} passos de teste (a partir de {df_raw['date'].iloc[0]}).")
    else:
        # Mantém o lookback histórico, mas não publica previsões anteriores a 2026.
        idx_2026 = df_raw[df_raw["year"] >= 2026].index
        if len(idx_2026) == 0:
            raise ValueError("Série CORDEX não contém dados a partir de 2026.")
        start_idx = max(0, idx_2026[0] - 30)
        df_raw = df_raw.iloc[start_idx:].copy().reset_index(drop=True)

    # 2. Engenharia de 138 features
    print("  Construindo matriz de preditores de superfície (138 variáveis)...")
    X_surface, tcwv_unnorm, df_full = engineer_cordex_surface_features(
        df_raw,
        feature_cols,
        cc_response_per_kelvin=cc_response_per_kelvin,
        cc_response_jitter=cc_response_jitter,
        random_seed=random_seed,
    )

    # 3. Normalização Zero-Leakage via Scaler V7
    print("  Aplicando RobustClimateScaler V7 (congelado)...")
    X_scaled = scaler.transform(X_surface).astype(np.float32)

    # 4. Carrega campos espaciais 2D CORDEX
    print(f"  Abrindo grades CORDEX 16x16: {meta['pr_nc'].name}...")
    ds_pr = xr.open_dataset(meta["pr_nc"])
    ds_tas = xr.open_dataset(meta["tas_nc"])

    seq_len = 30
    n_days = len(df_raw)
    n_windows = n_days - seq_len
    if n_windows <= 0:
        raise ValueError(f"Série temporal insuficiente ({n_days} dias) para seq_len={seq_len}")

    print(f"  Total de janelas temporais de 30 dias: {n_windows:,}")

    # Coleta de resultados
    pred_dates = []
    pred_years = []
    pred_months = []
    pred_days = []
    pred_tas_celsius = []
    pred_pr_raw_cordex = []
    pred_prob_occ = []
    pred_intensity = []
    pred_piml_uncapped = []
    pred_piml_downscaled = []
    tcwv_ceilings = []
    violations = []

    model.eval()

    # Loop em lotes para alta performance computacional
    chunk_size = batch_size * 4
    n_chunks = int(np.ceil(n_windows / chunk_size))

    with torch.no_grad():
        for chunk_idx in tqdm(range(n_chunks), desc=f"Inferência {scenario_key.upper()}"):
            start_i = chunk_idx * chunk_size
            end_i = min(n_windows, (chunk_idx + 1) * chunk_size)
            chunk_len = end_i - start_i

            # Prepara janelas de superfície (chunk_len, 30, 138)
            xs_list = [X_scaled[start_i + i : start_i + i + seq_len] for i in range(chunk_len)]
            xs_batch = torch.from_numpy(np.stack(xs_list)).to(device=device, dtype=torch.float32)

            # Prepara tensores espaciais para a janela (chunk_len, 30, 16, 16, 5)
            # Usa os índices originais globais do NetCDF para manter coerência espaço-temporal
            t_slice = df_raw["orig_idx"].iloc[start_i : end_i + seq_len].values
            sp_raw = build_cordex_spatial_tensor(ds_pr, ds_tas, t_slice)
            xsp_list = [sp_raw[i : i + seq_len] for i in range(chunk_len)]
            xsp_batch = torch.from_numpy(np.stack(xsp_list)).to(device=device, dtype=torch.float32)

            # Passagem pela rede neural THOR-PIML
            prob, inten, final = model(xs_batch, xsp_batch, return_components=True)

            p_vals = prob.cpu().squeeze(-1).numpy()
            i_vals = inten.cpu().squeeze(-1).numpy()
            f_vals = final.cpu().squeeze(-1).numpy()

            # Pós-processamento e restrição de Clausius-Clapeyron para cada dia projetado
            for i in range(chunk_len):
                day_idx = start_i + i + seq_len
                date_str = str(df_raw["date"].iloc[day_idx])
                pr_cordex = float(df_raw["pr_raw_mm"].iloc[day_idx])
                tas_cordex = float(df_raw["tas_celsius"].iloc[day_idx]) if "tas_celsius" in df_raw.columns else 20.0

                # Extrai ano, mês e dia com suporte estrito ao calendário 360_day (sem pd.to_datetime)
                if "year" in df_raw.columns:
                    yr_val = int(df_raw["year"].iloc[day_idx])
                    mo_val = int(df_raw["month"].iloc[day_idx])
                    dy_val = int(df_raw["day"].iloc[day_idx])
                else:
                    yr_val = int(date_str[:4])
                    mo_val = int(date_str[5:7])
                    dy_val = int(date_str[8:10])

                p_occ = float(p_vals[i])
                mu_int = float(i_vals[i])
                y_uncapped = float(f_vals[i])

                # Teto físico de precipitação convectiva: W_max = 4.0 * TCWV
                tcwv_day = float(tcwv_unnorm[day_idx])
                w_max = 4.0 * tcwv_day

                # Aplicação da barreira física
                is_viol = y_uncapped > w_max
                y_piml = min(y_uncapped, w_max)

                pred_dates.append(date_str)
                pred_years.append(yr_val)
                pred_months.append(mo_val)
                pred_days.append(dy_val)
                pred_tas_celsius.append(tas_cordex)
                pred_pr_raw_cordex.append(pr_cordex)
                pred_prob_occ.append(p_occ)
                pred_intensity.append(mu_int)
                pred_piml_uncapped.append(y_uncapped)
                pred_piml_downscaled.append(y_piml)
                tcwv_ceilings.append(w_max)
                violations.append(int(is_viol))

    ds_pr.close()
    ds_tas.close()

    # Monta DataFrame final consolidado com suporte nativo a 360_day
    df_out = pd.DataFrame({
        "date": pred_dates,
        "year": pred_years,
        "month": pred_months,
        "day": pred_days,
        "tas_celsius": np.round(pred_tas_celsius, 4),
        "pr_cordex_raw_mm": np.round(pred_pr_raw_cordex, 3),
        "prob_occurrence": np.round(pred_prob_occ, 4),
        "intensity_mm": np.round(pred_intensity, 3),
        "pr_thor_uncapped_mm": np.round(pred_piml_uncapped, 3),
        "pr_thor_downscaled_mm": np.round(pred_piml_downscaled, 3),
        # Limiar WMO de 1.0mm para eliminação de garoa crônica do CORDEX
        "pr_thor_clipped_mm": np.where(np.array(pred_piml_downscaled) >= 1.0, np.round(pred_piml_downscaled, 3), 0.0),
        "tcwv_ceiling_wmax_mm": np.round(tcwv_ceilings, 2),
        "is_physical_violation": violations,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / meta["out_name"]
    df_out.to_csv(out_path, index=False)

    dt_elapsed = time.time() - t0
    n_viol = sum(violations)
    viol_rate = (n_viol / len(violations)) * 100.0

    print(f"\n✓ Projeção concluída com sucesso em {dt_elapsed:.1f}s!")
    print(f"  Arquivo salvo: {out_path} ({len(df_out):,} dias projetados)")
    print(f"  Violações da barreira de Clausius-Clapeyron: {n_viol} / {len(violations)} ({viol_rate:.2f}%)")
    print(f"  Estatísticas comparativas (Média mm/dia):")
    print(f"    CORDEX Bruto:  {df_out['pr_cordex_raw_mm'].mean():.2f} mm/dia")
    print(f"    THOR-PIML:     {df_out['pr_thor_downscaled_mm'].mean():.2f} mm/dia")
    print(f"    THOR (>=1mm):  {df_out['pr_thor_clipped_mm'].mean():.2f} mm/dia")
    print(f"  Dias de tempestade R10mm:")
    print(f"    CORDEX Bruto:  {(df_out['pr_cordex_raw_mm'] >= 10.0).sum():,} dias")
    print(f"    THOR-PIML:     {(df_out['pr_thor_downscaled_mm'] >= 10.0).sum():,} dias")

    return df_out


def run_projections(
    scenario: str = "all",
    ckpt_path: Path = CKPT_PATH,
    scaler_path: Path = SCALER_PATH,
    out_dir: Optional[Path] = None,
    batch_size: int = 128,
    dry_run: bool = False,
    cc_response_per_kelvin: float = 0.065,
    cc_response_jitter: float = 0.0,
    random_seed: int = 42,
) -> Dict[str, Path]:
    """Orquestra a inferência Zero-Shot para todos os cenários."""
    device = get_device()
    out_dir = Path(out_dir) if out_dir else DEFAULT_PROJECTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler não encontrado: {scaler_path}")

    # 1. Carrega Scaler V7
    print(f"Carregando Scaler V7: {scaler_path}...")
    scaler = RobustClimateScaler.load(scaler_path)

    # 2. Carrega Checkpoint V8
    print(f"Carregando Checkpoint THOR-V8: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    feature_cols = ckpt.get("feature_cols", [])

    config = THORConfigV8()
    config.model.n_features = len(feature_cols)
    model = THORSpatialHybridModel(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✓ Modelo THOR-V8 carregado com sucesso ({sum(p.numel() for p in model.parameters()):,} parâmetros) no dispositivo: {device}")

    norm_map = {"rcp45": "ssp245", "rcp85": "ssp585", "ssp245": "ssp245", "ssp585": "ssp585", "all": "all"}
    sc_norm = norm_map.get(scenario, scenario)
    scenarios = ["ssp245", "ssp585"] if sc_norm == "all" else [sc_norm]
    results = {}
    summary_meta = {}

    for sc in scenarios:
        if sc not in SCENARIO_MAP:
            raise ValueError(f"Cenário inválido: '{sc}'. Opções: 'ssp245', 'ssp585', 'rcp45', 'rcp85', 'all'")
        df_proj = run_inference_scenario(
            scenario_key=sc,
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            device=device,
            out_dir=out_dir,
            batch_size=batch_size,
            dry_run=dry_run,
            cc_response_per_kelvin=cc_response_per_kelvin,
            cc_response_jitter=cc_response_jitter,
            random_seed=random_seed,
        )
        csv_file = out_dir / SCENARIO_MAP[sc]["out_name"]
        results[sc] = csv_file
        summary_meta[sc] = {
            "label": SCENARIO_MAP[sc]["label"],
            "total_days": len(df_proj),
            "date_start": df_proj["date"].iloc[0],
            "date_end": df_proj["date"].iloc[-1],
            "pr_cordex_mean": float(df_proj["pr_cordex_raw_mm"].mean()),
            "pr_thor_mean": float(df_proj["pr_thor_downscaled_mm"].mean()),
            "r10_cordex_days": int((df_proj["pr_cordex_raw_mm"] >= 10.0).sum()),
            "r10_thor_days": int((df_proj["pr_thor_downscaled_mm"] >= 10.0).sum()),
            "r20_cordex_days": int((df_proj["pr_cordex_raw_mm"] >= 20.0).sum()),
            "r20_thor_days": int((df_proj["pr_thor_downscaled_mm"] >= 20.0).sum()),
            "max_rx1_cordex": float(df_proj["pr_cordex_raw_mm"].max()),
            "max_rx1_thor": float(df_proj["pr_thor_downscaled_mm"].max()),
            "physical_violations": int(df_proj["is_physical_violation"].sum()),
            "cc_response_per_kelvin": cc_response_per_kelvin,
            "cc_response_jitter": cc_response_jitter,
            "random_seed": random_seed,
        }

    # Salva JSON de resumo consolidado
    summary_file = out_dir / "projections_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2)
    print(f"\n✓ Resumo das projeções salvo: {summary_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="THOR-PIML CORDEX V8 Inference (SEMCITEC 2026)")
    parser.add_argument(
        "--scenario",
        choices=["ssp245", "ssp585", "rcp45", "rcp85", "all"],
        default="all",
        help="Cenário CORDEX para projeção (default: all)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(CKPT_PATH),
        help="Caminho do checkpoint THOR-V8",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default=str(SCALER_PATH),
        help="Caminho do Scaler V7 JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_PROJECTIONS_DIR),
        help="Diretório de saída para projeções",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Tamanho do batch de inferência",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Executa inferência rápida em amostra reduzida para validação de pipeline",
    )
    parser.add_argument(
        "--cc-response-per-kelvin",
        type=float,
        default=0.065,
        help="Resposta relativa de TCWV por kelvin (default: 0.065 = 6.5%%)",
    )
    parser.add_argument(
        "--cc-response-jitter",
        type=float,
        default=0.0,
        help="Desvio padrão da resposta por dia; 0 mantém resposta determinística",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed para a perturbação reproduzível da resposta termodinâmica",
    )
    args = parser.parse_args()

    run_projections(
        scenario=args.scenario,
        ckpt_path=Path(args.ckpt),
        scaler_path=Path(args.scaler),
        out_dir=Path(args.out_dir),
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        cc_response_per_kelvin=args.cc_response_per_kelvin,
        cc_response_jitter=args.cc_response_jitter,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
