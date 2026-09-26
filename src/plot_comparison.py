"""
THOR-PIML — Geração da Figura 6 via Inferência Neural 100% Real
===============================================================
Executa a inferência direta dos modelos neurais treinados a partir de seus
checkpoints oficiais (.pt), sem NENHUM mock, placeholder ou dado sintético:

1. THOR-V8 Espacial (checkpoints/v8_hybrid_seed42.pt) -> CNN 2D + LSTM-TCN + CC Barrier
2. THOR-V7 Híbrido (checkpoints/v7_hybrid_v7_v3_seed42.pt) -> LSTM + TCN Gated
3. ResLSTM (checkpoints/v7_lstm_v7_v3_seed42.pt) -> LSTM Residual (Kratzert 2018)
4. TCN (checkpoints/v7_tcn_v7_v3_seed42.pt) -> TCN Causal (Bai 2018)
5. EQM (Empirical Quantile Mapping - Gudmundsson 2012) -> Ajustado no Treino

Período: Janeiro/2020 (31 dias de verão convectivo crítico no Teste Cego).
Saídas:
  - results/real_predictions_jan2020.csv (Predições numéricas brutas extraídas da rede)
  - results/figures/fig6_timeseries_summer_30days.png (300 DPI)
  - results/figures/fig6_timeseries_summer_30days.pdf
"""
from __future__ import annotations
import sys
import os
import copy
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR
from src.v7.config_v7 import THORConfigV8, THORConfigV7, build_primary_cols_v7, build_feature_cols_v7, LAG_DAYS_V7
from src.v7.pipeline_v7 import load_v7_frame
from src.preprocessing import RobustClimateScaler, engineer_temporal_lags
from src.v7.model_v8 import THORSpatialHybridModel
from src.v7.model_v7 import THORHybridModel

GT_V3 = DATA_DIR / "ground_truth_guarulhos_daily_v3.csv"
NC_PATH = DATA_DIR / "era5pl_domain_daily_1981_2026.nc"
OUT_FIG_DIR = RESULTS_DIR / "figures"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoints oficiais treinados
CKPT_V8 = CHECKPOINT_DIR / "v8_hybrid_seed42.pt"
CKPT_V7 = CHECKPOINT_DIR / "v7_hybrid_v7_v3_seed42.pt"
CKPT_LSTM = CHECKPOINT_DIR / "v7_lstm_v7_v3_seed42.pt"
CKPT_TCN = CHECKPOINT_DIR / "v7_tcn_v7_v3_seed42.pt"


def validate_required_files():
    """Validação estrita de arquivos (Anti-Mock Hard Rule)."""
    for fpath, name in [
        (GT_V3, "Dataset GT V3"),
        (NC_PATH, "NetCDF ERA5 Pressure Levels"),
        (CKPT_V8, "Checkpoint THOR-V8"),
        (CKPT_V7, "Checkpoint THOR-V7"),
        (CKPT_LSTM, "Checkpoint ResLSTM"),
        (CKPT_TCN, "Checkpoint TCN"),
    ]:
        if not fpath.exists():
            raise FileNotFoundError(
                f"\n[ERRO CRÍTICO] {name} não encontrado em: {fpath}\n"
                "Regra Canônica: Não é permitido mock ou fallback sintético.\n"
                "Certifique-se de que os dados e checkpoints estejam presentes antes de rodar."
            )


def extract_real_window_tensors(csv_path: Path, nc_path: Path, seq_len: int = 30):
    """Extrai e normaliza tensores reais de superfície e espaciais (Zero-Data-Leakage)."""
    import xarray as xr

    print("[1/4] Carregando e processando features tabulares de superfície...")
    df = load_v7_frame(csv_path)
    primary_cols = build_primary_cols_v7(list(df.columns))
    df = engineer_temporal_lags(df, target_cols=primary_cols, lags=LAG_DAYS_V7, drop_na=True)
    feature_cols = [c for c in build_feature_cols_v7(primary_cols, LAG_DAYS_V7) if c in df.columns]

    X_full = df[feature_cols].values.astype(np.float32)
    y_full = df["pr_target"].values.astype(np.float32)

    # Split estrito: normalizador treinado apenas nos primeiros 70% (1981-2012)
    n_train = int(len(df) * 0.70)
    scaler = RobustClimateScaler(method="minmax")
    scaler.fit(X_full[:n_train])
    X_norm = scaler.transform(X_full)

    # Localizar os 31 dias de Janeiro/2020 no dataset
    jan_mask = (df["date"] >= "2020-01-01") & (df["date"] <= "2020-01-31")
    idx_start = df.index[jan_mask][0]
    idx_end = df.index[jan_mask][-1]

    print(f"[2/4] Carregando campo espacial 2D de {nc_path.name}...")
    ds = xr.open_dataset(nc_path)
    vars_2d = ["z500", "u700", "v700", "q700", "w500"]
    arr_spatial = np.stack([ds[v].values for v in vars_2d], axis=-1).astype(np.float32)
    ds.close()

    # Normalização espacial (min-max por canal baseada no treino)
    mean_sp = arr_spatial[:n_train].mean(axis=(0, 1, 2), keepdims=True)
    std_sp = arr_spatial[:n_train].std(axis=(0, 1, 2), keepdims=True) + 1e-6
    arr_spatial_norm = (arr_spatial - mean_sp) / std_sp

    # Construção de janelas deslizantes temporais de seq_len (30 dias)
    xs_list, xsp_list, y_obs, dates_list = [], [], [], []
    for i in range(idx_start, idx_end + 1):
        xs_win = X_norm[i - seq_len + 1 : i + 1]  # (30, n_features)
        xsp_win = arr_spatial_norm[i - seq_len + 1 : i + 1]  # (30, H, W, C=5)

        xs_list.append(xs_win)
        xsp_list.append(xsp_win)
        y_obs.append(y_full[i])
        dates_list.append(df["date"].iloc[i])

    xs_tensor = torch.tensor(np.array(xs_list), dtype=torch.float32)
    xsp_tensor = torch.tensor(np.array(xsp_list), dtype=torch.float32)

    # Dados para treinamento e avaliação do EQM
    pr_primary = [c for c in primary_cols if c not in ("pr_grid_max", "pr_grid_std")]
    pr_col = "tcwv" if "tcwv" in pr_primary else pr_primary[0]
    tcwv_train = df[pr_col].values[:n_train]
    y_train = y_full[:n_train]
    tcwv_jan = df[pr_col].iloc[idx_start : idx_end + 1].values

    return (
        xs_tensor,
        xsp_tensor,
        np.array(y_obs),
        dates_list,
        len(feature_cols),
        tcwv_train,
        y_train,
        tcwv_jan,
    )


def run_real_inference():
    validate_required_files()

    print("=" * 70)
    print("THOR-PIML: INFERÊNCIA NEURAL REAL DOS CHECKPOINTS TREINADOS (SEM MOCK)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando em dispositivo: {device}")

    xs, xsp, obs_vals, dates_list, n_feat, tcwv_tr, y_tr, tcwv_jan = extract_real_window_tensors(
        GT_V3, NC_PATH, seq_len=30
    )
    xs = xs.to(device)
    xsp = xsp.to(device)
    print(f"✓ Tensores de teste carregados: {len(obs_vals)} dias (Janeiro/2020) | Features: {n_feat}")

    print("\n[3/4] Executando inferência direta em cada rede neural...")

    # 1. THOR-V8 Espacial
    print("  -> Carregando THOR-V8 (PIML Espaço-Temporal)...")
    cfg8 = THORConfigV8()
    cfg8.model.n_features = n_feat
    model_v8 = THORSpatialHybridModel(cfg8.model).to(device)
    ckpt8 = torch.load(CKPT_V8, map_location=device, weights_only=False)
    model_v8.load_state_dict(ckpt8["model_state_dict"])
    model_v8.eval()
    with torch.no_grad():
        _, _, final_v8 = model_v8(xs, xsp, return_components=True)
        v8_preds = final_v8.squeeze(-1).cpu().numpy()

    # 2. THOR-V7 Híbrido
    print("  -> Carregando THOR-V7 (Híbrido Temporal)...")
    cfg7 = THORConfigV7()
    cfg7.model.n_features = n_feat
    cfg7.model.use_lstm_branch = True
    cfg7.model.use_tcn_branch = True
    model_v7 = THORHybridModel(cfg7.model).to(device)
    ckpt7 = torch.load(CKPT_V7, map_location=device, weights_only=False)
    model_v7.load_state_dict(ckpt7["model_state_dict"])
    model_v7.eval()
    with torch.no_grad():
        _, _, final_v7 = model_v7(xs, return_components=True)
        v7_preds = final_v7.squeeze(-1).cpu().numpy()

    # 3. ResLSTM (Kratzert 2018)
    print("  -> Carregando ResLSTM (2018)...")
    cfg_lstm = copy.deepcopy(cfg7.model)
    cfg_lstm.use_lstm_branch = True
    cfg_lstm.use_tcn_branch = False
    model_lstm = THORHybridModel(cfg_lstm).to(device)
    ckpt_lstm = torch.load(CKPT_LSTM, map_location=device, weights_only=False)
    model_lstm.load_state_dict(ckpt_lstm["model_state_dict"])
    model_lstm.eval()
    with torch.no_grad():
        _, _, final_lstm = model_lstm(xs, return_components=True)
        lstm_preds = final_lstm.squeeze(-1).cpu().numpy()

    # 4. TCN (Bai 2018)
    print("  -> Carregando TCN (2018)...")
    cfg_tcn = copy.deepcopy(cfg7.model)
    cfg_tcn.use_lstm_branch = False
    cfg_tcn.use_tcn_branch = True
    model_tcn = THORHybridModel(cfg_tcn).to(device)
    ckpt_tcn = torch.load(CKPT_TCN, map_location=device, weights_only=False)
    model_tcn.load_state_dict(ckpt_tcn["model_state_dict"])
    model_tcn.eval()
    with torch.no_grad():
        _, _, final_tcn = model_tcn(xs, return_components=True)
        tcn_preds = final_tcn.squeeze(-1).cpu().numpy()

    # 5. EQM (Gudmundsson 2012)
    print("  -> Computando EQM (Gudmundsson 2012) via CDF Empírica...")
    qs = np.linspace(0.0001, 0.9999, 2000)
    obs_q = np.quantile(y_tr, qs)
    mod_q = np.quantile(tcwv_tr, qs)
    eqm_preds = np.interp(tcwv_jan, mod_q, obs_q)
    eqm_preds = np.clip(eqm_preds, 0.0, None)

    # Montar DataFrame com todas as predições numéricas reais
    df_results = pd.DataFrame({
        "date": pd.to_datetime(dates_list),
        "obs_chirps": obs_vals,
        "thor_v8_pred": v8_preds,
        "thor_v7_pred": v7_preds,
        "reslstm_pred": lstm_preds,
        "tcn_pred": tcn_preds,
        "eqm_pred": eqm_preds,
    })

    csv_out = RESULTS_DIR / "real_predictions_jan2020.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"\n✓ Predições numéricas brutas salvas em: {csv_out}")

    # =========================================================================
    # [4/4] Plotagem Científica Editorial da Figura 6 (Alta Resolução - 300 DPI)
    # =========================================================================
    print("\n[4/4] Gerando Figura 6 com as curvas de inferência 100% reais...")
    
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14.2, 7.8), gridspec_kw={"height_ratios": [2.3, 1.0]}, dpi=300
    )

    dates = df_results["date"].values
    d_start = pd.to_datetime(df_results["date"].iloc[0])
    d_end = pd.to_datetime(df_results["date"].iloc[-1])

    # Painel Superior: Séries Temporais Diárias
    ax1.bar(
        dates,
        df_results["obs_chirps"],
        width=0.55,
        color="#64748b",
        alpha=0.35,
        label="Observado (CHIRPS 5.5km)",
        edgecolor="#334155",
        linewidth=0.8,
    )
    ax1.plot(
        dates,
        df_results["eqm_pred"],
        color="#db2777",
        ls="--",
        lw=1.6,
        marker="x",
        markersize=4,
        label="EQM (2012)",
        alpha=0.85,
    )
    ax1.plot(
        dates,
        df_results["reslstm_pred"],
        color="#ea580c",
        ls="--",
        lw=1.8,
        marker="^",
        markersize=4.5,
        label="ResLSTM (2018)",
        alpha=0.85,
    )
    ax1.plot(
        dates,
        df_results["tcn_pred"],
        color="#8b5cf6",
        ls=":",
        lw=1.8,
        marker="v",
        markersize=4.5,
        label="TCN (2018)",
        alpha=0.85,
    )
    ax1.plot(
        dates,
        df_results["thor_v7_pred"],
        color="#38bdf8",
        ls="-.",
        lw=2.0,
        marker="s",
        markersize=4.5,
        label="THOR-V7 (Híbrido Temporal)",
        alpha=0.9,
    )
    ax1.plot(
        dates,
        df_results["thor_v8_pred"],
        color="#0284c7",
        ls="-",
        lw=2.6,
        marker="D",
        markersize=5.5,
        label="THOR-V8 (PIML Espaço-Temporal)",
        alpha=0.95,
    )

    ax1.set_title(
        "Comparação Temporal de Downscaling Diário — Janeiro/2020 (Guarulhos-SP)\n"
        "[Inferência Real Direta dos Checkpoints Treinados • Período Crítico de Verão Convectivo]",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylabel("Precipitação Diária (mm/dia)", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.6)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax1.set_xlim(d_start - pd.Timedelta(days=0.5), d_end + pd.Timedelta(days=0.5))
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Painel Inferior: Resíduos Reais (Predição - Observado)
    err_v8 = df_results["thor_v8_pred"] - df_results["obs_chirps"]
    err_lstm = df_results["reslstm_pred"] - df_results["obs_chirps"]
    err_tcn = df_results["tcn_pred"] - df_results["obs_chirps"]

    ax2.axhline(0, color="#0f172a", linestyle="-", linewidth=1.0, alpha=0.8)
    ax2.plot(
        dates,
        err_lstm,
        color="#ea580c",
        linestyle="--",
        linewidth=1.4,
        label="Resíduo Real ResLSTM (2018)",
    )
    ax2.plot(
        dates,
        err_tcn,
        color="#8b5cf6",
        linestyle=":",
        linewidth=1.4,
        label="Resíduo Real TCN (2018)",
    )
    ax2.plot(
        dates,
        err_v8,
        color="#0284c7",
        linestyle="-",
        linewidth=2.0,
        label="Resíduo Real THOR-V8 (PIML)",
    )

    ax2.set_ylabel("Resíduo (mm)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Dia do Mês (Janeiro/2020)", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax2.set_xlim(d_start - pd.Timedelta(days=0.5), d_end + pd.Timedelta(days=0.5))
    ax2.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    out_png = OUT_FIG_DIR / "fig6_timeseries_summer_30days.png"
    out_pdf = OUT_FIG_DIR / "fig6_timeseries_summer_30days.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"✓ Figura 6 (PNG 300 DPI) salva em: {out_png}")
    print(f"✓ Figura 6 (PDF Vetorial) salva em: {out_pdf}")
    print("=" * 70)
    print("✅ PROCESSO DE INFERÊNCIA REAL CONCLUÍDO COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    run_real_inference()
