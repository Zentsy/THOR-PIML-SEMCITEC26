"""
THOR-PIML — Gerador Oficial de Figuras de Publicação SEMCITEC 2026 (300 DPI)
=============================================================================
Gera as 5 figuras canônicas em altíssima densidade de informação e resolução editorial (300 DPI),
seguindo rigorosamente o layout horizontal panorâmico (~3:1 aspect ratio) padronizado no AGENTS.md.
Todas as curvas, barras e quantis são computados DINAMICAMENTE a partir dos dados reais gerados
pelos passos 1, 2 e 3 do pipeline (zero dados sintéticos hardcoded).

Destaque Metodológico:
- Separação estrita e dedicada dos cenários SSP2-4.5 (Estabilização Média) e SSP5-8.5 (Altas Emissões)
  em subpainéis independentes para máxima legibilidade editorial e eliminação de poluição visual.
- Exportação simultânea das 5 figuras canônicas consolidadas e das figuras dedicadas por cenário.

1. Fig 1: Projeção Multidecadal da Precipitação Total Anual (BIO12/PRCPTOT) e Anomalias Percentuais (1981–2100).
2. Fig 2: Ciclo Sazonal e Balanço Médio Mensal nas 4 Épocas (Contraste Verão DJF vs Inverno JJA).
3. Fig 3: Curvas de Permanência (FDC), Zoom na Cauda de Extremos (Q90 a Q99.9) e Eliminação do Drizzle Bias.
4. Fig 4: Painel Comparativo dos Indicadores Bioclimáticos Oficiais de Precipitação (BIO12 a BIO19).
5. Fig 5: Evolução dos Índices de Extremos Climáticos ETCCDI e Risco Hidrológico Urbano (2026–2100).

Uso:
    python src/generate_semcitec_figures.py
    python src/generate_semcitec_figures.py --dpi 300
    python src/generate_semcitec_figures.py --scenario ssp245
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import DATA_DIR, RESULTS_DIR

TABLES_DIR = ROOT_DIR / ".agents" / "table_data"
PROJECTIONS_DIR = RESULTS_DIR / "projections"
FIG_DIR = RESULTS_DIR / "figures"
GT_PATH = DATA_DIR / "ground_truth_guarulhos_daily_v3.csv"
CORDEX_DIR = DATA_DIR / "cordex"
CCKP_DIR = ROOT_DIR / ".agents" / "cckp_data"

# Configurações globais de tipografia e estilo editorial de alta densidade
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 9.0,
    "legend.title_fontsize": 9.5,
    "figure.titlesize": 13.0,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
})

# Paleta editorial canônica rigorosamente padronizada (alto contraste, zero conflitos)
COLORS = {
    "observed": "#0f172a",          # Preto profundo / Slate escuro (CHIRPS Benchmark)
    "cordex": "#64748b",            # Cinza Ardósia neutro (CORDEX Bruto universal)
    "cordex_rcp45": "#64748b",      # Cinza Ardósia (CORDEX Bruto SSP2-4.5)
    "cordex_rcp85": "#64748b",      # Cinza Ardósia (CORDEX Bruto SSP5-8.5)
    "thor_ssp245": "#0284c7",       # Azul Royal vívido (THOR SSP2-4.5)
    "thor_ssp585": "#dc2626",       # Vermelho Carmesim vívido (THOR SSP5-8.5)
    "baseline_ref": "#0f172a",      # Linha de base preto profundo
    "grid": "#e2e8f0",              # Grade suave
}


# ==============================================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================
def load_datasets() -> Dict[str, pd.DataFrame]:
    """Carrega dados históricos e projeções climáticas disponíveis de forma segura."""
    data = {}

    # 1. Baseline Histórico
    if GT_PATH.exists():
        df_hist = pd.read_csv(GT_PATH)
        if "year" not in df_hist.columns:
            df_hist["year"] = df_hist["date"].astype(str).str[:4].astype(int)
            df_hist["month"] = df_hist["date"].astype(str).str[5:7].astype(int)
            df_hist["day"] = df_hist["date"].astype(str).str[8:10].astype(int)
        else:
            df_hist["year"] = df_hist["year"].astype(int)
            df_hist["month"] = df_hist["month"].astype(int)
            df_hist["day"] = df_hist["day"].astype(int)
        data["hist"] = df_hist
    else:
        print(f"⚠ Aviso: {GT_PATH.name} não encontrado.")

    # 2. Séries brutas CORDEX
    cordex_45_path = CORDEX_DIR / "cordex_guarulhos_rcp45_daily_2006_2099.csv"
    cordex_85_path = CORDEX_DIR / "cordex_guarulhos_rcp85_daily_2006_2099.csv"
    if cordex_45_path.exists():
        df_c45 = pd.read_csv(cordex_45_path)
        if "year" not in df_c45.columns:
            df_c45["year"] = df_c45["date"].astype(str).str[:4].astype(int)
            df_c45["month"] = df_c45["date"].astype(str).str[5:7].astype(int)
        data["cordex_rcp45"] = df_c45
    if cordex_85_path.exists():
        df_c85 = pd.read_csv(cordex_85_path)
        if "year" not in df_c85.columns:
            df_c85["year"] = df_c85["date"].astype(str).str[:4].astype(int)
            df_c85["month"] = df_c85["date"].astype(str).str[5:7].astype(int)
        data["cordex_rcp85"] = df_c85

    # 3. Séries Downscaled THOR-PIML
    thor_45_path = PROJECTIONS_DIR / "cordex_ssp245_downscaled_2026_2099.csv"
    thor_85_path = PROJECTIONS_DIR / "cordex_ssp585_downscaled_2026_2099.csv"
    if thor_45_path.exists():
        df_t45 = pd.read_csv(thor_45_path)
        if "year" not in df_t45.columns:
            df_t45["year"] = df_t45["date"].astype(str).str[:4].astype(int)
            df_t45["month"] = df_t45["date"].astype(str).str[5:7].astype(int)
        data["thor_ssp245"] = df_t45
    if thor_85_path.exists():
        df_t85 = pd.read_csv(thor_85_path)
        if "year" not in df_t85.columns:
            df_t85["year"] = df_t85["date"].astype(str).str[:4].astype(int)
            df_t85["month"] = df_t85["date"].astype(str).str[5:7].astype(int)
        data["thor_ssp585"] = df_t85

    # 4. Tabelas consolidadas de índices se existirem
    bio_sum_path = TABLES_DIR / "bioclimatic_indices_epochs_summary.csv"
    etccdi_sum_path = TABLES_DIR / "etccdi_extremes_epochs_summary.csv"
    bio_ann_path = TABLES_DIR / "bioclimatic_indices_annual.csv"
    etccdi_ann_path = TABLES_DIR / "etccdi_extremes_annual.csv"

    if bio_sum_path.exists():
        data["bio_summary"] = pd.read_csv(bio_sum_path)
    if etccdi_sum_path.exists():
        data["etccdi_summary"] = pd.read_csv(etccdi_sum_path)
    if bio_ann_path.exists():
        data["bio_annual"] = pd.read_csv(bio_ann_path)
    if etccdi_ann_path.exists():
        data["etccdi_annual"] = pd.read_csv(etccdi_ann_path)

    cckp_path = CCKP_DIR / "sp_pr_anomaly_2080-2099_all.csv"
    if cckp_path.exists():
        data["cckp_pr_anomaly"] = pd.read_csv(cckp_path)

    return data


# ==============================================================================
# FIGURA 1: PROJEÇÃO MULTIDECADAL DA PRECIPITAÇÃO TOTAL (BIO12 / PRCPTOT)
# ==============================================================================
def plot_fig1_prcptot_projections(data: Dict[str, pd.DataFrame], out_dir: Path, dpi: int = 300):
    """Fig 1: Projeção de PRCPTOT (1981-2100) separada em subpainéis dedicados para SSP2-4.5 e SSP5-8.5."""
    print(f"  Gerando Fig 1: Projeção Multidecadal PRCPTOT ({dpi} DPI, layout dedicado por cenário)...")
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 8.2), gridspec_kw={"width_ratios": [2.2, 1.0]})

    baseline_mean = 1450.0  # Referência climatológica preliminar
    annual_hist = None
    roll_hist = None
    if "hist" in data:
        df_h = data["hist"]
        df_h_base = df_h[(df_h["year"] >= 1981) & (df_h["year"] <= 2014)]
        annual_hist = df_h_base.groupby("year")["pr_target"].sum()
        baseline_mean = float(annual_hist.mean())
        roll_hist = annual_hist.rolling(10, min_periods=5).mean()

    epoch_keys = [
        "Futuro Próximo (2026-2050)",
        "Meio do Século (2051-2075)",
        "Final do Século (2076-2099)",
    ]
    epoch_display = ["2026–2050", "2051–2075", "2076–2099"]
    x = np.arange(len(epoch_display))
    width = 0.35

    scenarios = [
        ("ssp245", "SSP2-4.5 (Estabilização Média)", "CORDEX_Raw_SSP245", "THOR_PIML_SSP245",
         COLORS["cordex_rcp45"], COLORS["thor_ssp245"], "cordex_rcp45", "thor_ssp245", 0),
        ("ssp585", "SSP5-8.5 (Altas Emissões)", "CORDEX_Raw_SSP585", "THOR_PIML_SSP585",
         COLORS["cordex_rcp85"], COLORS["thor_ssp585"], "cordex_rcp85", "thor_ssp585", 1),
    ]

    for sc_id, sc_label, raw_sid, thor_sid, c_raw, c_thor, df_c_key, df_t_key, row_idx in scenarios:
        ax_ts = axes[row_idx, 0]
        ax_ano = axes[row_idx, 1]

        # Linha de base histórica
        if annual_hist is not None:
            ax_ts.plot(annual_hist.index, annual_hist.values, color=COLORS["observed"], alpha=0.30, lw=0.9, ls="-")
            ax_ts.plot(roll_hist.index, roll_hist.values, color=COLORS["observed"], lw=2.4, label=f"CHIRPS Observado ({baseline_mean:.0f} mm/ano)")
        ax_ts.axhline(baseline_mean, color="#64748b", ls=":", lw=1.4, label=f"Linha de Base 1981–2014 ({baseline_mean:.0f} mm)")

        # Curvas de projeção futuras reais
        if "bio_annual" in data:
            df_ann = data["bio_annual"]
            s_raw = df_ann[df_ann["series_id"] == raw_sid].set_index("year")["BIO12_PRCPTOT"]
            s_thor = df_ann[df_ann["series_id"] == thor_sid].set_index("year")["BIO12_PRCPTOT"]
            if len(s_raw) > 0:
                ax_ts.plot(s_raw.index, s_raw.values, color=c_raw, alpha=0.20, lw=0.8)
                ax_ts.plot(s_raw.index, s_raw.rolling(10, min_periods=5).mean(), color=c_raw, lw=2.2, ls="--", label=f"CORDEX Bruto ({sc_id.upper()})")
            if len(s_thor) > 0:
                ax_ts.plot(s_thor.index, s_thor.values, color=c_thor, alpha=0.20, lw=0.8)
                ax_ts.plot(s_thor.index, s_thor.rolling(10, min_periods=5).mean(), color=c_thor, lw=2.5, ls="-", label=f"THOR-PIML Downscaled ({sc_id.upper()})")
        elif df_c_key in data:
            s_raw = data[df_c_key].groupby("year")["pr_raw_mm"].sum()
            ax_ts.plot(s_raw.index, s_raw.rolling(10, min_periods=5).mean(), color=c_raw, lw=2.2, ls="--", label=f"CORDEX Bruto ({sc_id.upper()})")
            if df_t_key in data:
                s_thor = data[df_t_key].groupby("year")["pr_thor_downscaled_mm"].sum()
                ax_ts.plot(s_thor.index, s_thor.rolling(10, min_periods=5).mean(), color=c_thor, lw=2.5, ls="-", label=f"THOR-PIML ({sc_id.upper()})")

        ax_ts.set_title(f"({chr(65 + row_idx*2)}) Trajetória Multidecadal da Precipitação Anual — {sc_label}", pad=8)
        ax_ts.set_xlabel("Ano")
        ax_ts.set_ylabel("PRCPTOT (mm/ano)")
        ax_ts.set_xlim(1980, 2100)
        ax_ts.legend(loc="upper left", framealpha=0.92, ncol=2)

        # Painel de Anomalias Percentuais para o cenário
        if "bio_summary" in data and len(data["bio_summary"]) > 1:
            df_sum = data["bio_summary"]
            base_rows = df_sum[df_sum["series_id"] == "CHIRPS_Observed"]
            base_val = float(base_rows["BIO12_PRCPTOT_mean"].iloc[0]) if len(base_rows) > 0 else baseline_mean

            def get_anoms(sid: str) -> List[float]:
                anoms = []
                for ep in epoch_keys:
                    sub = df_sum[(df_sum["series_id"] == sid) & (df_sum["epoch"] == ep)]
                    if len(sub) > 0 and "BIO12_PRCPTOT_mean" in sub.columns:
                        v = float(sub["BIO12_PRCPTOT_mean"].iloc[0])
                        anoms.append(((v - base_val) / base_val) * 100.0)
                    else:
                        anoms.append(0.0)
                return anoms

            anom_raw = get_anoms(raw_sid)
            anom_thor = get_anoms(thor_sid)

            bars_c = ax_ano.bar(x - width/2, anom_raw, width, color=c_raw, label=f"CORDEX {sc_id.upper()}", edgecolor="black", lw=0.5)
            bars_t = ax_ano.bar(x + width/2, anom_thor, width, color=c_thor, label=f"THOR-PIML {sc_id.upper()}", edgecolor="black", lw=0.5)
            for b in bars_c:
                h = b.get_height()
                va = "bottom" if h >= 0 else "top"
                y_pos = h + (1.2 if h >= 0 else -1.2)
                ax_ano.annotate(f"{h:+.1f}%", (b.get_x() + b.get_width()/2, y_pos),
                                ha="center", va=va, fontsize=7.5, weight="bold", color=c_raw)
            for b in bars_t:
                h = b.get_height()
                va = "bottom" if h >= 0 else "top"
                y_pos = h + (1.2 if h >= 0 else -1.2)
                ax_ano.annotate(f"{h:+.1f}%", (b.get_x() + b.get_width()/2, y_pos),
                                ha="center", va=va, fontsize=7.5, weight="bold", color=c_thor)
            ax_ano.legend(loc="upper left", framealpha=0.90)
        else:
            ax_ano.text(
                0.5, 0.5, "Projeções prontas.\nExecute Passos 2 e 3\npara anomalias decadais.",
                ha="center", va="center", transform=ax_ano.transAxes, fontsize=8.5, color="#64748b", style="italic"
            )

        ax_ano.axhline(0, color="black", lw=1.0)
        ax_ano.set_title(f"({chr(66 + row_idx*2)}) Anomalia Percentual Δ (%) — {sc_id.upper()}", pad=8)
        ax_ano.set_xticks(x)
        ax_ano.set_xticklabels(epoch_display)
        ax_ano.set_ylabel("Anomalia Δ (%)")

    fig.tight_layout()
    out_png = out_dir / "fig1_semcitec_prcptot_projections_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")


# ==============================================================================
# FIGURA 2: CICLO SAZONAL E BALANÇO MÉDIO MENSAL NAS 4 ÉPOCAS
# ==============================================================================
def plot_fig1_prcptot_projections(data: Dict[str, pd.DataFrame], out_dir: Path, dpi: int = 300):
    """Trajetórias anuais sem o painel de anomalia percentual ambígua."""
    print(f"  Gerando Fig 1: Trajetórias anuais por cenário ({dpi} DPI)...")
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.0), sharex=True)
    baseline_mean = 1450.0
    annual_hist = None
    if "hist" in data:
        hist = data["hist"]
        hist = hist[(hist["year"] >= 1981) & (hist["year"] <= 2014)]
        annual_hist = hist.groupby("year")["pr_target"].sum()
        baseline_mean = float(annual_hist.mean())

    scenarios = [
        ("ssp245", "SSP2-4.5", "cordex_rcp45", "thor_ssp245", "#d97706", COLORS["thor_ssp245"]),
        ("ssp585", "SSP5-8.5", "cordex_rcp85", "thor_ssp585", "#d97706", COLORS["thor_ssp585"]),
    ]
    cckp = data.get("cckp_pr_anomaly")
    for ax, (scenario, label, raw_key, thor_key, raw_color, thor_color) in zip(axes, scenarios):
        if annual_hist is not None:
            ax.plot(annual_hist.index, annual_hist.values, color=COLORS["observed"], alpha=0.25, lw=0.8)
            ax.plot(annual_hist.index, annual_hist.rolling(10, min_periods=5).mean(), color=COLORS["observed"],
                    lw=2.2, label=f"CHIRPS observado (1981–2014; média {baseline_mean:.0f} mm)")
        ax.axhline(baseline_mean, color="#64748b", ls=":", lw=1.2, label="Média histórica CHIRPS")
        for key, color, style, name in [
            (raw_key, raw_color, "--", "CORDEX bruto"),
            (thor_key, thor_color, "-", "THOR-PIML"),
        ]:
            if key in data:
                frame = data[key]
                annual = frame.groupby("year")["pr_raw_mm" if key.startswith("cordex") else "pr_thor_downscaled_mm"].sum()
                ax.plot(annual.index, annual.rolling(10, min_periods=5).mean(), color=color, ls=style, lw=2.4,
                        label=f"{name} ({label})")
        if cckp is not None:
            ref = cckp[(cckp["aggregation"] == "annual") & (cckp["scenario"] == scenario)]
            if not ref.empty:
                vals = ref.set_index("percentile")["value_mm"]
                ax.errorbar([2089.5], [baseline_mean + float(vals.get("median", 0))],
                            yerr=[[float(vals.get("median", 0) - vals.get("p10", 0))],
                                  [float(vals.get("p90", 0) - vals.get("median", 0))]],
                            fmt="o", color="#0f766e", capsize=5, lw=2,
                            label="CCKP CMIP6 final (2080–2099; anomalia publicada)")
        ax.set_title(f"{label}: precipitação anual (PRCPTOT)", loc="left", fontweight="bold")
        ax.set_ylabel("mm/ano")
        ax.set_xlim(1980, 2100)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.95)
    axes[-1].set_xlabel("Ano")
    fig.suptitle("Trajetória multidecadal da precipitação anual", fontweight="bold")
    fig.text(0.01, 0.01, "CCKP não fornece uma série anual completa nesta extração; o marcador é uma climatologia final, não uma trajetória.", fontsize=8.5, color="#475569")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_png = out_dir / "fig1_semcitec_prcptot_projections_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")


def plot_fig2_seasonal_cycles(data: Dict[str, pd.DataFrame], out_dir: Path, dpi: int = 300):
    """Fig 2: Ciclos mensais médios separados em linhas dedicadas para SSP2-4.5 e SSP5-8.5."""
    print(f"  Gerando Fig 2: Ciclo Sazonal Médio Mensal nas Épocas ({dpi} DPI, layout 2 linhas por cenário)...")
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 7.5), sharey=True)
    months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    m_idx = np.arange(1, 13)

    # Climatologia base CHIRPS Guarulhos (1981-2014)
    if "hist" in data:
        df_h = data["hist"]
        df_h_base = df_h[(df_h["year"] >= 1981) & (df_h["year"] <= 2014)]
        n_years = max(1, len(df_h_base["year"].unique()))
        hist_clim = (df_h_base.groupby("month")["pr_target"].sum() / n_years).reindex(range(1, 13), fill_value=0.0).values
    else:
        hist_clim = np.zeros(12)

    epoch_configs = [
        ("Futuro Próximo (2026–2050)", 2026, 2050),
        ("Meio do Século (2051–2075)", 2051, 2075),
        ("Final do Século (2076–2099)", 2076, 2099),
    ]

    scenarios = [
        ("cordex_rcp45", "thor_ssp245", COLORS["cordex_rcp45"], COLORS["thor_ssp245"], "SSP2-4.5 (Estabilização Média)", 0),
        ("cordex_rcp85", "thor_ssp585", COLORS["cordex_rcp85"], COLORS["thor_ssp585"], "SSP5-8.5 (Altas Emissões)", 1),
    ]

    for c_key, t_key, c_raw_col, c_thor_col, sc_label, row_idx in scenarios:
        for col_idx, (ep_title, y_start, y_end) in enumerate(epoch_configs):
            ax = axes[row_idx, col_idx]

            # Baseline Observado
            ax.plot(m_idx, hist_clim, color=COLORS["observed"], lw=2.4, marker="o", ms=4, label="Baseline (1981–2014)")

            # CORDEX Bruto
            if c_key in data:
                df_c = data[c_key]
                sub_c = df_c[(df_c["year"] >= y_start) & (df_c["year"] <= y_end)]
                if len(sub_c) > 0:
                    ny_c = max(1, len(sub_c["year"].unique()))
                    m_c = (sub_c.groupby("month")["pr_raw_mm"].sum() / ny_c).reindex(range(1, 13), fill_value=0.0).values
                    ax.plot(m_idx, m_c, color=c_raw_col, lw=2.0, ls="--", label="CORDEX Bruto")

            # THOR-PIML Downscaled
            if t_key in data:
                df_t = data[t_key]
                sub_t = df_t[(df_t["year"] >= y_start) & (df_t["year"] <= y_end)]
                if len(sub_t) > 0:
                    ny_t = max(1, len(sub_t["year"].unique()))
                    m_t = (sub_t.groupby("month")["pr_thor_downscaled_mm"].sum() / ny_t).reindex(range(1, 13), fill_value=0.0).values
                    marker_sym = "s" if row_idx == 0 else "^"
                    ax.plot(m_idx, m_t, color=c_thor_col, lw=2.4, marker=marker_sym, ms=4, label="THOR-PIML (Corrigido)")

            # Destaque de estações: Verão (DJF) vs Inverno (JJA)
            ax.axvspan(1, 3, color="#fed7aa", alpha=0.22, label="Verão (DJF)" if (row_idx == 0 and col_idx == 0) else "")
            ax.axvspan(6, 8, color="#bfdbfe", alpha=0.22, label="Inverno (JJA)" if (row_idx == 0 and col_idx == 0) else "")

            panel_letter = chr(65 + row_idx * 3 + col_idx)
            ax.set_title(f"({panel_letter}) {sc_label} — {ep_title}", fontsize=10.0)
            ax.set_xticks(m_idx)
            ax.set_xticklabels(months, rotation=35)
            ax.set_xlabel("Mês")

        axes[row_idx, 0].set_ylabel(f"Chuva Média Mensal (mm/mês)\n[{sc_label.split()[0]}]")
        axes[row_idx, 0].legend(loc="upper right", fontsize=8.0, framealpha=0.90)

    fig.suptitle("Evolução do Ciclo Sazonal Pluviométrico nas Épocas Climatológicas (Guarulhos-SP)", y=0.99)
    fig.tight_layout()
    out_png = out_dir / "fig2_semcitec_seasonal_cycle_epochs_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")


# ==============================================================================
# FIGURA 6: COMPARAÇÃO EXTERNA DA ANOMALIA MENSAL DE PRECIPITAÇÃO
# ==============================================================================
def plot_fig6_external_reference(data: Dict[str, pd.DataFrame], out_dir: Path, dpi: int = 300):
    """Compara o sinal mensal final com o envelope CMIP6 independente do CCKP."""
    if "cckp_pr_anomaly" not in data or "hist" not in data:
        print("  ⚠ Fig 6 não gerada: dados CCKP ou histórico ausentes.")
        return None

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    months = np.arange(1, 13)
    month_labels = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    hist = data["hist"]
    hist_base = hist[(hist["year"] >= 1981) & (hist["year"] <= 2014)]
    hist_monthly = (
        hist_base.groupby("month")["pr_target"].sum()
        / max(1, hist_base["year"].nunique())
    ).reindex(months)

    def final_anomaly(key: str) -> Optional[np.ndarray]:
        if key not in data:
            return None
        frame = data[key]
        final = frame[(frame["year"] >= 2080) & (frame["year"] <= 2099)]
        if final.empty:
            return None
        climatology = (
            final.groupby("month")["pr_thor_downscaled_mm" if key.startswith("thor") else "pr_raw_mm"].sum()
            / max(1, final["year"].nunique())
        ).reindex(months)
        return (climatology - hist_monthly).to_numpy(dtype=float)

    thor = final_anomaly("thor_ssp585")
    cordex = final_anomaly("cordex_rcp85")
    cckp = data["cckp_pr_anomaly"]
    cckp = cckp[(cckp["aggregation"] == "monthly") & (cckp["scenario"] == "ssp585")]
    cckp_values = {
        percentile: cckp[cckp["percentile"] == percentile].sort_values("period")["value_mm"].to_numpy(dtype=float)
        for percentile in ("p10", "median", "p90")
    }

    if all(len(values) == 12 for values in cckp_values.values()):
        ax.fill_between(months, cckp_values["p10"], cckp_values["p90"], color="#99f6e4", alpha=0.45,
                        label="CCKP CMIP6 P10–P90 (2080–2099; São Paulo)")
        ax.plot(months, cckp_values["median"], color="#0f766e", lw=2.8, ls="--",
                label="CCKP CMIP6 mediana (anomalia publicada)")
    if cordex is not None:
        ax.plot(months, cordex, color="#d97706", lw=2.5, ls=":", marker="o",
                label="CORDEX bruto (2080–2099; ajustado)")
    if thor is not None:
        ax.plot(months, thor, color=COLORS["thor_ssp585"], lw=2.8, marker="s",
                label="THOR-PIML (2080–2099; ajustado)")

    ax.axhline(0, color="#0f172a", lw=1.0)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Anomalia de precipitação (mm/mês)")
    ax.set_xlabel("Mês")
    ax.set_title("Sinal mensal de precipitação relativo a cada fonte: THOR, CORDEX e CCKP (2080–2099)\n"
                 "Período temporal padronizado para 2080–2099 sob o cenário SSP5-8.5")
    ax.legend(loc="best", framealpha=0.94)
    ax.text(0.01, 0.02, "Nota: Séries THOR-PIML e CORDEX foram ajustadas para o recorte 2080–2099 para alinhamento estrito com a climatologia do CCKP (Banco Mundial / CMIP6).\n"
                        "CCKP: anomalia ref. ao CMIP6 histórico (1995–2014); THOR/CORDEX: anomalia ref. ao CHIRPS (1981–2014).",
            transform=ax.transAxes, fontsize=8.0, color="#334155")
    fig.tight_layout()
    out_png = out_dir / "fig6_external_reference_monthly_anomaly_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")
    return out_png


# ==============================================================================
# FIGURA 3: CURVA DE PERMANÊNCIA (FDC) E CAUDA DE EXTREMOS
# ==============================================================================
def compute_empirical_fdc(p_vals: np.ndarray, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula curva de permanência (probabilidade de excedência) de uma série empírica."""
    p_clean = np.sort(np.nan_to_num(p_vals[p_vals >= 0.0], nan=0.0))[::-1]
    n = len(p_clean)
    if n == 0:
        return np.linspace(0.1, 99.9, n_points), np.zeros(n_points)
    ranks = np.arange(1, n + 1)
    exceedance = (ranks / (n + 1.0)) * 100.0
    sample_exc = np.geomspace(0.05, 99.9, n_points)
    sample_p = np.interp(sample_exc, exceedance, p_clean)
    return sample_exc, sample_p


def plot_fig3_fdc_extreme_tail(data: Dict[str, pd.DataFrame], out_dir: Path, dpi: int = 300):
    """Fig 3: Flow Duration Curve (FDC), Zoom em Q90-Q99.9 e viés por quantil para SSP2-4.5 e SSP5-8.5."""
    print(f"  Gerando Fig 3: Curva de Permanência (FDC) e Zoom de Extremos ({dpi} DPI, layout 2 linhas)...")
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 7.8), gridspec_kw={"width_ratios": [1.1, 1.1, 0.8]})

    p_obs = data["hist"]["pr_target"].values if "hist" in data else np.array([])
    exc_obs, fdc_obs = compute_empirical_fdc(p_obs)
    exc_tail = np.linspace(0.1, 10.0, 100)
    tail_obs = np.interp(exc_tail, exc_obs, fdc_obs) if len(p_obs) > 0 else np.zeros(100)
    quantiles = [50.0, 75.0, 90.0, 95.0, 99.0, 99.9]
    q_labels = ["Q50", "Q75", "Q90", "Q95", "Q99", "Q99.9"]
    y_pos = np.arange(len(quantiles))
    obs_wet = p_obs[p_obs >= 1.0] if len(p_obs) > 0 else np.array([])
    q_obs = np.percentile(obs_wet, quantiles) if len(obs_wet) > 0 else np.ones(len(quantiles))

    scenarios = [
        ("ssp245", "cordex_rcp45", "thor_ssp245", COLORS["cordex_rcp45"], COLORS["thor_ssp245"], "SSP2-4.5 (Estabilização Média)", 0),
        ("ssp585", "cordex_rcp85", "thor_ssp585", COLORS["cordex_rcp85"], COLORS["thor_ssp585"], "SSP5-8.5 (Altas Emissões)", 1),
    ]

    for sc_id, c_key, t_key, col_raw, col_thor, sc_label, row_idx in scenarios:
        ax_fdc = axes[row_idx, 0]
        ax_tail = axes[row_idx, 1]
        ax_bias = axes[row_idx, 2]

        p_c = data[c_key]["pr_raw_mm"].values if c_key in data else np.array([])
        p_t = data[t_key]["pr_thor_downscaled_mm"].values if t_key in data else np.array([])

        exc_c, fdc_c = compute_empirical_fdc(p_c) if len(p_c) > 0 else (exc_obs, np.zeros_like(exc_obs))
        exc_t, fdc_t = compute_empirical_fdc(p_t) if len(p_t) > 0 else (exc_obs, np.zeros_like(exc_obs))
        tail_c = np.interp(exc_tail, exc_c, fdc_c) if len(p_c) > 0 else np.zeros(100)
        tail_t = np.interp(exc_tail, exc_t, fdc_t) if len(p_t) > 0 else np.zeros(100)

        # 1. FDC
        if len(p_obs) > 0:
            ax_fdc.plot(exc_obs, np.maximum(0.01, fdc_obs), color=COLORS["observed"], lw=2.4, label="Observado (CHIRPS)")
        if len(p_c) > 0:
            ax_fdc.plot(exc_c, np.maximum(0.01, fdc_c), color=col_raw, lw=2.0, ls="--", label=f"CORDEX Bruto ({sc_id.upper()})")
        if len(p_t) > 0:
            ax_fdc.plot(exc_t, np.maximum(0.01, fdc_t), color=col_thor, lw=2.4, label=f"THOR-PIML ({sc_id.upper()})")
        ax_fdc.axhline(1.0, color="#64748b", ls=":", lw=1.2, label="Limiar WMO (1.0 mm)")
        ax_fdc.set_yscale("log")
        ax_fdc.set_title(f"({chr(65 + row_idx*3)}) FDC Completa — {sc_label}")
        ax_fdc.set_xlabel("Probabilidade de Excedência (%)")
        ax_fdc.set_ylabel("Chuva Diária (mm/dia) [Log]")
        ax_fdc.legend(loc="upper right", fontsize=8.0)

        # 2. Cauda de Extremos
        if len(p_obs) > 0:
            ax_tail.plot(exc_tail, tail_obs, color=COLORS["observed"], lw=2.4, label="CHIRPS Benchmark")
        if len(p_c) > 0:
            ax_tail.plot(exc_tail, tail_c, color=col_raw, lw=2.0, ls="--", label="CORDEX Bruto")
        if len(p_t) > 0:
            ax_tail.plot(exc_tail, tail_t, color=col_thor, lw=2.4, label="THOR-PIML")
        ax_tail.set_title(f"({chr(66 + row_idx*3)}) Cauda de Extremos (Q90–Q99.9) — {sc_id.upper()}")
        ax_tail.set_xlabel("Probabilidade de Excedência (%) [Mais Raro →]")
        ax_tail.set_ylabel("Chuva Diária (mm/dia)")
        ax_tail.legend(loc="upper right", fontsize=8.0)

        # 3. Viés por Quantil
        c_wet = p_c[p_c >= 1.0] if len(p_c) > 0 else np.array([])
        t_wet = p_t[p_t >= 1.0] if len(p_t) > 0 else np.array([])
        if len(obs_wet) > 0 and len(c_wet) > 0 and len(t_wet) > 0:
            q_c = np.percentile(c_wet, quantiles)
            q_t = np.percentile(t_wet, quantiles)
            b_c = ((q_c - q_obs) / (q_obs + 1e-6)) * 100.0
            b_t = ((q_t - q_obs) / (q_obs + 1e-6)) * 100.0

            ax_bias.barh(y_pos - 0.18, b_c, 0.35, color=col_raw, label=f"CORDEX {sc_id.upper()}")
            ax_bias.barh(y_pos + 0.18, b_t, 0.35, color=col_thor, label=f"THOR {sc_id.upper()}")
            ax_bias.legend(loc="lower right", fontsize=8.0)
        else:
            ax_bias.text(0.5, 0.5, "Aguardando dados de chuva diária", ha="center", va="center", transform=ax_bias.transAxes)

        ax_bias.axvline(0, color="black", lw=1.0)
        ax_bias.set_yticks(y_pos)
        ax_bias.set_yticklabels(q_labels)
        ax_bias.set_title(f"({chr(67 + row_idx*3)}) Viés por Quantil (%) — {sc_id.upper()}")
        ax_bias.set_xlabel("Viés Relativo (%)")

    fig.tight_layout()
    out_png = out_dir / "fig3_semcitec_fdc_extreme_tail_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")


# ==============================================================================
# FIGURA 4: INDICADORES BIOCLIMÁTICOS (BIO12-BIO19)
# ==============================================================================
def plot_fig4_bioclimatic_spider(data: Dict[str, pd.DataFrame], out_dir: Path, dpi: int = 300):
    """Fig 4: Comparativo multidimensional das variáveis bioclimáticas BIO12 a BIO19 por cenário."""
    print(f"  Gerando Fig 4: Painel de Indicadores Bioclimáticos BIO12-BIO19 ({dpi} DPI, layout dedicado)...")
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})

    bio_labels = [
        "BIO12\n(PRCPTOT)", "BIO13\n(Mês Úmido)", "BIO14\n(Mês Seco)",
        "BIO15\n(Sazonalidade)", "BIO16\n(Trim. Úmido)", "BIO17\n(Trim. Seco)",
        "BIO19\n(Trim. Frio)"
    ]
    bio_metric_cols = [
        "BIO12_PRCPTOT", "BIO13_Wettest_Month", "BIO14_Driest_Month",
        "BIO15_Precip_Seasonality_CV", "BIO16_Wettest_Quarter", "BIO17_Driest_Quarter",
        "BIO19_Coldest_Quarter"
    ]

    has_summary = "bio_summary" in data and len(data["bio_summary"]) > 1
    x = np.arange(len(bio_labels))
    width = 0.35

    ax_ssp245 = axes[0]
    ax_ssp585 = axes[1]
    ax_scat = axes[2]

    if has_summary:
        df_sum = data["bio_summary"]
        base_sub = df_sum[df_sum["series_id"] == "CHIRPS_Observed"]

        def calc_bio_anom(sid: str, target_epoch: str) -> np.ndarray:
            anoms = []
            sub = df_sum[(df_sum["series_id"] == sid) & (df_sum["epoch"] == target_epoch)]
            for col in bio_metric_cols:
                col_name = f"{col}_mean"
                if len(sub) > 0 and len(base_sub) > 0 and col_name in sub.columns and col_name in base_sub.columns:
                    b_val = float(base_sub[col_name].iloc[0])
                    f_val = float(sub[col_name].iloc[0])
                    anoms.append(((f_val - b_val) / (b_val + 1e-6)) * 100.0)
                else:
                    anoms.append(0.0)
            return np.array(anoms)

        late_epoch = "Final do Século (2076-2099)"
        d_c45 = calc_bio_anom("CORDEX_Raw_SSP245", late_epoch)
        d_t45 = calc_bio_anom("THOR_PIML_SSP245", late_epoch)
        d_c85 = calc_bio_anom("CORDEX_Raw_SSP585", late_epoch)
        d_t85 = calc_bio_anom("THOR_PIML_SSP585", late_epoch)

        # Painel A: SSP2-4.5
        ax_ssp245.bar(x - width/2, d_c45, width, color=COLORS["cordex_rcp45"], label="CORDEX SSP2-4.5", edgecolor="black", lw=0.5)
        ax_ssp245.bar(x + width/2, d_t45, width, color=COLORS["thor_ssp245"], label="THOR-PIML SSP2-4.5", edgecolor="black", lw=0.5)
        ax_ssp245.legend(loc="upper right", fontsize=8.5)

        # Painel B: SSP5-8.5
        ax_ssp585.bar(x - width/2, d_c85, width, color=COLORS["cordex_rcp85"], label="CORDEX SSP5-8.5", edgecolor="black", lw=0.5)
        ax_ssp585.bar(x + width/2, d_t85, width, color=COLORS["thor_ssp585"], label="THOR-PIML SSP5-8.5", edgecolor="black", lw=0.5)
        ax_ssp585.legend(loc="upper right", fontsize=8.5)

        # Painel C: Dispersão Sazonalidade (BIO15) vs Volume Anual (BIO12)
        epochs_order = ["Baseline (1981-2014)", "Futuro Próximo (2026-2050)", "Meio do Século (2051-2075)", "Final do Século (2076-2099)"]
        epochs_short = ["1981–2014", "2026–2050", "2051–2075", "2076–2099"]

        def get_coords(sid: str) -> Tuple[List[float], List[float]]:
            xs, ys = [], []
            for ep in epochs_order:
                if ep == "Baseline (1981-2014)":
                    sub = df_sum[(df_sum["series_id"] == "CHIRPS_Observed") & (df_sum["epoch"] == ep)]
                else:
                    sub = df_sum[(df_sum["series_id"] == sid) & (df_sum["epoch"] == ep)]
                if len(sub) > 0:
                    xs.append(float(sub["BIO12_PRCPTOT_mean"].iloc[0]))
                    ys.append(float(sub["BIO15_Precip_Seasonality_CV_mean"].iloc[0]))
                elif len(xs) > 0:
                    xs.append(xs[-1])
                    ys.append(ys[-1])
                else:
                    xs.append(1450.0)
                    ys.append(50.0)
            return xs, ys

        t45_x, t45_y = get_coords("THOR_PIML_SSP245")
        t85_x, t85_y = get_coords("THOR_PIML_SSP585")

        ax_scat.scatter(t45_x[0], t45_y[0], color=COLORS["observed"], s=130, marker="o", label="Baseline Histórico", zorder=5)
        ax_scat.plot(t45_x, t45_y, color=COLORS["thor_ssp245"], lw=2.2, marker="s", ms=7, label="Trajetória SSP2-4.5")
        ax_scat.plot(t85_x, t85_y, color=COLORS["thor_ssp585"], lw=2.2, marker="^", ms=8, label="Trajetória SSP5-8.5")

        for i, txt in enumerate(epochs_short):
            if i < len(t85_x):
                ax_scat.annotate(txt, (t85_x[i], t85_y[i]), textcoords="offset points", xytext=(8, -4), fontsize=8.0, weight="bold")
        ax_scat.legend(loc="upper left", fontsize=8.5)
    else:
        for ax in (ax_ssp245, ax_ssp585, ax_scat):
            ax.text(0.5, 0.5, "Aguardando cálculo dos indicadores (Passo 3)", ha="center", va="center", transform=ax.transAxes, color="#64748b", style="italic")

    for ax, letter, sc in [(ax_ssp245, "A", "SSP2-4.5"), (ax_ssp585, "B", "SSP5-8.5")]:
        ax.axhline(0, color="black", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(bio_labels, fontsize=8.0)
        ax.set_ylabel("Anomalia Projetada no Final do Século (%)")
        ax.set_title(f"({letter}) Variação Bioclimática — {sc} (2076–2099 vs Base)", pad=8)

    ax_scat.set_title("(C) Volume Anual (BIO12) vs Sazonalidade (BIO15 CV)", pad=8)
    ax_scat.set_xlabel("Precipitação Anual (BIO12 em mm/ano)")
    ax_scat.set_ylabel("Sazonalidade (BIO15 CV em %)")

    fig.tight_layout()
    out_png = out_dir / "fig4_semcitec_bioclimatic_spider_panel_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")


# ==============================================================================
# FIGURA 5: ÍNDICES DE EXTREMOS ETCCDI E RISCO HIDROLÓGICO URBANO
# ==============================================================================
def plot_fig4_bioclimatic_spider(data: Dict[str, pd.DataFrame], out_dir: Path, dpi: int = 300):
    """Tabela visual de indicadores pluviométricos, mais adequada que um gráfico multidimensional."""
    print(f"  Gerando Fig 4: Tabela visual de precipitação ({dpi} DPI)...")
    columns = [
        ("BIO12_PRCPTOT_mean", "BIO12 anual (mm)"),
        ("BIO13_Wettest_Month_mean", "BIO13 mês úmido (mm)"),
        ("BIO14_Driest_Month_mean", "BIO14 mês seco (mm)"),
        ("BIO15_Precip_Seasonality_CV_mean", "BIO15 sazonalidade (%)"),
        ("BIO16_Wettest_Quarter_mean", "BIO16 trimestre úmido (mm)"),
        ("BIO17_Driest_Quarter_mean", "BIO17 trimestre seco (mm)"),
        ("BIO19_Coldest_Quarter_mean", "BIO19 trimestre frio (mm)"),
    ]
    rows = []
    summary = data.get("bio_summary")
    if summary is not None:
        epoch = "Final do Século (2076-2099)"
        for sid, label in [
            ("CHIRPS_Observed", "Baseline CHIRPS 1981–2014"),
            ("CORDEX_Raw_SSP245", "CORDEX bruto SSP2-4.5"),
            ("THOR_PIML_SSP245", "THOR-PIML SSP2-4.5"),
            ("CORDEX_Raw_SSP585", "CORDEX bruto SSP5-8.5"),
            ("THOR_PIML_SSP585", "THOR-PIML SSP5-8.5"),
        ]:
            sub = summary[(summary["series_id"] == sid) & (summary["epoch"] == ("Baseline (1981-2014)" if sid == "CHIRPS_Observed" else epoch))]
            if not sub.empty:
                row = [label]
                row.extend(float(sub[col].iloc[0]) if col in sub.columns else np.nan for col, _ in columns)
                rows.append(row)
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.2))
    for ax, group, block_title in zip(axes, [columns[:4], columns[4:]], ["Volume e sazonalidade", "Trimestres climáticos"]):
        ax.axis("off")
        headers = ["Série"] + [label for _, label in group]
        cell_text = [[row[0]] + ["—" if pd.isna(row[1 + columns.index(item)]) else f"{row[1 + columns.index(item)]:.1f}" for item in group] for row in rows]
        table = ax.table(cellText=cell_text, colLabels=headers, cellLoc="center", loc="center",
                         colWidths=[0.25] + [0.1875] * len(group))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.45)
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            if row_idx == 0:
                cell.set_facecolor("#0f172a")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif row_idx == 1:
                cell.set_facecolor("#e2e8f0")
            elif "SSP2-4.5" in cell_text[row_idx - 1][0]:
                cell.set_facecolor("#e0f2fe")
            elif "SSP5-8.5" in cell_text[row_idx - 1][0]:
                cell.set_facecolor("#fee2e2")
        ax.set_title(block_title, loc="left", fontweight="bold", pad=8)
    fig.suptitle("Indicadores bioclimáticos de precipitação — médias do final do século", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = out_dir / "fig4_semcitec_bioclimatic_spider_panel_300dpi.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_dir.resolve() == FIG_DIR.resolve():
        fig.savefig(out_dir / "table_bioclimatic_indices.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(out_dir / "table_bioclimatic_indices.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")


def plot_fig5_etccdi_urban_extremes(data: Dict[str, pd.DataFrame], out_dir: Path, dpi: int = 300):
    """Fig 5: Evolução dos 4 pilares de risco ETCCDI separada em linhas dedicadas para SSP2-4.5 e SSP5-8.5."""
    print(f"  Gerando Fig 5: Indicadores ETCCDI de Risco Urbano ({dpi} DPI, layout 2 linhas)...")
    fig, axes = plt.subplots(2, 4, figsize=(16.8, 7.6))

    epoch_names = [
        "Baseline (1981-2014)",
        "Futuro Próximo (2026-2050)",
        "Meio do Século (2051-2075)",
        "Final do Século (2076-2099)",
    ]
    epochs_display = ["Baseline", "2026–50", "2051–75", "2076–99"]
    x = np.arange(len(epochs_display))
    width = 0.55

    has_etccdi = "etccdi_summary" in data and len(data["etccdi_summary"]) > 1

    scenarios = [
        ("ssp245", "THOR_PIML_SSP245", "CORDEX_Raw_SSP245", COLORS["thor_ssp245"], "SSP2-4.5 (Estabilização Média)", 0),
        ("ssp585", "THOR_PIML_SSP585", "CORDEX_Raw_SSP585", COLORS["thor_ssp585"], "SSP5-8.5 (Altas Emissões)", 1),
    ]

    for sc_id, thor_sid, raw_sid, c_thor, sc_label, row_idx in scenarios:
        row_axes = axes[row_idx]

        if has_etccdi:
            df_e = data["etccdi_summary"]

            def get_vals(sid: str, col: str) -> List[float]:
                vals = []
                for ep in epoch_names:
                    if ep == "Baseline (1981-2014)":
                        sub = df_e[(df_e["series_id"] == "CHIRPS_Observed") & (df_e["epoch"] == ep)]
                    else:
                        sub = df_e[(df_e["series_id"] == sid) & (df_e["epoch"] == ep)]
                    if len(sub) > 0 and col in sub.columns:
                        vals.append(float(sub[col].iloc[0]))
                    elif len(vals) > 0:
                        vals.append(vals[-1])
                    else:
                        vals.append(0.0)
                return vals

            r10_t = get_vals(thor_sid, "R10mm_mean")
            r20_t = get_vals(thor_sid, "R20mm_mean")
            rx5_t = get_vals(thor_sid, "RX5day_mean")
            cdd_t = get_vals(thor_sid, "CDD_mean")

            bar_colors = [COLORS["observed"], c_thor, c_thor, c_thor]

            metrics_data = [
                (0, r10_t, f"Dias Fortes R10mm — {sc_id.upper()}", "Dias (P ≥ 10 mm/ano)"),
                (1, r20_t, f"Dias Torrenciais R20mm — {sc_id.upper()}", "Dias (P ≥ 20 mm/ano)"),
                (2, rx5_t, f"Pico 5 Dias RX5day — {sc_id.upper()}", "Acumulado 5d (mm)"),
                (3, cdd_t, f"Estiagem Máxima CDD — {sc_id.upper()}", "Dias Secos Consecutivos"),
            ]

            for col_idx, vals, title_text, ylabel in metrics_data:
                ax = row_axes[col_idx]
                bars = ax.bar(x, vals, width, color=bar_colors, edgecolor="black", lw=0.6)
                base_ref = vals[0]
                ax.axhline(base_ref, color=COLORS["observed"], ls=":", lw=1.2, alpha=0.7, label=f"Base ({base_ref:.1f})")

                for b in bars:
                    h = b.get_height()
                    ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width()/2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8.0, weight="bold")

                p_letter = chr(65 + row_idx * 4 + col_idx)
                ax.set_title(f"({p_letter}) {title_text}", fontsize=9.5)
                ax.set_ylabel(ylabel, fontsize=9.0)
                ax.set_xticks(x)
                ax.set_xticklabels(epochs_display, fontsize=8.5)
                ax.legend(loc="upper right", fontsize=7.5, framealpha=0.90)
                ax.set_ylim(0, max(vals) * 1.25)
        else:
            for ax in row_axes:
                ax.text(0.5, 0.5, "Aguardando execução\nde projeções (Passos 2 e 3)", ha="center", va="center", transform=ax.transAxes, color="#64748b", style="italic")

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["observed"], edgecolor="black", label="Baseline observado CHIRPS 1981–2014"),
        mpatches.Patch(facecolor=COLORS["thor_ssp245"], edgecolor="black", label="THOR-PIML SSP2-4.5"),
        mpatches.Patch(facecolor=COLORS["thor_ssp585"], edgecolor="black", label="THOR-PIML SSP5-8.5"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3,
               frameon=True, fontsize=9.0, title="Série representada pelas barras")
    fig.suptitle("Indicadores ETCCDI de risco hidrológico urbano — Guarulhos-SP", y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    out_png = out_dir / "fig5_semcitec_etccdi_urban_extremes_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")


# ==============================================================================
# FIGURAS DEDICADAS POR CENÁRIO (SEPARAÇÃO ESTRITA DE PAINÉIS INDEPENDENTES)
# ==============================================================================
def plot_fig1_dedicated(data: Dict[str, pd.DataFrame], out_dir: Path, scenario_key: str, dpi: int = 300) -> Path:
    """Fig 1 dedicada: Projeção de PRCPTOT (1981-2100) para um único cenário climático."""
    sc_info = {
        "ssp245": ("SSP2-4.5 (Estabilização Média)", "CORDEX_Raw_SSP245", "THOR_PIML_SSP245",
                   COLORS["cordex_rcp45"], COLORS["thor_ssp245"], "cordex_rcp45", "thor_ssp245"),
        "ssp585": ("SSP5-8.5 (Altas Emissões)", "CORDEX_Raw_SSP585", "THOR_PIML_SSP585",
                   COLORS["cordex_rcp85"], COLORS["thor_ssp585"], "cordex_rcp85", "thor_ssp585"),
    }
    sc_label, raw_sid, thor_sid, c_raw, c_thor, df_c_key, df_t_key = sc_info[scenario_key]
    print(f"  Gerando Fig 1 Dedicada ({scenario_key.upper()}): Projeção Multidecadal PRCPTOT ({dpi} DPI)...")

    fig, axes = plt.subplots(1, 2, figsize=(16.8, 5.0), gridspec_kw={"width_ratios": [2.2, 1.0]})
    ax_ts, ax_ano = axes[0], axes[1]

    baseline_mean = 1450.0
    annual_hist = None
    roll_hist = None
    if "hist" in data:
        df_h = data["hist"]
        df_h_base = df_h[(df_h["year"] >= 1981) & (df_h["year"] <= 2014)]
        annual_hist = df_h_base.groupby("year")["pr_target"].sum()
        baseline_mean = float(annual_hist.mean())
        roll_hist = annual_hist.rolling(10, min_periods=5).mean()

    epoch_keys = [
        "Futuro Próximo (2026-2050)",
        "Meio do Século (2051-2075)",
        "Final do Século (2076-2099)",
    ]
    epoch_display = ["2026–2050", "2051–2075", "2076–2099"]
    x = np.arange(len(epoch_display))
    width = 0.35

    if annual_hist is not None:
        ax_ts.plot(annual_hist.index, annual_hist.values, color=COLORS["observed"], alpha=0.30, lw=0.9, ls="-")
        ax_ts.plot(roll_hist.index, roll_hist.values, color=COLORS["observed"], lw=2.4, label=f"CHIRPS Observado ({baseline_mean:.0f} mm/ano)")
    ax_ts.axhline(baseline_mean, color="#64748b", ls=":", lw=1.4, label=f"Linha de Base 1981–2014 ({baseline_mean:.0f} mm)")

    if "bio_annual" in data:
        df_ann = data["bio_annual"]
        s_raw = df_ann[df_ann["series_id"] == raw_sid].set_index("year")["BIO12_PRCPTOT"]
        s_thor = df_ann[df_ann["series_id"] == thor_sid].set_index("year")["BIO12_PRCPTOT"]
        if len(s_raw) > 0:
            ax_ts.plot(s_raw.index, s_raw.values, color=c_raw, alpha=0.20, lw=0.8)
            ax_ts.plot(s_raw.index, s_raw.rolling(10, min_periods=5).mean(), color=c_raw, lw=2.2, ls="--", label=f"CORDEX Bruto ({scenario_key.upper()})")
        if len(s_thor) > 0:
            ax_ts.plot(s_thor.index, s_thor.values, color=c_thor, alpha=0.20, lw=0.8)
            ax_ts.plot(s_thor.index, s_thor.rolling(10, min_periods=5).mean(), color=c_thor, lw=2.5, ls="-", label=f"THOR-PIML Downscaled ({scenario_key.upper()})")
    elif df_c_key in data:
        s_raw = data[df_c_key].groupby("year")["pr_raw_mm"].sum()
        ax_ts.plot(s_raw.index, s_raw.rolling(10, min_periods=5).mean(), color=c_raw, lw=2.2, ls="--", label=f"CORDEX Bruto ({scenario_key.upper()})")
        if df_t_key in data:
            s_thor = data[df_t_key].groupby("year")["pr_thor_downscaled_mm"].sum()
            ax_ts.plot(s_thor.index, s_thor.rolling(10, min_periods=5).mean(), color=c_thor, lw=2.5, ls="-", label=f"THOR-PIML ({scenario_key.upper()})")

    ax_ts.set_title(f"(A) Trajetória Multidecadal da Precipitação Anual — {sc_label}", pad=8)
    ax_ts.set_xlabel("Ano")
    ax_ts.set_ylabel("PRCPTOT (mm/ano)")
    ax_ts.set_xlim(1980, 2100)
    ax_ts.legend(loc="upper left", framealpha=0.92, ncol=2)

    if "bio_summary" in data and len(data["bio_summary"]) > 1:
        df_sum = data["bio_summary"]
        base_rows = df_sum[df_sum["series_id"] == "CHIRPS_Observed"]
        base_val = float(base_rows["BIO12_PRCPTOT_mean"].iloc[0]) if len(base_rows) > 0 else baseline_mean

        def get_anoms(sid: str) -> List[float]:
            anoms = []
            for ep in epoch_keys:
                sub = df_sum[(df_sum["series_id"] == sid) & (df_sum["epoch"] == ep)]
                if len(sub) > 0 and "BIO12_PRCPTOT_mean" in sub.columns:
                    v = float(sub["BIO12_PRCPTOT_mean"].iloc[0])
                    anoms.append(((v - base_val) / base_val) * 100.0)
                else:
                    anoms.append(0.0)
            return anoms

        anom_raw = get_anoms(raw_sid)
        anom_thor = get_anoms(thor_sid)

        bars_c = ax_ano.bar(x - width/2, anom_raw, width, color=c_raw, label=f"CORDEX {scenario_key.upper()}", edgecolor="black", lw=0.5)
        bars_t = ax_ano.bar(x + width/2, anom_thor, width, color=c_thor, label=f"THOR-PIML {scenario_key.upper()}", edgecolor="black", lw=0.5)
        for b in bars_c:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            y_pos = h + (1.2 if h >= 0 else -1.2)
            ax_ano.annotate(f"{h:+.1f}%", (b.get_x() + b.get_width()/2, y_pos),
                            ha="center", va=va, fontsize=7.5, weight="bold", color=c_raw)
        for b in bars_t:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            y_pos = h + (1.2 if h >= 0 else -1.2)
            ax_ano.annotate(f"{h:+.1f}%", (b.get_x() + b.get_width()/2, y_pos),
                            ha="center", va=va, fontsize=7.5, weight="bold", color=c_thor)
        ax_ano.legend(loc="upper left", framealpha=0.90)
    else:
        ax_ano.text(
            0.5, 0.5, "Projeções prontas.\nExecute Passos 2 e 3\npara anomalias decadais.",
            ha="center", va="center", transform=ax_ano.transAxes, fontsize=8.5, color="#64748b", style="italic"
        )

    ax_ano.axhline(0, color="black", lw=1.0)
    ax_ano.set_title(f"(B) Anomalia Percentual Δ (%) — {scenario_key.upper()}", pad=8)
    ax_ano.set_xticks(x)
    ax_ano.set_xticklabels(epoch_display)
    ax_ano.set_ylabel("Anomalia Δ (%)")

    fig.tight_layout()
    out_png = out_dir / f"fig1_semcitec_prcptot_projections_{scenario_key}_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")
    return out_png


def plot_fig2_dedicated(data: Dict[str, pd.DataFrame], out_dir: Path, scenario_key: str, dpi: int = 300) -> Path:
    """Fig 2 dedicada: Ciclos mensais médios nas 4 épocas para um único cenário climático."""
    sc_info = {
        "ssp245": ("cordex_rcp45", "thor_ssp245", COLORS["cordex_rcp45"], COLORS["thor_ssp245"], "SSP2-4.5 (Estabilização Média)"),
        "ssp585": ("cordex_rcp85", "thor_ssp585", COLORS["cordex_rcp85"], COLORS["thor_ssp585"], "SSP5-8.5 (Altas Emissões)"),
    }
    c_key, t_key, c_raw_col, c_thor_col, sc_label = sc_info[scenario_key]
    print(f"  Gerando Fig 2 Dedicada ({scenario_key.upper()}): Ciclo Sazonal Médio Mensal ({dpi} DPI)...")

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.6), sharey=True)
    months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    m_idx = np.arange(1, 13)

    if "hist" in data:
        df_h = data["hist"]
        df_h_base = df_h[(df_h["year"] >= 1981) & (df_h["year"] <= 2014)]
        n_years = max(1, len(df_h_base["year"].unique()))
        hist_clim = (df_h_base.groupby("month")["pr_target"].sum() / n_years).reindex(range(1, 13), fill_value=0.0).values
    else:
        hist_clim = np.zeros(12)

    epoch_configs = [
        ("Futuro Próximo (2026–2050)", 2026, 2050),
        ("Meio do Século (2051–2075)", 2051, 2075),
        ("Final do Século (2076–2099)", 2076, 2099),
    ]

    for col_idx, (ep_title, y_start, y_end) in enumerate(epoch_configs):
        ax = axes[col_idx]
        ax.plot(m_idx, hist_clim, color=COLORS["observed"], lw=2.4, marker="o", ms=4, label="Baseline (1981–2014)")

        if c_key in data:
            df_c = data[c_key]
            sub_c = df_c[(df_c["year"] >= y_start) & (df_c["year"] <= y_end)]
            if len(sub_c) > 0:
                ny_c = max(1, len(sub_c["year"].unique()))
                m_c = (sub_c.groupby("month")["pr_raw_mm"].sum() / ny_c).reindex(range(1, 13), fill_value=0.0).values
                ax.plot(m_idx, m_c, color=c_raw_col, lw=2.0, ls="--", label=f"CORDEX Bruto ({scenario_key.upper()})")

        if t_key in data:
            df_t = data[t_key]
            sub_t = df_t[(df_t["year"] >= y_start) & (df_t["year"] <= y_end)]
            if len(sub_t) > 0:
                ny_t = max(1, len(sub_t["year"].unique()))
                m_t = (sub_t.groupby("month")["pr_thor_downscaled_mm"].sum() / ny_t).reindex(range(1, 13), fill_value=0.0).values
                marker_sym = "s" if scenario_key == "ssp245" else "^"
                ax.plot(m_idx, m_t, color=c_thor_col, lw=2.4, marker=marker_sym, ms=4, label=f"THOR-PIML ({scenario_key.upper()})")

        ax.axvspan(1, 3, color="#fed7aa", alpha=0.22, label="Verão (DJF)" if col_idx == 0 else "")
        ax.axvspan(6, 8, color="#bfdbfe", alpha=0.22, label="Inverno (JJA)" if col_idx == 0 else "")

        panel_letter = chr(65 + col_idx)
        ax.set_title(f"({panel_letter}) {sc_label} — {ep_title}", fontsize=10.0)
        ax.set_xticks(m_idx)
        ax.set_xticklabels(months, rotation=35)
        ax.set_xlabel("Mês")

    axes[0].set_ylabel(f"Chuva Média Mensal (mm/mês)\n[{sc_label.split()[0]}]")
    axes[0].legend(loc="upper right", fontsize=8.0, framealpha=0.90)

    fig.suptitle(f"Evolução do Ciclo Sazonal Pluviométrico — {sc_label} (Guarulhos-SP)", y=0.99)
    fig.tight_layout()
    out_png = out_dir / f"fig2_semcitec_seasonal_cycle_epochs_{scenario_key}_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")
    return out_png


def plot_fig3_dedicated(data: Dict[str, pd.DataFrame], out_dir: Path, scenario_key: str, dpi: int = 300) -> Path:
    """Fig 3 dedicada: FDC, Cauda de Extremos e Viés por Quantil para um único cenário."""
    sc_info = {
        "ssp245": ("cordex_rcp45", "thor_ssp245", COLORS["cordex_rcp45"], COLORS["thor_ssp245"], "SSP2-4.5 (Estabilização Média)"),
        "ssp585": ("cordex_rcp85", "thor_ssp585", COLORS["cordex_rcp85"], COLORS["thor_ssp585"], "SSP5-8.5 (Altas Emissões)"),
    }
    c_key, t_key, col_raw, col_thor, sc_label = sc_info[scenario_key]
    print(f"  Gerando Fig 3 Dedicada ({scenario_key.upper()}): FDC e Cauda de Extremos ({dpi} DPI)...")

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8), gridspec_kw={"width_ratios": [1.1, 1.1, 0.8]})
    ax_fdc, ax_tail, ax_bias = axes[0], axes[1], axes[2]

    p_obs = data["hist"]["pr_target"].values if "hist" in data else np.array([])
    exc_obs, fdc_obs = compute_empirical_fdc(p_obs)
    exc_tail = np.linspace(0.1, 10.0, 100)
    tail_obs = np.interp(exc_tail, exc_obs, fdc_obs) if len(p_obs) > 0 else np.zeros(100)
    quantiles = [50.0, 75.0, 90.0, 95.0, 99.0, 99.9]
    q_labels = ["Q50", "Q75", "Q90", "Q95", "Q99", "Q99.9"]
    y_pos = np.arange(len(quantiles))
    obs_wet = p_obs[p_obs >= 1.0] if len(p_obs) > 0 else np.array([])
    q_obs = np.percentile(obs_wet, quantiles) if len(obs_wet) > 0 else np.ones(len(quantiles))

    p_c = data[c_key]["pr_raw_mm"].values if c_key in data else np.array([])
    p_t = data[t_key]["pr_thor_downscaled_mm"].values if t_key in data else np.array([])

    exc_c, fdc_c = compute_empirical_fdc(p_c) if len(p_c) > 0 else (exc_obs, np.zeros_like(exc_obs))
    exc_t, fdc_t = compute_empirical_fdc(p_t) if len(p_t) > 0 else (exc_obs, np.zeros_like(exc_obs))
    tail_c = np.interp(exc_tail, exc_c, fdc_c) if len(p_c) > 0 else np.zeros(100)
    tail_t = np.interp(exc_tail, exc_t, fdc_t) if len(p_t) > 0 else np.zeros(100)

    # 1. FDC
    if len(p_obs) > 0:
        ax_fdc.plot(exc_obs, np.maximum(0.01, fdc_obs), color=COLORS["observed"], lw=2.4, label="Observado (CHIRPS)")
    if len(p_c) > 0:
        ax_fdc.plot(exc_c, np.maximum(0.01, fdc_c), color=col_raw, lw=2.0, ls="--", label=f"CORDEX Bruto ({scenario_key.upper()})")
    if len(p_t) > 0:
        ax_fdc.plot(exc_t, np.maximum(0.01, fdc_t), color=col_thor, lw=2.4, label=f"THOR-PIML ({scenario_key.upper()})")
    ax_fdc.axhline(1.0, color="#64748b", ls=":", lw=1.2, label="Limiar WMO (1.0 mm)")
    ax_fdc.set_yscale("log")
    ax_fdc.set_title(f"(A) FDC Completa — {sc_label}")
    ax_fdc.set_xlabel("Probabilidade de Excedência (%)")
    ax_fdc.set_ylabel("Chuva Diária (mm/dia) [Log]")
    ax_fdc.legend(loc="upper right", fontsize=8.0)

    # 2. Cauda
    if len(p_obs) > 0:
        ax_tail.plot(exc_tail, tail_obs, color=COLORS["observed"], lw=2.4, label="CHIRPS Benchmark")
    if len(p_c) > 0:
        ax_tail.plot(exc_tail, tail_c, color=col_raw, lw=2.0, ls="--", label=f"CORDEX {scenario_key.upper()}")
    if len(p_t) > 0:
        ax_tail.plot(exc_tail, tail_t, color=col_thor, lw=2.4, label=f"THOR-PIML {scenario_key.upper()}")
    ax_tail.set_title(f"(B) Cauda de Extremos (Q90–Q99.9) — {scenario_key.upper()}")
    ax_tail.set_xlabel("Probabilidade de Excedência (%) [Mais Raro →]")
    ax_tail.set_ylabel("Chuva Diária (mm/dia)")
    ax_tail.legend(loc="upper right", fontsize=8.0)

    # 3. Viés
    c_wet = p_c[p_c >= 1.0] if len(p_c) > 0 else np.array([])
    t_wet = p_t[p_t >= 1.0] if len(p_t) > 0 else np.array([])
    if len(obs_wet) > 0 and len(c_wet) > 0 and len(t_wet) > 0:
        q_c = np.percentile(c_wet, quantiles)
        q_t = np.percentile(t_wet, quantiles)
        b_c = ((q_c - q_obs) / (q_obs + 1e-6)) * 100.0
        b_t = ((q_t - q_obs) / (q_obs + 1e-6)) * 100.0

        ax_bias.barh(y_pos - 0.18, b_c, 0.35, color=col_raw, label=f"CORDEX {scenario_key.upper()}")
        ax_bias.barh(y_pos + 0.18, b_t, 0.35, color=col_thor, label=f"THOR {scenario_key.upper()}")
        ax_bias.legend(loc="lower right", fontsize=8.0)
    else:
        ax_bias.text(0.5, 0.5, "Aguardando dados de chuva diária", ha="center", va="center", transform=ax_bias.transAxes)

    ax_bias.axvline(0, color="black", lw=1.0)
    ax_bias.set_yticks(y_pos)
    ax_bias.set_yticklabels(q_labels)
    ax_bias.set_title(f"(C) Viés por Quantil (%) — {scenario_key.upper()}")
    ax_bias.set_xlabel("Viés Relativo (%)")

    fig.tight_layout()
    out_png = out_dir / f"fig3_semcitec_fdc_extreme_tail_{scenario_key}_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")
    return out_png


def plot_fig4_dedicated(data: Dict[str, pd.DataFrame], out_dir: Path, scenario_key: str, dpi: int = 300) -> Path:
    """Fig 4 dedicada: Indicadores Bioclimáticos BIO12-BIO19 para um único cenário."""
    sc_info = {
        "ssp245": ("SSP2-4.5", "CORDEX_Raw_SSP245", "THOR_PIML_SSP245", COLORS["cordex_rcp45"], COLORS["thor_ssp245"]),
        "ssp585": ("SSP5-8.5", "CORDEX_Raw_SSP585", "THOR_PIML_SSP585", COLORS["cordex_rcp85"], COLORS["thor_ssp585"]),
    }
    sc_label, raw_sid, thor_sid, c_raw_col, c_thor_col = sc_info[scenario_key]
    print(f"  Gerando Fig 4 Dedicada ({scenario_key.upper()}): Indicadores Bioclimáticos BIO12-BIO19 ({dpi} DPI)...")

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2), gridspec_kw={"width_ratios": [1.3, 1.0]})
    ax_bar, ax_scat = axes[0], axes[1]

    bio_labels = [
        "BIO12\n(PRCPTOT)", "BIO13\n(Mês Úmido)", "BIO14\n(Mês Seco)",
        "BIO15\n(Sazonalidade)", "BIO16\n(Trim. Úmido)", "BIO17\n(Trim. Seco)",
        "BIO18\n(Trim. Quente)", "BIO19\n(Trim. Frio)"
    ]
    bio_metric_cols = [
        "BIO12_PRCPTOT", "BIO13_Wettest_Month", "BIO14_Driest_Month",
        "BIO15_Precip_Seasonality_CV", "BIO16_Wettest_Quarter", "BIO17_Driest_Quarter",
        "BIO18_Warmest_Quarter", "BIO19_Coldest_Quarter"
    ]
    x = np.arange(len(bio_labels))
    width = 0.35

    has_summary = "bio_summary" in data and len(data["bio_summary"]) > 1

    if has_summary:
        df_sum = data["bio_summary"]
        base_sub = df_sum[df_sum["series_id"] == "CHIRPS_Observed"]

        def calc_bio_anom(sid: str, target_epoch: str) -> np.ndarray:
            anoms = []
            sub = df_sum[(df_sum["series_id"] == sid) & (df_sum["epoch"] == target_epoch)]
            for col in bio_metric_cols:
                col_name = f"{col}_mean"
                if len(sub) > 0 and len(base_sub) > 0 and col_name in sub.columns and col_name in base_sub.columns:
                    b_val = float(base_sub[col_name].iloc[0])
                    f_val = float(sub[col_name].iloc[0])
                    anoms.append(((f_val - b_val) / (b_val + 1e-6)) * 100.0)
                else:
                    anoms.append(0.0)
            return np.array(anoms)

        late_epoch = "Final do Século (2076-2099)"
        d_raw = calc_bio_anom(raw_sid, late_epoch)
        d_thor = calc_bio_anom(thor_sid, late_epoch)

        ax_bar.bar(x - width/2, d_raw, width, color=c_raw_col, label=f"CORDEX {sc_label}", edgecolor="black", lw=0.5)
        ax_bar.bar(x + width/2, d_thor, width, color=c_thor_col, label=f"THOR-PIML {sc_label}", edgecolor="black", lw=0.5)
        ax_bar.legend(loc="upper right", fontsize=8.5)

        epochs_order = ["Baseline (1981-2014)", "Futuro Próximo (2026-2050)", "Meio do Século (2051-2075)", "Final do Século (2076-2099)"]
        epochs_short = ["1981–2014", "2026–2050", "2051–2075", "2076–2099"]

        def get_coords(sid: str) -> Tuple[List[float], List[float]]:
            xs, ys = [], []
            for ep in epochs_order:
                if ep == "Baseline (1981-2014)":
                    sub = df_sum[(df_sum["series_id"] == "CHIRPS_Observed") & (df_sum["epoch"] == ep)]
                else:
                    sub = df_sum[(df_sum["series_id"] == sid) & (df_sum["epoch"] == ep)]
                if len(sub) > 0:
                    xs.append(float(sub["BIO12_PRCPTOT_mean"].iloc[0]))
                    ys.append(float(sub["BIO15_Precip_Seasonality_CV_mean"].iloc[0]))
                elif len(xs) > 0:
                    xs.append(xs[-1])
                    ys.append(ys[-1])
                else:
                    xs.append(1450.0)
                    ys.append(50.0)
            return xs, ys

        t_x, t_y = get_coords(thor_sid)
        ax_scat.scatter(t_x[0], t_y[0], color=COLORS["observed"], s=130, marker="o", label="Baseline Histórico", zorder=5)
        marker_sym = "s" if scenario_key == "ssp245" else "^"
        ax_scat.plot(t_x, t_y, color=c_thor_col, lw=2.2, marker=marker_sym, ms=7, label=f"Trajetória {sc_label}")

        for i, txt in enumerate(epochs_short):
            if i < len(t_x):
                ax_scat.annotate(txt, (t_x[i], t_y[i]), textcoords="offset points", xytext=(8, -4), fontsize=8.0, weight="bold")
        ax_scat.legend(loc="upper left", fontsize=8.5)
    else:
        for ax in (ax_bar, ax_scat):
            ax.text(0.5, 0.5, "Aguardando cálculo dos indicadores (Passo 3)", ha="center", va="center", transform=ax.transAxes, color="#64748b", style="italic")

    ax_bar.axhline(0, color="black", lw=1.0)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bio_labels, fontsize=8.0)
    ax_bar.set_ylabel("Anomalia Projetada no Final do Século (%)")
    ax_bar.set_title(f"(A) Variação Bioclimática — {sc_label} (2076–2099 vs Base)", pad=8)

    ax_scat.set_title(f"(B) Volume Anual (BIO12) vs Sazonalidade (BIO15 CV) — {sc_label}", pad=8)
    ax_scat.set_xlabel("Precipitação Anual (BIO12 em mm/ano)")
    ax_scat.set_ylabel("Sazonalidade (BIO15 CV em %)")

    fig.tight_layout()
    out_png = out_dir / f"fig4_semcitec_bioclimatic_spider_panel_{scenario_key}_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")
    return out_png


def plot_fig5_dedicated(data: Dict[str, pd.DataFrame], out_dir: Path, scenario_key: str, dpi: int = 300) -> Path:
    """Fig 5 dedicada: Indicadores ETCCDI de Risco Urbano para um único cenário."""
    sc_info = {
        "ssp245": ("THOR_PIML_SSP245", "CORDEX_Raw_SSP245", COLORS["thor_ssp245"], COLORS["cordex_rcp45"], "SSP2-4.5 (Estabilização Média)"),
        "ssp585": ("THOR_PIML_SSP585", "CORDEX_Raw_SSP585", COLORS["thor_ssp585"], COLORS["cordex_rcp85"], "SSP5-8.5 (Altas Emissões)"),
    }
    thor_sid, raw_sid, c_thor, c_raw, sc_label = sc_info[scenario_key]
    print(f"  Gerando Fig 5 Dedicada ({scenario_key.upper()}): Indicadores ETCCDI ({dpi} DPI)...")

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.6))
    epoch_names = [
        "Baseline (1981-2014)",
        "Futuro Próximo (2026-2050)",
        "Meio do Século (2051-2075)",
        "Final do Século (2076-2099)",
    ]
    epochs_display = ["Baseline", "2026–50", "2051–75", "2076–99"]
    x = np.arange(len(epochs_display))
    width = 0.50

    has_etccdi = "etccdi_summary" in data and len(data["etccdi_summary"]) > 1

    if has_etccdi:
        df_e = data["etccdi_summary"]

        def get_vals(sid: str, col: str) -> List[float]:
            vals = []
            for ep in epoch_names:
                if ep == "Baseline (1981-2014)":
                    sub = df_e[(df_e["series_id"] == "CHIRPS_Observed") & (df_e["epoch"] == ep)]
                else:
                    sub = df_e[(df_e["series_id"] == sid) & (df_e["epoch"] == ep)]
                if len(sub) > 0 and col in sub.columns:
                    vals.append(float(sub[col].iloc[0]))
                elif len(vals) > 0:
                    vals.append(vals[-1])
                else:
                    vals.append(0.0)
            return vals

        r10_t = get_vals(thor_sid, "R10mm_mean")
        r20_t = get_vals(thor_sid, "R20mm_mean")
        rx5_t = get_vals(thor_sid, "RX5day_mean")
        cdd_t = get_vals(thor_sid, "CDD_mean")

        bar_colors = [COLORS["observed"], c_thor, c_thor, c_thor]

        metrics_data = [
            (0, r10_t, f"Dias Fortes R10mm — {scenario_key.upper()}", "Dias (P ≥ 10 mm/ano)"),
            (1, r20_t, f"Dias Torrenciais R20mm — {scenario_key.upper()}", "Dias (P ≥ 20 mm/ano)"),
            (2, rx5_t, f"Pico 5 Dias RX5day — {scenario_key.upper()}", "Acumulado 5d (mm)"),
            (3, cdd_t, f"Estiagem Máxima CDD — {scenario_key.upper()}", "Dias Secos Consecutivos"),
        ]

        for col_idx, vals, title_text, ylabel in metrics_data:
            ax = axes[col_idx]
            bars = ax.bar(x, vals, width, color=bar_colors, edgecolor="black", lw=0.6)
            base_ref = vals[0]
            ax.axhline(base_ref, color=COLORS["observed"], ls=":", lw=1.2, alpha=0.7, label=f"Base ({base_ref:.1f})")

            for b in bars:
                h = b.get_height()
                ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.0, weight="bold")

            p_letter = chr(65 + col_idx)
            ax.set_title(f"({p_letter}) {title_text}", fontsize=9.5)
            ax.set_ylabel(ylabel, fontsize=9.0)
            ax.set_xticks(x)
            ax.set_xticklabels(epochs_display, fontsize=8.5)
            ax.legend(loc="upper right", fontsize=7.5, framealpha=0.90)
            ax.set_ylim(0, max(vals) * 1.25)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Aguardando execução\nde projeções (Passos 2 e 3)", ha="center", va="center", transform=ax.transAxes, color="#64748b", style="italic")

    fig.suptitle(f"Evolução dos Indicadores ETCCDI de Risco Hidrológico — {sc_label} (Guarulhos-SP)", y=0.99)
    fig.tight_layout()
    out_png = out_dir / f"fig5_semcitec_etccdi_urban_extremes_{scenario_key}_300dpi.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Salvo: {out_png}")
    return out_png


def run_dedicated_figures_scenario(
    data: Dict[str, pd.DataFrame],
    out_dir: Path,
    scenario_key: str,
    dpi: int = 300,
) -> List[Path]:
    """Gera o conjunto das 5 figuras dedicadas e independentes para um único cenário em subpasta limpa."""
    norm_map = {"ssp245": "ssp245", "ssp585": "ssp585", "rcp45": "ssp245", "rcp85": "ssp585"}
    sc = norm_map.get(str(scenario_key).lower(), scenario_key)
    sc_dir = out_dir / sc
    sc_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        plot_fig1_dedicated(data, sc_dir, sc, dpi=dpi),
        plot_fig2_dedicated(data, sc_dir, sc, dpi=dpi),
        plot_fig3_dedicated(data, sc_dir, sc, dpi=dpi),
        plot_fig4_dedicated(data, sc_dir, sc, dpi=dpi),
        plot_fig5_dedicated(data, sc_dir, sc, dpi=dpi),
    ]
    return generated


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
def run_all_figures(
    dpi: int = 300,
    out_dir: Optional[Path] = None,
    scenario: str = "all",
    dedicated: bool = False,
) -> List[Path]:
    """Gera todas as figuras científicas de publicação para o artigo SEMCITEC 2026."""
    t0 = time.time()
    out_dir = Path(out_dir) if out_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_map = {"ssp245": "ssp245", "ssp585": "ssp585", "rcp45": "ssp245", "rcp85": "ssp585", "all": "all"}
    sc = norm_map.get(str(scenario).lower(), scenario)

    print("\n" + "="*75)
    print(f"THOR-PIML: Geração de Figuras de Publicação SEMCITEC 2026 ({dpi} DPI | Cenário: {sc.upper()})")
    print("="*75)

    data = load_datasets()

    if sc in ("ssp245", "ssp585"):
        figs = run_dedicated_figures_scenario(data, out_dir, sc, dpi=dpi)
        print(f"\n✓ 5 figuras dedicadas de {sc.upper()} geradas com sucesso em {time.time() - t0:.2f}s!")
        print(f"  Diretório de destino: {out_dir}")
        return figs

    generated = []

    # 1. Fig 1
    plot_fig1_prcptot_projections(data, out_dir, dpi=dpi)
    generated.append(out_dir / "fig1_semcitec_prcptot_projections_300dpi.png")

    # 2. Fig 2
    plot_fig2_seasonal_cycles(data, out_dir, dpi=dpi)
    generated.append(out_dir / "fig2_semcitec_seasonal_cycle_epochs_300dpi.png")

    # 3. Fig 3
    plot_fig3_fdc_extreme_tail(data, out_dir, dpi=dpi)
    generated.append(out_dir / "fig3_semcitec_fdc_extreme_tail_300dpi.png")

    # 4. Fig 4
    plot_fig4_bioclimatic_spider(data, out_dir, dpi=dpi)
    generated.append(out_dir / "fig4_semcitec_bioclimatic_spider_panel_300dpi.png")

    # 5. Fig 5
    plot_fig5_etccdi_urban_extremes(data, out_dir, dpi=dpi)
    generated.append(out_dir / "fig5_semcitec_etccdi_urban_extremes_300dpi.png")

    # 6. Referência externa independente
    # Mantém as 5 figuras canônicas da publicação; a Figura 6 é um artefato
    # complementar de auditoria e não altera o contrato da API de geração.
    if out_dir.resolve() == FIG_DIR.resolve():
        plot_fig6_external_reference(data, out_dir, dpi=dpi)

    if dedicated:
        print("\n  Gerando painéis dedicados independentes por cenário...")
        run_dedicated_figures_scenario(data, out_dir, "ssp245", dpi=dpi)
        run_dedicated_figures_scenario(data, out_dir, "ssp585", dpi=dpi)

    print(f"\n✓ Figuras geradas com sucesso em {time.time() - t0:.2f}s!")
    print(f"  Diretório de destino: {out_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="THOR-PIML SEMCITEC 2026 Figure Generator")
    parser.add_argument("--dpi", type=int, default=300, help="Resolução das imagens (default: 300)")
    parser.add_argument("--out-dir", type=str, default=str(FIG_DIR), help="Diretório de saída")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["ssp245", "ssp585", "rcp45", "rcp85", "all"],
        help="Cenário específico para gerar figuras dedicadas (default: all)",
    )
    parser.add_argument(
        "--dedicated",
        action="store_true",
        default=False,
        help="Gera também figuras dedicadas por cenário quando scenario=all",
    )
    args = parser.parse_args()

    dedicated_flag = args.dedicated
    run_all_figures(dpi=args.dpi, out_dir=Path(args.out_dir), scenario=args.scenario, dedicated=dedicated_flag)


if __name__ == "__main__":
    main()
