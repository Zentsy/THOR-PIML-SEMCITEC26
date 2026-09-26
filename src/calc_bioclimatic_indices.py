"""
THOR-PIML — Cálculo de Indicadores Bioclimáticos & Extremos ETCCDI (SEMCITEC 2026)
===================================================================================
Calcula a suíte completa de variáveis bioclimáticas oficiais de precipitação (BIO12 a BIO19 - WMO/WorldClim)
e os índices de extremos climáticos recomendados pela OMM/ETCCDI (Expert Team on Climate Change Detection and Indices).

Épocas Climatológicas Avaliadas:
1. Linha de Base Histórica (Baseline): 1981–2014 (34 anos, ERA5-Land / CHIRPS)
2. Futuro Próximo: 2026–2050 (25 anos)
3. Meio do Século: 2051–2075 (25 anos)
4. Final do Século: 2076–2099 (24 anos; o CORDEX disponível termina em 2099-12-30)

Cenários & Séries Comparadas:
- Baseline Observado (CHIRPS / Ground Truth V3)
- CORDEX Bruto (MOHC-HadGEM2-ES / INPE-Eta SAM-22) — SSP2-4.5
- THOR-PIML Downscaled — SSP2-4.5
- CORDEX Bruto (MOHC-HadGEM2-ES / INPE-Eta SAM-22) — SSP5-8.5
- THOR-PIML Downscaled — SSP5-8.5

Variáveis Bioclimáticas (WorldClim):
- BIO12: Precipitação Anual Total (PRCPTOT em mm/ano)
- BIO13: Precipitação do Mês Mais Chuvoso (mm)
- BIO14: Precipitação do Mês Mais Seco (mm)
- BIO15: Sazonalidade da Precipitação (Coeficiente de Variação CV em %)
- BIO16: Precipitação do Trimestre Mais Úmido (mm)
- BIO17: Precipitação do Trimestre Mais Seco (mm)
- BIO18: Precipitação do Trimestre Mais Quente (mm)
- BIO19: Precipitação do Trimestre Mais Frio (mm)

Índices ETCCDI:
- R10mm, R20mm (dias/ano com chuva >= 10mm e >= 20mm)
- RX1day (máxima precipitação diária anual em mm)
- RX5day (máxima precipitação acumulada em 5 dias consecutivos em mm)
- CDD (maior sequência de dias secos consecutivos < 1.0mm)
- CWD (maior sequência de dias úmidos consecutivos >= 1.0mm)
- R95p, R99p (acumulado e percentual anual de chuva em dias muito e extremamente úmidos > p95 e > p99)

Saídas em results/tables/:
- bioclimatic_indices_annual.csv
- bioclimatic_indices_epochs_summary.csv
- etccdi_extremes_annual.csv
- etccdi_extremes_epochs_summary.csv
- tabela_resumo_semcitec_2026.md

Uso:
    python src/calc_bioclimatic_indices.py
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

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import DATA_DIR, RESULTS_DIR

GT_PATH = DATA_DIR / "ground_truth_guarulhos_daily_v3.csv"
CORDEX_DIR = DATA_DIR / "cordex"
PROJECTIONS_DIR = RESULTS_DIR / "projections"
TABLES_DIR = ROOT_DIR / ".agents" / "table_data"

# Definição canônica das 4 épocas climatológicas
EPOCHS = {
    "Baseline (1981-2014)": (1981, 2014),
    "Futuro Próximo (2026-2050)": (2026, 2050),
    "Meio do Século (2051-2075)": (2051, 2075),
    "Final do Século (2076-2099)": (2076, 2099),
}


# ==============================================================================
# FUNÇÕES MATEMÁTICAS: BIO12 A BIO19
# ==============================================================================
def calc_bioclimatic_for_year(
    df_year: pd.DataFrame,
    pr_col: str,
    temp_col: str,
) -> Dict[str, float]:
    """Calcula BIO12 a BIO19 para um único ano a partir de séries diárias."""
    # Agrupamento mensal
    monthly = df_year.groupby("month").agg(
        p_sum=(pr_col, "sum"),
        t_mean=(temp_col, "mean"),
    ).reindex(range(1, 13))

    p_m = monthly["p_sum"].values.astype(float)
    t_m = monthly["t_mean"].values.astype(float)

    # Preenchimento defensivo se faltar algum mês
    p_m = np.nan_to_num(p_m, nan=0.0)
    t_m = np.nan_to_num(t_m, nan=20.0)

    # BIO12: Precipitação Total Anual
    bio12 = float(np.sum(p_m))

    # BIO13: Mês Mais Chuvoso
    bio13 = float(np.max(p_m))

    # BIO14: Mês Mais Seco
    bio14 = float(np.min(p_m))

    # BIO15: Sazonalidade da Precipitação (Coeficiente de Variação em %)
    # CV = (std / mean) * 100
    mean_p = np.mean(p_m)
    std_p = np.std(p_m, ddof=1) if len(p_m) > 1 else 0.0
    bio15 = float((std_p / (mean_p + 1e-6)) * 100.0)

    # Cálculo dos 12 trimestres móveis (3 meses consecutivos, com wrap-around anual)
    # Ex: Q1=[Jan,Fev,Mar], ..., Q12=[Dez,Jan,Fev]
    q_p = np.zeros(12, dtype=float)
    q_t = np.zeros(12, dtype=float)
    p_extended = np.concatenate([p_m, p_m[:2]])
    t_extended = np.concatenate([t_m, t_m[:2]])

    for q in range(12):
        q_p[q] = np.sum(p_extended[q : q + 3])
        q_t[q] = np.mean(t_extended[q : q + 3])

    # BIO16: Trimestre Mais Úmido
    bio16 = float(np.max(q_p))

    # BIO17: Trimestre Mais Seco
    bio17 = float(np.min(q_p))

    # BIO18: Trimestre Mais Quente
    idx_warmest = int(np.argmax(q_t))
    bio18 = float(q_p[idx_warmest])

    # BIO19: Trimestre Mais Frio
    idx_coldest = int(np.argmin(q_t))
    bio19 = float(q_p[idx_coldest])

    return {
        "BIO12_PRCPTOT": bio12,
        "BIO13_Wettest_Month": bio13,
        "BIO14_Driest_Month": bio14,
        "BIO15_Precip_Seasonality_CV": bio15,
        "BIO16_Wettest_Quarter": bio16,
        "BIO17_Driest_Quarter": bio17,
        "BIO18_Warmest_Quarter": bio18,
        "BIO19_Coldest_Quarter": bio19,
    }


# ==============================================================================
# FUNÇÕES MATEMÁTICAS: ÍNDICES DE EXTREMOS ETCCDI
# ==============================================================================
def calc_max_run_length(binary_mask: np.ndarray) -> int:
    """Calcula o comprimento máximo de uma sequência contínua de True."""
    if len(binary_mask) == 0:
        return 0
    max_run = 0
    curr_run = 0
    for val in binary_mask:
        if val:
            curr_run += 1
            if curr_run > max_run:
                max_run = curr_run
        else:
            curr_run = 0
    return max_run


def calc_etccdi_for_year(
    df_year: pd.DataFrame,
    pr_col: str,
    baseline_p95: float,
    baseline_p99: float,
) -> Dict[str, float]:
    """Calcula R10mm, R20mm, RX1day, RX5day, CDD, CWD, R95p e R99p para um ano."""
    p_vals = df_year[pr_col].values.astype(float)
    p_vals = np.maximum(0.0, np.nan_to_num(p_vals, nan=0.0))

    if len(p_vals) == 0:
        return {
            "R10mm": 0, "R20mm": 0, "RX1day": 0.0, "RX5day": 0.0,
            "CDD": 0, "CWD": 0, "R95p_mm": 0.0, "R95p_pct": 0.0,
            "R99p_mm": 0.0, "R99p_pct": 0.0,
        }

    # R10mm e R20mm
    r10 = int(np.sum(p_vals >= 10.0))
    r20 = int(np.sum(p_vals >= 20.0))

    # RX1day: Máxima diária
    rx1 = float(np.max(p_vals))

    # RX5day: Máxima acumulada em 5 dias consecutivos
    if len(p_vals) >= 5:
        s5 = pd.Series(p_vals).rolling(window=5, min_periods=5).sum()
        rx5 = float(s5.max())
    else:
        rx5 = float(np.sum(p_vals))

    # CDD (Consecutive Dry Days, P < 1.0mm)
    dry_mask = p_vals < 1.0
    cdd = calc_max_run_length(dry_mask)

    # CWD (Consecutive Wet Days, P >= 1.0mm)
    wet_mask = p_vals >= 1.0
    cwd = calc_max_run_length(wet_mask)

    # R95p e R99p: Volume acumulado em dias úmidos que excedem p95 e p99 da linha de base
    prcptot = float(np.sum(p_vals))
    wet_days = p_vals[wet_mask]

    very_wet_mask = wet_days > baseline_p95
    r95p_mm = float(np.sum(wet_days[very_wet_mask])) if len(wet_days) > 0 else 0.0
    r95p_pct = float((r95p_mm / (prcptot + 1e-6)) * 100.0)

    extreme_wet_mask = wet_days > baseline_p99
    r99p_mm = float(np.sum(wet_days[extreme_wet_mask])) if len(wet_days) > 0 else 0.0
    r99p_pct = float((r99p_mm / (prcptot + 1e-6)) * 100.0)

    return {
        "R10mm": r10,
        "R20mm": r20,
        "RX1day": rx1,
        "RX5day": rx5,
        "CDD": cdd,
        "CWD": cwd,
        "R95p_mm": r95p_mm,
        "R95p_pct": r95p_pct,
        "R99p_mm": r99p_mm,
        "R99p_pct": r99p_pct,
    }


# ==============================================================================
# PROCESSAMENTO DE SÉRIES COMPLETAS
# ==============================================================================
def get_baseline_thresholds(df_hist: pd.DataFrame, pr_col: str = "pr_target") -> Tuple[float, float]:
    """Extrai os percentis p95 e p99 dos dias úmidos (P >= 1.0mm) do período de base (1981-2014)."""
    df_base = df_hist[(df_hist["year"] >= 1981) & (df_hist["year"] <= 2014)]
    p_wet = df_base[df_base[pr_col] >= 1.0][pr_col].dropna().values
    if len(p_wet) == 0:
        return 25.0, 50.0
    p95 = float(np.percentile(p_wet, 95.0))
    p99 = float(np.percentile(p_wet, 99.0))
    print(f"  Limiares da Linha de Base Histórica (1981-2014, {len(p_wet):,} dias de chuva):")
    print(f"    p95 (dias muito úmidos):       {p95:.2f} mm/dia")
    print(f"    p99 (dias extremamente úmidos): {p99:.2f} mm/dia")
    return p95, p99


def compute_series_indices(
    df: pd.DataFrame,
    series_id: str,
    scenario: str,
    model_type: str,
    pr_col: str,
    temp_col: str,
    p95: float,
    p99: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula BIO e ETCCDI ano a ano para uma série temporal."""
    records_bio = []
    records_etccdi = []

    years = sorted(df["year"].unique())
    # Indicadores anuais não são válidos para amostras parciais, inclusive em dry-run.
    min_days_per_year = 300
    for yr in years:
        df_yr = df[df["year"] == yr]
        if len(df_yr) < min_days_per_year:
            continue

        bio_dict = calc_bioclimatic_for_year(df_yr, pr_col, temp_col)
        bio_dict.update({
            "series_id": series_id,
            "scenario": scenario,
            "model_type": model_type,
            "year": yr,
        })
        records_bio.append(bio_dict)

        etccdi_dict = calc_etccdi_for_year(df_yr, pr_col, p95, p99)
        etccdi_dict.update({
            "series_id": series_id,
            "scenario": scenario,
            "model_type": model_type,
            "year": yr,
        })
        records_etccdi.append(etccdi_dict)

    df_bio = pd.DataFrame(records_bio)
    df_etccdi = pd.DataFrame(records_etccdi)
    return df_bio, df_etccdi


def summarize_epochs(
    df_annual: pd.DataFrame,
    metric_cols: List[str],
) -> pd.DataFrame:
    """Agrega os índices anuais nas 4 épocas climatológicas, calculando média, std e anomalias."""
    # Classifica cada ano em sua respectiva época
    def assign_epoch(y: int) -> Optional[str]:
        for ep_name, (y_start, y_end) in EPOCHS.items():
            if y_start <= y <= y_end:
                return ep_name
        # Fallback para amostras de teste rápido / dry-run no início da série (2006-2025)
        if 2006 <= y < 2026:
            return "Futuro Próximo (2026-2050)"
        return None

    df = df_annual.copy()
    df["epoch"] = df["year"].apply(assign_epoch)
    df = df.dropna(subset=["epoch"])

    # Agregação por (series_id, scenario, model_type, epoch)
    grouped = df.groupby(["series_id", "scenario", "model_type", "epoch"], sort=False)
    mean_df = grouped[metric_cols].mean().reset_index()
    std_df = grouped[metric_cols].std().reset_index()

    # Mescla em formato tabular detalhado
    summary = pd.DataFrame()
    summary["series_id"] = mean_df["series_id"]
    summary["scenario"] = mean_df["scenario"]
    summary["model_type"] = mean_df["model_type"]
    summary["epoch"] = mean_df["epoch"]

    for col in metric_cols:
        summary[f"{col}_mean"] = np.round(mean_df[col], 2)
        summary[f"{col}_std"] = np.round(std_df[col], 2)

    return summary


def run_pipeline(
    projections_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Executa o pipeline completo de cálculo de indicadores bioclimáticos e extremos."""
    t0 = time.time()
    projections_dir = Path(projections_dir) if projections_dir else PROJECTIONS_DIR
    tables_dir = Path(tables_dir) if tables_dir else TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "="*75)
    print("THOR-PIML: Cálculo de Indicadores Bioclimáticos & Extremos (SEMCITEC 2026)")
    print("="*75)

    # 1. Carrega dados históricos (1981-2026)
    if not GT_PATH.exists():
        raise FileNotFoundError(f"Arquivo histórico não encontrado: {GT_PATH}")
    df_hist = pd.read_csv(GT_PATH)
    if "year" not in df_hist.columns:
        df_hist["year"] = df_hist["date"].astype(str).str[:4].astype(int)
        df_hist["month"] = df_hist["date"].astype(str).str[5:7].astype(int)
        df_hist["day"] = df_hist["date"].astype(str).str[8:10].astype(int)
    else:
        df_hist["year"] = df_hist["year"].astype(int)
        df_hist["month"] = df_hist["month"].astype(int)
        df_hist["day"] = df_hist["day"].astype(int)

    # Extrai limiares p95 e p99 da Linha de Base (1981-2014)
    p95_base, p99_base = get_baseline_thresholds(df_hist, pr_col="pr_target")

    all_bio_list = []
    all_etccdi_list = []

    # 2. Processa Linha de Base Observada (CHIRPS)
    print("\n[1/5] Processando Baseline Histórico (1981-2014)...")
    bio_hist, etccdi_hist = compute_series_indices(
        df=df_hist[(df_hist["year"] >= 1981) & (df_hist["year"] <= 2014)],
        series_id="CHIRPS_Observed",
        scenario="Historical",
        model_type="Observed_Baseline",
        pr_col="pr_target",
        temp_col="tmean",
        p95=p95_base,
        p99=p99_base,
    )
    all_bio_list.append(bio_hist)
    all_etccdi_list.append(etccdi_hist)

    # 3. Processa Cenários Futuros (2026-2100)
    scenarios = [
        ("ssp245", "SSP2-4.5", "cordex_guarulhos_rcp45_daily_2006_2099.csv", "cordex_ssp245_downscaled_2026_2099.csv"),
        ("ssp585", "SSP5-8.5", "cordex_guarulhos_rcp85_daily_2006_2099.csv", "cordex_ssp585_downscaled_2026_2099.csv"),
    ]

    step_num = 2
    for sc_key, sc_label, cordex_csv_name, proj_csv_name in scenarios:
        cordex_csv = CORDEX_DIR / cordex_csv_name
        proj_csv = projections_dir / proj_csv_name

        # Verifica se o arquivo do CORDEX pré-processado existe
        if not cordex_csv.exists():
            print(f"⚠ Aviso: {cordex_csv.name} não encontrado. Execute primeiro src/cordex_preprocessor.py")
            continue

        df_cordex = pd.read_csv(cordex_csv)
        if "year" not in df_cordex.columns:
            df_cordex["year"] = df_cordex["date"].astype(str).str[:4].astype(int)
            df_cordex["month"] = df_cordex["date"].astype(str).str[5:7].astype(int)
            df_cordex["day"] = df_cordex["date"].astype(str).str[8:10].astype(int)
        else:
            df_cordex["year"] = df_cordex["year"].astype(int)
            df_cordex["month"] = df_cordex["month"].astype(int)
            df_cordex["day"] = df_cordex["day"].astype(int)
        # Filtra período futuro 2026-2100
        df_cordex_fut = df_cordex[df_cordex["year"] >= 2026].copy()

        # A. CORDEX Bruto
        print(f"\n[{step_num}/5] Processando CORDEX Bruto: {sc_label} (2026-2100)...")
        bio_cordex, etccdi_cordex = compute_series_indices(
            df=df_cordex_fut,
            series_id=f"CORDEX_Raw_{sc_key.upper()}",
            scenario=sc_label,
            model_type="CORDEX_Raw",
            pr_col="pr_raw_mm",
            temp_col="tas_celsius",
            p95=p95_base,
            p99=p99_base,
        )
        all_bio_list.append(bio_cordex)
        all_etccdi_list.append(etccdi_cordex)
        step_num += 1

        # B. THOR-PIML Downscaled (se já disponível nas projeções)
        if proj_csv.exists():
            print(f"[{step_num}/5] Processando THOR-PIML Downscaled: {sc_label} (2026-2100)...")
            df_proj = pd.read_csv(proj_csv)
            if "year" not in df_proj.columns:
                df_proj["year"] = df_proj["date"].astype(str).str[:4].astype(int)
                df_proj["month"] = df_proj["date"].astype(str).str[5:7].astype(int)
                df_proj["day"] = df_proj["date"].astype(str).str[8:10].astype(int)
            else:
                df_proj["year"] = df_proj["year"].astype(int)
                df_proj["month"] = df_proj["month"].astype(int)
                df_proj["day"] = df_proj["day"].astype(int)
            # Associa temperatura do CORDEX para cálculo de BIO18/BIO19 se necessário
            if "tas_celsius" not in df_proj.columns:
                df_proj = df_proj.merge(df_cordex[["date", "tas_celsius"]], on="date", how="left")
            df_proj_fut = df_proj[df_proj["year"] >= 2026].copy()
            if len(df_proj_fut) == 0 and len(df_proj) > 0:
                df_proj_fut = df_proj.copy()

            bio_thor, etccdi_thor = compute_series_indices(
                df=df_proj_fut,
                series_id=f"THOR_PIML_{sc_key.upper()}",
                scenario=sc_label,
                model_type="THOR_PIML_Downscaled",
                pr_col="pr_thor_downscaled_mm",
                temp_col="tas_celsius",
                p95=p95_base,
                p99=p99_base,
            )
            all_bio_list.append(bio_thor)
            all_etccdi_list.append(etccdi_thor)
        else:
            print(f"ℹ Projeções THOR-PIML ({proj_csv.name}) ainda não computadas.")
            print("  Execute src/infer_cordex_v8.py para gerar os dados downscaled.")
        step_num += 1

    # 4. Concatena e Salva Tabelas Anuais
    df_bio_annual = pd.concat(all_bio_list, ignore_index=True)
    df_etccdi_annual = pd.concat(all_etccdi_list, ignore_index=True)

    bio_ann_csv = tables_dir / "bioclimatic_indices_annual.csv"
    etccdi_ann_csv = tables_dir / "etccdi_extremes_annual.csv"

    df_bio_annual.to_csv(bio_ann_csv, index=False)
    df_etccdi_annual.to_csv(etccdi_ann_csv, index=False)
    print(f"\n✓ Salvo: {bio_ann_csv} ({len(df_bio_annual)} registros)")
    print(f"✓ Salvo: {etccdi_ann_csv} ({len(df_etccdi_annual)} registros)")

    # 5. Agrega e Salva Resumos das 4 Épocas
    bio_cols = [
        "BIO12_PRCPTOT", "BIO13_Wettest_Month", "BIO14_Driest_Month",
        "BIO15_Precip_Seasonality_CV", "BIO16_Wettest_Quarter",
        "BIO17_Driest_Quarter", "BIO18_Warmest_Quarter", "BIO19_Coldest_Quarter"
    ]
    etccdi_cols = ["R10mm", "R20mm", "RX1day", "RX5day", "CDD", "CWD", "R95p_mm", "R95p_pct", "R99p_mm", "R99p_pct"]

    df_bio_summary = summarize_epochs(df_bio_annual, bio_cols)
    df_etccdi_summary = summarize_epochs(df_etccdi_annual, etccdi_cols)

    bio_sum_csv = tables_dir / "bioclimatic_indices_epochs_summary.csv"
    etccdi_sum_csv = tables_dir / "etccdi_extremes_epochs_summary.csv"

    df_bio_summary.to_csv(bio_sum_csv, index=False)
    df_etccdi_summary.to_csv(etccdi_sum_csv, index=False)
    print(f"✓ Salvo: {bio_sum_csv}")
    print(f"✓ Salvo: {etccdi_sum_csv}")

    # 6. Gera Relatório Executivo em Markdown
    md_report_path = (RESULTS_DIR / "tables" / "tabela_resumo_semcitec_2026.md"
                      if tables_dir.resolve() == TABLES_DIR.resolve()
                      else tables_dir / "tabela_resumo_semcitec_2026.md")
    md_report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_markdown_report(df_bio_summary, df_etccdi_summary, md_report_path)
    print(f"✓ Relatório Executivo Salvo: {md_report_path}")

    print(f"✓ Processamento de Indicadores concluído com sucesso em {time.time() - t0:.2f}s!")
    return {
        "bio_annual": bio_ann_csv,
        "etccdi_annual": etccdi_ann_csv,
        "bio_summary": bio_sum_csv,
        "etccdi_summary": etccdi_sum_csv,
        "md_report": md_report_path,
    }


def generate_markdown_report(df_bio: pd.DataFrame, df_etccdi: pd.DataFrame, out_path: Path):
    """Gera um relatório científico consolidado em Markdown das 4 épocas climatológicas."""
    lines = [
        "# RELATÓRIO TÉCNICO DE PROJEÇÃO CLIMÁTICA & INDICADORES BIOCLIMÁTICOS — SEMCITEC 2026",
        "",
        "**Local:** Guarulhos - SP, Brasil (Bacia do Rio Baquirivu-Guaçu / Alto Tietê)",
        f"**Data de Geração:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Metodologia:** Downscaling Estatístico com Física Informada (THOR-PIML V8)",
        "",
        "---",
        "",
        "## 1. Variáveis Bioclimáticas Oficiais de Precipitação (BIO12 a BIO19 - WorldClim/WMO)",
        "",
        "| Série / Modelo | Cenário | Época | BIO12 (mm/ano) | BIO13 (mm) | BIO14 (mm) | BIO15 (CV %) | BIO16 (mm) | BIO17 (mm) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in df_bio.iterrows():
        b12 = f"{row['BIO12_PRCPTOT_mean']:.1f} ± {row['BIO12_PRCPTOT_std']:.1f}"
        b13 = f"{row['BIO13_Wettest_Month_mean']:.1f}"
        b14 = f"{row['BIO14_Driest_Month_mean']:.1f}"
        b15 = f"{row['BIO15_Precip_Seasonality_CV_mean']:.1f}%"
        b16 = f"{row['BIO16_Wettest_Quarter_mean']:.1f}"
        b17 = f"{row['BIO17_Driest_Quarter_mean']:.1f}"
        lines.append(
            f"| **{row['series_id']}** | {row['scenario']} | {row['epoch']} | {b12} | {b13} | {b14} | {b15} | {b16} | {b17} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Indicadores de Extremos Climáticos (ETCCDI / OMM)",
        "",
        "| Série / Modelo | Cenário | Época | R10mm (dias) | R20mm (dias) | RX1day (mm) | RX5day (mm) | CDD (dias) | CWD (dias) | R95p (%) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, row in df_etccdi.iterrows():
        r10 = f"{row['R10mm_mean']:.1f} ± {row['R10mm_std']:.1f}"
        r20 = f"{row['R20mm_mean']:.1f} ± {row['R20mm_std']:.1f}"
        rx1 = f"{row['RX1day_mean']:.1f}"
        rx5 = f"{row['RX5day_mean']:.1f}"
        cdd = f"{row['CDD_mean']:.1f}"
        cwd = f"{row['CWD_mean']:.1f}"
        r95 = f"{row['R95p_pct_mean']:.1f}%"
        lines.append(
            f"| **{row['series_id']}** | {row['scenario']} | {row['epoch']} | {r10} | {r20} | {rx1} | {rx5} | {cdd} | {cwd} | {r95} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Síntese dos Principais Resultados Científicos",
        "",
        "1. **Comparação de regime:** BIO12–BIO19 e ETCCDI são calculados por ano completo e resumidos por época; anos incompletos são excluídos.",
        "2. **Extremos e persistência:** RX1day, RX5day, CDD, CWD, R95p e R99p devem ser interpretados comparando diretamente CORDEX, THOR e baseline, sem atribuir melhoria automática ao modelo.",
        "3. **Limitação de interpretação:** O relatório descreve estatísticas das séries geradas; não constitui validação de desempenho futuro nem comprova recuperação de extremos sem uma avaliação independente.",
        "",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="THOR-PIML Bioclimatic & ETCCDI Calculator (SEMCITEC 2026)")
    args = parser.parse_args()
    run_pipeline()


if __name__ == "__main__":
    main()
