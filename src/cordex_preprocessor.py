"""
THOR-PIML — CORDEX NetCDF Preprocessor (SEMCITEC 2026)
======================================================
Processa os 4 NetCDFs de projeção climática CORDEX SAM-22 (INPE-Eta / MOHC-HadGEM2-ES):
1. pr RCP 4.5  (SSP2-4.5)
2. pr RCP 8.5  (SSP5-8.5)
3. tas RCP 4.5 (SSP2-4.5)
4. tas RCP 8.5 (SSP5-8.5)

Executa:
- Conversão física de unidades:
    * pr (kg m-2 s-1) * 86400 -> mm/dia
    * tas (K) - 273.15 -> °C
- Extração de ponto de grade pontual de Guarulhos-SP (-23.43°S, -46.47°W)
- Extração de estatísticas espaciais da grade 16x16 (pr_grid_max, pr_grid_std, tas_grid_mean)
- Alinhamento de datas contínuas sob calendário 360_day (2006-01-01 a 2099-12-30, 33.840 dias)
- Salvamento em data/cordex/ com metadados de rastreabilidade científica.

Uso:
    python src/cordex_preprocessor.py
    python src/cordex_preprocessor.py --scenario all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import xarray as xr

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import DATA_DIR

DEFAULT_OUT_DIR = DATA_DIR / "cordex"

# Coordenadas geográficas oficiais de Guarulhos-SP (Bacia Hidrográfica Alto Tietê / Baquirivu-Guaçu)
GUARULHOS_LAT: float = -23.43
GUARULHOS_LON: float = -46.47

# Caminhos canônicos dos NetCDFs CORDEX no repositório
CORDEX_FILES = {
    "rcp45": {
        "pr": DATA_DIR / "SAM-22" / "pr-rcp45" / "pr_SAM-20_MOHC-HadGEM2-ES_rcp45_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
        "tas": DATA_DIR / "SAM-22" / "tas-rcp45" / "tas_SAM-20_MOHC-HadGEM2-ES_rcp45_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
        "ssp_name": "ssp245",
        "label": "SSP2-4.5 (Estabilização Média)",
    },
    "rcp85": {
        "pr": DATA_DIR / "SAM-22" / "pr-rcp85" / "pr_SAM-20_MOHC-HadGEM2-ES_rcp85_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
        "tas": DATA_DIR / "SAM-22" / "rcp8.5-TAS" / "tas_SAM-20_MOHC-HadGEM2-ES_rcp85_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
        "ssp_name": "ssp585",
        "label": "SSP5-8.5 (Altas Emissões)",
    },
}


def find_nearest_coord_idx(coords: np.ndarray, target: float) -> Tuple[int, float]:
    """Retorna índice e valor da coordenada mais próxima."""
    idx = int(np.argmin(np.abs(coords - target)))
    return idx, float(coords[idx])


def preprocess_scenario(
    scenario_key: str,
    out_dir: Path,
    target_lat: float = GUARULHOS_LAT,
    target_lon: float = GUARULHOS_LON,
) -> Path:
    """Extrai séries temporais de pr e tas para Guarulhos e calcula estatísticas espaciais."""
    meta = CORDEX_FILES[scenario_key]
    pr_path = meta["pr"]
    tas_path = meta["tas"]

    if not pr_path.exists():
        raise FileNotFoundError(f"Arquivo CORDEX 'pr' não encontrado: {pr_path}")
    if not tas_path.exists():
        raise FileNotFoundError(f"Arquivo CORDEX 'tas' não encontrado: {tas_path}")

    print(f"\n[{scenario_key.upper()}] Processando: {meta['label']}")
    print(f"  pr  NetCDF: {pr_path.name}")
    print(f"  tas NetCDF: {tas_path.name}")

    # Abre Datasets de forma lazy via xarray
    ds_pr = xr.open_dataset(pr_path)
    ds_tas = xr.open_dataset(tas_path)

    # Identificação das dimensões espaciais
    lat_arr = ds_pr["lat"].values
    lon_arr = ds_pr["lon"].values

    lat_idx, lat_val = find_nearest_coord_idx(lat_arr, target_lat)
    lon_idx, lon_val = find_nearest_coord_idx(lon_arr, target_lon)

    print(f"  Ponto Guarulhos alvo: ({target_lat:.4f}, {target_lon:.4f})")
    print(f"  Ponto de grade mais próximo: ({lat_val:.4f}, {lon_val:.4f}) [lat_idx={lat_idx}, lon_idx={lon_idx}]")

    # Extrai o vetor temporal
    times = ds_pr["time"].values
    n_steps = len(times)
    print(f"  Passos temporais: {n_steps} ({times[0]} a {times[-1]})")

    # Converte timestamps cftime 360_day para strings e inteiros
    date_strs = []
    years = []
    months = []
    days = []
    doys = []

    for t in times:
        y = t.year
        m = t.month
        d = t.day
        # No calendário 360_day, cada mês tem rigorosamente 30 dias
        doy = (m - 1) * 30 + d
        date_strs.append(f"{y:04d}-{m:02d}-{d:02d}")
        years.append(y)
        months.append(m)
        days.append(d)
        doys.append(doy)

    # Extrai dados no ponto de Guarulhos
    print("  Extraindo e convertendo variáveis locais...")
    pr_pt = ds_pr["pr"].isel(lat=lat_idx, lon=lon_idx).values.astype(np.float32)
    tas_pt = ds_tas["tas"].isel(lat=lat_idx, lon=lon_idx).values.astype(np.float32)

    # Conversão de unidades
    pr_local_mm = np.maximum(0.0, pr_pt * 86400.0)  # kg m-2 s-1 -> mm/dia
    tas_local_c = tas_pt - 273.15                   # K -> °C

    # Estatísticas espaciais na vizinhança local representativa de Guarulhos (subgrade 3x3, ~60x60 km)
    # Compatibilidade estrita de escala espacial com as células CHIRPS do treino histórico
    # evitando contaminação orográfica espúria do litoral de Santos e Serra do Mar da grade 16x16.
    subgrid_rad = 1  # 3x3 células centradas em (lat_idx, lon_idx)
    lat_min = max(0, lat_idx - subgrid_rad)
    lat_max = min(len(lat_arr), lat_idx + subgrid_rad + 1)
    lon_min = max(0, lon_idx - subgrid_rad)
    lon_max = min(len(lon_arr), lon_idx + subgrid_rad + 1)

    print(
        f"  Calculando estatísticas locais de subgrade {lat_max - lat_min}x{lon_max - lon_min} "
        f"(lat=[{lat_min}:{lat_max}], lon=[{lon_min}:{lon_max}] ~60x60 km)..."
    )
    sub_pr = ds_pr["pr"].isel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    sub_tas = ds_tas["tas"].isel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    pr_grid_max = (sub_pr.max(dim=["lat", "lon"]).values * 86400.0).astype(np.float32)
    pr_grid_std = (sub_pr.std(dim=["lat", "lon"]).values * 86400.0).astype(np.float32)
    tas_grid_mean = (sub_tas.mean(dim=["lat", "lon"]).values - 273.15).astype(np.float32)

    ds_pr.close()
    ds_tas.close()

    # Tratamento defensivo de NaNs (ex: último mês de 2099 em RCP4.5 ausente na grade bruta)
    def clean_series(arr: np.ndarray, name: str) -> np.ndarray:
        s = pd.Series(arr)
        n_nan = int(s.isna().sum())
        if n_nan > 0:
            print(f"  ⚠ Aviso: {n_nan} NaNs encontrados em '{name}'. Aplicando imputação causal (ffill/bfill)...")
            s = s.ffill().bfill()
        return s.values.astype(np.float32)

    pr_local_mm = clean_series(pr_local_mm, "pr_local_mm")
    tas_local_c = clean_series(tas_local_c, "tas_local_c")
    pr_grid_max = clean_series(pr_grid_max, "pr_grid_max")
    pr_grid_std = clean_series(pr_grid_std, "pr_grid_std")
    tas_grid_mean = clean_series(tas_grid_mean, "tas_grid_mean")

    # Monta DataFrame final padronizado
    df = pd.DataFrame({
        "date": date_strs,
        "year": years,
        "month": months,
        "day": days,
        "doy": doys,
        "pr_raw_mm": np.round(pr_local_mm, 4),
        "tas_celsius": np.round(tas_local_c, 4),
        "pr_grid_max": np.round(pr_grid_max, 4),
        "pr_grid_std": np.round(pr_grid_std, 4),
        "tas_grid_mean": np.round(tas_grid_mean, 4),
    })

    if df.isna().any().any():
        raise ValueError(f"Invariante violado: DataFrame final de {scenario_key} contém NaNs!")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"cordex_guarulhos_{scenario_key}_daily_2006_2099.csv"
    df.to_csv(out_csv, index=False)
    print(f"  ✓ Salvo: {out_csv} ({len(df):,} linhas)")

    # Cria cópia/alias com nomenclatura SSP para máxima conveniência
    ssp_alias = out_dir / f"cordex_guarulhos_{meta['ssp_name']}_daily_2006_2099.csv"
    if ssp_alias != out_csv:
        df.to_csv(ssp_alias, index=False)
        print(f"  ✓ Salvo alias SSP: {ssp_alias}")

    # Estatísticas sumárias do cenário
    print(f"  Resumo Estatístico {scenario_key.upper()} (2006-2099):")
    print(f"    Chuva média diária: {df['pr_raw_mm'].mean():.2f} mm/dia")
    print(f"    Temperatura média: {df['tas_celsius'].mean():.2f} °C")
    print(f"    Máxima diária (RX1day): {df['pr_raw_mm'].max():.2f} mm")
    print(f"    Dias de chuva forte (R10mm): {(df['pr_raw_mm'] >= 10.0).sum():,} dias")
    print(f"    Dias de chuva severa (R20mm): {(df['pr_raw_mm'] >= 20.0).sum():,} dias")

    return out_csv


def run_preprocessor(scenario: str = "all", out_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Executa o pré-processamento de todos os cenários solicitados."""
    t0 = time.time()
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_map = {"ssp245": "rcp45", "ssp585": "rcp85", "rcp45": "rcp45", "rcp85": "rcp85"}
    scenarios = ["rcp45", "rcp85"] if scenario == "all" else [norm_map.get(scenario, scenario)]
    results = {}

    for sc in scenarios:
        if sc not in CORDEX_FILES:
            raise ValueError(f"Cenário inválido: '{sc}'. Opções: 'rcp45', 'rcp85', 'ssp245', 'ssp585', 'all'")
        csv_path = preprocess_scenario(sc, out_dir=out_dir)
        results[sc] = csv_path

    # Salva manifesto de metadados
    manifest_path = out_dir / "cordex_preprocessing_metadata.json"
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_location": {
            "name": "Guarulhos-SP, Brasil",
            "lat": GUARULHOS_LAT,
            "lon": GUARULHOS_LON,
        },
        "model": "INPE-Eta regional climate model driven by MOHC-HadGEM2-ES",
        "domain": "CORDEX SAM-22 (0.20° resolution)",
        "calendar": "360_day",
        "period": "2006-01-01 to 2099-12-30 (33,840 daily steps)",
        "units": {
            "pr": "mm/dia (converted from kg m-2 s-1 via * 86400)",
            "tas": "celsius (converted from K via - 273.15)",
        },
        "files_generated": {k: str(v.name) for k, v in results.items()},
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Manifesto salvo: {manifest_path}")
    print(f"✓ Pré-processamento concluído em {time.time() - t0:.2f}s.")
    return results


def main():
    parser = argparse.ArgumentParser(description="THOR-PIML CORDEX Preprocessor (SEMCITEC 2026)")
    parser.add_argument(
        "--scenario",
        choices=["rcp45", "rcp85", "ssp245", "ssp585", "all"],
        default="all",
        help="Cenário CORDEX a processar (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Diretório de saída para CSVs processados",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=GUARULHOS_LAT,
        help="Latitude do ponto alvo",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=GUARULHOS_LON,
        help="Longitude do ponto alvo",
    )
    args = parser.parse_args()

    run_preprocessor(scenario=args.scenario, out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
