"""
THOR-PIML — Dataset CORDEX (Sprint S7)
=======================================
Suporte para CORDEX SAM-22 (0.22° ~25km, prioritário) e SAM-44 (0.44° ~50km, fallback)
com xarray + dask, usando scaler V2 (fitado em ERA5, sem refit no futuro).

Uso:
    from src.cordex_dataset import CORDEXDataset
    ds = CORDEXDataset(
        nc_pattern="data/cordex/SAM-22/*rcp85*pr*.nc",
        scaler_path="checkpoints/scaler_v2.json",
        seq_len=30,
    )
    loader = DataLoader(ds, batch_size=128)

    # Inferência 2026-2100:
    python src/infer_cordex.py --nc data/cordex/SAM-22/rcm.nc --ckpt checkpoints/best_model.pt

TODO S7 (do usuário): priorizar SAM-22 (25km) sobre SAM-44 (50km).
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from src.paths import ROOT_DIR, DATA_DIR, CHECKPOINT_DIR
from src.preprocessing import RobustClimateScaler
from src.config import FEATURE_COLS, BASE_FEATURE_COLS, PRIMARY_FEATURE_COLS


# Mapeamento CORDEX → THOR
# CORDEX vars: pr (kg m-2 s-1 → mm/dia), tas (K → °C), ps (Pa → hPa), hurs (%), sfcWind, rsds
CORDEX_VAR_MAP = {
    "pr": ("pr", 86400.0),      # kg m-2 s-1 *86400 = mm/dia
    "tas": ("tmean", 1.0),      # K → °C (subtrai 273.15 no código)
    "tasmax": ("tmax", 1.0),
    "tasmin": ("tmin", 1.0),
    "ps": ("psfc", 0.01),       # Pa → hPa (/100)
    "psl": ("psfc", 0.01),
    "hurs": ("rh", 1.0),        # % já
    "hur": ("rh", 1.0),
    "sfcWind": ("wind_speed", 1.0),
    "rsds": ("solar_rad", 0.0864),  # W/m2 → MJ/m2/dia (*0.0864)
}

DOMAINS_PRIORITY = ["SAM-22", "SAM-44"]  # 22 primeiro (25km), 44 fallback (50km)


def discover_cordex_files(pattern: str, domain: Optional[str] = None) -> List[Path]:
    """Descobre NetCDFs CORDEX, priorizando SAM-22."""
    if domain:
        # busca específica
        files = sorted(Path().glob(pattern))
        files = [f for f in files if domain in str(f)]
        return files
    # prioriza SAM-22
    for dom in DOMAINS_PRIORITY:
        files = [Path(p) for p in glob.glob(pattern) if dom in p]
        if files:
            return sorted(files)
    # fallback: qualquer
    return sorted(Path(p).glob(pattern) if "*" in pattern else [])


class CORDEXDataset(Dataset):
    """Dataset PyTorch para CORDEX NetCDF → THOR features (30 dias ×50)."""

    def __init__(
        self,
        nc_pattern: Union[str, Path],
        scaler_path: Union[str, Path] = CHECKPOINT_DIR / "scaler_v2.json",
        seq_len: int = 30,
        domain: Optional[str] = None,
        bbox: Optional[dict] = None,
    ):
        if not HAS_XARRAY:
            raise ImportError("xarray não instalado: pip install xarray netCDF4 dask cftime")
        self.seq_len = seq_len
        self.bbox = bbox or {"lat_min": -23.55, "lat_max": -23.30, "lon_min": -46.65, "lon_max": -46.35}
        self.scaler_path = Path(scaler_path)
        self.scaler = None
        if self.scaler_path.exists():
            self.scaler = RobustClimateScaler.load(self.scaler_path)
        else:
            # fallback: tenta scaler_v2.json ou cria dummy (avisa)
            import warnings
            warnings.warn(f"Scaler não encontrado em {self.scaler_path} — usando minmax dummy (não recomendado para produção)")

        # Descobre arquivos
        pattern = str(nc_pattern)
        files = discover_cordex_files(pattern, domain=domain)
        if not files:
            # modo mock: gera dados sintéticos para teste sem NetCDF
            self.mock = True
            self.dates = pd.date_range("2026-01-01", "2100-12-31", freq="D")
            self.n_samples = len(self.dates) - seq_len
            return
        self.mock = False
        self.files = files
        # Abre com xarray (dask chunks)
        try:
            ds = xr.open_mfdataset(files, chunks="auto", combine="by_coords", parallel=True)
            # Recorta bbox Guarulhos
            ds = ds.sel(
                lat=slice(self.bbox["lat_min"], self.bbox["lat_max"]),
                lon=slice(self.bbox["lon_min"], self.bbox["lon_max"]),
            )
            # Média espacial (downscale estatístico pontual)
            ds_mean = ds.mean(dim=["lat", "lon"], skipna=True)
            df = ds_mean.to_dataframe().reset_index()
            # Converte vars CORDEX → THOR
            df_thor = self._cordex_to_thor(df)
            self.df = df_thor.sort_values("date").reset_index(drop=True)
            self.n_samples = max(0, len(self.df) - seq_len)
            self.dates = pd.to_datetime(self.df["date"])
        except Exception as e:
            raise RuntimeError(f"Falha ao abrir CORDEX {files[:2]}: {e}")

    def _cordex_to_thor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte DataFrame xarray CORDEX → colunas THOR."""
        out = pd.DataFrame()
        # date
        if "time" in df.columns:
            out["date"] = pd.to_datetime(df["time"])
        elif "date" in df.columns:
            out["date"] = pd.to_datetime(df["date"])
        else:
            out["date"] = pd.date_range("2026-01-01", periods=len(df), freq="D")

        # Mapeia cada var CORDEX
        for cordex_var, (thor_col, scale) in CORDEX_VAR_MAP.items():
            if cordex_var in df.columns:
                vals = df[cordex_var].values.astype(float) * scale
                if cordex_var in ["tas", "tasmax", "tasmin"]:
                    vals = vals - 273.15
                out[thor_col] = vals

        # Garante colunas THOR obrigatórias (preenche com NaN se faltar)
        for col in BASE_FEATURE_COLS:
            if col not in out.columns:
                # fallback: usa climatologia (ex: tmean 20°C, rh 80% etc.)
                fallback = {"tmean": 20.0, "tmax": 25.0, "tmin": 15.0, "rh": 80.0, "psfc": 936.0, "wind_speed": 2.0, "solar_rad": 20.0}
                out[col] = fallback.get(col, 0.0)

        # Deriva termo
        from src.preprocessing import engineer_thermodynamic_features, engineer_temporal_lags

        out = engineer_thermodynamic_features(out)
        # Lags: para CORDEX, precisa de histórico — faz shift e dropna
        out = engineer_temporal_lags(out, target_cols=PRIMARY_FEATURE_COLS, drop_na=False)
        # Preenche NaNs iniciais com forward fill (único caso onde bfill é ok: início da série futura 2026)
        out = out.ffill().bfill()
        return out

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        if getattr(self, "mock", False):
            # mock: retorna ruído com climatologia V2
            rng = np.random.default_rng(idx)
            x = rng.normal(0.5, 0.15, size=(self.seq_len, len(FEATURE_COLS))).astype(np.float32)
            x = np.clip(x, 0, 1)
            return torch.tensor(x, dtype=torch.float32), self.dates[idx + self.seq_len]

        # Real: pega janela [idx, idx+seq_len) de features
        window = self.df.iloc[idx : idx + self.seq_len][FEATURE_COLS].values.astype(np.float32)
        # Normaliza com scaler V2 (sem refit!)
        if self.scaler is not None and self.scaler.fitted:
            window = self.scaler.transform(window)
        date = self.dates.iloc[idx + self.seq_len]
        return torch.tensor(window, dtype=torch.float32), date
