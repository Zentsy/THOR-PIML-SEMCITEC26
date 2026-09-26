"""
THOR-PIML V8 — Pipeline espacial (GT V3 + campos ERA5 diários)
===============================================================
Monta tensores duplos por janela:
  X_surface (B, 30, F)  — features tabulares do GT V3 (mesma lógica do pipeline_v7)
  X_spatial (B, 30, H, W, C=5) — z500, u700, v700, q700, w500 do
      data/era5pl_domain_daily_1981_2026.nc (média diária por célula)

Zero-leakage: normalização espacial (min-max por canal) fitada SÓ no treino;
split 70/15/15 cronológico idêntico ao V7/V6 (teste cego comparável).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import create_sliding_windows
from src.v7.config_v7 import LAG_DAYS_V7, build_feature_cols_v7, build_primary_cols_v7
from src.v7.pipeline_v7 import SPLIT_TRAIN, SPLIT_VAL, load_v7_frame
from src.preprocessing import engineer_temporal_lags

SPATIAL_VARS = ["z500", "u700", "v700", "q700", "w500"]


def load_spatial_stack(nc_path: Path) -> pd.DataFrame:
    """NC diário → DataFrame long (date, lat, lon, var...)."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    df = ds.to_dataframe().reset_index()
    time_col = next((c for c in ["time", "valid_time", "date"] if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"Sem coordenada temporal em {nc_path}: cols={list(df.columns)}")
    df["date"] = pd.to_datetime(df[time_col]).dt.normalize()
    ds.close()
    return df


def build_spatial_tensor(df: pd.DataFrame, dates: pd.Series) -> np.ndarray:
    """Alinha stack espacial às datas do pipeline tabular → (n_days, H, W, C)."""
    lat_name = "latitude" if "latitude" in df.columns else "lat"
    lon_name = "longitude" if "longitude" in df.columns else "lon"
    piv = df.pivot_table(index="date", columns=[lat_name, lon_name], values=SPATIAL_VARS)
    piv = piv.reindex(pd.to_datetime(dates).dt.normalize())
    missing = int(piv[SPATIAL_VARS[0]].isna().any(axis=1).sum())
    if missing > 0.02 * len(dates):
        raise ValueError(
            f"{missing} dias sem campo espacial (>2%) — checar período do era5pl NC "
            "vs GT (regra anti-mock: não interpolar mais que isso)"
        )
    piv = piv.interpolate(limit_direction="both")
    n = len(dates)
    H = df[lat_name].nunique()
    W = df[lon_name].nunique()
    arr = np.empty((n, H, W, len(SPATIAL_VARS)), dtype=np.float32)
    for ci, var in enumerate(SPATIAL_VARS):
        block = piv[var].values.reshape(n, H, W)
        arr[..., ci] = block
    return arr


def prepare_v8_arrays(
    csv_path: Path,
    spatial_nc: Path,
    lags: Optional[List[int]] = None,
    occurrence_threshold: float = 1.0,
) -> Dict:
    lags = LAG_DAYS_V7 if lags is None else lags
    df = load_v7_frame(csv_path)
    primary_cols = build_primary_cols_v7(list(df.columns))
    df = engineer_temporal_lags(df, target_cols=primary_cols, lags=lags, drop_na=True)
    feature_cols = [c for c in build_feature_cols_v7(primary_cols, lags) if c in df.columns]
    dates = df["date"].reset_index(drop=True)

    sp_df = load_spatial_stack(spatial_nc)
    spatial = build_spatial_tensor(sp_df, dates)

    X = df[feature_cols].values.astype(np.float32)
    y = df["pr_target"].values.astype(np.float32)
    return {
        "X": X, "spatial": spatial, "y": y, "dates": dates,
        "feature_cols": feature_cols, "lags": lags,
    }


def _windows_spatial(X, S, y, start, end, seq_len, thr):
    """Janelas 3D pares (surface, spatial) sobre [start, end)."""
    n_samples = end - start - seq_len
    Xw = np.lib.stride_tricks.sliding_window_view(X[start:end], seq_len, axis=0).copy()
    Sw = np.lib.stride_tricks.sliding_window_view(S[start:end], seq_len, axis=0).copy()
    Xw = np.ascontiguousarray(Xw.transpose(0, 2, 1))  # (N, T, F)
    # sliding_window_view põe a janela no FIM: (N, H, W, C, T) → (N, T, H, W, C)
    Sw = np.ascontiguousarray(Sw.transpose(0, 4, 1, 2, 3))
    y_reg = y[start + seq_len : end].reshape(-1, 1).astype(np.float32)
    y_class = (y_reg >= thr).astype(np.float32)
    return Xw[:n_samples], Sw[:n_samples], y_class, y_reg


class SpatialScaler:
    """Min-max por canal espacial (fit só no treino) + scaler tabular do V7."""

    def __init__(self):
        self.mins: Optional[np.ndarray] = None
        self.maxs: Optional[np.ndarray] = None

    def fit(self, S: np.ndarray):
        self.mins = S.reshape(-1, S.shape[-1]).min(axis=0)
        self.maxs = S.reshape(-1, S.shape[-1]).max(axis=0)

    def transform(self, S: np.ndarray) -> np.ndarray:
        rng = np.maximum(self.maxs - self.mins, 1e-8)
        return (S - self.mins) / rng


def prepare_v8_pipeline(
    csv_path: Path,
    spatial_nc: Path,
    config=None,
    occurrence_threshold: float = 1.0,
    lags: Optional[List[int]] = None,
):
    """Retorna DataLoaders (x_surf, x_spatial, y_class, y_reg) train/val/test + meta."""
    from src.v7.config_v7 import THORConfigV7
    if config is None:
        config = THORConfigV7()
    seq_len = config.model.seq_len
    bs = config.training.batch_size
    data = prepare_v8_arrays(csv_path, spatial_nc, lags=lags, occurrence_threshold=occurrence_threshold)
    X, S, y = data["X"], data["spatial"], data["y"]
    n = len(X)
    n_train = int(n * SPLIT_TRAIN)
    n_val = int(n * SPLIT_VAL)

    # Tabular: scaler do V7 (fit treino)
    from src.preprocessing import RobustClimateScaler

    tab = RobustClimateScaler(method="minmax")
    tab.fit(X[:n_train])
    Xn = tab.transform(X)
    # Espacial: fit treino
    spsc = SpatialScaler()
    spsc.fit(S[:n_train])
    Sn = spsc.transform(S)

    thr = occurrence_threshold
    tr = _windows_spatial(Xn, Sn, y, 0, n_train, seq_len, thr)
    buf_v = max(0, n_train - (seq_len - 1))
    va = _windows_spatial(Xn, Sn, y, buf_v, n_train + n_val, seq_len, thr)
    buf_t = max(0, n_train + n_val - (seq_len - 1))
    te = _windows_spatial(Xn, Sn, y, buf_t, n, seq_len, thr)

    def mk(t, shuffle):
        ds = TensorDataset(torch.tensor(t[0]), torch.tensor(t[1]), torch.tensor(t[2]), torch.tensor(t[3]))
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=config.training.num_workers, pin_memory=config.training.pin_memory,
        )

    loaders = (mk(tr, True), mk(va, False), mk(te, False))
    print(
        f"[V8 pipeline] {len(data['feature_cols'])} feats tabulares + espacial "
        f"{S.shape[1]}×{S.shape[2]}×{S.shape[3]} | janelas tr/va/te: {len(tr[0])}/{len(va[0])}/{len(te[0])}"
    )
    return loaders, tab, data["feature_cols"]
