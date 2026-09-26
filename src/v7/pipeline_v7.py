"""
THOR-PIML V7 — Pipeline de dados (zero-leakage) + CV temporal bloqueada
=======================================================================
- Features V7: 14 primárias V6 + colunas sinóticas/espaciais do GT V3 se
  existirem (detectadas automaticamente), lags [1,2,3,7,14] aplicados de verdade.
- Split cronológico 70/15/15 idêntico ao V6 → teste cego = 2019-09-02→2026-06-30
  (comparável com todos os números históricos).
- Scaler fit SÓ no treino (RobustClimateScaler), salvo em checkpoints/scaler_v7.json.
- CV temporal bloqueada (expanding window) sobre a região treino+validação:
  o teste cego NUNCA entra na seleção de modelo.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import THORConfig  # só para type-hint de compat
from src.preprocessing import (
    RobustClimateScaler,
    build_dataloaders,
    compute_dew_point,
    compute_specific_humidity,
    compute_vpd,
    create_sliding_windows,
    engineer_temporal_lags,
)
from src.v7.config_v7 import (
    LAG_DAYS_V7,
    build_feature_cols_v7,
    build_primary_cols_v7,
)

SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15  # teste cego = 15% final (2019-09-02 → 2026-06-30 no GT V2/V3)


def load_v7_frame(csv_path: Path) -> pd.DataFrame:
    """Carrega GT (V2 ou V3), ordena por data, deriva termo + sazonalidade."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    tmean = df["tmean"].values.astype(np.float32)
    rh = df["rh"].values.astype(np.float32)
    psfc = df["psfc"].values.astype(np.float32)
    df["dew_point"] = compute_dew_point(tmean, rh)
    df["vpd"] = compute_vpd(tmean, rh)
    df["specific_humidity"] = compute_specific_humidity(tmean, rh, psfc)
    doy = df["date"].dt.dayofyear.values.astype(np.float32)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    # cape/tcwv: V3 já tem REAIS nas colunas; V2 mantém proxy/constante do CSV.
    if "cape" not in df.columns:
        raise ValueError(
            "CSV sem coluna 'cape' — GT V2/V3 sempre tem (proxy ou real). Dados inconsistentes?"
        )
    if "tcwv" not in df.columns:
        df["tcwv"] = 20.0
    return df


def prepare_v7_arrays(
    csv_path: Path,
    lags: Optional[List[int]] = None,
    occurrence_threshold: float = 1.0,
) -> Dict:
    """Feature engineering completa → arrays 2D + metadados (sem split/scaler)."""
    lags = LAG_DAYS_V7 if lags is None else lags
    df = load_v7_frame(csv_path)
    primary_cols = build_primary_cols_v7(list(df.columns))
    df = engineer_temporal_lags(df, target_cols=primary_cols, lags=lags, drop_na=True)
    feature_cols = [c for c in build_feature_cols_v7(primary_cols, lags) if c in df.columns]
    n_declared = len(build_feature_cols_v7(primary_cols, lags))
    if len(feature_cols) != n_declared:
        # Regra hard (AGENTS.md): nunca silencioso
        print(
            f"[V7][AVISO] Features declaradas {n_declared} ≠ existentes {len(feature_cols)} "
            f"— colunas ausentes: {sorted(set(build_feature_cols_v7(primary_cols, lags)) - set(feature_cols))}"
        )
    assert df["pr_target"].notna().all(), "pr_target com NaN após lags"
    X = df[feature_cols].values.astype(np.float32)
    y = df["pr_target"].values.astype(np.float32)
    y_class = (y >= occurrence_threshold).astype(np.float32)
    dates = df["date"].reset_index(drop=True)
    return {
        "X": X, "y": y, "y_class": y_class, "dates": dates,
        "feature_cols": feature_cols, "lags": lags, "primary_cols": primary_cols,
    }


def _window_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    train_end: int,
    val_start: int,
    val_end: int,
    seq_len: int,
    occurrence_threshold: float,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, RobustClimateScaler]:
    """Scaler fit em [0, train_end); janelas com buffer de contexto para val."""
    scaler = RobustClimateScaler(method="minmax")
    scaler.fit(X[:train_end])
    x_train = scaler.transform(X[:train_end])
    buf = max(0, val_start - (seq_len - 1))
    x_val = scaler.transform(X[buf:val_end])
    tr = create_sliding_windows(x_train, y[:train_end], seq_len=seq_len, occurrence_threshold=occurrence_threshold)
    va = create_sliding_windows(x_val, y[buf:val_end], seq_len=seq_len, occurrence_threshold=occurrence_threshold)
    train_loader, val_loader, _ = build_dataloaders(
        tr, va, va, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader, scaler


def prepare_v7_pipeline(
    csv_path: Path,
    config=None,
    occurrence_threshold: float = 1.0,
    lags: Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, RobustClimateScaler, List[str]]:
    """Pipeline canônico V7: split 70/15/15 cronológico, loader de teste cego incluso."""
    from src.v7.config_v7 import THORConfigV7
    if config is None:
        config = THORConfigV7()
    seq_len = config.model.seq_len
    batch_size = config.training.batch_size
    data = prepare_v7_arrays(csv_path, lags=lags, occurrence_threshold=occurrence_threshold)
    X, y = data["X"], data["y"]
    n = len(X)
    n_train = int(n * SPLIT_TRAIN)
    n_val = int(n * SPLIT_VAL)

    train_loader, val_loader, scaler = _window_and_scale(
        X, y, n_train, n_train, n_train + n_val, seq_len,
        occurrence_threshold, batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    # Teste cego: buffer vem do fim do val (sem contaminação de scaler — só transform)
    buf_test = max(0, n_train + n_val - (seq_len - 1))
    x_test = scaler.transform(X[buf_test:])
    te = create_sliding_windows(x_test, y[buf_test:], seq_len=seq_len, occurrence_threshold=occurrence_threshold)
    test_loader, _, _ = build_dataloaders(
        te, te, te, batch_size=batch_size,
        num_workers=config.training.num_workers, pin_memory=config.training.pin_memory,
    )
    try:
        from src.paths import CHECKPOINT_DIR
        scaler.save(CHECKPOINT_DIR / "scaler_v7.json")
    except Exception:
        pass
    print(
        f"[V7 pipeline] {len(data['feature_cols'])} feats, lags {data['lags']}, "
        f"{n} dias | train até {data['dates'].iloc[n_train - 1].date()} | "
        f"val {data['dates'].iloc[n_train].date()}→{data['dates'].iloc[n_train + n_val - 1].date()} | "
        f"teste cego {data['dates'].iloc[n_train + n_val].date()}→{data['dates'].iloc[-1].date()}"
    )
    return train_loader, val_loader, test_loader, scaler, data["feature_cols"]


def blocked_temporal_folds(n_dev: int, n_folds: int = 5) -> List[Tuple[int, int, int]]:
    """Folds temporais em expanding window sobre a região dev (treino+val).

    Divide [0, n_dev) em n_folds blocos; fold i (i=1..n_folds-1) valida no
    bloco i e treina em tudo antes dele. Devolve (train_end, val_start, val_end)
    com val_start == train_end e val_end == início do próximo treino.
    """
    edges = np.linspace(0, n_dev, n_folds + 1).astype(int)
    folds = []
    for i in range(1, n_folds):
        folds.append((int(edges[i]), int(edges[i]), int(edges[i + 1])))
    return folds


def prepare_v7_cv_loaders(
    csv_path: Path,
    config=None,
    n_folds: int = 5,
    occurrence_threshold: float = 1.0,
    lags: Optional[List[int]] = None,
) -> Tuple[List[Tuple[DataLoader, DataLoader, Tuple[int, int, int]]], Dict]:
    """Loaders por fold — CV SÓ na região dev (primeiros 85%); teste cego intacto."""
    from src.v7.config_v7 import THORConfigV7
    if config is None:
        config = THORConfigV7()
    seq_len = config.model.seq_len
    batch_size = config.training.batch_size
    data = prepare_v7_arrays(csv_path, lags=lags, occurrence_threshold=occurrence_threshold)
    X, y = data["X"], data["y"]
    n_dev = int(len(X) * (SPLIT_TRAIN + SPLIT_VAL))
    folds = blocked_temporal_folds(n_dev, n_folds)
    loaders = []
    for train_end, val_start, val_end in folds:
        tr_loader, va_loader, _ = _window_and_scale(
            X, y, train_end, val_start, val_end, seq_len,
            occurrence_threshold, batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
        )
        loaders.append((tr_loader, va_loader, (train_end, val_start, val_end)))
    return loaders, data
