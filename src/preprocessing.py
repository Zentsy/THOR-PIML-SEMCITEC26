"""
THOR-PIML — Pré-processamento Zero-Leakage V2 (Sprint S2)
===========================================================
Carregamento do Ground Truth V2 (CHIRPS+ERA5+CEMADEN), derivação termodinâmica,
lags sem vazamento futuro, RobustClimateScaler versionado e janelamento 3D.

Mudanças S2 vs V1:
- engineer_temporal_lags: remove bfill().ffill() (vazamento futuro), agora descarta
  as primeiras max(LAG) linhas com NaN — rastreável, sem preencher com futuro.
- RobustClimateScaler: guarda feature_names, to_dict/from_dict, save/load JSON,
  denormalize por nome (não índice frágil), serializável para checkpoint CORDEX.
- prepare_zero_leakage_pipeline: suporta V2 (extra cols), descarta NaNs de lag
  antes do split, salva scaler_v2.json, sem buffers contaminados.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
try:
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    MinMaxScaler = None
    RobustScaler = None
    StandardScaler = None


class _NumpyMinMaxScaler:
    """Fallback puro NumPy caso scikit-learn não esteja instalado."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.data_range_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = np.maximum(self.data_max_ - self.data_min_, 1e-8)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.min_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.scale_


from src.config import (
    THORConfig,
    BASE_FEATURE_COLS,
    THERMO_FEATURE_COLS,
    PRIMARY_FEATURE_COLS,
    LAG_DAYS,
    FEATURE_COLS,
)


def compute_dew_point(
    tmean_c: np.ndarray,
    rh_pct: np.ndarray,
    a: float = 17.269,
    b: float = 237.3,
) -> np.ndarray:
    rh_clamped = np.clip(rh_pct, 1e-4, 100.0)
    gamma = np.log(rh_clamped / 100.0) + (a * tmean_c) / (b + tmean_c)
    t_d = (b * gamma) / (a - gamma)
    return t_d.astype(np.float32)


def compute_vpd(
    tmean_c: np.ndarray,
    rh_pct: np.ndarray,
    e0: float = 6.1078,
    a: float = 17.269,
    b: float = 237.3,
) -> np.ndarray:
    e_s = e0 * np.exp((a * tmean_c) / (b + tmean_c))
    rh_frac = np.clip(rh_pct / 100.0, 0.0, 1.0)
    vpd = e_s * (1.0 - rh_frac)
    return np.maximum(0.0, vpd).astype(np.float32)


def compute_specific_humidity(
    tmean_c: np.ndarray,
    rh_pct: np.ndarray,
    psfc_hpa: np.ndarray,
    e0: float = 6.1078,
    a: float = 17.269,
    b: float = 237.3,
) -> np.ndarray:
    e_s = e0 * np.exp((a * tmean_c) / (b + tmean_c))
    rh_frac = np.clip(rh_pct / 100.0, 0.0, 1.0)
    e_act = e_s * rh_frac
    psfc_safe = np.maximum(psfc_hpa, e_act + 1.0)
    q_g_kg = (622.0 * e_act) / (psfc_safe - 0.378 * e_act)
    return np.maximum(0.0, q_g_kg).astype(np.float32)


def engineer_thermodynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona dew_point, vpd, specific_humidity + sin/cos doy (V6) e year_norm legado (se habilitado)."""
    df_out = df.copy()
    tmean = df_out["tmean"].values.astype(np.float32)
    rh = df_out["rh"].values.astype(np.float32)
    psfc = df_out["psfc"].values.astype(np.float32)
    df_out["dew_point"] = compute_dew_point(tmean, rh)
    df_out["vpd"] = compute_vpd(tmean, rh)
    df_out["specific_humidity"] = compute_specific_humidity(tmean, rh, psfc)
    # Sazonalidade: sin/cos do day-of-year (V6 ALL-IN, sem leakage de year_norm)
    if "date" in df_out.columns:
        dates = pd.to_datetime(df_out["date"])
        doy = dates.dt.dayofyear.values.astype(np.float32)
        # sin/cos para capturar ciclo anual sem leakage temporal linear
        df_out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
        df_out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
        # Trend climático legado (year_norm) — só adiciona se não for V6 (V6 usa sin/cos)
        # Para compatibilidade, adiciona year_norm apenas se não existir sin_doy? Na verdade V5 precisa, V6 não
        # Vamos adicionar year_norm apenas se PRIMARY contém year_norm (detecta via config legado)
        try:
            from src.config import PRIMARY_FEATURE_COLS
            if "year_norm" in PRIMARY_FEATURE_COLS and "year_norm" not in df_out.columns:
                years = dates.dt.year.values.astype(np.float32)
                df_out["year_norm"] = (years - 1981.0) / 45.0
        except Exception:
            # fallback: não adiciona year_norm em V6
            pass
    else:
        df_out["sin_doy"] = 0.0
        df_out["cos_doy"] = 0.0
    # Preserva cape, tcwv se já existem (ERA5 REAL), senão cria proxy 0
    if "cape" not in df_out.columns:
        # proxy simples se não tem real: (tmax-tmin)*rh/100*10
        if "tmax" in df_out.columns and "tmin" in df_out.columns:
            tmax = df_out["tmax"].values.astype(np.float32)
            tmin = df_out["tmin"].values.astype(np.float32)
            rh_vals = df_out["rh"].values.astype(np.float32)
            df_out["cape"] = ((tmax - tmin) * rh_vals / 100.0 * 10.0).astype(np.float32)
        else:
            df_out["cape"] = 0.0
    if "tcwv" not in df_out.columns:
        df_out["tcwv"] = 20.0
    return df_out


def engineer_temporal_lags(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    lags: List[int] = LAG_DAYS,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Gera lags t-1, t-2, t-3, t-7 SEM vazamento futuro.

    V1 usava df.bfill().ffill() que preenchia os 7 primeiros dias com valores
    FUTUROS (dia 8). V2 descarta as primeiras max(lags) linhas com NaN — perda
    de 7 dias em 16.648 (<0.05%) é irrelevante e garante zero leakage.
    Se drop_na=False, mantém NaNs para que o caller decida (útil para debug).
    """
    df_out = df.copy()
    if target_cols is None:
        target_cols = PRIMARY_FEATURE_COLS
    lag_dict = {}
    for col in target_cols:
        for lag in lags:
            lag_dict[f"{col}_lag_{lag}"] = df_out[col].shift(lag)
    df_out = pd.concat([df_out, pd.DataFrame(lag_dict, index=df_out.index)], axis=1)
    if drop_na:
        # Descarta apenas onde algum lag é NaN (primeiras max(lags) linhas)
        df_out = df_out.dropna(subset=[f"{c}_lag_{l}" for c in target_cols for l in lags]).reset_index(drop=True)
    return df_out


class RobustClimateScaler:
    """Scaler versionado com zero-leakage e serialização para CORDEX.

    Guarda feature_names, data_min_/max_, e parâmetros internos do sklearn
    para que denormalize por nome (não índice) e possa ser salvo em JSON.
    """

    def __init__(self, method: str = "minmax", feature_names: Optional[List[str]] = None):
        self.method = method
        self.feature_names: Optional[List[str]] = feature_names
        if method == "minmax":
            if HAS_SKLEARN:
                self._scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                self._scaler = _NumpyMinMaxScaler(feature_range=(0, 1))
        elif method == "robust":
            if HAS_SKLEARN:
                self._scaler = RobustScaler()
            else:
                raise ImportError("scikit-learn é necessário para RobustScaler")
        elif method == "standard":
            if HAS_SKLEARN:
                self._scaler = StandardScaler()
            else:
                raise ImportError("scikit-learn é necessário para StandardScaler")
        else:
            raise ValueError(f"Método desconhecido: {method}")
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "RobustClimateScaler":
        self._scaler.fit(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        if feature_names is not None:
            self.feature_names = feature_names
        elif self.feature_names is None:
            # tenta inferir de FEATURE_COLS se shape coincidir
            if X.shape[1] == len(FEATURE_COLS):
                self.feature_names = FEATURE_COLS
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("RobustClimateScaler: .fit() em TREINO antes de .transform()!")
        return self._scaler.transform(X)

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        self.fit(X, feature_names=feature_names)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler não ajustado!")
        return self._scaler.inverse_transform(X)

    # --- Serialização (S2) ---

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "method": self.method,
            "feature_names": self.feature_names,
            "data_min": self.data_min_.tolist() if self.data_min_ is not None else None,
            "data_max": self.data_max_.tolist() if self.data_max_ is not None else None,
            "fitted": self.fitted,
        }
        # parâmetros do sklearn
        if hasattr(self._scaler, "data_min_"):
            d["sklearn_data_min"] = self._scaler.data_min_.tolist()
        if hasattr(self._scaler, "data_max_"):
            d["sklearn_data_max"] = self._scaler.data_max_.tolist()
        if hasattr(self._scaler, "center_"):
            d["sklearn_center"] = self._scaler.center_.tolist()
        if hasattr(self._scaler, "scale_"):
            d["sklearn_scale"] = self._scaler.scale_.tolist()
        if hasattr(self._scaler, "mean_"):
            d["sklearn_mean"] = self._scaler.mean_.tolist()
        if hasattr(self._scaler, "var_"):
            d["sklearn_var"] = self._scaler.var_.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RobustClimateScaler":
        obj = cls(method=d["method"], feature_names=d.get("feature_names"))
        obj.fitted = d.get("fitted", False)
        if d.get("data_min") is not None:
            obj.data_min_ = np.array(d["data_min"], dtype=np.float64)
        if d.get("data_max") is not None:
            obj.data_max_ = np.array(d["data_max"], dtype=np.float64)
        if not HAS_SKLEARN:
            obj._scaler = _NumpyMinMaxScaler()
            if obj.data_min_ is not None and obj.data_max_ is not None:
                obj._scaler.data_min_ = obj.data_min_
                obj._scaler.data_max_ = obj.data_max_
                obj._scaler.data_range_ = np.maximum(obj.data_max_ - obj.data_min_, 1e-8)
                if "sklearn_scale" in d and d["sklearn_scale"] is not None:
                    obj._scaler.scale_ = np.array(d["sklearn_scale"])
                else:
                    obj._scaler.scale_ = 1.0 / obj._scaler.data_range_
                obj._scaler.min_ = -obj._scaler.data_min_ * obj._scaler.scale_
            return obj
        # reconstrói scaler sklearn
        X_dummy = np.zeros((2, len(d["data_min"])), dtype=np.float64)
        # truque: fit em dummy e sobrescreve parâmetros
        obj._scaler.fit(X_dummy)
        if "sklearn_data_min" in d and hasattr(obj._scaler, "data_min_"):
            obj._scaler.data_min_ = np.array(d["sklearn_data_min"])
            obj._scaler.data_max_ = np.array(d["sklearn_data_max"])
            obj._scaler.data_range_ = obj._scaler.data_max_ - obj._scaler.data_min_
            obj._scaler.scale_ = 1.0 / obj._scaler.data_range_
            obj._scaler.min_ = -obj._scaler.data_min_ * obj._scaler.scale_
        if "sklearn_center" in d:
            obj._scaler.center_ = np.array(d["sklearn_center"])
            obj._scaler.scale_ = np.array(d["sklearn_scale"])
        if "sklearn_mean" in d:
            obj._scaler.mean_ = np.array(d["sklearn_mean"])
            obj._scaler.scale_ = np.array(d["sklearn_scale"])
            if "sklearn_var" in d:
                obj._scaler.var_ = np.array(d["sklearn_var"])
        return obj

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RobustClimateScaler":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    # --- Denormalização por nome (S2) ---

    def _index_of(self, feature: Union[str, int]) -> int:
        if isinstance(feature, int):
            return feature
        if self.feature_names is None:
            raise ValueError("feature_names não definido — use índice int ou defina feature_names")
        return self.feature_names.index(feature)

    def denormalize_column(self, col_tensor: torch.Tensor, col_idx: Union[int, str]) -> torch.Tensor:
        """Denormaliza coluna por nome ou índice (compatível com V1 que usava índice)."""
        if not self.fitted or self.data_min_ is None or self.data_max_ is None:
            return col_tensor
        if isinstance(col_idx, str):
            col_idx = self._index_of(col_idx)
        vmin = torch.tensor(self.data_min_[col_idx], device=col_tensor.device, dtype=col_tensor.dtype)
        vmax = torch.tensor(self.data_max_[col_idx], device=col_tensor.device, dtype=col_tensor.dtype)
        if self.method == "minmax":
            return col_tensor * (vmax - vmin) + vmin
        elif hasattr(self._scaler, "mean_") and hasattr(self._scaler, "scale_"):
            mean = torch.tensor(self._scaler.mean_[col_idx], device=col_tensor.device, dtype=col_tensor.dtype)
            scale = torch.tensor(self._scaler.scale_[col_idx], device=col_tensor.device, dtype=col_tensor.dtype)
            return col_tensor * scale + mean
        elif hasattr(self._scaler, "center_") and hasattr(self._scaler, "scale_"):
            center = torch.tensor(self._scaler.center_[col_idx], device=col_tensor.device, dtype=col_tensor.dtype)
            scale = torch.tensor(self._scaler.scale_[col_idx], device=col_tensor.device, dtype=col_tensor.dtype)
            return col_tensor * scale + center
        else:
            return col_tensor * (vmax - vmin) + vmin


def load_ground_truth_csv(
    csv_path: Union[str, Path],
    include_thermo: bool = True,
    include_lags: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Carrega ground_truth (V1 ou V2) e calcula features.

    V2 tem colunas extras pr_chirps, pr_cemaden, pr_provenance — são ignoradas
    para features, mas podem ser usadas para análise. termodinâmicas são sempre
    calculadas a partir de tmean/rh/psfc (ERA5 ou NASA fallback).
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if include_thermo:
        df = engineer_thermodynamic_features(df)
    if include_lags:
        df = engineer_temporal_lags(df, target_cols=PRIMARY_FEATURE_COLS if include_thermo else BASE_FEATURE_COLS)
    if include_thermo and include_lags:
        feature_cols = FEATURE_COLS
    elif include_thermo:
        feature_cols = PRIMARY_FEATURE_COLS
    else:
        feature_cols = BASE_FEATURE_COLS
    target_col = "pr_target"
    features = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    return features, target, df


def create_sliding_windows(
    features: np.ndarray,
    target: np.ndarray,
    seq_len: int = 30,
    occurrence_threshold: float = 0.1,  # V5 legado 0.1, V6 usa 1.0 (WMO ETCCDI)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(features) - seq_len
    if n_samples <= 0:
        raise ValueError(f"Dataset ({len(features)}) deve ser > seq_len ({seq_len})")
    shape = (n_samples, seq_len, features.shape[1])
    strides = (features.strides[0], features.strides[0], features.strides[1])
    X = np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides).copy()
    y_reg = target[seq_len:].reshape(-1, 1).astype(np.float32)
    # V6 fix: threshold configurável — 1.0mm unifica treino com ETCCDI (antes 0.1 causava CWD)
    y_class = (y_reg >= occurrence_threshold).astype(np.float32)
    return X, y_class, y_reg


def train_val_test_split(
    X: np.ndarray,
    y_class: np.ndarray,
    y_reg: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    X_train = X[:n_train]
    y_class_train = y_class[:n_train]
    y_reg_train = y_reg[:n_train]
    X_val = X[n_train : n_train + n_val]
    y_class_val = y_class[n_train : n_train + n_val]
    y_reg_val = y_reg[n_train : n_train + n_val]
    X_test = X[n_train + n_val :]
    y_class_test = y_class[n_train + n_val :]
    y_reg_test = y_reg[n_train + n_val :]
    return (X_train, y_class_train, y_reg_train), (X_val, y_class_val, y_reg_val), (X_test, y_class_test, y_reg_test)


def build_dataloaders(
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    X_tr, yc_tr, yr_tr = [torch.tensor(arr, dtype=torch.float32) for arr in train_data]
    X_va, yc_va, yr_va = [torch.tensor(arr, dtype=torch.float32) for arr in val_data]
    X_te, yc_te, yr_te = [torch.tensor(arr, dtype=torch.float32) for arr in test_data]
    train_ds = TensorDataset(X_tr, yc_tr, yr_tr)
    val_ds = TensorDataset(X_va, yc_va, yr_va)
    test_ds = TensorDataset(X_te, yc_te, yr_te)
    # S5: T4 otimizado — num_workers=4, pin_memory=True, persistent_workers
    # CPU fallback: num_workers=0 (sem multiprocessing)
    common_kwargs: Dict[str, Any] = {"pin_memory": pin_memory}
    if num_workers > 0:
        common_kwargs.update({"num_workers": num_workers, "persistent_workers": True})
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader


def prepare_zero_leakage_pipeline(
    csv_path: Union[str, Path],
    config: Optional[THORConfig] = None,
    scaler_method: str = "minmax",
    occurrence_threshold: Optional[float] = None,
    use_v6_features: bool = False,
    lags: Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, RobustClimateScaler, List[str]]:
    """Pipeline V2 zero-leakage sem bfill futuro.

    Passos:
    1. Carrega V1 ou V2 e calcula termo + lags (descarta 7 linhas iniciais com NaN)
    2. Split 2D cronológico (antes de normalizar)
    3. FIT só no train, TRANSFORM em train/val/test
    4. Janelas 3D com buffers de seq_len-1 do conjunto anterior (continuidade sem contaminação de scaler)
    5. Salva scaler_v2.json para CORDEX (S7)

    V6 extras:
    - occurrence_threshold: 0.1 (V5 legado) ou 1.0 (V6 WMO)
    - use_v6_features: se True, usa FEATURE_COLS_V6 (sem year_norm) e LAG_DAYS_V6
    - lags: override explícito (usado pelo run_evaluation para reproduzir o feature
      set EXATO de checkpoints antigos — ex.: V6d treinado com 70 feats porque o
      bug do lag-14 ainda existia)
    """
    if config is None:
        config = THORConfig()

    # Resolve feature set V6 vs V5
    if use_v6_features:
        try:
            from src.config_v6 import FEATURE_COLS_V6, PRIMARY_FEATURE_COLS_V6, LAG_DAYS_V6
            feature_cols_cfg = FEATURE_COLS_V6
            primary_cols = PRIMARY_FEATURE_COLS_V6
            resolved_lags = LAG_DAYS_V6 if lags is None else lags
            print(f"[Pipeline V6] Usando FEATURE_COLS_V6 (sem year_norm, {len(feature_cols_cfg)} feats, lags {resolved_lags})")
        except ImportError:
            feature_cols_cfg = FEATURE_COLS
            primary_cols = PRIMARY_FEATURE_COLS
            resolved_lags = LAG_DAYS if lags is None else lags
            print("[Pipeline V6] config_v6 não encontrado, fallback para V5")
    else:
        feature_cols_cfg = FEATURE_COLS
        primary_cols = PRIMARY_FEATURE_COLS
        resolved_lags = LAG_DAYS if lags is None else lags

    seq_len = config.model.seq_len
    batch_size = config.training.batch_size

    # occurrence threshold: prioriza config, depois arg, depois legado 0.1
    if occurrence_threshold is None:
        occurrence_threshold = getattr(getattr(config, 'model', None), 'occurrence_threshold_mm', None)
        if occurrence_threshold is None:
            occurrence_threshold = getattr(config, 'occurrence_threshold_mm', 0.1) if hasattr(config, 'occurrence_threshold_mm') else 0.1

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Engenharia termodinâmica — V6 opcionalmente ignora year_norm
    if use_v6_features:
        # V6 FINAL ALL-IN: calcula thermo + sin/cos doy + preserva cape/tcwv real se já existir
        df_out = df.copy()
        tmean = df_out["tmean"].values.astype(np.float32)
        rh = df_out["rh"].values.astype(np.float32)
        psfc = df_out["psfc"].values.astype(np.float32)
        df_out["dew_point"] = compute_dew_point(tmean, rh)
        df_out["vpd"] = compute_vpd(tmean, rh)
        df_out["specific_humidity"] = compute_specific_humidity(tmean, rh, psfc)
        # Sazonalidade V6: sin/cos dayofyear (sem year_norm leakage)
        if "date" in df_out.columns:
            dates = pd.to_datetime(df_out["date"])
            doy = dates.dt.dayofyear.values.astype(np.float32)
            df_out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
            df_out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
        else:
            df_out["sin_doy"] = 0.0
            df_out["cos_doy"] = 0.0
        # Preserva cape, tcwv se já existem (ERA5 REAL), senão proxy
        if "cape" not in df_out.columns:
            if "tmax" in df_out.columns and "tmin" in df_out.columns:
                tmax = df_out["tmax"].values.astype(np.float32)
                tmin = df_out["tmin"].values.astype(np.float32)
                rh_vals = df_out["rh"].values.astype(np.float32)
                df_out["cape"] = ((tmax - tmin) * rh_vals / 100.0 * 10.0).astype(np.float32)
            else:
                df_out["cape"] = 0.0
        if "tcwv" not in df_out.columns:
            df_out["tcwv"] = 20.0
        df = df_out
    else:
        df = engineer_thermodynamic_features(df)

    # FIX V7 (bug lag-14): antes chamava sem lags → default V5 [1,2,3,7] → as 14
    # colunas *_lag_14 nunca eram criadas e o "V6 84 feats" treinava com 70.
    df = engineer_temporal_lags(df, target_cols=primary_cols, lags=resolved_lags, drop_na=True)
    feature_cols = feature_cols_cfg
    # filtra apenas colunas que existem (para compatibilidade V6 sem year_norm)
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col = "pr_target"
    assert df[target_col].notna().all(), f"{target_col} com NaN após lags"
    raw_features = df[feature_cols].values.astype(np.float32)
    raw_target = df[target_col].values.astype(np.float32)
    n_samples = len(df)
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)
    feat_train = raw_features[:n_train]
    targ_train = raw_target[:n_train]
    buffer_val_start = max(0, n_train - (seq_len - 1))
    feat_val = raw_features[buffer_val_start : n_train + n_val]
    targ_val = raw_target[buffer_val_start : n_train + n_val]
    buffer_test_start = max(0, n_train + n_val - (seq_len - 1))
    feat_test = raw_features[buffer_test_start:]
    targ_test = raw_target[buffer_test_start:]
    scaler = RobustClimateScaler(method=scaler_method, feature_names=feature_cols)
    scaler.fit(feat_train)
    feat_train_norm = scaler.transform(feat_train)
    feat_val_norm = scaler.transform(feat_val)
    feat_test_norm = scaler.transform(feat_test)
    tr_data = create_sliding_windows(feat_train_norm, targ_train, seq_len=seq_len, occurrence_threshold=occurrence_threshold)
    va_data = create_sliding_windows(feat_val_norm, targ_val, seq_len=seq_len, occurrence_threshold=occurrence_threshold)
    te_data = create_sliding_windows(feat_test_norm, targ_test, seq_len=seq_len, occurrence_threshold=occurrence_threshold)
    train_loader, val_loader, test_loader = build_dataloaders(
        tr_data, va_data, te_data, batch_size=batch_size,
        num_workers=config.training.num_workers, pin_memory=config.training.pin_memory,
    )
    try:
        from src.paths import SCALER_JSON
        scaler.save(SCALER_JSON)
    except Exception:
        pass
    return train_loader, val_loader, test_loader, scaler, feature_cols
