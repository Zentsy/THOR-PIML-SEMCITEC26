"""
THOR-PIML — Avaliação Hidrológica V2 (Sprint S6)
=================================================
NSE, KGE, RMSE, MAE, Bias + ETCCDI + Brier, ROC-AUC, Reliability
+ geração de relatório bonito (Markdown + TXT).

S6 vs V1:
- Adiciona Brier Score, ROC-AUC, Precision, Recall para head de ocorrência
- Mantém NSE/KGE etc.
- THORReport agora gera Markdown bonito (tabela) além de __str__ TXT legado
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class THORReport:
    rmse: float
    mae: float
    bias: float
    nse: float
    kge: float
    accuracy_occ: float
    f1_occ: float
    physics_violation_rate: float
    # Novos S6
    brier_score: float = 0.0
    roc_auc: float = 0.0
    precision_occ: float = 0.0
    recall_occ: float = 0.0
    # ETCCDI
    sdii_obs: float = 0.0
    sdii_pred: float = 0.0
    r10mm_obs: int = 0
    r10mm_pred: int = 0
    r20mm_obs: int = 0
    r20mm_pred: int = 0
    cwd_obs: int = 0
    cwd_pred: int = 0
    cdd_obs: int = 0
    cdd_pred: int = 0
    qb95_obs: float = 0.0
    qb95_pred: float = 0.0
    qb95_bias_pct: float = 0.0
    qb99_obs: float = 0.0
    qb99_pred: float = 0.0
    qb99_bias_pct: float = 0.0
    # Meta
    n_samples: int = 0

    def __str__(self) -> str:
        return (
            f"+--------------------------------------------------+\n"
            f"|        THOR-PIML — Relatório de Avaliação        |\n"
            f"+--------------------------------------------------+\n"
            f"|  [Métricas Hidrológicas e Estatísticas]          |\n"
            f"|  • RMSE (mm/dia)        :  {self.rmse:>19.4f} |\n"
            f"|  • MAE  (mm/dia)        :  {self.mae:>19.4f} |\n"
            f"|  • Bias (mm/dia)        :  {self.bias:>19.4f} |\n"
            f"|  • NSE  (Nash-Sutcliffe):  {self.nse:>19.4f} |\n"
            f"|  • KGE  (Kling-Gupta)   :  {self.kge:>19.4f} |\n"
            f"|  • Acurácia Ocorrência  :  {self.accuracy_occ*100:>18.2f}% |\n"
            f"|  • F1-Score Ocorrência  :  {self.f1_occ:>19.4f} |\n"
            f"|  • Brier Score          :  {self.brier_score:>19.4f} |\n"
            f"|  • ROC-AUC              :  {self.roc_auc:>19.4f} |\n"
            f"|  • Violação Física (%)  :  {self.physics_violation_rate*100:>18.2f}% |\n"
            f"+--------------------------------------------------+\n"
            f"|  [Índices de Extremos Climáticos WMO / ETCCDI]   |\n"
            f"|  • SDII (Obs vs Pred)   : {self.sdii_obs:>6.2f} vs {self.sdii_pred:>6.2f} mm/d |\n"
            f"|  • R10mm (Dias >= 10mm) : {self.r10mm_obs:>6d} vs {self.r10mm_pred:>6d} dias |\n"
            f"|  • R20mm (Dias >= 20mm) : {self.r20mm_obs:>6d} vs {self.r20mm_pred:>6d} dias |\n"
            f"|  • CWD (Dias Chuv. Cons): {self.cwd_obs:>6d} vs {self.cwd_pred:>6d} dias |\n"
            f"|  • CDD (Dias Secos Cons): {self.cdd_obs:>6d} vs {self.cdd_pred:>6d} dias |\n"
            f"|  • Quantil 95% (Obs/Pred): {self.qb95_obs:>5.2f} / {self.qb95_pred:>5.2f} mm |\n"
            f"|  • Vício Quantil 95% (QB95): {self.qb95_bias_pct:>13.2f}% |\n"
            f"|  • Quantil 99% (Obs/Pred): {self.qb99_obs:>5.2f} / {self.qb99_pred:>5.2f} mm |\n"
            f"|  • Vício Quantil 99% (QB99): {self.qb99_bias_pct:>13.2f}% |\n"
            f"+--------------------------------------------------"
        )

    def to_markdown(self) -> str:
        # Tabela bonita com badges (sem LaTeX excessivo)
        def badge(val, thresh, higher_better=True):
            if higher_better:
                return "🟢" if val >= thresh else "🔴" if val < thresh - 0.1 else "🟡"
            else:
                return "🟢" if val <= thresh else "🔴" if val > thresh + 1 else "🟡"

        nse_badge = badge(self.nse, 0.5, True)
        kge_badge = badge(self.kge, 0.5, True)
        r20_recall = (self.r20mm_pred / max(self.r20mm_obs, 1) * 100) if self.r20mm_obs else 0
        r20_badge = "🟢" if r20_recall >= 30 else "🔴" if r20_recall == 0 else "🟡"
        qb99_ok = abs(self.qb99_bias_pct) <= 25
        qb99_badge = "🟢" if qb99_ok else "🔴" if abs(self.qb99_bias_pct) > 40 else "🟡"

        md = f"""# 📊 THOR-PIML — Relatório de Avaliação V2

> **Amostras:** {self.n_samples} dias (teste cego) • **Gerado por:** `src/evaluate.py` S6

## 1. Métricas Hidrológicas Principais

| Métrica | Valor | Alvo | Status |
|:---|---:|---|:---:|
| **NSE** (Nash-Sutcliffe) | `{self.nse:.4f}` | >0.50 | {nse_badge} |
| **KGE** (Kling-Gupta) | `{self.kge:.4f}` | >0.50 | {kge_badge} |
| **RMSE** (mm/dia) | `{self.rmse:.2f}` | <5.0 | {badge(-self.rmse, -5, True)} |
| **MAE** (mm/dia) | `{self.mae:.2f}` | <3.0 | {badge(-self.mae, -3, True)} |
| **Bias** (mm/dia) | `{self.bias:+.2f}` | ±1.0 | {"🟢" if abs(self.bias) < 1.0 else "🟡" if abs(self.bias) < 2 else "🔴"} |

## 2. Head de Ocorrência (Seco vs Chuvoso)

| Métrica | Valor | Alvo | Status |
|:---|---:|---|:---:|
| **Acurácia** | `{self.accuracy_occ*100:.2f}%` | >75% | {"🟢" if self.accuracy_occ >= 0.75 else "🔴"} |
| **F1-Score** | `{self.f1_occ:.4f}` | >0.70 | {"🟢" if self.f1_occ >= 0.70 else "🟡"} |
| **Precision** | `{self.precision_occ:.4f}` | >0.70 | {"🟢" if self.precision_occ >= 0.70 else "🟡"} |
| **Recall** | `{self.recall_occ:.4f}` | >0.70 | {"🟢" if self.recall_occ >= 0.70 else "🟡"} |
| **Brier Score** | `{self.brier_score:.4f}` | <0.20 | {"🟢" if self.brier_score < 0.20 else "🟡"} |
| **ROC-AUC** | `{self.roc_auc:.4f}` | >0.80 | {"🟢" if self.roc_auc >= 0.80 else "🟡"} |
| **Violação Física** | `{self.physics_violation_rate*100:.2f}%` | 0.00% | {"🟢" if self.physics_violation_rate == 0 else "🔴"} |

## 3. Extremos Climáticos (WMO ETCCDI)

| Índice | Observado | Predito | Viés / Recall | Status |
|:---|---:|---:|---|:---:|
| **SDII** (mm/d úmido) | `{self.sdii_obs:.2f}` | `{self.sdii_pred:.2f}` | `{self.sdii_pred - self.sdii_obs:+.2f}` | {"🟢" if abs(self.sdii_pred - self.sdii_obs) < 1 else "🟡"} |
| **R10mm** (dias ≥10mm) | `{self.r10mm_obs}` | `{self.r10mm_pred}` | recall `{self.r10mm_pred/max(self.r10mm_obs,1)*100:.1f}%` | {"🟢" if self.r10mm_pred/max(self.r10mm_obs,1) >= 0.5 else "🔴"} |
| **R20mm** (dias ≥20mm) | `{self.r20mm_obs}` | `{self.r20mm_pred}` | recall `{r20_recall:.1f}%` | {r20_badge} |
| **CWD** (máx chuvosos cons.) | `{self.cwd_obs}` | `{self.cwd_pred}` | `+{self.cwd_pred-self.cwd_obs}` | {"🟢" if abs(self.cwd_pred-self.cwd_obs) < 10 else "🔴"} |
| **CDD** (máx secos cons.) | `{self.cdd_obs}` | `{self.cdd_pred}` | `{self.cdd_pred-self.cdd_obs:+d}` | {"🟢" if abs(self.cdd_pred-self.cdd_obs) < 10 else "🟡"} |
| **QB95** (95º percentil) | `{self.qb95_obs:.2f} mm` | `{self.qb95_pred:.2f} mm` | `{self.qb95_bias_pct:+.1f}%` | {"🟢" if abs(self.qb95_bias_pct) < 20 else "🟡" if abs(self.qb95_bias_pct) < 35 else "🔴"} |
| **QB99** (99º percentil) | `{self.qb99_obs:.2f} mm` | `{self.qb99_pred:.2f} mm` | `{self.qb99_bias_pct:+.1f}%` | {qb99_badge} |

> **Legenda:** 🟢 Bom • 🟡 Atenção • 🔴 Crítico

---
*Dica: R20 recall 0% = modelo não aprendeu tempestade (V1). Meta V2: ≥30%. QB99 |viés|>40% = subestimação severa (V1: -48%).*
"""
        return md


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))

def calculate_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    if den == 0:
        return 0.0
    return float(1.0 - (num / den))

def calculate_kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true == 0 or mean_true == 0:
        return 0.0
    r = float(np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1])
    if np.isnan(r):
        r = 0.0
    alpha = std_pred / std_true
    beta = mean_pred / mean_true
    ed = np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(1.0 - ed)

def calculate_occurrence_metrics(y_class_true: np.ndarray, prob_occ: np.ndarray, threshold: float = 0.5):
    y_class_pred = (prob_occ >= threshold).astype(np.float32)
    acc = float(np.mean(y_class_true == y_class_pred))
    tp = np.sum((y_class_pred == 1.0) & (y_class_true == 1.0))
    fp = np.sum((y_class_pred == 1.0) & (y_class_true == 0.0))
    fn = np.sum((y_class_pred == 0.0) & (y_class_true == 1.0))
    tn = np.sum((y_class_pred == 0.0) & (y_class_true == 0.0))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    # Brier
    brier = float(np.mean((prob_occ - y_class_true) ** 2))
    # ROC-AUC (sem sklearn, via rank)
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = float(roc_auc_score(y_class_true, prob_occ))
    except Exception:
        # fallback Mann-Whitney
        pos = prob_occ[y_class_true == 1]
        neg = prob_occ[y_class_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            roc_auc = 0.5
        else:
            # aproximação rápida
            n_pos, n_neg = len(pos), len(neg)
            # usa rank
            all_scores = np.concatenate([pos, neg])
            all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
            order = np.argsort(all_scores)
            ranks = np.argsort(order).argsort()
            # soma ranks dos positivos
            sum_ranks_pos = np.sum(ranks[all_labels == 1])
            roc_auc = float((sum_ranks_pos - n_pos*(n_pos-1)/2) / (n_pos * n_neg))
            roc_auc = float(np.clip(roc_auc, 0, 1))
    return acc, float(f1), float(precision), float(recall), brier, float(roc_auc)

def calculate_sdii(y: np.ndarray, threshold: float = 1.0) -> float:
    wet_days = y[y >= threshold]
    if len(wet_days) == 0:
        return 0.0
    return float(np.mean(wet_days))

def calculate_r10mm(y: np.ndarray) -> int:
    return int(np.sum(y >= 10.0))

def calculate_r20mm(y: np.ndarray) -> int:
    return int(np.sum(y >= 20.0))

def calculate_cwd(y: np.ndarray, threshold: float = 1.0) -> int:
    max_cwd = 0
    current = 0
    for val in y:
        if val >= threshold:
            current += 1
            if current > max_cwd:
                max_cwd = current
        else:
            current = 0
    return int(max_cwd)

def calculate_cdd(y: np.ndarray, threshold: float = 1.0) -> int:
    max_cdd = 0
    current = 0
    for val in y:
        if val < threshold:
            current += 1
            if current > max_cdd:
                max_cdd = current
        else:
            current = 0
    return int(max_cdd)

def calculate_quantile_bias(y_true: np.ndarray, y_pred: np.ndarray, q: float):
    q_obs = float(np.percentile(y_true, q))
    q_pred = float(np.percentile(y_pred, q))
    bias_pct = float(((q_pred - q_obs) / (q_obs + 1e-8)) * 100.0)
    return q_obs, q_pred, bias_pct

def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_class_true: Optional[np.ndarray] = None,
    prob_occ: Optional[np.ndarray] = None,
    n_violations: int = 0,
) -> THORReport:
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    bias = calculate_bias(y_true, y_pred)
    nse = calculate_nse(y_true, y_pred)
    kge = calculate_kge(y_true, y_pred)
    if y_class_true is not None and prob_occ is not None:
        acc, f1, prec, rec, brier, roc_auc = calculate_occurrence_metrics(y_class_true, prob_occ)
    else:
        acc, f1, prec, rec, brier, roc_auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.5
    violation_rate = n_violations / max(len(y_true), 1)
    sdii_obs = calculate_sdii(y_true)
    sdii_pred = calculate_sdii(y_pred)
    r10mm_obs = calculate_r10mm(y_true)
    r10mm_pred = calculate_r10mm(y_pred)
    r20mm_obs = calculate_r20mm(y_true)
    r20mm_pred = calculate_r20mm(y_pred)
    cwd_obs = calculate_cwd(y_true)
    cwd_pred = calculate_cwd(y_pred)
    cdd_obs = calculate_cdd(y_true)
    cdd_pred = calculate_cdd(y_pred)
    qb95_obs, qb95_pred, qb95_bias_pct = calculate_quantile_bias(y_true, y_pred, 95.0)
    qb99_obs, qb99_pred, qb99_bias_pct = calculate_quantile_bias(y_true, y_pred, 99.0)
    return THORReport(
        rmse=rmse, mae=mae, bias=bias, nse=nse, kge=kge,
        accuracy_occ=acc, f1_occ=f1, physics_violation_rate=violation_rate,
        brier_score=brier, roc_auc=roc_auc, precision_occ=prec, recall_occ=rec,
        sdii_obs=sdii_obs, sdii_pred=sdii_pred,
        r10mm_obs=r10mm_obs, r10mm_pred=r10mm_pred,
        r20mm_obs=r20mm_obs, r20mm_pred=r20mm_pred,
        cwd_obs=cwd_obs, cwd_pred=cwd_pred,
        cdd_obs=cdd_obs, cdd_pred=cdd_pred,
        qb95_obs=qb95_obs, qb95_pred=qb95_pred, qb95_bias_pct=qb95_bias_pct,
        qb99_obs=qb99_obs, qb99_pred=qb99_pred, qb99_bias_pct=qb99_bias_pct,
        n_samples=len(y_true),
    )
