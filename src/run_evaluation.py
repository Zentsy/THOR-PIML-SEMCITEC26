"""
THOR-PIML — Avaliação Pós-Treino (Sprint S6)
=============================================
Gera relatório bonito + 6 plots de alta qualidade a partir de um checkpoint.

Uso (na T4 após treinar, ou local):
    python src/run_evaluation.py                          # usa checkpoints/best_model.pt + V2
    python src/run_evaluation.py --ckpt checkpoints/best_model.pt --csv data/ground_truth_guarulhos_daily_v2.csv
    python src/run_evaluation.py --pretty-only            # só re-gera plots sem re-avaliar

Saída:
    results/thor_metrics_summary.txt   (legado, para compatibilidade)
    results/RELATORIO_RESULTADOS_V2.md (novo, bonito, com badges)
    results/graficos/*.png (6 plots 300 DPI, estilo seaborn-v0_8-whitegrid)
"""
from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import CHECKPOINT_DIR, RESULTS_DIR, GRAFICOS_DIR, GROUND_TRUTH_V2, GROUND_TRUTH_V1

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import THORConfig
from src.preprocessing import prepare_zero_leakage_pipeline
from src.model import THORPIMLModel
from src.physics_loss import THORPhysicsLoss
from src.evaluate import full_evaluation
from src.utils import get_device, set_seed

# Estilo bonito S6
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 300
sns.set_palette("viridis")

COLORS = {
    "obs": "#0f172a",      # slate-900
    "pred": "#e11d48",     # rose-600
    "phys": "#7c3aed",     # violet
    "bce": "#059669",      # emerald
    "grid": "#e2e8f0",
}


def detect_thor_version(ckpt: dict) -> str:
    """Detecta a versão do modelo a partir do checkpoint.

    Ordem: marcador explícito 'thor_version' (V7+) → heurística pela config
    salva (V5 ModelConfig tem 'use_year_norm'; V6 ModelConfigV6 não tem).
    """
    if ckpt.get("thor_version"):
        return str(ckpt["thor_version"]).lower()
    cfg = ckpt.get("config") or {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    if isinstance(model_cfg, dict) and "use_year_norm" in model_cfg:
        return "v5"
    if isinstance(model_cfg, dict) and "lstm_hidden" in model_cfg:
        return "v6"
    return "v5"


def infer_lags_from_feature_names(feature_names: list) -> list | None:
    """Extrai os lags reais usados no treino a partir do scaler salvo no checkpoint.

    Necessário porque o V6d histórico treinou com 70 feats (bug lag-14) —
    recriar o pipeline com lags [1,2,3,7,14] quebraria o load_state_dict.
    """
    if not feature_names:
        return None
    lags = set()
    for name in feature_names:
        if "_lag_" in name:
            try:
                lags.add(int(name.rsplit("_lag_", 1)[1]))
            except ValueError:
                continue
    return sorted(lags) if lags else None


def evaluate_from_checkpoint(ckpt_path: Path, csv_path: Path, device: torch.device):
    # Avaliação determinística — seed fixa reprodutível (não confundir com seed aleatória de treino)
    set_seed(42, eval_mode=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) if ckpt_path.exists() else {}
    version = detect_thor_version(ckpt)
    scaler_state = ckpt.get("scaler_state") or {}
    feature_names = scaler_state.get("feature_names") or []
    ckpt_lags = infer_lags_from_feature_names(feature_names)

    if version == "v6":
        from src.config_v6 import THORConfigV6
        from src.model_v6 import THORPIMLModelV6
        from src.physics_loss_v6 import THORLossV6
        config = THORConfigV6()
        occ = float((ckpt.get("config") or {}).get("model", {}).get("occurrence_threshold_mm", 1.0))
        config.model.occurrence_threshold_mm = occ
        train_loader, val_loader, test_loader, scaler, feature_cols = prepare_zero_leakage_pipeline(
            csv_path, config=config, scaler_method="minmax",
            use_v6_features=True, lags=ckpt_lags, occurrence_threshold=occ,
        )
        model = THORPIMLModelV6(config.model)
        loss_cls = THORLossV6
        model_ctor = lambda cfg: THORPIMLModelV6(cfg)  # noqa: E731
    elif version == "v7":
        from src.v7.config_v7 import THORConfigV7
        from src.v7.pipeline_v7 import prepare_v7_pipeline
        from src.v7.model_v7 import THORHybridModel
        from src.v7.physics_loss_v7 import THORLossV7
        config = THORConfigV7()
        train_loader, val_loader, test_loader, scaler, feature_cols = prepare_v7_pipeline(
            csv_path, config=config, occurrence_threshold=1.0, lags=ckpt_lags,
        )
        config.model.n_features = len(feature_cols)
        model_ctor = lambda cfg: THORHybridModel(cfg)  # noqa: E731
        model = model_ctor(config.model)
        loss_cls = THORLossV7
    else:
        config = THORConfig()
        occ = float((ckpt.get("config") or {}).get("model", {}).get("occurrence_threshold_mm", 0.1))
        train_loader, val_loader, test_loader, scaler, feature_cols = prepare_zero_leakage_pipeline(
            csv_path, config=config, scaler_method="minmax", occurrence_threshold=occ,
        )
        model_ctor = lambda cfg: THORPIMLModel(cfg)  # noqa: E731
        model = model_ctor(config.model)
        loss_cls = THORPhysicsLoss

    df_raw = pd.read_csv(csv_path)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.sort_values('date').reset_index(drop=True)

    X_test = test_loader.dataset.tensors[0].to(device)
    y_class_test = test_loader.dataset.tensors[1].cpu().numpy().ravel()
    y_true_test = test_loader.dataset.tensors[2].cpu().numpy().ravel()
    dates_test = df_raw['date'].iloc[-len(y_true_test):].reset_index(drop=True)

    # Reconcilia n_features com o pipeline real (mesma regra do run_thor_training)
    if getattr(config.model, "n_features", None) != len(feature_cols):
        print(f"[Eval FIX] n_features {config.model.n_features} → {len(feature_cols)} (checkpoint {version})")
        config.model.n_features = len(feature_cols)
        model = model_ctor(config.model)

    loss_fn = loss_cls(scaler=scaler, training_config=config.training)

    model = model.to(device)
    if ckpt_path.exists():
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Checkpoint carregado: {ckpt_path} (versão {version}, epoch {ckpt.get('epoch','?')}, val_loss {ckpt.get('val_loss',0):.4f})")
        history = ckpt.get("history", None)
    else:
        print(f"⚠ Checkpoint não encontrado: {ckpt_path} — usando modelo aleatório (apenas para teste de plots)")
        history = None

    model.eval()
    with torch.no_grad():
        prob_occ, intensity, y_pred_t = model(X_test, return_components=True)
        # W_max com psfc (S3)
        try:
            from src.preprocessing import FEATURE_COLS
            # usa nomes se scaler tem feature_names
            tmean = loss_fn._denorm(X_test, "tmean", 0)
            rh = loss_fn._denorm(X_test, "rh", 3)
            psfc = loss_fn._denorm(X_test, "psfc", 4)
            w_max = loss_fn.compute_thermodynamic_limit_wmax(tmean, rh, psfc)
        except Exception:
            tmean = loss_fn.denormalize_feature(X_test, 0)
            rh = loss_fn.denormalize_feature(X_test, 3)
            w_max = loss_fn.compute_thermodynamic_limit_wmax(tmean, rh, None)
        n_viol = int((y_pred_t > w_max).sum().item())

    y_pred = y_pred_t.cpu().numpy().ravel()
    prob_occ_np = prob_occ.cpu().numpy().ravel()

    # Diagnóstico S6: report cru (sem clip) + report com clip 1mm para R10/CWD
    # Clip 1mm: final<1mm → 0 (corrige garoa infinita sem re-treino, ver ANALISE_DEGRADACAO)
    y_pred_clipped = np.where(y_pred < 1.0, 0.0, y_pred)
    report = full_evaluation(y_true_test, y_pred, y_class_test, prob_occ_np, n_violations=n_viol)
    report_clipped = full_evaluation(y_true_test, y_pred_clipped, y_class_test, prob_occ_np, n_violations=n_viol)
    # Guarda clipped para plots extras (não sobrescreve report principal)
    report._clipped = report_clipped  # type: ignore
    return report, history, (y_true_test, y_pred, y_class_test, prob_occ_np, dates_test)


def plot_all(report, history, data_tuple, out_dir: Path):
    y_true_test, y_pred, y_class_test, prob_occ_np, dates_test = data_tuple
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Fig 1: Loss total (se history existe) ---
    if history and "train_loss" in history:
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, history["train_loss"], label="Treino", color=COLORS["pred"], lw=2.5)
        ax.plot(epochs, history["val_loss"], label="Validação", color=COLORS["obs"], lw=2.5, ls="--")
        ax.set_title("Evolução da Loss Composta (α·Focal + β·MSE·w_storm + λ·Physics)", fontweight="bold")
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.legend(frameon=True, facecolor="white")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig1_loss_total.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        # Fig 2-4: BCE, MSE, Physics
        for key, title, ylabel, color in [
            ("bce", "Head de Ocorrência (Focal Loss γ=2.0)", "BCE/Focal", COLORS["bce"]),
            ("mse", "Head de Intensidade (MSE ponderado storm×3)", "MSE (mm²)", COLORS["pred"]),
            ("physics", "Restrição Física PIML (Softplus β=1.0)", "Physics Loss", COLORS["phys"]),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(epochs, history[f"train_{key}"], label="Treino", color=color, lw=2.5)
            if f"val_{key}" in history:
                ax.plot(epochs, history[f"val_{key}"], label="Validação", color=COLORS["obs"], lw=2, ls="--")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Época")
            ax.set_ylabel(ylabel)
            ax.legend(frameon=True, facecolor="white")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"fig_{key}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    # --- Fig 5: Downscaling com datas reais (01/01/2024-15/02/2024) ---
    mask = (dates_test >= "2024-01-01") & (dates_test <= "2024-02-15")
    idx = np.where(mask)[0]
    if len(idx) > 10:
        sub_dates = dates_test.iloc[idx]
        sub_true = y_true_test[idx]
        sub_pred = y_pred[idx]
        sub_class_true = y_class_test[idx]
        sub_prob = prob_occ_np[idx]
        window_report = full_evaluation(sub_true, sub_pred, sub_class_true, sub_prob)

        fig, ax = plt.subplots(figsize=(15, 6.5))
        ax.plot(sub_dates, sub_true, label="Observado (CHIRPS+CEMADEN)", color=COLORS["obs"], lw=2.8, marker="o", ms=4)
        ax.plot(sub_dates, sub_pred, label="THOR-PIML V2", color=COLORS["pred"], lw=2.2, marker="s", ms=4, ls="--")
        exact = (sub_true == 0) & (sub_pred < 0.5)
        if exact.any():
            ax.scatter(sub_dates[exact], np.zeros(exact.sum()), color="#10b981", s=90, zorder=5, label=f"Acerto seco ({exact.sum()} dias)")
        ax.set_title("Downscaling Diário com Datas Reais (01/01/2024–15/02/2024)", fontweight="bold", fontsize=14)
        ax.set_xlabel("Data", fontweight="bold")
        ax.set_ylabel("Precipitação (mm/dia)", fontweight="bold")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%b'))
        plt.xticks(rotation=45, fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
        box = f"Janela 01/01–15/02/2024:\nRMSE {window_report.rmse:.2f} mm\nMAE {window_report.mae:.2f} mm\nNSE {window_report.nse:.2f} • R20 {window_report.r20mm_pred}/{window_report.r20mm_obs}"
        ax.text(0.02, 0.96, box, transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor="#cbd5e1"))
        fig.tight_layout()
        fig.savefig(out_dir / "fig5_downscaling.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Fig 6: Scatter 1:1 ---
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(y_true_test, y_pred, alpha=0.35, color=COLORS["pred"], s=18, edgecolors="none", label="Dias (teste)")
    m = max(float(y_true_test.max()), float(y_pred.max())) + 5
    ax.plot([0, m], [0, m], color=COLORS["obs"], ls="--", lw=2, label="1:1 ideal")
    ax.set_title("Observado vs Predito (Teste Cego)", fontweight="bold")
    ax.set_xlabel("Observado (mm/dia) — CHIRPS+CEMADEN")
    ax.set_ylabel("Predito THOR-PIML (mm/dia)")
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.legend(frameon=True, facecolor="white")
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f"NSE {report.nse:.2f}\nKGE {report.kge:.2f}\nRMSE {report.rmse:.2f} mm\nR20 {report.r20mm_pred}/{report.r20mm_obs}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cbd5e1"))
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 7: Matriz de confusão bonita ---
    y_pred_class = (prob_occ_np >= 0.5).astype(int)
    cm = confusion_matrix(y_class_test, y_pred_class)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None] * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    annot = np.array([[f"{cm[i,j]:,}\n{cm_pct[i,j]:.1f}%" for j in range(2)] for i in range(2)])
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred Seco", "Pred Chuvoso"], yticklabels=["Obs Seco", "Obs Chuvoso"], annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"Matriz de Confusão — Acurácia {report.accuracy_occ*100:.1f}% • F1 {report.f1_occ:.2f}", fontweight="bold")
    ax.set_xlabel("Predito (threshold 0.5)")
    ax.set_ylabel("Observado")
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 8: Reliability / QQ ---
    # QQ plot quantis
    qs = np.linspace(0.5, 0.99, 20)
    q_obs = np.percentile(y_true_test, qs * 100)
    q_pred = np.percentile(y_pred, qs * 100)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(q_obs, q_pred, color=COLORS["phys"], s=60, edgecolors="white", linewidths=1.2, label="Quantis 50–99%")
    ax.plot([0, max(q_obs.max(), q_pred.max())], [0, max(q_obs.max(), q_pred.max())], color=COLORS["obs"], ls="--", lw=2, label="1:1")
    ax.set_title("Q-Q Plot — Calibração de Extremos (QB95, QB99)", fontweight="bold")
    ax.set_xlabel("Quantil Observado (mm)")
    ax.set_ylabel("Quantil Predito (mm)")
    ax.legend(frameon=True, facecolor="white")
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f"QB95 {report.qb95_bias_pct:+.1f}%\nQB99 {report.qb99_bias_pct:+.1f}%", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cbd5e1"))
    fig.tight_layout()
    fig.savefig(out_dir / "fig8_qq.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 9: Reliability diagram (Brier) ---
    # Bin prob_occ em 10 bins
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_counts = []
    bin_acc = []
    for i in range(10):
        mask_bin = (prob_occ_np >= bins[i]) & (prob_occ_np < bins[i+1] if i < 9 else prob_occ_np <= 1)
        if mask_bin.sum() > 0:
            bin_acc.append(float(np.mean(y_class_test[mask_bin])))
            bin_counts.append(int(mask_bin.sum()))
        else:
            bin_acc.append(np.nan)
            bin_counts.append(0)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 1], [0, 1], color=COLORS["obs"], ls="--", lw=2, label="Perfeito")
    ax.plot(bin_centers, bin_acc, marker="o", color=COLORS["pred"], lw=2.5, ms=8, label="THOR-PIML")
    # tamanho do ponto proporcional ao count
    sizes = np.array(bin_counts) / max(max(bin_counts), 1) * 300 + 30
    ax.scatter(bin_centers, bin_acc, s=sizes, color=COLORS["pred"], alpha=0.6, edgecolors="white")
    ax.set_title(f"Reliability Diagram — Brier {report.brier_score:.3f} • ROC-AUC {report.roc_auc:.3f}", fontweight="bold")
    ax.set_xlabel("Probabilidade Predita (bin)")
    ax.set_ylabel("Frequência Observada")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=True, facecolor="white")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ 7-9 plots salvos em {out_dir}")


def save_reports(report, history, csv_path: Path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # TXT legado
    with open(RESULTS_DIR / "thor_metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(report))
    # Markdown bonito V2
    md = report.to_markdown()
    # Adiciona diagnóstico com clip 1mm (S6 correção garoa)
    clipped = getattr(report, "_clipped", None)
    if clipped is not None:
        md += f"\n\n## 3b. Diagnóstico com Clip 1mm (pós-processamento, sem re-treino)\n\n> `y_pred<1mm → 0` corrige CWD inflado sem mudar NSE/RMSE (só contagens).\n\n| Índice | Sem Clip | Com Clip 1mm | Δ |\n|:---|---:|---:|---:|\n| **R10mm** | {report.r10mm_pred} | {clipped.r10mm_pred} | {clipped.r10mm_pred - report.r10mm_pred:+d} |\n| **R20mm** | {report.r20mm_pred} | {clipped.r20mm_pred} | {clipped.r20mm_pred - report.r20mm_pred:+d} |\n| **CWD** | {report.cwd_pred} | {clipped.cwd_pred} | {clipped.cwd_pred - report.cwd_pred:+d} |\n| **CDD** | {report.cdd_pred} | {clipped.cdd_pred} | {clipped.cdd_pred - report.cdd_pred:+d} |\n| **SDII** | {report.sdii_pred:.2f} | {clipped.sdii_pred:.2f} | {clipped.sdii_pred - report.sdii_pred:+.2f} |\n"
    # Adiciona seção de histórico se houver
    if history and "train_loss" in history:
        md += f"\n\n## 4. Histórico de Treino\n\n- **Épocas:** {len(history['train_loss'])}\n- **Train loss final:** `{history['train_loss'][-1]:.4f}`\n- **Val loss final:** `{history['val_loss'][-1]:.4f}`\n"
    md += f"\n\n---\n*Dataset:* `{csv_path.name}` • *Checkpoint:* `checkpoints/best_model.pt` • *Gerado por:* `src/run_evaluation.py` S6\n"
    with open(RESULTS_DIR / "RELATORIO_RESULTADOS_V2.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✓ Relatórios: {RESULTS_DIR / 'RELATORIO_RESULTADOS_V2.md'}")
    print(md[:600])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(CHECKPOINT_DIR / "best_model.pt"))
    parser.add_argument("--csv", type=str, default=str(GROUND_TRUTH_V2 if GROUND_TRUTH_V2.exists() else GROUND_TRUTH_V1))
    parser.add_argument("--pretty-only", action="store_true", help="só re-gera markdown a partir do último txt (não precisa checkpoint)")
    args = parser.parse_args()

    device = get_device()
    ckpt_path = Path(args.ckpt)
    csv_path = Path(args.csv)

    print(f"=== THOR-PIML V2 — Avaliação ({csv_path.name}) ===")
    report, history, data_tuple = evaluate_from_checkpoint(ckpt_path, csv_path, device)
    print("\n" + str(report) + "\n")
    save_reports(report, history, csv_path)
    plot_all(report, history, data_tuple, GRAFICOS_DIR)
    print("\n✓ Avaliação V2 concluída! Ver results/RELATORIO_RESULTADOS_V2.md e results/graficos/")

if __name__ == "__main__":
    main()
