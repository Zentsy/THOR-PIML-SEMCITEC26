"""
THOR-PIML — Execução do Treinamento Oficial (1981-2026)
======================================================
Executa 100 épocas de treinamento da rede THOR-PIML de alta capacidade (97.858 parâmetros)
sobre o dataset oficial de 45.5 anos (16.648 dias) e gera os gráficos e relatórios finais.
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import ROOT_DIR as CANONICAL_ROOT, CHECKPOINT_DIR, RESULTS_DIR, GROUND_TRUTH_V1, GROUND_TRUTH_V2
# Usa paths canônicos (S0)
ROOT_DIR = CANONICAL_ROOT

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.config import THORConfig
from src.preprocessing import (
    load_ground_truth_csv,
    RobustClimateScaler,
    create_sliding_windows,
    train_val_test_split,
    build_dataloaders,
    prepare_zero_leakage_pipeline,
)
from src.model import THORPIMLModel
from src.physics_loss import THORPhysicsLoss
from src.train import train
from src.evaluate import full_evaluation
from src.utils import set_seed, get_device, model_summary


def run_official_training(seed: int | None = None, use_v6: bool = False, v6_loss: bool = False, occ_thresh: float | None = None):
    print("=== THOR-PIML: Execução Oficial do Treinamento (1981-2026) ===")
    if use_v6:
        from src.config_v6 import THORConfigV6
        config = THORConfigV6()
        print(f"[V6] Usando THORConfigV6 (quick-win) — {config.model.n_features} feats, hidden {config.model.lstm_hidden}×{config.model.lstm_layers}, attn={config.model.use_attention}")
    else:
        config = THORConfig()

    # Seed: usa do config (None=aleatória) ou do argumento CLI
    use_seed = seed if seed is not None else config.training.seed
    actual_seed = set_seed(use_seed)
    print(f"Seed usada: {actual_seed} ({'aleatória' if use_seed is None else 'fixa'})")
    device = get_device()

    csv_path = GROUND_TRUTH_V2 if GROUND_TRUTH_V2.exists() else GROUND_TRUTH_V1
    print(f"Dataset: {csv_path.name} ({'V2 CHIRPS+ERA5' if csv_path == GROUND_TRUTH_V2 else 'V1 NASA POWER legado'})")

    # Threshold de ocorrência: CLI > config > padrão
    occ_thresh_cfg = occ_thresh
    if occ_thresh_cfg is None:
        occ_thresh_cfg = getattr(getattr(config, 'model', None), 'occurrence_threshold_mm', 0.1)

    train_loader, val_loader, test_loader, feature_scaler, feature_cols = prepare_zero_leakage_pipeline(
        csv_path, config=config, scaler_method="minmax",
        occurrence_threshold=occ_thresh_cfg,
        use_v6_features=use_v6,
    )
    print(f"Occurrence threshold: {occ_thresh_cfg} mm (V5=0.1, V6=1.0) | Features: {len(feature_cols)}")
    # FIX: modelo V6 ALL-IN tem n_features=84 mas pipeline pode filtrar para 70 se alguma feature não existir no CSV
    # Ajusta config para bater com dados reais, evitando RuntimeError input.size(-1) must be equal to input_size
    if len(feature_cols) != config.model.n_features:
        print(f"[FIX] Ajustando model.n_features {config.model.n_features} → {len(feature_cols)} para bater com pipeline (alguma feature extra não existe no CSV)")
        config.model.n_features = len(feature_cols)

    X_test_tensor = test_loader.dataset.tensors[0].to(device)
    y_class_test = test_loader.dataset.tensors[1].cpu().numpy().ravel()
    y_true_test = test_loader.dataset.tensors[2].cpu().numpy().ravel()

    # Instanciar Modelo
    if use_v6:
        from src.model_v6 import THORPIMLModelV6
        model = THORPIMLModelV6(config.model).to(device)
    else:
        model = THORPIMLModel(config.model).to(device)
    print("\n" + model_summary(model))

    # Loss
    if v6_loss or use_v6:
        from src.physics_loss_v6 import THORLossV6
        loss_fn = THORLossV6(scaler=feature_scaler, training_config=config.training, occurrence_threshold=occ_thresh_cfg)
        print(f"[V6 Loss] THORLossV6 — occ thresh {occ_thresh_cfg}mm, masked intensity, dry BCE")
    else:
        loss_fn = THORPhysicsLoss(scaler=feature_scaler, training_config=config.training)

    # Treinar (S5: passa scaler + thor_config para checkpoint versionado, AMP automático)
    checkpoint_dir = CHECKPOINT_DIR
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config.training,
        checkpoint_dir=checkpoint_dir,
        device=device,
        scaler=feature_scaler,
        thor_config=config,
    )

    # Avaliação no Teste (Carregar Melhor Modelo)
    best_ckpt = checkpoint_dir / "best_model.pt"
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Modelo recarregado a partir do melhor checkpoint (best_model.pt).")

    model.eval()
    with torch.no_grad():
        prob_occ, intensity, y_pred_tensor = model(X_test_tensor, return_components=True)
        tmean_celsius = loss_fn.denormalize_feature(X_test_tensor, loss_fn.temp_idx)
        rh_pct = loss_fn.denormalize_feature(X_test_tensor, loss_fn.rh_idx)
        w_max_test = loss_fn.compute_thermodynamic_limit_wmax(tmean_celsius, rh_pct)
        n_violations = int((y_pred_tensor > w_max_test).sum().item())

    y_pred_test = y_pred_tensor.cpu().numpy().ravel()
    prob_occ_test = prob_occ.cpu().numpy().ravel()

    report = full_evaluation(
        y_true_test, y_pred_test, y_class_test, prob_occ_test, n_violations=n_violations
    )
    print("\n" + str(report))

    # Gerar Gráfico de Downscaling de 30 dias para o relatório
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    n_days = 30
    days_x = np.arange(1, n_days + 1)
    y_true_sub = y_true_test[:n_days]
    y_pred_sub = y_pred_test[:n_days]

    plt.figure(figsize=(14, 6), dpi=300)
    plt.plot(days_x, y_true_sub, label="Observado (Ground Truth)", color="#111111", lw=2.5, marker="o", ms=5)
    plt.plot(days_x, y_pred_sub, label="Predito THOR-PIML (Dual-Head Hurdle)", color="#ff6b6b", lw=2.2, marker="s", ms=5, ls="--")

    dry_days = days_x[y_true_sub == 0.0]
    plt.scatter(dry_days, np.zeros_like(dry_days), color="#2ee59d", s=80, zorder=5, label="Dias Secos (Zero Hurdle Exact)")

    plt.title("Downscaling e Correção de Precipitação — Janela de 30 Dias (Guarulhos-SP)", fontweight="bold", fontsize=14)
    plt.xlabel("Dia")
    plt.ylabel("Precipitação (mm/dia)")
    plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    metrics_text = (
        f"NSE = {report.nse:.4f}\n"
        f"KGE = {report.kge:.4f}\n"
        f"RMSE = {report.rmse:.2f} mm\n"
        f"MAE = {report.mae:.2f} mm\n"
        f"Violação Física = {report.physics_violation_rate*100:.2f}%"
    )
    plt.gca().text(
        0.02, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#ccc")
    )

    plot_path = results_dir / "thor_downscaling_30days_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nGráfico salvo em: {plot_path}")

    # Salvar Relatório de Métricas em TXT
    metrics_txt = results_dir / "thor_metrics_summary.txt"
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(str(report))
    print(f"Métricas salvas em: {metrics_txt}")


if __name__ == "__main__":
    import argparse

    def parse_seed(value):
        if value is None:
            return None
        if isinstance(value, str) and value.lower() in ("none", "null", "nil", ""):
            return None
        try:
            return int(value)
        except:
            raise argparse.ArgumentTypeError(f"seed deve ser int ou None, recebeu {value}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=parse_seed, default=None, help="Seed (None=aleatória, int=fixa). Use --seed None ou omita para aleatória. Default usa config.training.seed (None)")
    parser.add_argument("--v6", action="store_true", help="Usa config V6a (50 feats, 96×2, sem year_norm, sem atenção)")
    parser.add_argument("--v6-loss", action="store_true", help="Usa loss V6 reformulada (masked intensity + dry BCE)")
    parser.add_argument("--occ-thresh", type=float, default=None, help="Threshold ocorrência mm (0.1 V5, 1.0 V6). Default vem da config")
    args = parser.parse_args()
    run_official_training(seed=args.seed, use_v6=args.v6, v6_loss=args.v6_loss or args.v6, occ_thresh=args.occ_thresh)
