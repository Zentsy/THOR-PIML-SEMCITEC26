"""
THOR-PIML V8 — Entrypoint da híbrida espacial (CNN 2D + LSTM/TCN)
==================================================================
Requer GT V3 + data/era5pl_domain_daily_1981_2026.nc (lightning.ai).

    python -m src.v7.run_v8 --model hybrid --seed 42
    python -m src.v7.run_v8 --smoke-synthetic     # valida shapes com tensor sintético
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.paths import CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR
from src.evaluate import full_evaluation
from src.utils import get_device, model_summary, set_seed
from src.v7.config_v7 import THORConfigV8
from src.v7.model_v8 import THORSpatialHybridModel
from src.v7.physics_loss_v8 import THORLossV8
from src.v7.pipeline_v8 import prepare_v8_pipeline
from src.v7.train_v7 import train_v7

GT_V3 = DATA_DIR / "ground_truth_guarulhos_daily_v3.csv"
SPATIAL_NC = DATA_DIR / "era5pl_domain_daily_1981_2026.nc"


def train_v8(model, train_loader, val_loader, loss_fn, config, ckpt_path, device,
             scaler, variant, feature_cols, seed):
    """Wrapper: train_v7 adaptado a batches (x_surf, x_spatial, yc, yr)."""
    from src.v7.train_v7 import EarlyStoppingV7, _autocast_ctx, _grad_scaler, validate_v7
    from src.utils import logger, save_checkpoint
    from dataclasses import asdict

    model = model.to(device)
    tcfg = config.training
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=tcfg.max_lr,
        total_steps=tcfg.epochs * max(len(train_loader), 1),
        pct_start=0.3, div_factor=max(tcfg.max_lr / tcfg.lr, 1.0),
        final_div_factor=max(tcfg.max_lr / tcfg.min_lr, 1.0),
    )
    early = EarlyStoppingV7(patience=tcfg.patience, mode="max")
    amp = _grad_scaler(device, tcfg.use_amp)
    best = -float("inf")
    from tqdm.auto import tqdm

    for epoch in tqdm(range(1, tcfg.epochs + 1), desc=f"THOR-V8 [{variant}]", unit="epoch"):
        model.train()
        for batch in train_loader:
            xs, xsp, yc, yr = (t.to(device, non_blocking=True) for t in batch)
            optimizer.zero_grad(set_to_none=True)
            stepped = True
            if amp is not None:
                with _autocast_ctx(device):
                    prob, inten, final = model(xs, xsp, return_components=True)
                    loss, _ = loss_fn(prob, inten, final, yc, yr, xs)
                amp.scale(loss).backward()
                if tcfg.grad_clip > 0:
                    amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                scale_before = amp.get_scale()
                amp.step(optimizer)
                amp.update()
                stepped = amp.get_scale() >= scale_before  # False se step foi pulado (inf/nan)
            else:
                prob, inten, final = model(xs, xsp, return_components=True)
                loss, _ = loss_fn(prob, inten, final, yc, yr, xs)
                loss.backward()
                if tcfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                optimizer.step()
            if stepped:
                scheduler.step()
        # val: reusa validate_v7 com loader de pares → adaptar batches
        model.eval()
        losses, preds, trues = [], [], []
        with torch.no_grad():
            for xs, xsp, yc, yr in val_loader:
                xs, xsp, yc, yr = xs.to(device), xsp.to(device), yc.to(device), yr.to(device)
                prob, inten, final = model(xs, xsp, return_components=True)
                loss, _ = loss_fn(prob, inten, final, yc, yr, xs)
                losses.append(loss.item())
                preds.append(final.squeeze(-1).float().cpu().numpy())
                trues.append(yr.squeeze(-1).float().cpu().numpy())
        val_loss = float(np.mean(losses))
        from src.v7.train_v7 import _kge
        kge = _kge(np.concatenate(trues), np.concatenate(preds))
        if not np.isnan(kge) and kge > best:
            best = kge
            save_checkpoint(
                model, optimizer, epoch, val_loss, ckpt_path,
                extra={
                    "thor_version": "v8", "model_variant": variant,
                    "feature_cols": feature_cols, "history": {"val_kge": [kge]},
                    "seed": seed, "checkpoint_metric": f"kge={kge:.4f}",
                },
                scaler_state=scaler.to_dict() if scaler is not None else None,
                config_dict=asdict(config),
            )
        if early.step(kge):
            logger.info(f"V8 early stop na época {epoch} (melhor val KGE: {best:.4f})")
            break
    logger.info(f"Treino V8 [{variant}] concluído — melhor val KGE {best:.4f}")
    return best


@torch.no_grad()
def predict(model, loader, device, gamma: float = 1.0):
    model.eval()
    probs, preds, trues = [], [], []
    for xs, xsp, yc, yr in loader:
        xs, xsp = xs.to(device), xsp.to(device)
        prob, inten, final = model(xs, xsp, return_components=True)
        p = prob.squeeze(-1).float().cpu().numpy()
        i = inten.squeeze(-1).float().cpu().numpy()
        # Calibração suave contínua: f = (p^gamma) * i (sem degrau duro, preserva SDII e CWD)
        f = (p ** gamma) * i if gamma != 1.0 else (p * i)
        probs.append(p)
        preds.append(f)
        trues.append(yr.squeeze(-1).float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(preds), np.concatenate(trues)


def smoke_synthetic(device):
    """Valida shapes/forward/backward do V8 com tensor sintético (SÓ teste, não dado)."""
    cfg = THORConfigV8()
    cfg.model.n_features = 93  # 84 + 9 sinóticas V3
    model = THORSpatialHybridModel(cfg.model).to(device)
    xs = torch.randn(2, 30, 93, device=device)
    xsp = torch.randn(2, 30, 25, 33, 5, device=device)
    out = model(xs, xsp)
    print(f"✓ V8 forward OK: {out.shape} | params: {sum(p.numel() for p in model.parameters()):,}")
    print(model_summary(model).splitlines()[-2])


def resolve_spatial_nc(custom_path: str | None = None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
    for cand in [
        DATA_DIR / "era5pl_domain_daily_1981_2026.nc",
        DATA_DIR / "noaa_pl_domain_daily_1981_2026.nc",
    ]:
        if cand.exists():
            print(f"[V8 spatial] Usando campo 2D: {cand.name}")
            return cand
    raise SystemExit(
        f"❌ Nenhum arquivo NetCDF 2D encontrado em {DATA_DIR}.\n"
        "   Para gerar em 2 minutos via OPeNDAP sem fila, rode:\n"
        "   python data_prep/fetch_noaa_pl.py"
    )


def main():
    parser = argparse.ArgumentParser(description="THOR-PIML V8 híbrida espacial")
    parser.add_argument("--model", choices=["hybrid", "lstm", "tcn"], default="hybrid")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--csv", type=str, default=str(GT_V3))
    parser.add_argument("--spatial-nc", type=str, default=None)
    parser.add_argument("--smoke-synthetic", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default config=4); 0 costuma ser mais rápido (dataset in-RAM)")
    parser.add_argument("--physics", type=float, default=0.05,
                        help="lambda_physics (barreira CC com TCWV real; 0=off)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="fator de calibração suave de probabilidade (default 1.0)")
    parser.add_argument("--eval-only", action="store_true",
                        help="só avalia o checkpoint salvo sem re-treinar")
    args = parser.parse_args()

    device = get_device()
    if args.smoke_synthetic:
        smoke_synthetic(device)
        return

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"❌ {csv_path} não existe.")
    nc_path = resolve_spatial_nc(args.spatial_nc)

    config = THORConfigV8()
    if args.epochs:
        config.training.epochs = args.epochs
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    config.model.use_lstm_branch = args.model in ("hybrid", "lstm")
    config.model.use_tcn_branch = args.model in ("hybrid", "tcn")
    config.training.lambda_physics = args.physics
    seed = set_seed(args.seed)
    print(f"=== THOR-V8 [{args.model}] seed={seed} | física CC λ={args.physics} | gamma={args.gamma} ===")

    (train_loader, val_loader, test_loader), scaler, feature_cols = prepare_v8_pipeline(
        csv_path, nc_path, config=config,
        occurrence_threshold=config.model.occurrence_threshold_mm,
    )
    config.model.n_features = len(feature_cols)
    model = THORSpatialHybridModel(config.model).to(device)
    print(model_summary(model))
    loss_fn = THORLossV8(scaler=scaler, training_config=config.training)

    ckpt_path = CHECKPOINT_DIR / f"v8_{args.model}_seed{seed}.pt"
    if not args.eval_only:
        train_v8(model, train_loader, val_loader, loss_fn, config, ckpt_path,
                 device, scaler, args.model, feature_cols, seed)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    probs, preds, trues = predict(model, test_loader, device, gamma=args.gamma)
    report = full_evaluation(trues, preds, (trues >= 1.0).astype(float), probs)
    print("\n" + str(report))

    from src.v7.run import log_experiment

    log_experiment({
        "ts": f"{datetime.now():%Y-%m-%d %H:%M}",
        "parte": "v8",
        "variant": args.model,
        "loss": "v8",
        "physics_lambda": args.physics,
        "data": "v3",
        "seed": seed,
        "cv": None,
        "ckpt": ckpt_path.name,
        "metrics": {
            "nse": report.nse, "kge": report.kge, "rmse": report.rmse,
            "mae": report.mae, "bias": report.bias, "f1_occ": report.f1_occ,
            "accuracy_occ": report.accuracy_occ, "brier_score": report.brier_score,
            "roc_auc": report.roc_auc,
            "sdii_obs": report.sdii_obs, "sdii_pred": report.sdii_pred,
            "qb95_bias_pct": report.qb95_bias_pct, "qb99_bias_pct": report.qb99_bias_pct,
            "r10mm_obs": report.r10mm_obs, "r10mm_pred": report.r10mm_pred,
            "r20mm_obs": report.r20mm_obs, "r20mm_pred": report.r20mm_pred,
            "cwd_obs": report.cwd_obs, "cwd_pred": report.cwd_pred,
            "cdd_obs": report.cdd_obs, "cdd_pred": report.cdd_pred,
        },
    })
    print(f"\n✓ Checkpoint: {ckpt_path}")
    print("✓ Log: results/experiments_v7.md (+ experiments_v7.jsonl)")
    print("✓ Dashboard: python -m src.v7.results_viewer")


if __name__ == "__main__":
    main()
