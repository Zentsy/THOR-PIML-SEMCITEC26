"""
THOR-PIML V7 — Entrypoint de treino/avaliação da arquitetura híbrida
====================================================================
Uso (T4 @ lightning.ai ou CPU local para smoke):
    python -m src.v7.run --model hybrid --loss v7 --data v2 --seed 42
    python -m src.v7.run --model hybrid --cv 5            # CV temporal bloqueada + final
    python -m src.v7.run --model lstm                      # ablação: só LSTM (=V6d + fixes)
    python -m src.v7.run --model tcn                       # ablação: só TCN
    python -m src.v7.run --model hybrid --loss v6d         # ablação: loss V6d (sem termos novos)
    python -m src.v7.run --data v3 ...                     # quando GT V3 existir (sinótica real)
    python -m src.v7.run --smoke                           # 2 épocas CPU, valida shapes

O teste cego (2019-09-02 → 2026-06-30) só é AVALIADO no final; CV roda só na
região dev (primeiros 85%).
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
import json

import numpy as np
import torch

from src.paths import CHECKPOINT_DIR, DATA_DIR, GROUND_TRUTH_V2, RESULTS_DIR
from src.evaluate import full_evaluation
from src.utils import get_device, model_summary, set_seed
from src.v7.config_v7 import THORConfigV7
from src.v7.model_v7 import THORHybridModel
from src.v7.physics_loss_v7 import THORLossV7
from src.v7.pipeline_v7 import prepare_v7_cv_loaders, prepare_v7_pipeline
from src.v7.train_v7 import train_v7, validate_v7

EXPERIMENTS_LOG = RESULTS_DIR / "experiments_v7.md"
EXPERIMENTS_JSONL = RESULTS_DIR / "experiments_v7.jsonl"

_MD_HEADER = (
    "# Log de Experimentos V7/V8 (era híbrida)\n\n"
    "| Data | Parte | Variante | Loss | Dados | Seed | CV KGE (m±s) | NSE | KGE | RMSE | MAE | Bias | F1 | SDII (obs/pred) | QB95% | QB99% | R10 (o/p) | R20 (o/p) | CWD (o/p) |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def log_experiment(entry: dict) -> None:
    """Registra uma run em DOIS formatos (mesma informação):

    - results/experiments_v7.jsonl — 1 JSON por linha, estruturado. Fonte da verdade,
      consumida pelo dashboard: python -m src.v7.results_viewer
    - results/experiments_v7.md   — tabela humana, pronta pro paper.

    A diferenciação V7/V8/ablações vive nos campos: parte, variant, loss, data, seed
    (mesma tabela única de propósito — é a ablação do paper — mas com tag por parte).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    m = entry["metrics"]
    cv = entry.get("cv")
    cv_str = f"{cv['kge_mean']:.3f}±{cv['kge_std']:.3f}" if cv else "—"
    if not EXPERIMENTS_LOG.exists():
        EXPERIMENTS_LOG.write_text(_MD_HEADER, encoding="utf-8")
    row = (
        f"| {entry['ts']} | {entry['parte']} | {entry['variant']} | {entry['loss']} "
        f"| {entry['data']} | {entry['seed']} | {cv_str} "
        f"| {m['nse']:.3f} | {m['kge']:.3f} | {m['rmse']:.2f} | {m['mae']:.2f} "
        f"| {m['bias']:+.2f} | {m['f1_occ']:.3f} | {m['sdii_obs']:.2f}/{m['sdii_pred']:.2f} "
        f"| {m['qb95_bias_pct']:+.1f}% | {m['qb99_bias_pct']:+.1f}% "
        f"| {m['r10mm_obs']}/{m['r10mm_pred']} | {m['r20mm_obs']}/{m['r20mm_pred']} "
        f"| {m['cwd_obs']}/{m['cwd_pred']} |"
    )
    with open(EXPERIMENTS_LOG, "a", encoding="utf-8") as f:
        f.write(row + "\n")
    with open(EXPERIMENTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def resolve_csv(data_version: str) -> Path:
    if data_version == "v3":
        p = DATA_DIR / "ground_truth_guarulhos_daily_v3.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"GT V3 não existe: {p}. Rode os scripts de data_prep (fetch_era5_pl.py, "
                "fetch_era5_single.py, extract_chirps_grid.py, build_gt_v3.py) na lightning.ai primeiro."
            )
        return p
    if GROUND_TRUTH_V2.exists():
        return GROUND_TRUTH_V2
    raise FileNotFoundError(f"GT V2 não encontrado: {GROUND_TRUTH_V2}")


def build_model(config: THORConfigV7, variant: str) -> THORHybridModel:
    config.model.use_lstm_branch = variant in ("hybrid", "lstm")
    config.model.use_tcn_branch = variant in ("hybrid", "tcn")
    return THORHybridModel(config.model)


def build_loss(config: THORConfigV7, loss_variant: str, scaler) -> THORLossV7:
    if loss_variant == "v6d":
        # Ablação: comportamento exatamente V6d — desliga os termos novos
        config.training.lambda_extreme_fn = 0.0
        config.training.lambda_var_match = 0.0
    elif loss_variant != "v7":
        raise ValueError(f"Loss desconhecida: {loss_variant} (use v7 | v6d)")
    return THORLossV7(scaler=scaler, training_config=config.training)


@torch.no_grad()
def predict_loader(model, loader, device) -> dict:
    model.eval()
    probs, preds, trues = [], [], []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        prob, _, final = model(x, return_components=True)
        probs.append(prob.squeeze(-1).float().cpu().numpy())
        preds.append(final.squeeze(-1).float().cpu().numpy())
        trues.append(batch[2].squeeze(-1).float().cpu().numpy())
    return {
        "prob": np.concatenate(probs),
        "pred": np.concatenate(preds),
        "true": np.concatenate(trues),
    }


def append_experiment_log(row: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not EXPERIMENTS_LOG.exists():
        EXPERIMENTS_LOG.write_text(
            "# Log de Experimentos V7 (híbrida LSTM+TCN)\n\n"
            "| Data | Variante | Loss | Data | Seed | CV KGE (m±s) | NSE | KGE | RMSE | MAE | Bias | F1 | SDII (obs/pred) | QB95% | QB99% | R10 (o/p) | R20 (o/p) | CWD (o/p) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    with open(EXPERIMENTS_LOG, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def _cv_cache_path(variant: str, loss_variant: str, data_version: str, seed, n_folds: int, cv_epochs: int) -> Path:
    return RESULTS_DIR / "cv_cache" / f"cv_{variant}_{loss_variant}_{data_version}_seed{seed}_f{n_folds}_e{cv_epochs}.json"


def run_cv(config: THORConfigV7, csv_path: Path, variant: str, loss_variant: str, data_version: str,
           n_folds: int, cv_epochs: int, device, seed: int, use_cache: bool = True) -> dict:
    print(f"\n=== CV temporal bloqueada ({n_folds} blocos, {cv_epochs} épocas/fold) — teste cego INTACTO ===")
    loaders, data = prepare_v7_cv_loaders(
        csv_path, config=config,
        occurrence_threshold=config.model.occurrence_threshold_mm,
        lags=None,
    )
    config.model.n_features = len(data["feature_cols"])

    # Cache por fold: cada fold CONCLUÍDO é gravado em results/cv_cache/ na hora.
    # Interrupção (studio cair, modo interruptible, etc) → reexecutar o MESMO comando
    # retoma dos folds salvos. Deletar o arquivo (ou --cv-no-cache) força CV do zero.
    cache_path = _cv_cache_path(variant, loss_variant, data_version, seed, n_folds, cv_epochs)
    cache: dict = {"folds": {}}
    if use_cache and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            done = sorted(cache.get("folds", {}), key=int)
            print(f"[CV] cache: {cache_path.name} → folds já prontos: {done or 'nenhum'}")
        except Exception as e:
            print(f"[CV] ⚠ cache corrompido ({e}) — recomeçando do zero")
            cache = {"folds": {}}

    summaries: dict = {}
    for fold_i, (tr_loader, va_loader, bounds) in enumerate(loaders, start=1):
        key = str(fold_i)
        cached = cache.get("folds", {}).get(key) if use_cache else None
        if cached is not None:
            print(f"--- Fold {fold_i}/{len(loaders)} — CACHEADO (KGE {cached['kge']:.3f}) — sem retreino ---")
            summaries[key] = cached
            continue
        print(f"\n--- Fold {fold_i}/{len(loaders)} (treino até idx {bounds[0]}, val {bounds[1]}:{bounds[2]}) ---")
        set_seed(seed)
        model = build_model(config, variant).to(device)
        loss_fn = build_loss(config, loss_variant, scaler=None)
        # Nota: loss V7 não usa scaler no forward (physics off) — None ok na CV
        fold_cfg = copy.deepcopy(config)
        fold_cfg.training.epochs = cv_epochs
        ckpt_fold = CHECKPOINT_DIR / f"v7_cv_fold{fold_i}.pt"
        train_v7(model, tr_loader, va_loader, loss_fn, fold_cfg, ckpt_fold,
                 device=device, variant=variant, seed=seed)
        ckpt = torch.load(ckpt_fold, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        out = predict_loader(model, va_loader, device)
        rep = full_evaluation(out["true"], out["pred"], (out["true"] >= 1.0).astype(float), out["prob"])
        summary = {
            "kge": float(rep.kge),
            "r20_recall": float(rep.r20mm_pred / max(rep.r20mm_obs, 1) * 100),
            "qb99_bias_pct": float(rep.qb99_bias_pct),
            "sdii_obs": float(rep.sdii_obs), "sdii_pred": float(rep.sdii_pred),
        }
        summaries[key] = summary
        cache.setdefault("folds", {})[key] = summary
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        print(f"Fold {fold_i}: KGE {summary['kge']:.3f} | R20 recall {summary['r20_recall']:.0f}% "
              f"| QB99 {summary['qb99_bias_pct']:+.1f}% | SDII {summary['sdii_obs']:.2f}/{summary['sdii_pred']:.2f}")
        ckpt_fold.unlink(missing_ok=True)

    if not summaries:
        raise RuntimeError("CV sem folds — nada para agregar (cache vazio e nenhum fold treinado?)")
    kges = [s["kge"] for s in summaries.values()]
    r20s = [s["r20_recall"] for s in summaries.values()]
    out_summary = {
        "kge_mean": float(np.mean(kges)),
        "kge_std": float(np.std(kges)),
        "r20_recall_mean": float(np.mean(r20s)),
        "folds": len(kges),
    }
    print(f"\nCV resumo ({out_summary['folds']} folds): KGE {out_summary['kge_mean']:.3f} ± {out_summary['kge_std']:.3f} "
          f"| R20 recall médio {out_summary['r20_recall_mean']:.0f}%")
    return out_summary


def main():
    parser = argparse.ArgumentParser(description="THOR-PIML V7 híbrida LSTM+TCN")
    parser.add_argument("--model", choices=["hybrid", "lstm", "tcn"], default="hybrid")
    parser.add_argument("--loss", choices=["v7", "v6d"], default="v7")
    parser.add_argument("--data", choices=["v2", "v3"], default="v2")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cv", type=int, default=0, help="Nº de blocos da CV temporal (0=off, 5=recomendado)")
    parser.add_argument("--cv-epochs", type=int, default=60)
    parser.add_argument("--cv-no-cache", action="store_true",
                        help="ignora/deleta cache de folds e refaz a CV do zero")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default config=4). Dataset é pequeno e mora na RAM: "
                             "0 costuma ser MAIS rápido na T4 (menos overhead de IPC/worker processes).")
    parser.add_argument("--smoke", action="store_true", help="2 épocas, CPU-friendly, ckpt smoke_v7.pt")
    args = parser.parse_args()

    config = THORConfigV7()
    if args.epochs:
        config.training.epochs = args.epochs
    if args.seq_len:
        config.model.seq_len = args.seq_len
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.smoke:
        config.training.epochs = 2
        config.training.num_workers = 0
        config.training.use_amp = False
        config.training.scheduler = "cosine"  # OneCycleLR com 2 épocas é instável p/ smoke

    csv_path = resolve_csv(args.data)
    seed = set_seed(args.seed)
    device = get_device()
    print(f"=== THOR-V7 [{args.model}] loss={args.loss} data={args.data} seed={seed} ===")

    train_loader, val_loader, test_loader, scaler, feature_cols = prepare_v7_pipeline(
        csv_path, config=config, occurrence_threshold=config.model.occurrence_threshold_mm
    )
    if config.model.n_features != len(feature_cols):
        print(
            f"[V7][FIX] n_features {config.model.n_features} → {len(feature_cols)} "
            f"(pipeline real — logado, nunca silencioso)"
        )
        config.model.n_features = len(feature_cols)

    model = build_model(config, args.model).to(device)
    print(model_summary(model))
    loss_fn = build_loss(config, args.loss, scaler=scaler)

    cv_summary = None
    if args.cv > 1:
        if args.cv_no_cache:
            cache_path = _cv_cache_path(args.model, args.loss, args.data, seed, args.cv, args.cv_epochs)
            cache_path.unlink(missing_ok=True)
            print(f"[CV] --cv-no-cache: cache {cache_path.name} removido")
        cv_summary = run_cv(
            config, csv_path, args.model, args.loss, args.data,
            args.cv, args.cv_epochs, device, seed,
            use_cache=not args.cv_no_cache,
        )

    ckpt_name = "smoke_v7.pt" if args.smoke else f"v7_{args.model}_{args.loss}_{args.data}_seed{seed}.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    history = train_v7(
        model, train_loader, val_loader, loss_fn, config, ckpt_path,
        device=device, scaler=scaler, variant=args.model,
        feature_cols=feature_cols, lags=None, seed=seed,
    )

    # Recarrega melhor época e avalia o teste cego UMA vez (relatório final)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    out = predict_loader(model, test_loader, device)
    y_class = (out["true"] >= config.model.occurrence_threshold_mm).astype(float)
    report = full_evaluation(out["true"], out["pred"], y_class, out["prob"])
    print("\n" + str(report))

    if not args.smoke:
        log_experiment({
            "ts": f"{datetime.now():%Y-%m-%d %H:%M}",
            "parte": "v7",
            "variant": args.model,
            "loss": args.loss,
            "data": args.data,
            "seed": seed,
            "cv": cv_summary,
            "ckpt": ckpt_name,
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
        print(f"\n✓ Log: {EXPERIMENTS_LOG} (+ {EXPERIMENTS_JSONL.name})\n✓ Checkpoint: {ckpt_path}")
        print(f"✓ Dashboard: python -m src.v7.results_viewer")
    del history


if __name__ == "__main__":
    main()
