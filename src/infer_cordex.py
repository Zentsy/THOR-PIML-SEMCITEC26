"""
THOR-PIML — Inferência CORDEX 2026-2100 (Sprint S7)
===================================================
Downscaling estatístico: CORDEX 50km/25km → Guarulhos pontual (CHIRPS 5km)

Uso:
    python src/infer_cordex.py --nc "data/cordex/SAM-22/*rcp85*.nc" --ckpt checkpoints/best_model.pt --out results/cordex_downscaled_2026_2100.csv
    # mock sem NetCDF (teste):
    python src/infer_cordex.py --mock --ckpt checkpoints/best_model.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import pandas as pd
from tqdm.auto import tqdm

from src.paths import CHECKPOINT_DIR, RESULTS_DIR
from src.model import THORPIMLModel
from src.config import THORConfig
from src.cordex_dataset import CORDEXDataset
from src.utils import get_device


def infer(nc_pattern: str, ckpt_path: Path, out_path: Path, batch_size: int = 128, mock: bool = False):
    device = get_device()
    config = THORConfig()

    # Dataset CORDEX (SAM-22 prioritário, SAM-44 fallback)
    if mock:
        print("Modo mock (sem NetCDF) — gerando série sintética 2026-2100")
        ds = CORDEXDataset(nc_pattern="mock", seq_len=config.model.seq_len, scaler_path=CHECKPOINT_DIR / "scaler_v2.json")
    else:
        ds = CORDEXDataset(nc_pattern=nc_pattern, seq_len=config.model.seq_len, scaler_path=CHECKPOINT_DIR / "scaler_v2.json")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Modelo
    model = THORPIMLModel(config.model).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Modelo carregado: {ckpt_path} (epoch {ckpt.get('epoch','?')})")
    else:
        print(f"⚠ Checkpoint não encontrado: {ckpt_path} — usando pesos aleatórios (teste)")

    model.eval()
    dates, preds, probs = [], [], []
    with torch.no_grad():
        for x_batch, d_batch in tqdm(loader, desc="Downscaling CORDEX"):
            x_batch = x_batch.to(device)
            prob, intensity, final = model(x_batch, return_components=True)
            # final já é prob*intensity (S4 contínuo)
            dates.extend([pd.to_datetime(d) for d in d_batch])
            preds.extend(final.cpu().numpy().ravel().tolist())
            probs.extend(prob.cpu().numpy().ravel().tolist())

    out = pd.DataFrame({"date": dates, "pr_downscaled_mm": preds, "prob_occ": probs})
    # Pós-processamento: clip 1mm para dias secos (recomendação S6)
    out["pr_downscaled_clipped_mm"] = out["pr_downscaled_mm"].where(out["pr_downscaled_mm"] >= 1.0, 0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"✓ Downscaling salvo: {out_path} ({len(out)} dias)")
    print(out.head().to_string())
    # Estatística rápida
    print(f"\nEstatística 2026-2100:")
    print(f"  Média: {out['pr_downscaled_mm'].mean():.2f} mm/dia")
    print(f"  R10: {(out['pr_downscaled_mm'] >= 10).sum()} dias")
    print(f"  R20: {(out['pr_downscaled_mm'] >= 20).sum()} dias")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=str, default="data/cordex/SAM-22/*.nc", help="padrão NetCDF CORDEX (SAM-22 prioritário, SAM-44 fallback)")
    parser.add_argument("--ckpt", type=str, default=str(CHECKPOINT_DIR / "best_model.pt"))
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "cordex_downscaled_2026_2100.csv"))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--mock", action="store_true", help="sem NetCDF, gera mock sintético")
    args = parser.parse_args()
    infer(args.nc, Path(args.ckpt), Path(args.out), batch_size=args.batch, mock=args.mock)
