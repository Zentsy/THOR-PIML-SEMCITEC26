"""
THOR-PIML — Caminhos Canônicos do Projeto (Sprint S0)
========================================================
Centraliza todos os Paths para portabilidade Linux/Windows/Lightning.AI.
Nenhum arquivo deve hardcodar r"c:\\Users\\..." ou ".venv/Lib/site-packages".

Uso:
    from src.paths import ROOT_DIR, DATA_DIR, DATA_PREP_DIR, CHECKPOINT_DIR, RESULTS_DIR, GRAFICOS_DIR
"""
from __future__ import annotations
from pathlib import Path

# Raiz do projeto = pasta que contém src/, data/, etc.
# src/paths.py -> resolve().parent = src/, .parent.parent = THOR-PIML/
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

SRC_DIR: Path = ROOT_DIR / "src"
DATA_DIR: Path = ROOT_DIR / "data"
DATA_PREP_DIR: Path = ROOT_DIR / "data_prep"
CHECKPOINT_DIR: Path = ROOT_DIR / "checkpoints"
RESULTS_DIR: Path = ROOT_DIR / "results"
GRAFICOS_DIR: Path = RESULTS_DIR / "graficos"
DOCS_DIR: Path = ROOT_DIR / "docs"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"

# Arquivos canônicos (v1 legado e v2 futuro)
GROUND_TRUTH_V1: Path = DATA_DIR / "ground_truth_guarulhos_daily.csv"
GROUND_TRUTH_V2: Path = DATA_DIR / "ground_truth_guarulhos_daily_v2.csv"
SCALER_JSON: Path = CHECKPOINT_DIR / "scaler_v2.json"
BEST_MODEL_PT: Path = CHECKPOINT_DIR / "best_model.pt"

__all__ = [
    "ROOT_DIR",
    "SRC_DIR",
    "DATA_DIR",
    "DATA_PREP_DIR",
    "CHECKPOINT_DIR",
    "RESULTS_DIR",
    "GRAFICOS_DIR",
    "DOCS_DIR",
    "NOTEBOOKS_DIR",
    "GROUND_TRUTH_V1",
    "GROUND_TRUTH_V2",
    "SCALER_JSON",
    "BEST_MODEL_PT",
]
