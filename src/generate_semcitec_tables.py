"""Gera tabelas editoriais do SEMCITEC em PNG/PDF, sem depender de CSV na leitura."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLE_DATA = ROOT / ".agents" / "table_data"
TABLES = ROOT / "results" / "tables"


def export_table(frame: pd.DataFrame, filename: str, title: str, dpi: int = 300) -> None:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: "—" if pd.isna(value) else f"{value:.1f}")
    fig_height = max(3.5, 0.42 * (len(display) + 2))
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")
    table = ax.table(cellText=display.astype(str).values.tolist(), colLabels=list(display.columns),
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#0f172a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    ax.set_title(title, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(TABLES / f"{filename}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(TABLES / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    bio = pd.read_csv(TABLE_DATA / "bioclimatic_indices_epochs_summary.csv")
    etccdi = pd.read_csv(TABLE_DATA / "etccdi_extremes_epochs_summary.csv")
    bio_cols = ["series_id", "scenario", "epoch", "BIO12_PRCPTOT_mean", "BIO13_Wettest_Month_mean",
                "BIO14_Driest_Month_mean", "BIO15_Precip_Seasonality_CV_mean", "BIO16_Wettest_Quarter_mean",
                "BIO17_Driest_Quarter_mean", "BIO19_Coldest_Quarter_mean"]
    bio_cols = [column for column in bio_cols if column in bio.columns]
    export_table(bio[bio_cols], "bioclimatic_indices_epochs_summary", "Tabela de variáveis bioclimáticas de precipitação")
    etccdi_cols = ["series_id", "scenario", "epoch", "R10mm_mean", "R20mm_mean", "RX1day_mean", "RX5day_mean",
                   "CDD_mean", "CWD_mean", "R95p_mm_mean", "R99p_mm_mean"]
    etccdi_cols = [column for column in etccdi_cols if column in etccdi.columns]
    export_table(etccdi[etccdi_cols], "etccdi_extremes_epochs_summary", "Tabela de índices ETCCDI e persistência de chuva")
    print("Tabelas editoriais geradas em", TABLES)


if __name__ == "__main__":
    main()
