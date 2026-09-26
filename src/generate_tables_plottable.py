"""
THOR-PIML: Geração de Tabelas com PLOTTABLE (Opção 1)
=====================================================
Gera Tabela 3 e Tabela 4 utilizando a biblioteca Plottable com
estilo editorial e a paleta oficial do projeto.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from plottable import Table, ColumnDefinition

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "tables" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "thor": "#1E40AF",
    "ssp119": "#008FA3",
    "ssp126": "#6FAE6F",
    "ssp245": "#F2C230",
    "ssp370": "#F27421",
    "ssp585": "#A62D2D",
    "dark_text": "#0F172A",
    "header_text": "#0B2538",
    "gray_text": "#334155",
    "bg_stripe": "#F1F5F9",
    "border_col": "#CBD5E1",
}


def make_table_3_plottable():
    data = [
        {
            "Módulo / Componente": "Decodificador Hurdle",
            "Equação": r"$\hat{y} = p_{\mathrm{occ}} \times \mu_{\mathrm{int}}$",
            "Parâmetros e Variáveis": "$p_{\mathrm{occ}} = \sigma(z_{\mathrm{occ}}) \in (0,1)$\n$\mu_{\mathrm{int}} = \mathrm{Softplus}(z_{\mathrm{int}}) > 0$",
            "Papel Físico / Neural": "Desacopla a ocorrência (WMO $\geq$ 1 mm) do\nvolume e intensidade da precipitação.",
        },
        {
            "Módulo / Componente": "Pressão de Saturação",
            "Equação": r"$e_s(T) = 6{,}11 \cdot \exp\left( \frac{17{,}27 T}{237{,}3 + T} \right)$",
            "Parâmetros e Variáveis": "$T$: temperatura a 2m (°C)\n$e_s$: pressão de saturação (hPa)",
            "Papel Físico / Neural": "Capacidade máxima teórica de vapor\nd'água retido antes do ponto de orvalho.",
        },
        {
            "Módulo / Componente": "Teto Convectivo",
            "Equação": r"$W_{\max} = 4{,}0 \times \mathrm{TCWV}$",
            "Parâmetros e Variáveis": "$\mathrm{TCWV}$: Coluna Total de Vapor\nd'água integrada na coluna (mm)",
            "Papel Físico / Neural": "Barreira física superior de Clausius-Clapeyron\npara chuva máxima diária na bacia.",
        },
        {
            "Módulo / Componente": "Barreira Física PIML",
            "Equação": r"$\mathcal{L}_{\mathrm{phys}} = \lambda \cdot \left[ \mathrm{Softplus}\left( \frac{\hat{y} - W_{\max}}{\delta} \right) \right]^2$",
            "Parâmetros e Variáveis": "$\lambda_{\mathrm{phys}} = 1{,}0$ ; $\delta = 1{,}0$\nPenalização quadrática diferenciável",
            "Papel Físico / Neural": "Anula alucinações de super-chuva sem zerar\nos gradientes de aprendizagem neural.",
        },
        {
            "Módulo / Componente": "Perda Multiobjetivo",
            "Equação": r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{occ}} + \mathcal{L}_{\mathrm{int}} + \lambda_{\mathrm{var}}\mathcal{L}_{\mathrm{var}} + \mathcal{L}_{\mathrm{phys}}$",
            "Parâmetros e Variáveis": "BCE ponderada + Log-Cosh assimétrica\n$\lambda_{\mathrm{var}} = 0{,}10$ (peso de cauda de picos)",
            "Papel Físico / Neural": "Otimização harmônica de ocorrência, volume,\ncauda extrema e consistência termodinâmica.",
        },
    ]

    df = pd.DataFrame(data)

    col_defs = [
        ColumnDefinition("Módulo / Componente", width=2.2, textprops={"ha": "left", "weight": "bold", "color": PALETTE["header_text"], "size": 10.5}),
        ColumnDefinition("Equação", width=3.2, textprops={"ha": "left", "color": PALETTE["dark_text"], "size": 10.5}),
        ColumnDefinition("Parâmetros e Variáveis", width=3.2, textprops={"ha": "left", "color": PALETTE["gray_text"], "size": 9.5}),
        ColumnDefinition("Papel Físico / Neural", width=3.8, textprops={"ha": "left", "color": PALETTE["gray_text"], "size": 9.5}),
    ]

    fig, ax = plt.subplots(figsize=(13.0, 5.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Título do Card
    plt.suptitle(
        "Tabela 3 — Formulações matemáticas e restrições físicas do THOR-PIML",
        fontsize=13, fontweight="bold", color=PALETTE["thor"], y=0.98, ha="center"
    )

    tab = Table(
        df,
        ax=ax,
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        odd_row_color=PALETTE["bg_stripe"],
        even_row_color="#FFFFFF",
        col_label_divider_kw={"color": PALETTE["thor"], "linewidth": 2.0},
        row_divider_kw={"color": PALETTE["border_col"], "linewidth": 0.8},
        textprops={"size": 10},
    )

    out_file = OUT_DIR / "tabela_3_plottable.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300, facecolor="#FFFFFF")
    plt.close()
    print(f"✓ Plottable Tabela 3 salva em: {out_file}")


def make_table_4_plottable():
    data = [
        {
            "Código": "BIO12",
            "Indicador / Definição": "Precipitação Anual Total (PRCPTOT)",
            "Expressão Matemática": r"$P_{\mathrm{total}} = \sum_{m=1}^{12} P_m$",
            "Unidade": "mm/ano",
            "Relevância Hidrológica e Urbana (Guarulhos)": "Balanço hídrico anual e recarga dos aquíferos da bacia do Alto Tietê.",
        },
        {
            "Código": "BIO13 / BIO14",
            "Indicador / Definição": "Mês Mais Chuvoso / Mais Seco",
            "Expressão Matemática": r"$\max(P_m)$  e  $\min(P_m)$",
            "Unidade": "mm/mês",
            "Relevância Hidrológica e Urbana (Guarulhos)": "Contraste entre pico de cheias do verão e estiagem severa de inverno.",
        },
        {
            "Código": "BIO15",
            "Indicador / Definição": "Sazonalidade da Precipitação (CV)",
            "Expressão Matemática": r"$CV = (\sigma_m / \mu_m) \times 100$",
            "Unidade": "%",
            "Relevância Hidrológica e Urbana (Guarulhos)": "Grau de irregularidade e concentração temporal das chuvas ao longo do ano.",
        },
        {
            "Código": "BIO16 / BIO17",
            "Indicador / Definição": "Trimestre Mais Úmido (DJF) / Seco (JJA)",
            "Expressão Matemática": r"$\max \sum P_k$  e  $\min \sum P_k$",
            "Unidade": "mm/trim",
            "Relevância Hidrológica e Urbana (Guarulhos)": "Auditoria de viés sazonal do CORDEX e restauração de amplitude no THOR.",
        },
        {
            "Código": "RX1day / RX5day",
            "Indicador / Definição": "Máxima Precipitação em 1 e 5 Dias",
            "Expressão Matemática": r"$\max(P_{\mathrm{dia}})$  e  $\max(P_{\mathrm{5d}})$",
            "Unidade": "mm",
            "Relevância Hidrológica e Urbana (Guarulhos)": "Gatilhos de transbordamento do Rio Baquirivu-Guaçu e Rodovia Dutra.",
        },
        {
            "Código": "R10mm / R20mm",
            "Indicador / Definição": "Dias com Chuva Moderada e Forte",
            "Expressão Matemática": r"$\sum [P \geq 10]$  e  $\sum [P \geq 20]$",
            "Unidade": "dias/ano",
            "Relevância Hidrológica e Urbana (Guarulhos)": "Frequência de sobrecarga da rede municipal de microdrenagem pluvial.",
        },
        {
            "Código": "CDD / CWD",
            "Indicador / Definição": "Dias Consecutivos Secos / Chuvosos",
            "Expressão Matemática": r"$\max(\mathrm{run}_{P < 1})$  e  $\max(\mathrm{run}_{P \geq 1})$",
            "Unidade": "dias",
            "Relevância Hidrológica e Urbana (Guarulhos)": "Diagnóstico de estiagens e auditoria da 'garoa crônica' do CORDEX.",
        },
    ]

    df = pd.DataFrame(data)

    col_defs = [
        ColumnDefinition("Código", width=1.6, textprops={"ha": "center", "weight": "bold", "color": PALETTE["ssp370"], "size": 10}),
        ColumnDefinition("Indicador / Definição", width=3.0, textprops={"ha": "left", "weight": "bold", "color": PALETTE["header_text"], "size": 9.8}),
        ColumnDefinition("Expressão Matemática", width=2.4, textprops={"ha": "left", "color": PALETTE["dark_text"], "size": 10}),
        ColumnDefinition("Unidade", width=1.2, textprops={"ha": "center", "color": PALETTE["gray_text"], "size": 9.5}),
        ColumnDefinition("Relevância Hidrológica e Urbana (Guarulhos)", width=4.6, textprops={"ha": "left", "color": PALETTE["gray_text"], "size": 9.5}),
    ]

    fig, ax = plt.subplots(figsize=(13.8, 6.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    plt.suptitle(
        "Tabela 4 — Indicadores bioclimáticos (WMO) e extremos hidrológicos (ETCCDI) avaliados",
        fontsize=13, fontweight="bold", color=PALETTE["ssp245"], y=0.98, ha="center"
    )

    tab = Table(
        df,
        ax=ax,
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        odd_row_color=PALETTE["bg_stripe"],
        even_row_color="#FFFFFF",
        col_label_divider_kw={"color": PALETTE["ssp245"], "linewidth": 2.0},
        row_divider_kw={"color": PALETTE["border_col"], "linewidth": 0.8},
        textprops={"size": 9.8},
    )

    out_file = OUT_DIR / "tabela_4_plottable.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300, facecolor="#FFFFFF")
    plt.close()
    print(f"✓ Plottable Tabela 4 salva em: {out_file}")


if __name__ == "__main__":
    make_table_3_plottable()
    make_table_4_plottable()
