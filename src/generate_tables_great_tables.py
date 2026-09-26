"""
THOR-PIML: Geração de Tabelas com GREAT_TABLES (Opção 2)
========================================================
Gera Tabela 3 e Tabela 4 utilizando a biblioteca great_tables
com formatação editorial de publicação científica.
"""

import sys
from pathlib import Path
import pandas as pd
from great_tables import GT, md, html, style, loc

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
    "gray_text": "#475569",
    "bg_stripe": "#F8FAFC",
    "border_col": "#E2E8F0",
}


def make_table_3_gt():
    data = [
        {
            "modulo": "Decodificador Hurdle (Dual-Head)",
            "equacao": "ŷ = p_occ × μ_int",
            "params": "p_occ = σ(z_occ) ∈ (0,1)<br>μ_int = Softplus(z_int) > 0",
            "papel": "Desacopla a probabilidade estocástica de ocorrência (WMO ≥ 1 mm) da magnitude do volume precipitado.",
        },
        {
            "modulo": "Pressão de Saturação (Magnus-Tetens)",
            "equacao": "e_s(T) = 6,1078 · exp( 17,269·T / (237,3 + T) )",
            "params": "T: temperatura do ar a 2m (°C)<br>e_s: pressão de saturação (hPa)",
            "papel": "Estima a capacidade termodinâmica máxima teórica de vapor d'água na atmosfera antes da condensação.",
        },
        {
            "modulo": "Teto Convectivo (Clausius-Clapeyron)",
            "equacao": "W_max = 4,0 × TCWV",
            "params": "TCWV: Coluna Total de Vapor d'água integrada (mm)",
            "papel": "Barreira física superior para a precipitação diária máxima precipitável na bacia hidrográfica.",
        },
        {
            "modulo": "Barreira Física PIML",
            "equacao": "L_phys = λ_phys · [ Softplus( (ŷ - W_max)/δ ) ]²",
            "params": "λ_phys = 1,0 ; δ = 1,0<br>Penalização quadrática diferenciável",
            "papel": "Impede alucinações numéricas e superestimativas sem zerar os gradientes de aprendizagem neural.",
        },
        {
            "modulo": "Função de Perda Multiobjetivo",
            "equacao": "L_total = L_occ + L_int + λ_var·L_var + L_phys",
            "params": "L_occ: BCE ponderada<br>L_int: Log-Cosh assimétrica<br>λ_var = 0,10 (peso de cauda)",
            "papel": "Otimização conjunta de acurácia de ocorrência, volume, representação de picos e consistência física.",
        },
    ]

    df = pd.DataFrame(data)

    gt_tbl = (
        GT(df)
        .tab_header(
            title="Tabela 3 — Formulações matemáticas e restrições físicas da arquitetura THOR-PIML",
            subtitle="Acoplamento estocástico Hurdle, barreira termodinâmica de Clausius-Clapeyron e perda multiobjetivo"
        )
        .cols_label(
            modulo=html("<b>Componente / Módulo</b>"),
            equacao=html("<b>Equação Matemática</b>"),
            params=html("<b>Parâmetros e Variáveis</b>"),
            papel=html("<b>Papel Físico / Neural</b>"),
        )
        .fmt_markdown(columns=["modulo", "equacao", "params", "papel"])
        .tab_options(
            table_font_names=["Arial", "Helvetica", "sans-serif"],
            table_font_size="13px",
            heading_title_font_size="17px",
            heading_subtitle_font_size="13px",
            heading_background_color="#FFFFFF",
            column_labels_background_color="#F1F5F9",
            table_body_hlines_color=PALETTE["border_col"],
            table_body_hlines_width="1px",
            row_striping_include_table_body=True,
            row_striping_background_color=PALETTE["bg_stripe"],
        )
    )

    out_file = OUT_DIR / "tabela_3_great_tables.png"
    gt_tbl.gtsave(str(out_file))
    print(f"✓ Great Tables Tabela 3 salva em: {out_file}")


def make_table_4_gt():
    data = [
        {
            "codigo": "BIO12",
            "indicador": "Precipitação Anual Total (PRCPTOT)",
            "expressao": "P_total = Σ P_m (m=1..12)",
            "unidade": "mm/ano",
            "relevancia": "Balanço hídrico macro e recarga dos aquíferos da bacia hidrográfica do Alto Tietê.",
        },
        {
            "codigo": "BIO13 / BIO14",
            "indicador": "Mês Mais Chuvoso / Mês Mais Seco",
            "expressao": "max(P_m)  e  min(P_m)",
            "unidade": "mm/mês",
            "relevancia": "Identificação do contraste entre o pico de cheias do verão e a severidade da estiagem no inverno.",
        },
        {
            "codigo": "BIO15",
            "indicador": "Sazonalidade da Precipitação (CV)",
            "expressao": "CV = (σ_m / μ_m) × 100",
            "unidade": "%",
            "relevancia": "Grau de irregularidade e concentração temporal da chuva ao longo dos meses do ano.",
        },
        {
            "codigo": "BIO16 / BIO17",
            "indicador": "Trimestre Mais Úmido (DJF) / Mais Seco (JJA)",
            "expressao": "max Σ P_k  e  min Σ P_k",
            "unidade": "mm/trim",
            "relevancia": "Auditoria do viés sazonal do CORDEX e verificação da restauração da amplitude pelo THOR-PIML.",
        },
        {
            "codigo": "RX1day / RX5day",
            "indicador": "Precipitação Máxima em 1 Dia e 5 Dias Consecutivos",
            "expressao": "max(P_dia)  e  max(P_acum5d)",
            "unidade": "mm",
            "relevancia": "Gatilhos críticos de transbordamento do Rio Baquirivu-Guaçu e inundações na Rodovia Presidente Dutra.",
        },
        {
            "codigo": "R10mm / R20mm",
            "indicador": "Dias com Chuva Moderada (≥ 10 mm) e Forte (≥ 20 mm)",
            "expressao": "Σ [P ≥ 10 mm]  e  Σ [P ≥ 20 mm]",
            "unidade": "dias/ano",
            "relevancia": "Frequência de sobrecarga na rede municipal de microdrenagem e galerias de águas pluviais.",
        },
        {
            "codigo": "CDD / CWD",
            "indicador": "Dias Secos Consecutivos / Dias Chuvosos Consecutivos",
            "expressao": "max(run_P<1)  e  max(run_P≥1)",
            "unidade": "dias",
            "relevancia": "Diagnóstico de secas severas e auditoria do viés de 'garoa crônica' (persistência úmida do CORDEX).",
        },
    ]

    df = pd.DataFrame(data)

    gt_tbl = (
        GT(df)
        .tab_header(
            title="Tabela 4 — Indicadores bioclimáticos (WMO) e índices de extremos hidrológicos (ETCCDI) avaliados",
            subtitle="Métricas para avaliação do regime hídrico municipal, balanço sazonal e vulnerabilidade a cheias urbanas"
        )
        .cols_label(
            codigo=html("<b>Código</b>"),
            indicador=html("<b>Indicador / Definição</b>"),
            expressao=html("<b>Expressão Matemática</b>"),
            unidade=html("<b>Unidade</b>"),
            relevancia=html("<b>Relevância Hidrológica e Urbana em Guarulhos</b>"),
        )
        .fmt_markdown(columns=["codigo", "indicador", "expressao", "unidade", "relevancia"])
        .tab_options(
            table_font_names=["Arial", "Helvetica", "sans-serif"],
            table_font_size="13px",
            heading_title_font_size="17px",
            heading_subtitle_font_size="13px",
            heading_background_color="#FFFFFF",
            column_labels_background_color="#F1F5F9",
            table_body_hlines_color=PALETTE["border_col"],
            table_body_hlines_width="1px",
            row_striping_include_table_body=True,
            row_striping_background_color=PALETTE["bg_stripe"],
        )
    )

    out_file = OUT_DIR / "tabela_4_great_tables.png"
    gt_tbl.gtsave(str(out_file))
    print(f"✓ Great Tables Tabela 4 salva em: {out_file}")


if __name__ == "__main__":
    make_table_3_gt()
    make_table_4_gt()
