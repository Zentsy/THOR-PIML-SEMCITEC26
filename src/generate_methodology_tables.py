"""
THOR-PIML: Geração de Tabelas Metodológicas Editoriais (SEMCITEC 2026)
=====================================================================
Gera 3 tabelas em formato de card gráfico moderno (300 DPI),
seguindo rigorosamente a paleta do projeto:
- SSP1-1.9: #008FA3
- SSP1-2.6: #6FAE6F
- SSP2-4.5: #F2C230
- SSP3-7.0: #F27421
- SSP5-8.5: #A62D2D
- THOR Exclusivo: #1E40AF (Royal Cobalt Blue) / #2563EB
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configura encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paleta do Projeto
PALETTE = {
    "ssp119": "#008FA3",
    "ssp126": "#6FAE6F",
    "ssp245": "#F2C230",
    "ssp370": "#F27421",
    "ssp585": "#A62D2D",
    "thor": "#1E40AF",        # Royal Cobalt Blue
    "thor_light": "#EBF2FF",
    "dark_text": "#0F172A",
    "header_text": "#0B2538",
    "gray_text": "#475569",
    "border_gray": "#E2E8F0",
    "card_bg": "#FFFFFF",
    "page_bg": "#F8FAFC",
}


def draw_card_base(ax, width, height, title_text, subtitle_text="", accent_color="#008FA3"):
    """Desenha a moldura do card moderno com cabeçalho perfeitamente espaçado."""
    ax.set_facecolor(PALETTE["page_bg"])
    
    # Card principal com cantos arredondados
    card = patches.FancyBboxPatch(
        (0.015, 0.015), width - 0.03, height - 0.03,
        boxstyle=patches.BoxStyle("Round", pad=0.012, rounding_size=0.025),
        facecolor=PALETTE["card_bg"],
        edgecolor=PALETTE["border_gray"],
        linewidth=1.8,
        zorder=1
    )
    ax.add_patch(card)
    
    # Título principal
    ax.text(
        0.045, height - 0.050, title_text,
        fontsize=15.5, fontweight="bold", color=PALETTE["header_text"],
        ha="left", va="top", zorder=3, fontfamily="sans-serif"
    )
    
    # Linha de destaque sob o título (accent bar)
    accent_bar = patches.Rectangle(
        (0.045, height - 0.092), 0.22, 0.005,
        facecolor=accent_color, edgecolor="none", zorder=3
    )
    ax.add_patch(accent_bar)
    
    # Subtítulo logo abaixo da accent bar
    if subtitle_text:
        ax.text(
            0.045, height - 0.110, subtitle_text,
            fontsize=10.5, color=PALETTE["gray_text"],
            ha="left", va="top", zorder=3, fontfamily="sans-serif"
        )


def render_table_2_datasets():
    """Gera a Tabela 2: Síntese das bases de dados hidrometeorológicas e climáticas."""
    fig, ax = plt.subplots(figsize=(14.5, 6.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    draw_card_base(
        ax, 1.0, 1.0,
        title_text="Tabela 2 — Bases de dados hidrometeorológicas e climáticas",
        subtitle_text="Séries observadas, reanálises atmosféricas e projeções dinâmicas sob protocolo Zero Leakage",
        accent_color=PALETTE["ssp119"]
    )
    
    cols = [
        {"name": "Base / Modelo", "x": 0.045, "w": 0.18, "ha": "left"},
        {"name": "Variáveis Utilizadas", "x": 0.245, "w": 0.28, "ha": "left"},
        {"name": "Resolução", "x": 0.540, "w": 0.12, "ha": "center"},
        {"name": "Período", "x": 0.675, "w": 0.11, "ha": "center"},
        {"name": "Finalidade no Estudo", "x": 0.795, "w": 0.16, "ha": "left"},
    ]
    
    header_y = 0.79
    ax.plot([0.045, 0.955], [header_y - 0.03, header_y - 0.03], color=PALETTE["ssp119"], linewidth=1.5, zorder=3)
    for col in cols:
        ax.text(
            col["x"] if col["ha"] == "left" else col["x"] + col["w"]/2,
            header_y, col["name"],
            fontsize=11.5, fontweight="bold", color=PALETTE["header_text"],
            ha=col["ha"], va="center", zorder=3
        )
        
    rows = [
        {
            "badge": "CHIRPS v2.0", "badge_col": "#0F766E", "badge_bg": "#E6F6F4",
            "vars": "Precipitação diária de alta resolução\n(alvo observacional local)",
            "res": "0,05° (~5,5 km)", "period": "1981–2014",
            "fin": "Linha de Base Observada (Ground Truth histórico municipal)"
        },
        {
            "badge": "ERA5-Land / Global", "badge_col": PALETTE["thor"], "badge_bg": PALETTE["thor_light"],
            "vars": "84 preditores de superfície (Land) +\n5 níveis sinóticos 2D (ERA5)",
            "res": "0,10° a 0,25°\n(~9 a ~28 km)", "period": "1981–2026",
            "fin": "Calibração e Teste Histórico Independente do THOR-PIML"
        },
        {
            "badge": "CORDEX SAM-22", "badge_col": PALETTE["ssp370"], "badge_bg": "#FEF2EA",
            "vars": "Precipitação (pr) e Temperatura 2m (tas)\nINPE-Eta orientado por HadGEM2-ES",
            "res": "0,20° (~22 km)", "period": "2026–2099",
            "fin": "Forçamento Dinâmico Futuro (Cenários SSP2-4.5 e SSP5-8.5)"
        },
        {
            "badge": "CMIP6 (CCKP)", "badge_col": PALETTE["ssp119"], "badge_bg": "#E6F7F9",
            "vars": "Anomalias anuais e mensais de chuva\n(Percentis P10, Mediana e P90)",
            "res": "0,25° (~28 km)", "period": "2080–2099",
            "fin": "Referência Multimodelo Global Independente (Banco Mundial)"
        }
    ]
    
    row_y = 0.65
    row_h = 0.13
    
    for i, r in enumerate(rows):
        y_c = row_y - i * row_h
        if i % 2 == 1:
            stripe = patches.Rectangle((0.035, y_c - 0.052), 0.93, row_h, facecolor="#F8FAFC", edgecolor="none", zorder=2)
            ax.add_patch(stripe)
            
        badge = patches.FancyBboxPatch(
            (cols[0]["x"], y_c - 0.030), cols[0]["w"] * 0.92, 0.060,
            boxstyle=patches.BoxStyle("Round", pad=0.005, rounding_size=0.015),
            facecolor=r["badge_bg"], edgecolor="none", zorder=3
        )
        ax.add_patch(badge)
        ax.text(
            cols[0]["x"] + cols[0]["w"] * 0.46, y_c, r["badge"],
            fontsize=10.5, fontweight="bold", color=r["badge_col"],
            ha="center", va="center", zorder=4
        )
        
        ax.text(cols[1]["x"], y_c, r["vars"], fontsize=10, color=PALETTE["dark_text"], ha="left", va="center", zorder=3)
        ax.text(cols[2]["x"] + cols[2]["w"]/2, y_c, r["res"], fontsize=10, color=PALETTE["dark_text"], ha="center", va="center", zorder=3)
        ax.text(cols[3]["x"] + cols[3]["w"]/2, y_c, r["period"], fontsize=10.5, fontweight="bold", color=PALETTE["dark_text"], ha="center", va="center", zorder=3)
        ax.text(cols[4]["x"], y_c, r["fin"], fontsize=9.5, color=PALETTE["gray_text"], ha="left", va="center", zorder=3)
        
        ax.plot([0.045, 0.955], [y_c - 0.060, y_c - 0.060], color=PALETTE["border_gray"], linewidth=0.8, zorder=2)
        
    out_path = OUT_DIR / "tabela_2_bases_dados_metodologia_300dpi.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Tabela 2 salva: {out_path}")


def render_table_3_equations():
    """Gera a Tabela 3: Formulações matemáticas centrais da arquitetura THOR-PIML."""
    fig, ax = plt.subplots(figsize=(15.5, 7.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    draw_card_base(
        ax, 1.0, 1.0,
        title_text="Tabela 3 — Formulações matemáticas e restrições físicas do THOR-PIML",
        subtitle_text="Acoplamento estocástico Hurdle, barreira termodinâmica de Clausius-Clapeyron e perda multiobjetivo",
        accent_color=PALETTE["thor"]
    )
    
    cols = [
        {"name": "Componente / Módulo", "x": 0.045, "w": 0.18, "ha": "left"},
        {"name": "Equação Matemática", "x": 0.240, "w": 0.26, "ha": "left"},
        {"name": "Parâmetros e Variáveis", "x": 0.515, "w": 0.22, "ha": "left"},
        {"name": "Princípio Físico / Papel Neural", "x": 0.745, "w": 0.21, "ha": "left"},
    ]
    
    header_y = 0.80
    ax.plot([0.045, 0.955], [header_y - 0.028, header_y - 0.028], color=PALETTE["thor"], linewidth=1.5, zorder=3)
    for col in cols:
        ax.text(
            col["x"] if col["ha"] == "left" else col["x"] + col["w"]/2,
            header_y, col["name"],
            fontsize=11.5, fontweight="bold", color=PALETTE["header_text"],
            ha=col["ha"], va="center", zorder=3
        )
        
    rows = [
        {
            "badge": "Decodificador Hurdle", "badge_col": PALETTE["thor"], "badge_bg": PALETTE["thor_light"],
            "eq": r"$\hat{y} = p_{\mathrm{occ}} \times \mu_{\mathrm{int}}$",
            "params": r"$p_{\mathrm{occ}} = \sigma(z_{\mathrm{occ}}) \in (0,1)$" + "\n" + r"$\mu_{\mathrm{int}} = \mathrm{Softplus}(z_{\mathrm{int}}) > 0$",
            "role": "Desacopla probabilidade de ocorrência\n(WMO ≥ 1 mm) da intensidade da chuva."
        },
        {
            "badge": "Pressão de Saturação", "badge_col": PALETTE["ssp119"], "badge_bg": "#E6F7F9",
            "eq": r"$e_s(T) = 6{,}1078 \cdot \exp\left( \frac{17{,}269 \cdot T}{237{,}3 + T} \right)$",
            "params": r"$T$: temperatura do ar a 2m ($^\circ$C)" + "\n" + r"$e_s$: pressão em hPa (Magnus-Tetens)",
            "role": "Calcula a capacidade teórica máxima de\nvapor atmosférico antes da condensação."
        },
        {
            "badge": "Teto Convectivo", "badge_col": "#059669", "badge_bg": "#ECFDF5",
            "eq": r"$W_{\max} = 4{,}0 \times \mathrm{TCWV}$",
            "params": r"$\mathrm{TCWV}$: Coluna Total de Vapor" + "\n" + r"d'água integrada na coluna (mm)",
            "role": "Barreira física de Clausius-Clapeyron\npara limite máximo diário de precipitação."
        },
        {
            "badge": "Barreira Física PIML", "badge_col": "#7C3AED", "badge_bg": "#F5F3FF",
            "eq": r"$\mathcal{L}_{\mathrm{phys}} = \lambda_{\mathrm{phys}} \cdot \left[ \mathrm{Softplus}\left( \frac{\hat{y} - W_{\max}}{\delta} \right) \right]^2$",
            "params": r"$\lambda_{\mathrm{phys}} = 1{,}0$ ; $\delta = 1{,}0$" + "\nPenalização quadrática diferenciável",
            "role": "Gradiente suave sem 'dead units' que\nanula alucinações de super-chuva."
        },
        {
            "badge": "Perda Multiobjetivo", "badge_col": PALETTE["dark_text"], "badge_bg": "#F1F5F9",
            "eq": r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{occ}} + \mathcal{L}_{\mathrm{int}} + \lambda_{\mathrm{var}}\mathcal{L}_{\mathrm{var}} + \mathcal{L}_{\mathrm{phys}}$",
            "params": r"BCE ponderada + Log-Cosh assimétrica" + "\n" + r"$\lambda_{\mathrm{var}} = 0{,}10$ (Preservação de picos)",
            "role": "Otimização conjunta de acurácia, cauda\nde tempestades e consistência termodinâmica."
        }
    ]
    
    row_y = 0.67
    row_h = 0.120
    
    for i, r in enumerate(rows):
        y_c = row_y - i * row_h
        if i % 2 == 1:
            stripe = patches.Rectangle((0.035, y_c - 0.050), 0.93, row_h, facecolor="#F8FAFC", edgecolor="none", zorder=2)
            ax.add_patch(stripe)
            
        badge = patches.FancyBboxPatch(
            (cols[0]["x"], y_c - 0.028), cols[0]["w"] * 0.94, 0.056,
            boxstyle=patches.BoxStyle("Round", pad=0.005, rounding_size=0.015),
            facecolor=r["badge_bg"], edgecolor="none", zorder=3
        )
        ax.add_patch(badge)
        ax.text(
            cols[0]["x"] + cols[0]["w"] * 0.47, y_c, r["badge"],
            fontsize=10.5, fontweight="bold", color=r["badge_col"],
            ha="center", va="center", zorder=4
        )
        
        ax.text(cols[1]["x"], y_c, r["eq"], fontsize=11, color=PALETTE["dark_text"], ha="left", va="center", zorder=3)
        ax.text(cols[2]["x"], y_c, r["params"], fontsize=9.5, color=PALETTE["dark_text"], ha="left", va="center", zorder=3)
        ax.text(cols[3]["x"], y_c, r["role"], fontsize=9.2, color=PALETTE["gray_text"], ha="left", va="center", zorder=3)
        
        ax.plot([0.045, 0.955], [y_c - 0.058, y_c - 0.058], color=PALETTE["border_gray"], linewidth=0.8, zorder=2)
        
    out_path = OUT_DIR / "tabela_3_equacoes_piml_metodologia_300dpi.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Tabela 3 salva: {out_path}")


def render_table_4_indicators():
    """Gera a Tabela 4: Indicadores Bioclimáticos e Extremos Climáticos (ETCCDI)."""
    fig, ax = plt.subplots(figsize=(15.5, 9.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    draw_card_base(
        ax, 1.0, 1.0,
        title_text="Tabela 4 — Indicadores bioclimáticos (WMO) e extremos ETCCDI avaliados",
        subtitle_text="Métricas para avaliação do regime hídrico municipal, balanço sazonal e vulnerabilidade a cheias urbanas",
        accent_color=PALETTE["ssp245"]
    )
    
    cols = [
        {"name": "Código", "x": 0.045, "w": 0.13, "ha": "center"},
        {"name": "Indicador / Definição", "x": 0.190, "w": 0.25, "ha": "left"},
        {"name": "Expressão Matemática", "x": 0.450, "w": 0.18, "ha": "left"},
        {"name": "Unidade", "x": 0.640, "w": 0.09, "ha": "center"},
        {"name": "Relevância Hidrológica e Urbana em Guarulhos", "x": 0.740, "w": 0.22, "ha": "left"},
    ]
    
    header_y = 0.82
    ax.plot([0.045, 0.955], [header_y - 0.025, header_y - 0.025], color=PALETTE["ssp245"], linewidth=1.5, zorder=3)
    for col in cols:
        ax.text(
            col["x"] if col["ha"] == "left" else col["x"] + col["w"]/2,
            header_y, col["name"],
            fontsize=11, fontweight="bold", color=PALETTE["header_text"],
            ha=col["ha"], va="center", zorder=3
        )
        
    rows = [
        {
            "code": "BIO12", "col": PALETTE["ssp119"], "bg": "#E6F7F9",
            "name": "Precipitação Anual Total (PRCPTOT)",
            "eq": r"$P_{\mathrm{total}} = \sum_{m=1}^{12} P_m$",
            "unit": "mm/ano",
            "rel": "Balanço hídrico e recarga de aquíferos do Alto Tietê."
        },
        {
            "code": "BIO13 / BIO14", "col": PALETTE["ssp126"], "bg": "#EFF7F0",
            "name": "Mês Mais Chuvoso / Mais Seco",
            "eq": r"$\max(P_m)$  e  $\min(P_m)$",
            "unit": "mm/mês",
            "rel": "Pico de cheias de verão vs. estiagem severa de inverno."
        },
        {
            "code": "BIO15", "col": "#D99B00", "bg": "#FEF9E6",
            "name": "Sazonalidade da Chuva",
            "eq": r"$CV = (\sigma_m / \mu_m) \times 100$",
            "unit": "%",
            "rel": "Irregularidade e concentração temporal do regime pluvial."
        },
        {
            "code": "BIO16 / BIO17", "col": PALETTE["ssp370"], "bg": "#FEF1E9",
            "name": "Trimestre Mais Úmido (DJF) / Seco (JJA)",
            "eq": r"$\max \sum_{k=m}^{m+2} P_k$  /  $\min \sum P_k$",
            "unit": "mm/trim",
            "rel": "Auditoria de correção de viés sazonal frente ao CORDEX."
        },
        {
            "code": "RX1day / RX5day", "col": PALETTE["ssp585"], "bg": "#FDECEC",
            "name": "Máxima Precipitação em 1 e 5 Dias",
            "eq": r"$\max(P_{\mathrm{dia}})$  /  $\max(P_{\mathrm{acum5d}})$",
            "unit": "mm",
            "rel": "Gatilhos de transbordamento do Baquirivu e Rodovia Dutra."
        },
        {
            "code": "R10mm / R20mm", "col": PALETTE["dark_text"], "bg": "#F1F5F9",
            "name": "Dias com Chuva Forte / Tempestades",
            "eq": r"$\sum [P \geq 10\mathrm{mm}]$  /  $\sum [P \geq 20\mathrm{mm}]$",
            "unit": "dias/ano",
            "rel": "Frequência de sobrecarga da microdrenagem municipal."
        },
        {
            "code": "CWD / CDD", "col": PALETTE["thor"], "bg": PALETTE["thor_light"],
            "name": "Dias Consecutivos Chuvosos / Secos",
            "eq": r"$\max(\mathrm{run}_{P \geq 1})$  /  $\max(\mathrm{run}_{P < 1})$",
            "unit": "dias",
            "rel": "Persistência úmida e diagnóstico de garoa crônica."
        }
    ]
    
    row_y = 0.73
    row_h = 0.088
    
    for i, r in enumerate(rows):
        y_c = row_y - i * row_h
        if i % 2 == 1:
            stripe = patches.Rectangle((0.035, y_c - 0.038), 0.93, row_h, facecolor="#F8FAFC", edgecolor="none", zorder=2)
            ax.add_patch(stripe)
            
        badge = patches.FancyBboxPatch(
            (cols[0]["x"] - 0.005, y_c - 0.022), cols[0]["w"] * 1.08, 0.044,
            boxstyle=patches.BoxStyle("Round", pad=0.005, rounding_size=0.012),
            facecolor=r["bg"], edgecolor="none", zorder=3
        )
        ax.add_patch(badge)
        ax.text(
            cols[0]["x"] + cols[0]["w"] * 0.54, y_c, r["code"],
            fontsize=10, fontweight="bold", color=r["col"],
            ha="center", va="center", zorder=4
        )
        
        ax.text(cols[1]["x"], y_c, r["name"], fontsize=10, fontweight="bold", color=PALETTE["header_text"], ha="left", va="center", zorder=3)
        ax.text(cols[2]["x"], y_c, r["eq"], fontsize=10.5, color=PALETTE["dark_text"], ha="left", va="center", zorder=3)
        ax.text(cols[3]["x"] + cols[3]["w"]/2, y_c, r["unit"], fontsize=10, color=PALETTE["gray_text"], ha="center", va="center", zorder=3)
        ax.text(cols[4]["x"], y_c, r["rel"], fontsize=9.2, color=PALETTE["gray_text"], ha="left", va="center", zorder=3)
        
        ax.plot([0.045, 0.955], [y_c - 0.042, y_c - 0.042], color=PALETTE["border_gray"], linewidth=0.8, zorder=2)
        
    out_path = OUT_DIR / "tabela_4_indicadores_bioclimaticos_etccdi_300dpi.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Tabela 4 salva: {out_path}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("THOR-PIML: RENDERIZANDO TABELAS METODOLÓGICAS OFICIAIS (300 DPI)")
    print("="*70)
    render_table_2_datasets()
    render_table_3_equations()
    render_table_4_indicators()
    print("="*70 + "\n")
