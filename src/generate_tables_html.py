"""
THOR-PIML: Geração de Quadros com HTML5 + CSS + KaTeX
=====================================================
Gera Quadro 3 e Quadro 4 no padrão de publicação editorial:
- Sem títulos internos (inseridos diretamente pelo autor no Google Docs)
- Cartão flutuante com cantos arredondados e sombra suave difusa (box-shadow)
- Container de tabela delimitado com cantos arredondados internos
- KaTeX vetorial para todas as equações matemáticas
- Badges e destaques na paleta oficial do projeto (IPCC SSPs e THOR Royal Blue)
- Renderização em 300 DPI via nokap (Headless Chromium)
"""

import sys
from pathlib import Path
import nokap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
COMP_DIR = BASE_DIR / "results" / "tables" / "comparison"
TABLES_DIR = BASE_DIR / "results" / "tables"
COMP_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def render_html_quadro_3():
    html_content = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
        ],
        throwOnError: false
    });"></script>
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        background-color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    #capture-wrapper {
        background-color: #FFFFFF;
        padding: 24px 28px;
        display: inline-block;
    }
    .card {
        background: #FFFFFF;
        width: 1020px;
        border-radius: 20px;
        border: 1.5px solid #E2E8F0;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.14), 0 6px 16px -4px rgba(0, 0, 0, 0.08);
        padding: 18px 20px;
    }
    .table-container {
        border: 1.5px solid #CBD5E1;
        border-radius: 12px;
        overflow: hidden;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }
    thead tr {
        background-color: #F1F5F9;
        border-bottom: 2.5px solid #1E40AF; /* THOR Royal Blue */
    }
    th {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        padding: 14px 14px;
        text-align: left;
        letter-spacing: -0.2px;
    }
    tbody tr {
        border-bottom: 1px solid #E2E8F0;
        background-color: #FFFFFF;
    }
    tbody tr:nth-child(even) {
        background-color: #F8FAFC;
    }
    tbody tr:last-child {
        border-bottom: none;
    }
    td {
        padding: 14px 14px;
        font-size: 15px;
        color: #1E293B;
        vertical-align: middle;
        line-height: 1.42;
    }
    .badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.1px;
    }
    .badge-hurdle { background: #EBF2FF; color: #1E40AF; }
    .badge-sat { background: #E6F7F9; color: #008FA3; }
    .badge-conv { background: #EFF7F0; color: #2E7D32; }
    .badge-piml { background: #F5F3FF; color: #6D28D9; }
    .badge-loss { background: #F1F5F9; color: #0F172A; }
    .math-cell {
        font-size: 16px;
        color: #0F172A;
        text-align: left;
        padding-right: 8px;
    }
    .katex-display {
        margin: 2px 0 !important;
    }
    .params-cell {
        font-size: 14px;
        color: #334155;
        line-height: 1.45;
        white-space: normal;
    }
    .role-cell {
        font-size: 14px;
        color: #334155;
        line-height: 1.45;
    }
</style>
</head>
<body>
    <div id="capture-wrapper">
        <div class="card">
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 17%;">Componente / Módulo</th>
                            <th style="width: 38%;">Equação Matemática</th>
                            <th style="width: 21%;">Parâmetros e Variáveis</th>
                            <th style="width: 24%;">Papel Físico / Neural</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><span class="badge badge-hurdle">Decodificador Hurdle</span></td>
                            <td class="math-cell">$$\hat{y} = p_{\text{occ}} \times \mu_{\text{int}}$$</td>
                            <td class="params-cell"><span style="white-space: nowrap;">$p_{\text{occ}} = \sigma(z_{\text{occ}}) \in (0,1)$</span><br><span style="white-space: nowrap;">$\mu_{\text{int}} = \text{Softplus}(z_{\text{int}}) > 0$</span></td>
                            <td class="role-cell">Desacopla a probabilidade estocástica de ocorrência ($P \ge 1{,}0$ mm) da magnitude e volume de chuva.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-sat">Pressão de Saturação</span></td>
                            <td class="math-cell">$$e_s(T) = 6{,}1078 \cdot \exp\left( \frac{17{,}269 \cdot T}{237{,}3 + T} \right)$$</td>
                            <td class="params-cell">$T$: temperatura do ar a 2m (°C)<br>$e_s$: pressão de saturação (hPa)</td>
                            <td class="role-cell">Estima a capacidade higrométrica teórica de retenção de vapor d'água antes da condensação.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-conv">Teto Convectivo</span></td>
                            <td class="math-cell">$$W_{\max} = 4{,}0 \times \text{TCWV}$$</td>
                            <td class="params-cell">$\text{TCWV}$: Coluna Total de Vapor d'água integrada (mm)</td>
                            <td class="role-cell">Barreira de Clausius-Clapeyron para o limite máximo diário precipitável na bacia.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-piml">Barreira Física PIML</span></td>
                            <td class="math-cell">$$\mathcal{L}_{\text{phys}} = \lambda_{\text{phys}} \cdot \left[ \text{Softplus}\left( \frac{\hat{y} - W_{\max}}{\delta} \right) \right]^2$$</td>
                            <td class="params-cell"><span style="white-space: nowrap;">$\lambda_{\text{phys}} = 1{,}0$ ; $\delta = 1{,}0$</span><br>Penalização quadrática diferenciável</td>
                            <td class="role-cell">Elimina alucinações de super-chuva sem zerar os gradientes de aprendizagem neural.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-loss">Perda Multiobjetivo</span></td>
                            <td class="math-cell">$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{occ}} + \mathcal{L}_{\text{int}} + \lambda_{\text{var}}\mathcal{L}_{\text{var}} + \mathcal{L}_{\text{phys}}$$</td>
                            <td class="params-cell"><span style="white-space: nowrap;">BCE + Log-Cosh assimétrica</span><br><span style="white-space: nowrap;">$\lambda_{\text{var}} = 0{,}10$ (peso de cauda)</span></td>
                            <td class="role-cell">Otimização conjunta de acurácia de ocorrência, volume, representação de extremos e consistência física.</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>"""

    comp_file = COMP_DIR / "tabela_3_html.png"
    nokap.from_html(html_content, str(comp_file), selector="#capture-wrapper", delay=1.5, zoom=3.0)
    print(f"✓ Quadro 3 salvo em comparação: {comp_file}")

    # Salva também como arquivo oficial pronto para uso
    for out_name in ["tabela_3_equacoes_piml_metodologia_300dpi.png", "quadro_3_equacoes_piml_metodologia_300dpi.png"]:
        dest = TABLES_DIR / out_name
        dest.write_bytes(comp_file.read_bytes())
        print(f"✓ Quadro 3 copiado para: {dest}")


def render_html_quadro_4():
    html_content = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
        ],
        throwOnError: false
    });"></script>
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        background-color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    #capture-wrapper {
        background-color: #FFFFFF;
        padding: 24px 28px;
        display: inline-block;
    }
    .card {
        background: #FFFFFF;
        width: 980px;
        border-radius: 20px;
        border: 1.5px solid #E2E8F0;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.14), 0 6px 16px -4px rgba(0, 0, 0, 0.08);
        padding: 18px 20px;
    }
    .table-container {
        border: 1.5px solid #CBD5E1;
        border-radius: 12px;
        overflow: hidden;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }
    thead tr {
        background-color: #F1F5F9;
        border-bottom: 2.5px solid #F2C230; /* SSP2-4.5 Amber */
    }
    th {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        padding: 14px 14px;
        text-align: left;
        letter-spacing: -0.2px;
    }
    tbody tr {
        border-bottom: 1px solid #E2E8F0;
        background-color: #FFFFFF;
    }
    tbody tr:nth-child(even) {
        background-color: #F8FAFC;
    }
    tbody tr:last-child {
        border-bottom: none;
    }
    td {
        padding: 14px 14px;
        font-size: 15px;
        color: #1E293B;
        vertical-align: middle;
        line-height: 1.42;
    }
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13.5px;
        text-align: center;
        letter-spacing: 0.1px;
    }
    .b-bio12 { background: #E6F7F9; color: #008FA3; }
    .b-bio13 { background: #EFF7F0; color: #2E7D32; }
    .b-bio15 { background: #FEF9E6; color: #B45309; }
    .b-bio16 { background: #FEF1E9; color: #C2410C; }
    .b-rx { background: #FDECEC; color: #A62D2D; }
    .b-r { background: #F1F5F9; color: #0F172A; }
    .b-run { background: #EBF2FF; color: #1E40AF; }
    .math-cell {
        font-size: 16px;
        color: #0F172A;
        white-space: nowrap;
    }
    .desc-cell {
        font-weight: 700;
        color: #0F172A;
        font-size: 15px;
    }
    .unit-cell {
        text-align: center;
        color: #334155;
        font-weight: 700;
        font-size: 14.5px;
    }
    .rel-cell {
        font-size: 14px;
        color: #334155;
        line-height: 1.45;
    }
</style>
</head>
<body>
    <div id="capture-wrapper">
        <div class="card">
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 13%; text-align: center;">Código</th>
                            <th style="width: 23%;">Indicador / Definição</th>
                            <th style="width: 26%;">Expressão Matemática</th>
                            <th style="width: 9%; text-align: center;">Unidade</th>
                            <th style="width: 29%;">Relevância Hidrológica e Urbana (Guarulhos)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="text-align: center;"><span class="badge b-bio12">BIO12</span></td>
                            <td class="desc-cell">Precipitação Anual Total (PRCPTOT)</td>
                            <td class="math-cell">$$P_{\text{total}} = \sum_{m=1}^{12} P_m$$</td>
                            <td class="unit-cell">mm/ano</td>
                            <td class="rel-cell">Balanço hídrico macro e recarga dos aquíferos da bacia hidrográfica do Alto Tietê.</td>
                        </tr>
                        <tr>
                            <td style="text-align: center;"><span class="badge b-bio13">BIO13 / BIO14</span></td>
                            <td class="desc-cell">Mês Mais Chuvoso / Mês Mais Seco</td>
                            <td class="math-cell">$\max(P_m)$ &nbsp;e&nbsp; $\min(P_m)$</td>
                            <td class="unit-cell">mm/mês</td>
                            <td class="rel-cell">Contraste entre o pico de cheias do verão e a severidade da estiagem no inverno.</td>
                        </tr>
                        <tr>
                            <td style="text-align: center;"><span class="badge b-bio15">BIO15</span></td>
                            <td class="desc-cell">Sazonalidade da Precipitação (CV)</td>
                            <td class="math-cell">$$CV = \left( \frac{\sigma_m}{\mu_m} \right) \times 100$$</td>
                            <td class="unit-cell">%</td>
                            <td class="rel-cell">Grau de irregularidade e concentração temporal da chuva ao longo dos meses do ano.</td>
                        </tr>
                        <tr>
                            <td style="text-align: center;"><span class="badge b-bio16">BIO16 / BIO17</span></td>
                            <td class="desc-cell">Trimestre Mais Úmido (DJF) / Seco (JJA)</td>
                            <td class="math-cell">$\max \sum_{k=m}^{m+2} P_k$ &nbsp;/&nbsp; $\min \sum P_k$</td>
                            <td class="unit-cell">mm/trim</td>
                            <td class="rel-cell">Auditoria do viés sazonal do CORDEX e verificação da restauração da amplitude pelo THOR.</td>
                        </tr>
                        <tr>
                            <td style="text-align: center;"><span class="badge b-rx">RX1day / RX5day</span></td>
                            <td class="desc-cell">Precipitação Máxima em 1 e 5 Dias</td>
                            <td class="math-cell">$\max(P_{\text{dia}})$ &nbsp;/&nbsp; $\max(P_{\text{5d}})$</td>
                            <td class="unit-cell">mm</td>
                            <td class="rel-cell">Gatilhos críticos de transbordamento do Rio Baquirivu-Guaçu e inundações na Rodovia Presidente Dutra.</td>
                        </tr>
                        <tr>
                            <td style="text-align: center;"><span class="badge b-r">R10mm / R20mm</span></td>
                            <td class="desc-cell">Dias com Chuva Moderada e Forte</td>
                            <td class="math-cell">$\sum [P \ge 10]$ &nbsp;/&nbsp; $\sum [P \ge 20]$</td>
                            <td class="unit-cell">dias/ano</td>
                            <td class="rel-cell">Frequência de sobrecarga na rede municipal de microdrenagem e galerias de águas pluviais.</td>
                        </tr>
                        <tr>
                            <td style="text-align: center;"><span class="badge b-run">CDD / CWD</span></td>
                            <td class="desc-cell">Dias Secos / Chuvosos Consecutivos</td>
                            <td class="math-cell">$\max(\text{run}_{P < 1})$ &nbsp;/&nbsp; $\max(\text{run}_{P \ge 1})$</td>
                            <td class="unit-cell">dias</td>
                            <td class="rel-cell">Diagnóstico de secas severas e auditoria do viés de "garoa crônica" (persistência úmida do CORDEX).</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>"""

    comp_file = COMP_DIR / "tabela_4_html.png"
    nokap.from_html(html_content, str(comp_file), selector="#capture-wrapper", delay=1.5, zoom=3.0)
    print(f"✓ Quadro 4 salvo em comparação: {comp_file}")

    # Salva também como arquivo oficial pronto para uso
    for out_name in ["tabela_4_indicadores_bioclimaticos_etccdi_300dpi.png", "quadro_4_indicadores_bioclimaticos_etccdi_300dpi.png"]:
        dest = TABLES_DIR / out_name
        dest.write_bytes(comp_file.read_bytes())
        print(f"✓ Quadro 4 copiado para: {dest}")


def render_html_quadro_5():
    html_content = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        background-color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    #capture-wrapper {
        background-color: #FFFFFF;
        padding: 24px 28px;
        display: inline-block;
    }
    .card {
        background: #FFFFFF;
        width: 1080px;
        border-radius: 20px;
        border: 1.5px solid #E2E8F0;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.14), 0 6px 16px -4px rgba(0, 0, 0, 0.08);
        padding: 18px 20px;
    }
    .table-container {
        border: 1.5px solid #CBD5E1;
        border-radius: 12px;
        overflow: hidden;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }
    thead tr:first-child {
        background-color: #F1F5F9;
        border-bottom: 1px solid #CBD5E1;
    }
    thead tr:nth-child(2) {
        background-color: #F8FAFC;
        border-bottom: 2.5px solid #1E40AF; /* THOR Royal Blue */
    }
    th {
        font-size: 14.5px;
        font-weight: 700;
        color: #0F172A;
        padding: 10px 10px;
        text-align: center;
        letter-spacing: -0.2px;
    }
    tbody tr.epoch-row {
        background-color: #E2E8F0;
        border-top: 2px solid #94A3B8;
        border-bottom: 1.5px solid #94A3B8;
    }
    tbody tr.epoch-row td {
        font-size: 13.5px;
        font-weight: 800;
        color: #0F172A;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 8px 12px;
        text-align: left;
    }
    tbody tr.data-row {
        border-bottom: 1px solid #E2E8F0;
        background-color: #FFFFFF;
    }
    tbody tr.data-row:nth-child(even) {
        background-color: #F8FAFC;
    }
    tbody tr.data-row:last-child {
        border-bottom: none;
    }
    td {
        padding: 7px 10px;
        font-size: 13.5px;
        color: #1E293B;
        vertical-align: middle;
        line-height: 1.35;
    }
    td.ind-cell {
        text-align: left;
        font-weight: 600;
        color: #0F172A;
    }
    td.num-cell {
        text-align: center;
        font-variant-numeric: tabular-nums;
    }
    .badge {
        display: inline-block;
        padding: 3px 7px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 12px;
        margin-right: 6px;
        letter-spacing: 0.1px;
    }
    .b-bio12 { background: #E6F7F9; color: #008FA3; }
    .b-bio13 { background: #EFF7F0; color: #2E7D32; }
    .b-bio15 { background: #FEF9E6; color: #B45309; }
    .b-bio16 { background: #FEF1E9; color: #C2410C; }
    .b-rx { background: #FDECEC; color: #A62D2D; }
    .b-r { background: #F1F5F9; color: #0F172A; }
    .b-run { background: #EBF2FF; color: #1E40AF; }
</style>
</head>
<body>
    <div id="capture-wrapper">
        <div class="card">
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th rowspan="2" style="width: 28%; text-align: left; padding-left: 14px;">Indicador Hidroclimático</th>
                            <th rowspan="2" style="width: 18%;">Linha de Base<br><span style="font-weight: 600; font-size: 13px; color: #475569;">CHIRPS (1981–2014)</span></th>
                            <th colspan="2" style="width: 27%; border-left: 1px solid #CBD5E1; border-right: 1px solid #CBD5E1;">Cenário SSP2-4.5</th>
                            <th colspan="2" style="width: 27%;">Cenário SSP5-8.5</th>
                        </tr>
                        <tr>
                            <th style="width: 13.5%; border-left: 1px solid #CBD5E1; font-size: 13px; color: #334155;">CORDEX Bruto</th>
                            <th style="width: 13.5%; font-size: 13px; color: #1E40AF; font-weight: 800;">THOR-PIML</th>
                            <th style="width: 13.5%; border-left: 1px solid #CBD5E1; font-size: 13px; color: #334155;">CORDEX Bruto</th>
                            <th style="width: 13.5%; font-size: 13px; color: #DC2626; font-weight: 800;">THOR-PIML</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- ÉPOCA 1: FUTURO PRÓXIMO -->
                        <tr class="epoch-row">
                            <td colspan="6">Época 1: Futuro Próximo (2026–2050)</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio12">BIO12</span> Precipitação Total (mm/ano)</td>
                            <td class="num-cell">1545,4 ± 207,9</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">1268,1 ± 441,1</td>
                            <td class="num-cell" style="font-weight: 700; color: #1E40AF;">1371,1 ± 121,7</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">1211,7 ± 386,1</td>
                            <td class="num-cell" style="font-weight: 700; color: #DC2626;">1528,0 ± 141,3</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio13">BIO13</span> Mês Mais Chuvoso (mm)</td>
                            <td class="num-cell">313,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">243,4</td>
                            <td class="num-cell">237,6</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">240,0</td>
                            <td class="num-cell">266,1</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio13">BIO14</span> Mês Mais Seco (mm)</td>
                            <td class="num-cell">16,5</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">25,1</td>
                            <td class="num-cell">19,9</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">20,5</td>
                            <td class="num-cell">27,6</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio15">BIO15</span> Sazonalidade (CV %)</td>
                            <td class="num-cell">73,5%</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">66,5%</td>
                            <td class="num-cell">68,7%</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">70,4%</td>
                            <td class="num-cell">65,6%</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio16">BIO16</span> Trimestre Mais Úmido (mm)</td>
                            <td class="num-cell">745,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">518,5</td>
                            <td class="num-cell">634,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">500,4</td>
                            <td class="num-cell">696,5</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio16">BIO17</span> Trimestre Mais Seco (mm)</td>
                            <td class="num-cell">107,6</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">151,0</td>
                            <td class="num-cell">80,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">143,7</td>
                            <td class="num-cell">106,5</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-r">R10mm</span> Dias com Chuva &ge; 10 mm (dias)</td>
                            <td class="num-cell">57,5 ± 8,9</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">45,4 ± 18,6</td>
                            <td class="num-cell">32,7 ± 7,2</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">42,5 ± 14,9</td>
                            <td class="num-cell">40,2 ± 7,6</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-r">R20mm</span> Dias com Chuva &ge; 20 mm (dias)</td>
                            <td class="num-cell">24,0 ± 5,1</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">15,3 ± 8,3</td>
                            <td class="num-cell">6,5 ± 3,2</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">14,8 ± 7,9</td>
                            <td class="num-cell">9,0 ± 3,9</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-rx">RX1day</span> Máx. Precipitação Diária (mm)</td>
                            <td class="num-cell">65,6</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">46,9</td>
                            <td class="num-cell">28,4</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">46,4</td>
                            <td class="num-cell">29,1</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-rx">RX5day</span> Máx. Acumulada em 5 Dias (mm)</td>
                            <td class="num-cell">124,2</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">108,9</td>
                            <td class="num-cell">79,2</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">111,5</td>
                            <td class="num-cell">84,4</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-run">CDD</span> Dias Secos Consecutivos (dias)</td>
                            <td class="num-cell">27,3</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">27,8</td>
                            <td class="num-cell">26,4</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">27,1</td>
                            <td class="num-cell">13,6</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-run">CWD</span> Dias Úmidos Consecutivos (dias)</td>
                            <td class="num-cell">10,4</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">10,9</td>
                            <td class="num-cell">125,4</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">11,0</td>
                            <td class="num-cell">128,2</td>
                        </tr>

                        <!-- ÉPOCA 2: MEIO DO SÉCULO -->
                        <tr class="epoch-row">
                            <td colspan="6">Época 2: Meio do Século (2051–2075)</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio12">BIO12</span> Precipitação Total (mm/ano)</td>
                            <td class="num-cell">1545,4 ± 207,9</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">1017,7 ± 418,1</td>
                            <td class="num-cell" style="font-weight: 700; color: #1E40AF;">1629,6 ± 174,8</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">960,4 ± 337,1</td>
                            <td class="num-cell" style="font-weight: 700; color: #DC2626;">1907,2 ± 204,2</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio13">BIO13</span> Mês Mais Chuvoso (mm)</td>
                            <td class="num-cell">313,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">201,9</td>
                            <td class="num-cell">277,6</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">211,5</td>
                            <td class="num-cell">300,2</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio13">BIO14</span> Mês Mais Seco (mm)</td>
                            <td class="num-cell">16,5</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">8,6</td>
                            <td class="num-cell">25,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">6,7</td>
                            <td class="num-cell">40,7</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio15">BIO15</span> Sazonalidade (CV %)</td>
                            <td class="num-cell">73,5%</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">74,8%</td>
                            <td class="num-cell">66,2%</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">81,6%</td>
                            <td class="num-cell">56,2%</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio16">BIO16</span> Trimestre Mais Úmido (mm)</td>
                            <td class="num-cell">745,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">448,5</td>
                            <td class="num-cell">728,7</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">438,4</td>
                            <td class="num-cell">799,6</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio16">BIO17</span> Trimestre Mais Seco (mm)</td>
                            <td class="num-cell">107,6</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">98,7</td>
                            <td class="num-cell">97,3</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">86,5</td>
                            <td class="num-cell">160,9</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-r">R10mm</span> Dias com Chuva &ge; 10 mm (dias)</td>
                            <td class="num-cell">57,5 ± 8,9</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">36,2 ± 18,3</td>
                            <td class="num-cell">47,3 ± 9,7</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">32,1 ± 13,2</td>
                            <td class="num-cell">58,2 ± 12,2</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-r">R20mm</span> Dias com Chuva &ge; 20 mm (dias)</td>
                            <td class="num-cell">24,0 ± 5,1</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">12,3 ± 8,6</td>
                            <td class="num-cell">11,5 ± 4,5</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">11,6 ± 6,6</td>
                            <td class="num-cell">15,2 ± 4,9</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-rx">RX1day</span> Máx. Precipitação Diária (mm)</td>
                            <td class="num-cell">65,6</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">40,9</td>
                            <td class="num-cell">29,8</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">47,0</td>
                            <td class="num-cell">30,3</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-rx">RX5day</span> Máx. Acumulada em 5 Dias (mm)</td>
                            <td class="num-cell">124,2</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">94,3</td>
                            <td class="num-cell">85,7</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">111,0</td>
                            <td class="num-cell">91,1</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-run">CDD</span> Dias Secos Consecutivos (dias)</td>
                            <td class="num-cell">27,3</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">35,6</td>
                            <td class="num-cell">18,5</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">32,0</td>
                            <td class="num-cell">8,2</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-run">CWD</span> Dias Úmidos Consecutivos (dias)</td>
                            <td class="num-cell">10,4</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">10,3</td>
                            <td class="num-cell">124,8</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">9,5</td>
                            <td class="num-cell">133,4</td>
                        </tr>

                        <!-- ÉPOCA 3: FINAL DO SÉCULO -->
                        <tr class="epoch-row">
                            <td colspan="6">Época 3: Final do Século (2076–2099)</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio12">BIO12</span> Precipitação Total (mm/ano)</td>
                            <td class="num-cell">1545,4 ± 207,9</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">1101,5 ± 463,6</td>
                            <td class="num-cell" style="font-weight: 700; color: #1E40AF;">1640,9 ± 183,7</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">1021,7 ± 490,9</td>
                            <td class="num-cell" style="font-weight: 700; color: #DC2626;">2365,9 ± 214,2</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio13">BIO13</span> Mês Mais Chuvoso (mm)</td>
                            <td class="num-cell">313,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">219,8</td>
                            <td class="num-cell">276,8</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">255,1</td>
                            <td class="num-cell">335,1</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio13">BIO14</span> Mês Mais Seco (mm)</td>
                            <td class="num-cell">16,5</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">14,1</td>
                            <td class="num-cell">26,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">7,3</td>
                            <td class="num-cell">59,6</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio15">BIO15</span> Sazonalidade (CV %)</td>
                            <td class="num-cell">73,5%</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">76,2%</td>
                            <td class="num-cell">65,2%</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">94,9%</td>
                            <td class="num-cell">46,5%</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio16">BIO16</span> Trimestre Mais Úmido (mm)</td>
                            <td class="num-cell">745,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">483,6</td>
                            <td class="num-cell">737,2</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">507,5</td>
                            <td class="num-cell">907,9</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio16">BIO17</span> Trimestre Mais Seco (mm)</td>
                            <td class="num-cell">107,6</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">111,2</td>
                            <td class="num-cell">108,3</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">68,7</td>
                            <td class="num-cell">258,2</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-r">R10mm</span> Dias com Chuva &ge; 10 mm (dias)</td>
                            <td class="num-cell">57,5 ± 8,9</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">39,0 ± 18,3</td>
                            <td class="num-cell">46,2 ± 9,5</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">33,3 ± 17,4</td>
                            <td class="num-cell">83,2 ± 12,7</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-r">R20mm</span> Dias com Chuva &ge; 20 mm (dias)</td>
                            <td class="num-cell">24,0 ± 5,1</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">12,4 ± 8,7</td>
                            <td class="num-cell">10,1 ± 3,9</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">13,8 ± 8,9</td>
                            <td class="num-cell">21,8 ± 6,1</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-rx">RX1day</span> Máx. Precipitação Diária (mm)</td>
                            <td class="num-cell">65,6</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">48,1</td>
                            <td class="num-cell">29,1</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">57,2</td>
                            <td class="num-cell">30,3</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-rx">RX5day</span> Máx. Acumulada em 5 Dias (mm)</td>
                            <td class="num-cell">124,2</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">105,6</td>
                            <td class="num-cell">82,7</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">124,3</td>
                            <td class="num-cell">97,6</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-run">CDD</span> Dias Secos Consecutivos (dias)</td>
                            <td class="num-cell">27,3</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">34,9</td>
                            <td class="num-cell">14,1</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">38,6</td>
                            <td class="num-cell">5,6</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-run">CWD</span> Dias Úmidos Consecutivos (dias)</td>
                            <td class="num-cell">10,4</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">9,5</td>
                            <td class="num-cell">131,4</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">9,8</td>
                            <td class="num-cell">148,9</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>"""

    comp_file = COMP_DIR / "tabela_5_html.png"
    nokap.from_html(html_content, str(comp_file), selector="#capture-wrapper", delay=1.5, zoom=3.0)
    print(f"✓ Quadro 5 salvo em comparação: {comp_file}")

    # Salva também como arquivo oficial pronto para uso
    for out_name in ["tabela_5_indicadores_bioclimaticos_extremos_300dpi.png", "quadro_5_indicadores_bioclimaticos_extremos_300dpi.png"]:
        dest = TABLES_DIR / out_name
        dest.write_bytes(comp_file.read_bytes())
        print(f"✓ Quadro 5 (completo) copiado para: {dest}")


def render_html_quadro_5_compact():
    html_content = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        background-color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    #capture-wrapper {
        background-color: #FFFFFF;
        padding: 24px 28px;
        display: inline-block;
    }
    .card {
        background: #FFFFFF;
        width: 1080px;
        border-radius: 20px;
        border: 1.5px solid #E2E8F0;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.14), 0 6px 16px -4px rgba(0, 0, 0, 0.08);
        padding: 18px 20px;
    }
    .table-container {
        border: 1.5px solid #CBD5E1;
        border-radius: 12px;
        overflow: hidden;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }
    thead tr:first-child {
        background-color: #F1F5F9;
        border-bottom: 1px solid #CBD5E1;
    }
    thead tr:nth-child(2) {
        background-color: #F8FAFC;
        border-bottom: 2.5px solid #1E40AF; /* THOR Royal Blue */
    }
    th {
        font-size: 14.5px;
        font-weight: 700;
        color: #0F172A;
        padding: 11px 10px;
        text-align: center;
        letter-spacing: -0.2px;
    }
    tbody tr.epoch-row {
        background-color: #E2E8F0;
        border-top: 2px solid #94A3B8;
        border-bottom: 1.5px solid #94A3B8;
    }
    tbody tr.epoch-row td {
        font-size: 14px;
        font-weight: 800;
        color: #0F172A;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 9px 14px;
        text-align: left;
    }
    tbody tr.data-row {
        border-bottom: 1px solid #E2E8F0;
        background-color: #FFFFFF;
    }
    tbody tr.data-row:nth-child(even) {
        background-color: #F8FAFC;
    }
    tbody tr.data-row:last-child {
        border-bottom: none;
    }
    td {
        padding: 8px 10px;
        font-size: 14px;
        color: #1E293B;
        vertical-align: middle;
        line-height: 1.4;
    }
    td.ind-cell {
        text-align: left;
        font-weight: 600;
        color: #0F172A;
    }
    td.num-cell {
        text-align: center;
        font-variant-numeric: tabular-nums;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 12.5px;
        margin-right: 6px;
        letter-spacing: 0.1px;
    }
    .b-bio12 { background: #E6F7F9; color: #008FA3; }
    .b-bio13 { background: #EFF7F0; color: #2E7D32; }
    .b-bio15 { background: #FEF9E6; color: #B45309; }
    .b-bio16 { background: #FEF1E9; color: #C2410C; }
    .b-rx { background: #FDECEC; color: #A62D2D; }
    .b-r { background: #F1F5F9; color: #0F172A; }
    .b-run { background: #EBF2FF; color: #1E40AF; }
</style>
</head>
<body>
    <div id="capture-wrapper">
        <div class="card">
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th rowspan="2" style="width: 29%; text-align: left; padding-left: 14px;">Indicador Hidroclimático</th>
                            <th rowspan="2" style="width: 19%;">Linha de Base<br><span style="font-weight: 600; font-size: 13px; color: #475569;">CHIRPS (1981–2014)</span></th>
                            <th colspan="2" style="width: 26%; border-left: 1px solid #CBD5E1; border-right: 1px solid #CBD5E1;">Cenário SSP2-4.5</th>
                            <th colspan="2" style="width: 26%;">Cenário SSP5-8.5</th>
                        </tr>
                        <tr>
                            <th style="width: 13%; border-left: 1px solid #CBD5E1; font-size: 13px; color: #334155;">CORDEX Bruto</th>
                            <th style="width: 13%; font-size: 13px; color: #1E40AF; font-weight: 800;">THOR-PIML</th>
                            <th style="width: 13%; border-left: 1px solid #CBD5E1; font-size: 13px; color: #334155;">CORDEX Bruto</th>
                            <th style="width: 13%; font-size: 13px; color: #DC2626; font-weight: 800;">THOR-PIML</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr class="epoch-row">
                            <td colspan="6">Horizonte de Longo Prazo: Final do Século (2076–2099)</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio12">BIO12</span> Precipitação Total Anual (mm/ano)</td>
                            <td class="num-cell">1545,4 ± 207,9</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">1101,5 ± 463,6</td>
                            <td class="num-cell" style="font-weight: 700; color: #1E40AF;">1640,9 ± 183,7</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">1021,7 ± 490,9</td>
                            <td class="num-cell" style="font-weight: 700; color: #DC2626;">2365,9 ± 214,2</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio13">BIO13</span> Mês Mais Chuvoso (mm)</td>
                            <td class="num-cell">313,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">219,8</td>
                            <td class="num-cell">276,8</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">255,1</td>
                            <td class="num-cell">335,1</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio13">BIO14</span> Mês Mais Seco (mm)</td>
                            <td class="num-cell">16,5</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">14,1</td>
                            <td class="num-cell">26,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">7,3</td>
                            <td class="num-cell">59,6</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio15">BIO15</span> Sazonalidade da Precipitação (CV %)</td>
                            <td class="num-cell">73,5%</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">76,2%</td>
                            <td class="num-cell">65,2%</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">94,9%</td>
                            <td class="num-cell">46,5%</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio16">BIO16</span> Trimestre Mais Úmido (mm)</td>
                            <td class="num-cell">745,0</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">483,6</td>
                            <td class="num-cell">737,2</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">507,5</td>
                            <td class="num-cell">907,9</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-bio16">BIO17</span> Trimestre Mais Seco (mm)</td>
                            <td class="num-cell">107,6</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">111,2</td>
                            <td class="num-cell">108,3</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">68,7</td>
                            <td class="num-cell">258,2</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-r">R10mm</span> Dias com Chuva &ge; 10 mm (dias/ano)</td>
                            <td class="num-cell">57,5 ± 8,9</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">39,0 ± 18,3</td>
                            <td class="num-cell">46,2 ± 9,5</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">33,3 ± 17,4</td>
                            <td class="num-cell">83,2 ± 12,7</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-r">R20mm</span> Dias com Chuva &ge; 20 mm (dias/ano)</td>
                            <td class="num-cell">24,0 ± 5,1</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">12,4 ± 8,7</td>
                            <td class="num-cell">10,1 ± 3,9</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">13,8 ± 8,9</td>
                            <td class="num-cell">21,8 ± 6,1</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-rx">RX1day</span> Máx. Precipitação Diária (mm)</td>
                            <td class="num-cell">65,6</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">48,1</td>
                            <td class="num-cell">29,1</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">57,2</td>
                            <td class="num-cell">30,3</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-rx">RX5day</span> Máx. Acumulada em 5 Dias (mm)</td>
                            <td class="num-cell">124,2</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">105,6</td>
                            <td class="num-cell">82,7</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">124,3</td>
                            <td class="num-cell">97,6</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-run">CDD</span> Dias Secos Consecutivos (dias)</td>
                            <td class="num-cell">27,3</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">34,9</td>
                            <td class="num-cell">14,1</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">38,6</td>
                            <td class="num-cell">5,6</td>
                        </tr>
                        <tr class="data-row">
                            <td class="ind-cell"><span class="badge b-run">CWD</span> Dias Úmidos Consecutivos (dias)</td>
                            <td class="num-cell">10,4</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">9,5</td>
                            <td class="num-cell">131,4</td>
                            <td class="num-cell" style="border-left: 1px solid #E2E8F0;">9,8</td>
                            <td class="num-cell">148,9</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>"""

    comp_file = COMP_DIR / "tabela_5_compacta_html.png"
    nokap.from_html(html_content, str(comp_file), selector="#capture-wrapper", delay=1.5, zoom=3.0)
    print(f"✓ Quadro 5 (compacto) salvo em comparação: {comp_file}")

    for out_name in ["tabela_5_final_seculo_300dpi.png", "quadro_5_final_seculo_300dpi.png"]:
        dest = TABLES_DIR / out_name
        dest.write_bytes(comp_file.read_bytes())
        print(f"✓ Quadro 5 (compacto) copiado para: {dest}")


if __name__ == "__main__":
    render_html_quadro_3()
    render_html_quadro_4()
    render_html_quadro_5()
    render_html_quadro_5_compact()
