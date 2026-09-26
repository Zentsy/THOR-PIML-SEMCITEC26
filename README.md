# THOR-PIML — SEMCITEC 2026

**Projeção do Regime Pluviométrico e Indicadores Bioclimáticos sob Cenários SSP (2026–2100) em Guarulhos-SP via Downscaling com o Modelo THOR-PIML**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Visão Geral

Este repositório contém o código-fonte, pesos de modelo congelados, rotinas de inferência e dados de projeção climática secular (2026–2099) do projeto **THOR-PIML** (*Taylor-Hurdle Optimized Regional Physics-Informed Machine Learning*) para o município de Guarulhos-SP.

O trabalho investiga a aplicação de uma arquitetura neural híbrida com física informada para realizar o *downscaling* estatístico das projeções do modelo regional **CORDEX SAM-22 (Eta-HadGEM2-ES)** sob os cenários socioeconômicos **SSP2-4.5** (estabilização) e **SSP5-8.5** (altas emissões), avaliando:
1. **Recuperação da sazonalidade** frente ao viés de secamento (*dry bias*) sistemático do modelo forçador regional;
2. **Consistência termodinâmica** via restrição de Clausius-Clapeyron baseada na coluna de vapor d'água atmosférico integrado ($\text{TCWV}$);
3. **Indicadores Bioclimáticos Oficiais (BIO12 a BIO19)** e **Índices de Extremos Hidrológicos (ETCCDI)** ao longo de 4 épocas climatológicas:
   * *Baseline Histórica:* 1981–2014 (CHIRPS / ERA5-Land)
   * *Futuro Próximo:* 2026–2050
   * *Meio do Século:* 2051–2075
   * *Final do Século:* 2076–2099

---

## 🏗️ Arquitetura do Modelo (THOR-PIML V8)

A arquitetura desacopla a dinâmica sinótica e a probabilidade de chuva por meio de blocos especializados:

* **Codificador Sinótico 2D:** Convoluções espaciais sobre campos de pressão em níveis atmosféricos ($z_{500}, u_{700}, v_{700}, q_{700}, w_{500}$) em grade $25 \times 33$ ($6^\circ \times 8^\circ$).
* **Tronco Temporal Bi-Branch ($T = 30\text{ dias}$):**
  * *Ramo ResLSTM:* LSTM residual de 2 camadas com memória hidrológica multiescala;
  * *Ramo TCN Multiescala:* Convoluções causais dilatadas ($k \in \{3,5,7\}, d \in \{1,2,4,8\}$) para captura de frentes frias e linhas de instabilidade;
  * *Fusão Adaptativa Gated:* Ponderação dinâmica entre memória de solo e gatilhos convectivos rápidos;
  * *Atenção Causal Mascarada:* SDPA com 8 cabeças de atenção e máscara triangular inferior.
* **Decodificador Hurdle Dual-Head:**
  * *Cabeça de Ocorrência:* $\text{Sigmoid}(z_{\text{occ}}) \to p_{\text{occ}} \in [0, 1]$ (limiar WMO $\ge 1{,}0\text{ mm}$);
  * *Cabeça de Intensidade:* $\text{Softplus}(z_{\text{int}}) \to \mu_{\text{int}} \ge 0\text{ mm/dia}$;
  * *Predição:* $\hat{y} = p_{\text{occ}} \times \mu_{\text{int}}$.
* **Barreira Física Diferenciável (PIML):**
  * Teto termodinâmico convectivo: $W_{\max} = 4{,}0 \times \text{TCWV}$;
  * Função de penalização quadrática suave: $\mathcal{L}_{\text{phys}} = \lambda_{\text{phys}} \cdot [\text{Softplus}(\hat{y} - W_{\max})]^2$ (Zero violações físicas, $0{,}00\%$).

---

## 📁 Estrutura do Repositório

```bash
THOR-PIML-Semcitec2026/
├── checkpoints/              # Checkpoints pré-treinados e congelados
│   ├── v8_hybrid_seed42.pt   # Pesos oficiais do modelo THOR-PIML V8
│   └── scaler_v7.json        # Parâmetros de escalonamento robusto (treino)
├── data/                     # Dados de entrada
│   ├── ground_truth_guarulhos_daily_v3.csv  # Linha de base observada (1981–2026)
│   └── cordex/               # Séries diárias extraídas do CORDEX SAM-22
├── results/                  # Saídas e produtos oficiais
│   ├── figures/              # Figuras em alta resolução (300 DPI)
│   │   ├── fig1_semcitec_prcptot_projections_300dpi.png
│   │   ├── fig2_semcitec_seasonal_cycle_epochs_300dpi.png
│   │   ├── fig3_semcitec_fdc_extreme_tail_300dpi.png
│   │   ├── fig5_semcitec_etccdi_urban_extremes_300dpi.png
│   │   ├── fig6_external_reference_monthly_anomaly_300dpi.png
│   │   └── fig_thor_v8_architecture.png
│   ├── tables/               # Quadros de indicadores bioclimáticos e extremos
│   │   ├── quadro_3_equacoes_piml_metodologia_300dpi.png
│   │   ├── quadro_4_indicadores_bioclimaticos_etccdi_300dpi.png
│   │   ├── quadro_5_final_seculo_300dpi.png
│   │   ├── quadro_5_indicadores_bioclimaticos_extremos_300dpi.png
│   │   └── tabela_resumo_semcitec_2026.md
│   └── projections/          # CSVs com as séries diárias projetadas (2026–2099)
│       ├── cordex_ssp245_downscaled_2026_2099.csv
│       ├── cordex_ssp585_downscaled_2026_2099.csv
│       └── projections_summary.json
├── src/                      # Código-fonte modular
│   ├── calc_bioclimatic_indices.py
│   ├── cordex_preprocessor.py
│   ├── generate_semcitec_figures.py
│   ├── generate_semcitec_tables.py
│   ├── infer_cordex_v8.py
│   ├── model.py
│   └── physics_loss.py
├── tests/                    # Testes de integração do pipeline
├── run_semcitec_pipeline.py  # Script de execução e reprodução completa
├── requirements.txt          # Dependências do ambiente
└── README.md
```

---

## 🚀 Instalação e Reprodução

### 1. Clonar o Repositório e Criar Ambiente Virtual

```bash
git clone https://github.com/<SEU-USUARIO>/<SEU-REPOSITORIO>.git
cd <SEU-REPOSITORIO>

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Executar o Pipeline Completo

Para reproduzir a inferência secular, o cálculo dos indicadores bioclimáticos (BIO12–BIO19) e a geração de todas as figuras e tabelas de publicação em 300 DPI:

```bash
python run_semcitec_pipeline.py
```

---

## 📊 Principais Resultados

* **Superação do Viés de Secamento:** Enquanto o modelo CORDEX regional bruto projeta queda artificial na precipitação anual ($1.021{,}7\text{ mm/ano}$ no final do século no SSP5-8.5), o THOR-PIML recupera a resposta termodinâmica de maior disponibilidade de vapor ($\text{TCWV}$), alcançando $2.365{,}9\text{ mm/ano}$ com pico de verão concentrado (BIO16 de até $907{,}9\text{ mm}$).
* **Conformidade Física:** 0,00% de violações da barreira de Clausius-Clapeyron ao longo de 53.280 predições diárias independentes.
* **Estabilização de Extremos:** A cabeça Hurdle conteve o limiar diário pontual ($RX1\text{day}$) em $\sim 29\text{ a }30\text{ mm/dia}$, enquanto eventos acumulados severos ($R10\text{mm}$ e $R20\text{mm}$) apresentaram crescimento contínuo.

---

## 📄 Licença e Citação

Este projeto é disponibilizado sob a licença [MIT](LICENSE).

Se utilizar este código ou os dados gerados em sua pesquisa, por favor cite:

```bibtex
@inproceedings{thor_piml_semcitec2026,
  author    = {Silva, L. et al.},
  title     = {THOR-PIML: Arquitetura Neural Híbrida com Física Informada para Projeção do Regime Pluviométrico sob Cenários SSP (2026–2100) em Guarulhos-SP},
  booktitle = {Anais da SEMCITEC 2026},
  year      = {2026},
  address   = {Guarulhos, Brasil}
}
```
