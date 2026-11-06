# THOR-PIML: Modelagem Híbrida com Física Informada para Projeção de Indicadores Bioclimáticos e Correção de Viés Sazonal em Guarulhos-SP sob Cenários SSP (2026–2099)

🌐 **Versão em Português** | [English Version](README_EN.md)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status: Under Review](https://img.shields.io/badge/Status-3°%20Place-green.svg)]()
[![License: Academic Review Only](https://img.shields.io/badge/License-Academic%20Review%20Only-lightgrey.svg)](LICENSE)
[![Propriedade](https://img.shields.io/badge/Propriedade-Autores%20%26%20Orientadores-003E47.svg)]()

> ⚠️ **Aviso de Submissão e Verificação Acadêmica:**  
> Este repositório destina-se exclusivamente à auditoria de reprodutibilidade, consulta técnica de figuras e verificação experimental por parte da banca examinadora e do comitê científico da SEMCITEC 2026.  
> 
> **Autores e Titularidade:** Victor Gabriel de Jesus Cursi & Caroline Raquel Silva Macris
> **Orientadores:** Me. Luiz Carlos Rodrigues de Medeiros & prof. Maria Cristina Tagliari Diniz
> **Filiação Acadêmica:** Graduandos no Centro Universitário Eniac (Guarulhos, SP)  
> **Propriedade Intelectual:** Exclusiva dos Autores e Orientadores. Todos os direitos sobre os códigos, pesos e formulações estão reservados. Proibida a redistribuição comercial ou institucional não autorizada.

---

## 📌 Visão Geral

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

## 🔒 Propriedade Intelectual & Acesso Restrito ao Modelo

> [!IMPORTANT]
> **Aviso de Proteção Intelectual e Titularidade Tecnológica:**  
> Os pesos neurais treinados (.pt), a parametrização profunda do tronco híbrido ResLSTM-TCN, a formulação termodinâmica de perda e o código-fonte proprietário de modelagem e treinamento constituem **propriedade intelectual exclusiva e intransferível dos AUTORES e ORIENTADORES**. O Centro Universitário ENIAC figura estritamente como filiação acadêmica dos alunos graduandos, não possuindo propriedade patrimonial, direitos autorais ou tecnológicos sobre o modelo.
> 
> Para resguardar a tecnologia desenvolvida e evitar apropriação indevida ou exploração comercial não autorizada, os arquivos de modelo bruto e rotinas de treino foram retidos sob custódia privada. Pesquisadores, órgãos públicos e entidades interessadas em auditoria técnica, cooperações científicas ou licenciamento institucional devem entrar em contato formalmente com os autores.

---

## 📁 Estrutura do Repositório & Produtos Disponibilizados

```text
THOR-PIML-Semcitec2026/
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
├── requirements.txt          # Dependências do ambiente
└── README.md
```

## 📊 Principais Resultados

* **Superação do Viés de Secamento:** Enquanto o modelo CORDEX regional bruto projeta queda artificial na precipitação anual ($1.021{,}7\text{ mm/ano}$ no final do século no SSP5-8.5), o THOR-PIML recupera a resposta termodinâmica de maior disponibilidade de vapor ($\text{TCWV}$), alcançando $2.365{,}9\text{ mm/ano}$ com pico de verão concentrado (BIO16 de até $907{,}9\text{ mm}$).
* **Conformidade Física:** 0,00% de violações da barreira de Clausius-Clapeyron ao longo de 53.280 predições diárias independentes.
* **Estabilização de Extremos:** A cabeça Hurdle conteve o limiar diário pontual ($RX1\text{day}$) em $\sim 29\text{ a }30\text{ mm/dia}$, enquanto eventos acumulados severos ($R10\text{mm}$ e $R20\text{mm}$) apresentaram crescimento contínuo.

---

## 📄 Termo de Uso e Licença Acadêmica

Este repositório é disponibilizado sob **Licença de Avaliação Acadêmica Exclusiva (Todos os Direitos Reservados)**.

<!--
Citação oficial reservada para os anais:
@inproceedings{thor_piml_semcitec2026,
  author    = {Cursi, Victor Gabriel de Jesus and Macris, Caroline Raquel Silva},
  title     = {THOR-PIML: Modelagem Híbrida com Física Informada para Projeção de Indicadores Bioclimáticos e Correção de Viés Sazonal em Guarulhos-SP sob Cenários SSP (2026–2099)},
  booktitle = {Anais da SEMCITEC 2026},
  year      = {2026},
  address   = {Guarulhos, Brasil}
}
-->
