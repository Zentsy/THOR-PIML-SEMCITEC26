# THOR-PIML: Physics-Informed Hybrid Modeling for Bioclimatic Indicator Projections and Seasonal Bias Correction in Guarulhos-SP under SSP Scenarios (2026–2099)

🌐 [Versão em Português](README.md) | **English Version**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-3rd%20Place%20(Track%202)-green.svg)]()
[![License: Academic Review Only](https://img.shields.io/badge/License-Academic%20Review%20Only-lightgrey.svg)](LICENSE)
[![Ownership](https://img.shields.io/badge/Ownership-Authors%20%26%20Advisors-003E47.svg)]()

> ⚠️ **Academic Submission & Technical Verification Notice:**  
> This repository is exclusively dedicated to reproducibility auditing, technical figure inspection, and experimental verification for the examination committee and scientific board of SEMCITEC 2026.  
> 
> **Authors & Ownership:** Victor Gabriel de Jesus Cursi & Caroline Raquel Silva Macris  
> **Advisors:** M.Sc. Luiz Carlos Rodrigues de Medeiros & Prof. Maria Cristina Tagliari Diniz  
> **Academic Affiliation:** Undergraduate Students at Centro Universitário Eniac (Guarulhos, SP, Brazil)  
> **Intellectual Property:** Strictly and exclusively owned by the Authors and Advisors. All rights reserved on code, model representations, and formulations. Commercial or unauthorized institutional redistribution is strictly prohibited.

---

## 📌 Overview

This repository provides reproducible code routines, figures, tables, and secular climate projection datasets (2026–2099) for the **THOR-PIML** (*Taylor-Hurdle Optimized Regional Physics-Informed Machine Learning*) framework applied to Guarulhos-SP, Brazil.

The research investigates a physics-informed hybrid neural architecture to perform daily statistical downscaling and bias correction of the regional **CORDEX SAM-22 (Eta-HadGEM2-ES)** model under shared socioeconomic pathways **SSP2-4.5** (stabilization) and **SSP5-8.5** (high emissions), evaluating:
1. **Seasonal bias recovery** against the chronic artificial dry bias of the driving regional model;
2. **Thermodynamic physical consistency** enforced by an atmospheric precipitable water ceiling based on integrated total column water vapour ($\text{TCWV}$);
3. **Official Bioclimatic Indicators (BIO12 to BIO19)** and **Hydrological Extreme Indices (ETCCDI)** across 4 climatological epochs:
   * *Historical Baseline:* 1981–2014 (CHIRPS / ERA5-Land)
   * *Near Future:* 2026–2050
   * *Mid-Century:* 2051–2075
   * *Late Century:* 2076–2099

---

## 🏗️ Model Architecture (THOR-PIML V8)

The neural formulation decouples synoptic atmospheric dynamics from localized precipitation probability through specialized components:

* **2D Synoptic Encoder:** Spatial convolutions over pressure levels ($z_{500}, u_{700}, v_{700}, q_{700}, w_{500}$) across a $25 \times 33$ regional grid ($6^\circ \times 8^\circ$).
* **Bi-Branch Temporal Trunk ($T = 30\text{ days}$):**
  * *ResLSTM Branch:* 2-layer residual LSTM capturing multi-scale slow soil moisture memory;
  * *Multi-Scale TCN Branch:* Dilated causal convolutions ($k \in \{3,5,7\}, d \in \{1,2,4,8\}$) for frontal triggers and convective lines;
  * *Adaptive Gated Fusion:* Dynamic soft gating between inertial soil memory and fast atmospheric triggers;
  * *Masked Causal Attention:* Scaled dot-product attention with 8 heads and causal lower-triangular mask.
* **Hurdle Dual-Head Decoder:**
  * *Occurrence Head:* $\text{Sigmoid}(z_{\text{occ}}) \to p_{\text{occ}} \in [0, 1]$ (WMO threshold $\ge 1.0\text{ mm}$);
  * *Intensity Head:* $\text{Softplus}(z_{\text{int}}) \to \mu_{\text{int}} \ge 0\text{ mm/day}$;
  * *Prediction:* $\hat{y} = p_{\text{occ}} \times \mu_{\text{int}}$.
* **Differentiable Physics Barrier (PIML):**
  * Atmospheric precipitable water ceiling: $W_{\max} = 4.0 \times \text{TCWV}$;
  * Quadratic smooth penalty: $\mathcal{L}_{\text{phys}} = \lambda_{\text{phys}} \cdot [\text{Softplus}(\hat{y} - W_{\max})]^2$ (Zero physical violations, $0.00\%$).

---

## 🔒 Intellectual Property & Proprietary Model Access

> [!IMPORTANT]
> **Proprietary Technology and Intellectual Property Notice:**  
> The trained neural weight checkpoints (`.pt`), the deep ResLSTM-TCN hybrid architecture, the thermodynamic loss formulation, and all proprietary modeling/training code constitute the **exclusive and non-transferable intellectual property of the AUTHORS and ADVISORS**. Centro Universitário ENIAC serves strictly as the academic affiliation of the student authors, holding no patrimonial ownership, patent claims, or technological rights over the model.
> 
> To safeguard the technology against unauthorized commercial exploitation or infringement, raw binary models and training routines are retained under private custody. Public bodies and researchers seeking technical auditing, scientific cooperation, or licensing should contact the authors directly.

---

## 📁 Repository Structure & Provided Artifacts

```text
THOR-PIML-Semcitec2026/
├── data/                     # Input datasets
│   ├── ground_truth_guarulhos_daily_v3.csv  # Observed baseline (1981–2026)
│   └── cordex/               # Daily series extracted from CORDEX SAM-22
├── results/                  # Official outputs and products
│   ├── figures/              # High-resolution figures (300 DPI)
│   │   ├── fig1_semcitec_prcptot_projections_300dpi.png
│   │   ├── fig2_semcitec_seasonal_cycle_epochs_300dpi.png
│   │   ├── fig3_semcitec_fdc_extreme_tail_300dpi.png
│   │   ├── fig5_semcitec_etccdi_urban_extremes_300dpi.png
│   │   ├── fig6_external_reference_monthly_anomaly_300dpi.png
│   │   └── fig_thor_v8_architecture.png
│   ├── tables/               # Bioclimatic indicators & extreme indices tables
│   │   ├── quadro_3_equacoes_piml_metodologia_300dpi.png
│   │   ├── quadro_4_indicadores_bioclimaticos_etccdi_300dpi.png
│   │   ├── quadro_5_final_seculo_300dpi.png
│   │   ├── quadro_5_indicadores_bioclimaticos_extremos_300dpi.png
│   │   └── tabela_resumo_semcitec_2026.md
│   └── projections/          # Downscaled daily projection CSVs (2026–2099)
│       ├── cordex_ssp245_downscaled_2026_2099.csv
│       ├── cordex_ssp585_downscaled_2026_2099.csv
│       └── projections_summary.json
├── requirements.txt          # Environment dependencies
└── README.md
```

## 📊 Key Findings

* **Overcoming Artificial Dry Bias:** While the raw regional CORDEX model predicts an artificial drop in annual precipitation ($1,021.7\text{ mm/year}$ at late century under SSP5-8.5), THOR-PIML recovers the thermodynamic Clausius-Clapeyron response linked to vapor availability ($\text{TCWV}$), reaching $2,365.9\text{ mm/year}$ with a concentrated summer peak (BIO16 up to $907.9\text{ mm}$).
* **Physical Compliance:** 0.00% violations of the Clausius-Clapeyron barrier across 53,280 independent daily predictions.
* **Extremes Stabilization:** The Hurdle dual-head stabilized point daily peaks ($RX1\text{day}$) at $\sim 29\text{--}30\text{ mm/day}$, while moderate accumulated precipitation events ($R10\text{mm}$ and $R20\text{mm}$) exhibited steady growth.

---

## 📄 Terms of Use & Academic License

This repository is made available under the **Restricted Academic Evaluation License (All Rights Reserved)**.

Commercial, governmental, or unauthorized institutional redistribution without prior express written permission from the authors and advisors is strictly prohibited.

<!--
Official conference citation:
@inproceedings{thor_piml_semcitec2026,
  author    = {Cursi, Victor Gabriel de Jesus and Macris, Caroline Raquel Silva},
  title     = {THOR-PIML: Physics-Informed Hybrid Modeling for Bioclimatic Indicator Projections and Seasonal Bias Correction in Guarulhos-SP under SSP Scenarios (2026–2099)},
  booktitle = {Proceedings of SEMCITEC 2026},
  year      = {2026},
  address   = {Guarulhos, Brazil}
}
-->
