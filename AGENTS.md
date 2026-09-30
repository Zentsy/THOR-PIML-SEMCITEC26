# AGENTS.md — THOR-PIML (SEMCITEC 2026)

Physics-Informed Machine Learning for Regional Precipitation Downscaling and Bioclimatic Projections (Guarulhos-SP, Brazil).

**Canonical Name:** THOR-PIML (*Taylor-Hurdle Optimized Regional Physics-Informed Machine Learning*)  
**Paper Title:** *"THOR-PIML: Modelagem Híbrida com Física Informada para Projeção de Indicadores Bioclimáticos e Correção de Viés Sazonal em Guarulhos-SP sob Cenários SSP (2026–2099)"*  
**English Title:** *"THOR-PIML: Physics-Informed Hybrid Modeling for Bioclimatic Indicator Projections and Seasonal Bias Correction in Guarulhos-SP under SSP Scenarios (2026–2099)"*  
**Target Venue:** SEMCITEC 2026 (Semana de Ciência e Tecnologia do IFSP Guarulhos)  
**Simulation Horizon:** 2026–2099 (74 anos, calendário 360_day = 26.640 passos/cenário, 53.280 passos totais)

---

## 1. Architecture & Neural Formulation (THOR-PIML V8)

- **2D Synoptic Encoder:** Spatial convolutions over ERA5 pressure levels ($z_{500}, u_{700}, v_{700}, q_{700}, w_{500}$) in a $25 \times 33$ grid ($6^\circ \times 8^\circ$) $\to \mathbf{Z}_{\text{syn}} \in \mathbb{R}^{30 \times 64}$.
- **Surface Predictors:** 138 engineered features (local surface meteorology, multi-scale temporal lags $t-1$ to $t-14$, and CORDEX regional surface forcings).
- **Bi-Branch Temporal Trunk ($T = 30\text{ dias}$, 148 features):**
  - **ResLSTM Branch:** 2-layer LSTM (hidden 128) + Residual Skip Connection + LayerNorm for 14-day hydrological memory.
  - **Multi-Scale TCN Branch:** Causal dilated Conv1D ($k \in \{3,5,7\}, d \in \{1,2,4,8\}$) for frontal triggers and squall lines.
  - **Adaptive Gated Fusion:** Dynamic soft gating $\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \mathbf{h}_{\text{lstm}} + (1 - \mathbf{g}) \odot \mathbf{h}_{\text{tcn}}$.
  - **Causal Attention:** 8-head SDPA attention with lower-triangular causal mask $\to \mathbf{h}_T \in \mathbb{R}^{128}$.
- **Hurdle Dual-Head Predictor:**
  - **Occurrence Head:** $\text{Linear}(128 \to 64) \to \text{SiLU} \to \text{Linear}(64 \to 1) \to \text{Sigmoid} \to p_{\text{occ}} \in [0, 1]$ (WMO threshold $\ge 1{,}0\text{ mm/dia}$).
  - **Intensity Head:** $\text{Linear}(128 \to 96) \to \text{SiLU} \to \text{Linear}(96 \to 1) \to \text{Softplus} \to \mu_{\text{int}} \ge 0\text{ mm/dia}$.
  - **Final Output:** $\hat{y} = p_{\text{occ}} \times \mu_{\text{int}}$.
- **Thermodynamic Physics Constraint (PIML):**
  - **Atmospheric Precipitable Water Ceiling:** $W_{\max} = 4{,}0 \times \text{TCWV}$ (denormalized real Total Column Water Vapour in mm).
  - **Quadratic Softplus Barrier:** $\mathcal{L}_{\text{phys}} = \lambda_{\text{phys}} \cdot \left[ \text{Softplus}(\hat{y} - W_{\max}) \right]^2$
  - **Guarantee:** Zero dead gradients, convex penalty above boundary, **0.00% physical violation rate** across all 53,280 projection steps.

---

## 2. Scientific Invariants & Guardrails

1. **Zero Data Leakage:** `RobustClimateScaler` fits strictly on historical training data (1981–2014); validation, test, and future projection sets only get `.transform()`. No future lookahead, no `year_norm`, no bidirectional temporal leakage.
2. **No Silent Mocking:** All ingestion scripts and data loaders fail loudly if raw NetCDFs or ground truth observations are missing.
3. **Strict Scope Separation:**
   - *CONIC Scope:* Model architecture, loss function formulations, ablation studies, and historical test benchmarks (1981–2026).
   - *SEMCITEC Scope:* Downscaling CORDEX SAM-22 (Eta-HadGEM2-ES), bias correction, climate projection under SSP2-4.5 and SSP5-8.5 (2026–2099), bioclimatic shifts (BIO12–BIO19), ETCCDI extreme indices, CMIP6 external benchmark comparison, and urban drainage implications.
4. **Publication Standards:** All architecture and evaluation diagrams follow high-density publication layouts at 300 DPI with standardized color palettes (Black Baseline, Slate Gray Raw CORDEX, Royal Blue THOR SSP2-4.5, Crimson Red THOR SSP5-8.5).

---

## 3. Climatological Epochs & Indicators

### 4 Climatological Epochs
1. **Linha de Base Histórica:** 1981–2014 (34 anos — CHIRPS / ERA5-Land)
2. **Futuro Próximo:** 2026–2050 (25 anos)
3. **Meio do Século:** 2051–2075 (25 anos)
4. **Final do Século:** 2076–2099 (24 anos)

### Official Bioclimatic Indicators (WMO / WorldClim)
- **BIO12 (Precipitação Total Anual):** $\text{PRCPTOT} = \sum_{m=1}^{12} P_m$ (mm/ano)
- **BIO13 (Precipitação do Mês Mais Chuvoso):** $\max(P_m)$ (mm)
- **BIO14 (Precipitação do Mês Mais Seco):** $\min(P_m)$ (mm)
- **BIO15 (Sazonalidade da Precipitação):** $CV = (\sigma_m / \mu_m) \times 100$ (%)
- **BIO16 (Precipitação do Trimestre Mais Úmido):** Acumulado de verão (DJF em mm)
- **BIO17 (Precipitação do Trimestre Mais Seco):** Acumulado de inverno (JJA em mm)
- **BIO18 (Precipitação do Trimestre Mais Quente):** Chuva no pico térmico sazonal
- **BIO19 (Precipitação do Trimestre Mais Frio):** Chuva no período frio do ano

### Extreme Precipitation Indices (ETCCDI)
- **Magnitudes & Dias Críticos:** $R10\text{mm}$, $R20\text{mm}$, $RX1\text{day}$, $RX5\text{day}$, $R95p$, $R99p$.
- **Persistência Seca/Úmida:** $CDD$ (*Consecutive Dry Days*), $CWD$ (*Consecutive Wet Days*).

---

## 4. Current Project Checkpoint & Verified Artifacts

- **Data Status:**
  - 4 CORDEX NetCDFs (INPE-Eta at $0{,}20^\circ$, driven by HadGEM2-ES) covering 2006–2099 downloaded and verified.
  - Subgrid $3 \times 3$ extracted and centered at Guarulhos-SP ($-23{,}43^\circ\text{S}, -46{,}47^\circ\text{W}$).
  - Inferred dataset: 53,280 daily projection steps (2026–2099) saved in `results/projections/`.
- **Pretrained Checkpoint:** Frozen weights `checkpoints/v8_hybrid_seed42.pt` and `checkpoints/scaler_v7.json`.
- **Publication Figures:** 5 figures at 300 DPI generated and verified in `results/figures/`:
  - `fig1_semcitec_prcptot_projections_300dpi.png` (BIO12 Annual Trends)
  - `fig2_semcitec_seasonal_cycle_epochs_300dpi.png` (Monthly Sazonality Across Epochs)
  - `fig3_semcitec_fdc_extreme_tail_300dpi.png` (Flow Duration Curve & Extremes Tail)
  - `fig5_semcitec_etccdi_urban_extremes_300dpi.png` (ETCCDI Multi-Panel Comparison)
  - `fig6_external_reference_monthly_anomaly_300dpi.png` (CMIP6 CCKP Multi-Model Benchmark Audit)
- **Manuscript Text:**
  - Sections 4.1, 4.2, 4.3, 5, Resumo/Abstract, and Methodology Tables written, formatted, and strictly aligned with ABNT NBR 10520 and NBR 6023.

---

## 5. Key Scientific Findings & Limitations

1. **Correção do Viés de Secamento:** O CORDEX bruto projeta secamento irreal sob SSP5-8.5 ($960\text{ mm/ano}$ no meio do século), decorrente do viés histórico do HadGEM2-ES. O THOR-PIML recupera a resposta termodinâmica de Clausius-Clapeyron, projetando $1.907\text{ mm/ano}$ e $2.366\text{ mm/ano}$ no final do século.
2. **Preservação do Regime Sazonal:** O modelo preserva o ciclo unimodal típico do Sudeste (verão úmido DJF vs inverno seco JJA), evitando o achatamento irreal do modelo regional forçador.
3. **Estabilidade de Extremos em 24h:** O modelo estabiliza o $RX1\text{day}$ em $\sim 30\text{ mm/dia}$, eliminando alucinações numéricas e divergências explosivas, oferecendo parâmetros seguros para dimensionamento de microdrenagem urbana.
4. **Limitação Metodológica Reconhecida:** Valores elevados de persistência úmida ($CWD \approx 130\text{--}148\text{ dias}$) refletem a forçante regional de baixa resolução (sinal ininterrupto de umidade/garoa), devendo ser analisados criticamente por projetistas hidráulicos.
