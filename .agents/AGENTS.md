# AGENTS.md — THOR-PIML

Physics-Informed ML for regional precipitation downscaling (Guarulhos, Brazil).
**Canonical Name:** THOR-PIML (*Taylor-Hurdle Optimized Regional Physics-Informed Machine Learning*)
**Paper Title:** *"Auditoria de transferência de domínio e correção de vieses sazonais na precipitação de Guarulhos-SP sob cenários SSP"*

## Architecture & Neural Formulation
- **2D Synoptic Encoder:** Spatial convolutions over ERA5 pressure levels ($z_{500}, u_{700}, v_{700}, q_{700}, w_{500}$) in a $25 \times 33$ grid ($6^\circ \times 8^\circ$) $\to \mathbf{Z}_{\text{syn}} \in \mathbb{R}^{30 \times 64}$.
- **Surface Predictors:** 138 V8 features, including local predictors, lagged variables and engineered CORDEX features.
- **Bi-Branch Temporal Trunk ($30 \times 148$):**
  - **ResLSTM Branch:** 2-layer LSTM (hidden 128) + Residual Skip + LayerNorm for 14-day hydrological memory.
  - **Multi-Scale TCN Branch:** Causal dilated Conv1D ($k \in \{3,5,7\}, d \in \{1,2,4,8\}$) for frontal triggers and squall lines.
  - **Adaptive Gated Fusion:** $\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \mathbf{h}_{\text{lstm}} + (1 - \mathbf{g}) \odot \mathbf{h}_{\text{tcn}}$.
  - **Causal Attention:** 8-head SDPA attention with lower-triangular causal mask $\to \mathbf{h}_T \in \mathbb{R}^{128}$.
- **Hurdle Dual-Head Predictor:**
  - Occurrence Head: $\text{Linear}(128 \to 64) \to \text{SiLU} \to \text{Linear}(64 \to 1) \to \text{Sigmoid} \to p_{\text{occ}} \in [0, 1]$ (WMO threshold $\ge 1.0\text{ mm}$).
  - Intensity Head: $\text{Linear}(128 \to 96) \to \text{SiLU} \to \text{Linear}(96 \to 1) \to \text{Softplus} \to \mu_{\text{int}} \ge 0\text{ mm/d}$.
  - Target Prediction: $\hat{y} = p_{\text{occ}} \times \mu_{\text{int}}$.
- **Thermodynamic Physics Constraint (PIML):**
  - Atmospheric precipitable water ceiling: $W_{\max} = 4.0 \times \text{TCWV}$ (denormalized real TCWV in mm).
  - Quadratic Softplus barrier: $\mathcal{L}_{\text{phys}} = \lambda_{\text{phys}} \cdot [\text{Softplus}(\hat{y} - W_{\max})]^2$ (Zero dead gradients, 0.00% physical violation rate).

## Hard Rules & Scientific Invariants
1. **Zero Data Leakage:** `RobustClimateScaler` fits on train only; val, test, and future projections only get `.transform()`. No `year_norm`, no future lookahead, no bidirectional encoders over time.
2. **No Silent Mock Fallback:** All data loaders must fail loudly if real data is missing. Mocking requires explicit flags.
3. **Repository Scope & Cleanliness:** Never mix conference scopes (CONIC methodology vs SEMCITEC SSP projections). Public/official paper repositories must contain strictly canonical code, weights, ground truth, PNG figures, and 1-command reproduction scripts (`reproduce_paper_results.py`).
4. **Publication Visuals:** All architecture and evaluation diagrams must use high-density, horizontal (~3:1 aspect ratio) publication layouts at 300 DPI.

## Current Active Scope (SEMCITEC 2026)
- **Topic:** Auditoria de transferência de domínio: capacidade do THOR-PIML de recuperar a estrutura sazonal local ao receber CORDEX futuro, sem confundir plausibilidade sazonal com validação quantitativa da amplitude.
- **Formato do Artigo:** Relatório completo / Artigo expandido de ~25 páginas.
- **Narrativa & Foco Científico:**
  - **Foco em Agregações Estatísticas e Regimes:** Não focar em previsão determinística de dia individual (o clima futuro é estocástico/caótico). O modelo foca em **acumulados mensais, sazonais (DJF vs JJA), curvas de permanência, indicadores pluviométricos e distribuição de extremos**.
  - **Cenários Avaliados (2026–2099):** SSP2-4.5 (estabilização intermediária) vs SSP5-8.5 (altas emissões / extremo).
  - **Horizonte Temporal em 4 Épocas:**
    1. *Linha de Base Histórica:* 1981–2014 (ERA5-Land / CHIRPS)
    2. *Futuro Próximo:* 2026–2050
    3. *Meio do Século:* 2051–2075
    4. *Final do Século:* 2076–2099
- **Indicadores Pluviométricos Apresentados (BIO12–BIO17 e BIO19):**
  - **BIO12 (Precipitação Total Anual):** $\text{PRCPTOT} = \sum_{m=1}^{12} P_m$ em mm/ano.
  - **BIO13 (Precipitação do Mês Mais Chuvoso):** $\max(P_m)$ — pico de cheias do verão.
  - **BIO14 (Precipitação do Mês Mais Seco):** $\min(P_m)$ — severidade da estiagem de inverno.
  - **BIO15 (Sazonalidade da Precipitação):** Coeficiente de Variação ($CV = \frac{\sigma_m}{\mu_m} \times 100$) — irregularidade temporal do regime de chuvas.
  - **BIO16 (Precipitação do Trimestre Mais Úmido):** Acumulado do trimestre mais chuvoso (ex: DJF / Verão).
  - **BIO17 (Precipitação do Trimestre Mais Seco):** Acumulado do trimestre mais seco (ex: JJA / Inverno).
  - **BIO19 (Precipitação do Trimestre Mais Frio):** Chuva durante o período frio do ano.
- **Índices de Extremos Complementares (ETCCDI):** $R10\text{mm}, R20\text{mm}, RX1\text{day}, RX5\text{day}, CDD, CWD, R95p, R99p$.

## Active Progress & Current Checkpoint (SEMCITEC 2026)
- **Data Status (Downloaded & Verified):**
  - All 4 CORDEX projection NetCDFs (INPE-Eta at $0.20^\circ$ spatial resolution, driven by MOHC-HadGEM2-ES) covering 2006 to 2099 are downloaded and verified in `THOR-PIML-Semcitec2026/data/`:
    1. `pr` (Precipitation flux) — RCP 4.5 (SSP2-4.5)
    2. `pr` (Precipitation flux) — RCP 8.5 (SSP5-8.5)
    3. `tas` (2m Air Temperature) — RCP 4.5 (SSP2-4.5)
    4. `tas` (2m Air Temperature) — RCP 8.5 (SSP5-8.5)
  - Clipped Bounding Box: North: -22, South: -25, West: -48, East: -45 (centering Guarulhos-SP at $-23.43^\circ\text{S}, -46.47^\circ\text{W}$).
  - Time extent: 33,840 continuous daily steps from 2006-01-01 to 2099-12-30.

- **Environment & Tools Ready:**
  - Python 3.11 environment equipped with: `torch`, `xarray`, `netCDF4`, `cftime`, `pandas`, `numpy`, `matplotlib`, `scipy`.
  - Checkpoints frozen and ready: `checkpoints/v8_hybrid_seed42.pt` and `scaler_v7.json`.
  - Historical baseline dataset: `data/ground_truth_guarulhos_daily_v3.csv` ($1981\text{–}2026$).

- **Execution Roadmap & Current Checkpoint (SEMCITEC 2026):**
  1. **Step 1 (CORDEX Ingestion & Preprocessing):** [COMPLETED]
     - 4 NetCDFs parsed, subgrid 3x3 extracted for Guarulhos (-23.43°S, -46.47°W), flux conversion executed, CSVs saved in `data/cordex/`.
  2. **Step 2 (THOR-PIML Zero-Shot Inference):** [COMPLETED]
    - 26,640 daily steps published for each scenario using frozen weights `v8_hybrid_seed42.pt` (causal lookback T=30d) for 2026–2099. Physical barrier violations: 0.00%. Projection CSVs are intermediate data in `results/projections/`; editorial tables are PNG/PDF.
  3. **Step 3 (Bioclimatic & ETCCDI Computation):** [COMPLETED]
    - Pluviometric indicators BIO12–BIO17 and BIO19, plus ETCCDI indices, computed across 4 epochs (Baseline 1981–2014, Near Future 2026–2050, Mid-Century 2051–2075, Late-Century 2076–2099). Editorial tables are PNG/PDF in `results/tables/`; intermediate CSVs are in `.agents/table_data/`.
  4. **Step 4 (Publication Figures & 25-Page Paper Writing):**
     - **Figures (4.1):** [COMPLETED] All 5 high-density 300 DPI figures generated with standardized high-contrast palette (Black Baseline, Slate Gray Raw CORDEX, Royal Blue THOR SSP2-4.5, Crimson Red THOR SSP5-8.5). Saved in `results/figures/`.
    - **Paper Writing (4.2):** [ACTIVE NEXT STEP] Draft the technical report integrating methodology, QC audit, seasonal transfer result, CCKP comparison, TCWV sensitivity and documented limitations. Do not claim bias correction, amplitude validation or extreme-event recovery without an independent validation protocol.

  ## Final Scientific Interpretation
  - Historical benchmark: THOR-V8 is competitive on the CONIC historical test, but daily skill does not guarantee future transfer.
  - Seasonal structure: THOR preserves the observed wet-summer/dry-winter ordering, while raw CORDEX produces a late-century seasonal inversion for Guarulhos.
  - Future amplitude: CCKP CMIP6 is an independent comparison for Sao Paulo. THOR's late-century DJF anomaly is above the CCKP central estimate and approximately above its P90 approximation; this is divergence, not proof of failure.
  - Sensitivity: changing the explicit TCWV response from 6.5%/K to 0%/K lowers DJF anomaly from about +169 mm to +133 mm. The term explains part, not all, of the excess; 0%/K is a contrafactual control.
  - Limitations: future circulation and convection fields are synthetic/proxied; extremes and persistence remain weak; CCKP historical P10/P90 describe model spread, not model error against reanalysis.

