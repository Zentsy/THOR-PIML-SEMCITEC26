# Relatório Técnico de Problemas e Inconsistências — THOR-PIML SEMCITEC 2026

> **Atualização de auditoria — 2026-09-05**
>
> Este documento preserva o diagnóstico original e registra abaixo o plano executado, as correções aplicadas e o estado verificável dos artefatos. O diagnóstico de problemas científicos do modelo não foi removido nem convertido em conclusão de desempenho.

**Escopo:** Pipeline de Downscaling Estatístico e Projeções Climáticas (2026–2100) em Guarulhos-SP.  
**Arquitetura:** `THORSpatialHybridModel` (V8) + Preditores de Superfície (138 variáveis) + Tensor Espacial 2D CORDEX.  
**Natureza do Documento:** Diagnóstico estrito de falhas de código, limitações numéricas e anomalias estatísticas identificadas. Não contém propostas de resolução.

---

## 1. Sobrescrita de Dados pelo Teste Unitário Dry-Run

* **Arquivo de Origem:** `tests/test_semcitec_pipeline.py` (linha 330, método `test_13_master_pipeline_dry_run`) e `run_semcitec_pipeline.py`.
* **Descrição Técnica:**
  O método `test_13_master_pipeline_dry_run` dispara a execução do pipeline mestre através do comando `subprocess.run([sys.executable, "run_semcitec_pipeline.py", "--dry-run"])`.
  No script orquestrador `run_semcitec_pipeline.py`, a função `run_step_2()` repassa o argumento `dry_run=True` para `infer_cordex_v8.py`. Dentro deste módulo, a flag `--dry-run` recorta a série temporal para apenas 120 passos brutos a partir de 2026, gerando 90 passos de janela deslizante efetiva ($120 - 30\text{ dias de lookback} = 90\text{ dias}$, período de 2026-01-01 a 2026-03-30).
  A saída desse teste é gravada diretamente sobre o caminho oficial de produção:
  `results/projections/cordex_ssp245_downscaled_2026_2099.csv` e `cordex_ssp585_downscaled_2026_2099.csv`.
  Como consequência, qualquer execução prévia completa da inferência secular (33.810 passos) é sobrescrita e reduzida a um arquivo de 90 linhas toda vez que a suíte de testes unitários é rodada.

---

## 2. Inconsistência Temporal e Truncamento nos Indicadores e Figuras

* **Arquivos de Origem:** `src/calc_bioclimatic_indices.py`, `src/generate_semcitec_figures.py` e artefatos em `results/`.
* **Descrição Técnica:**
  1. **Tabelas de Indicadores (`results/tables/`):** O script `calc_bioclimatic_indices.py` agrega dados mensais para os cálculos de BIO12 a BIO19 através de `df_year.groupby("month")`. Ao processar a projeção truncada em 90 dias (janeiro, fevereiro e março de 2026), os meses de abril a dezembro resultam em `NaN` e são imputados com `0.0 mm`.
     * `BIO12 (PRCPTOT)` é computado como a soma de apenas 3 meses ($654.4\text{ mm}$).
     * `BIO14 (Mês Mais Seco)` assume valor idêntico a $0.0\text{ mm}$.
     * `BIO17 (Trimestre Mais Seco)` assume valor idêntico a $0.0\text{ mm}$.
     * `BIO15 (Sazonalidade CV)` salta artificialmente para $181.2\%$.
     * Como apenas o ano de 2026 existe no arquivo truncado, o desvio padrão interanual nas tabelas das épocas resulta em `NaN`.
  2. **Tabela Anual (`bioclimatic_indices_annual.csv`):** Enquanto as séries brutas do CORDEX contêm 74 anos completos (2026 a 2099), as séries do THOR-PIML contêm apenas 1 único ano (2026).
  3. **Figuras Editoriais (`results/figures/`):**
     * Na **Figura 1** (`fig1_semcitec_prcptot_projections_300dpi.png`), a linha temporal do CORDEX exibe 74 anos de projeção contínua, enquanto a linha do THOR-PIML é interrompida no primeiro ano, gerando uma visualização assimétrica com ausência total de dados nas épocas 2 (2051–2075) e 3 (2076–2100).
     * Nas **Figuras 2 a 5**, o ciclo sazonal, as curvas de permanência e as teias bioclimáticas do THOR refletem exclusivamente o trimestre de verão de 2026, e não o regime multidecadal projetado.

---

## 3. Drizzle Bias Crônico e Impossibilidade Matemática de Zero Chuva

* **Arquivos de Origem:** `src/v7/model_v8.py` (linhas 117–128 e 146–154) e `src/infer_cordex_v8.py` (linha 423).
* **Descrição Técnica:**
  A arquitetura neural do THOR-PIML V8 formula a predição através de duas cabeças paralelas:
  $$\hat{y} = p_{\text{occ}} \times \mu_{\text{int}}$$
  Onde:
  * $p_{\text{occ}} = \text{Sigmoid}(z) \in (0, 1)$
  * $\mu_{\text{int}} = \text{Softplus}(z) \in (0, +\infty)$
  
  O produto de duas funções estritamente positivas impede que o modelo produza saída exatamente igual a zero. Na série secular de 33.810 passos inferidos:
  * O menor valor de precipitação gerado pelo modelo é de **$0.266\text{ mm/dia}$**.
  * Teste de sensibilidade sintético: Quando a precipitação CORDEX de entrada é forçada para **$0.0\text{ mm}$** e a grade para $0.0$, a rede ainda assim infere $p_{\text{occ}} = 0.1789$ e $\mu_{\text{int}} = 15.15\text{ mm}$, gerando uma garoa de fundo ininterrupta de **$2.71\text{ mm/dia}$**.
  * A fração de dias com chuva $\ge 1.0\text{ mm}$ no modelo é de **$75.2\%$**, enquanto na base observada real (CHIRPS 1981–2026) é de apenas **$34.1\%$** e no CORDEX bruto é de **$32.1\%$**.
  * A fração de dias secos ($< 1.0\text{ mm}$) colapsa de $65.9\%$ (observado) para **$24.8\%$** no THOR.
  * O corte aplicado em `src/infer_cordex_v8.py` (`np.where(pred >= 1.0, pred, 0.0)`) tem efeito quase nulo, pois três quartos dos dias geram valores superiores a $1.0\text{ mm}$.
  * O índice de dias secos consecutivos ($CDD$) é severamente subestimado e o índice de dias úmidos consecutivos ($CWD$) atinge uma média anômala de **126 dias consecutivos de chuva**.

---

## 4. Colapso Estrutural da Cauda de Extremos Pluviométricos

* **Arquivos de Origem:** `src/v7/model_v8.py` (linhas 124–128), `src/infer_cordex_v8.py` e `src/calc_bioclimatic_indices.py` (linhas 220–226).
* **Descrição Técnica:**
  A camada `intensity_head` da rede neural opera com limites superiores aprendidos durante o treinamento histórico onde a regularização e funções de perda focadas em KGE/NSE amorteceram a variância de picos convectivos.
  * O valor máximo gerado pela cabeça de intensidade em toda a série secular é de **$38.79\text{ mm}$**.
  * Multiplicado pela probabilidade de ocorrência (cujo valor máximo observado é $p_{\text{occ}} = 0.8516$), o teto absoluto da precipitação diária do THOR-PIML é de **$31.98\text{ mm/dia}$**.
  * O percentil de dias muito úmidos da linha de base histórica observada (CHIRPS 1981–2014) é de **$p_{95} = 36.78\text{ mm/dia}$**.
  * O percentil de dias extremamente úmidos da linha de base é de **$p_{99} = 55.74\text{ mm/dia}$**.
  * Como a saída do modelo nunca atinge $32.0\text{ mm}$, nenhum dia projetado satisfaz a condição `wet_days > baseline_p95`.
  * Os índices oficiais do ETCCDI **$R95p$** e **$R99p$** (volume acumulado e percentual em dias muito e extremamente úmidos) resultam estritamente em **$0.00\text{ mm}$ e $0.00\%$** em todos os anos entre 2026 e 2099.
  * O índice $RX1\text{day}$ (máxima precipitação diária em 24h) estabiliza em torno de 25 a 31 mm/ano, não capturando eventos convectivos severos (que no histórico CHIRPS atingem $134.6\text{ mm}$ e no CORDEX bruto atingem $87.6\text{ mm}$).

---

## 5. Desacoplamento Físico entre a Saída do THOR e a Precipitação do RCM

* **Arquivos de Origem:** `src/infer_cordex_v8.py` (método `engineer_cordex_surface_features`, linhas 112–210).
* **Descrição Técnica:**
  A correlação linear de Pearson calculada sobre os 33.810 dias entre a chuva prevista pelo THOR e a chuva do CORDEX bruto resulta em:
  * Correlação diária: **$r = -0.0059$** ($R^2 < 0.0001$).
  * Correlação anual: **$r = -0.344$**.
  * Correlação diária entre THOR e temperatura do ar (`tas`): **$r = +0.575$**.
  * Correlação anual entre THOR e temperatura do ar (`tas`): **$r = +0.933$**.
  
  A resposta da rede neural é governada pelo ciclo anual de temperatura e pelo dia do ano ($doy$). Um teste de sensibilidade injetando um evento pluviométrico severo de $80\text{ mm/dia}$ no dia alvo aumentou a previsão do modelo de $2.71\text{ mm}$ para apenas $11.22\text{ mm}$, indicando que a dinâmica de precipitação do modelo regional CORDEX é quase totalmente atenuada pela estrutura de entrada da rede.

---

## 6. Inversão Forçada do Sinal Climático Secular por Parametrização Sintética

* **Arquivos de Origem:** `src/infer_cordex_v8.py` (linhas 154–160).
* **Descrição Técnica:**
  O modelo regional dinâmico Eta-HadGEM2-ES projeta redução sistemática dos acumulados anuais de chuva para a bacia de Guarulhos ao longo do século:
  * RCP4.5: $1268.1\text{ mm/ano}$ (2026–2050) $\to$ $1017.7\text{ mm/ano}$ (2051–2075) $\to$ $1101.5\text{ mm/ano}$ (2076–2099).
  * RCP8.5: $1211.7\text{ mm/ano}$ (2026–2050) $\to$ $960.4\text{ mm/ano}$ (2051–2075) $\to$ $1021.7\text{ mm/ano}$ (2076–2099).
  
  O modelo THOR-PIML projeta a tendência contrária:
  * RCP4.5 / SSP2-4.5: $1371.1\text{ mm/ano}$ (2026–2050) $\to$ $1629.6\text{ mm/ano}$ (2051–2075) $\to$ $1640.9\text{ mm/ano}$ (2076–2099) — aumento de $+20\%$ a $+27\%$.
  
  Essa divergência é causada pela formulação da coluna total de vapor d'água ($TCWV$) em `src/infer_cordex_v8.py`:
  $$\text{TCWV} = \text{TCWV}_{\text{base}}(doy) \times [1 + 0.065 \times (tas - 20.0)] + 3.0 \times \tanh(pr / 10.0)$$
  O aquecimento secular simulado pelo CORDEX eleva $tas$ de $20.77^\circ\text{C}$ para quase $25.00^\circ\text{C}$ no final do século. A amplificação de $+6.5\%/^\circ\text{C}$ embutida na fórmula eleva o $TCWV$ injetado na rede em até $+32\%$. Como a rede foi treinada associando maiores teores de umidade a maior precipitação, a projeção secular de chuva sobe como consequência direta da formulação da feature de entrada, invertendo o sinal projetado pelo RCM.

---

## 7. Inoperância Prática da Restrição Física de Clausius-Clapeyron

* **Arquivos de Origem:** `src/infer_cordex_v8.py` (linhas 386–393).
* **Descrição Técnica:**
  A barreira física de água precipitável calcula:
  $$W_{\max} = 4.0 \times \text{TCWV}$$
  Durante toda a série de projeção (2006–2099):
  * Média de $W_{\max}$: **$113.6\text{ mm}$**.
  * Valor mínimo absoluto de $W_{\max}$: **$20.00\text{ mm}$** (ocorrido no pico da estiagem de inverno).
  * Valor máximo da precipitação bruta sem restrição ($\hat{y}_{\text{uncapped}}$) gerado pela rede: **$31.98\text{ mm}$** (ocorrido no verão, onde $W_{\max} > 100\text{ mm}$).
  * Nos dias em que $W_{\max}$ atinge seu mínimo ($20.00\text{ mm}$), a rede produz saídas inferiores a $1.0\text{ mm}$.
  * A folga mínima observada entre o teto e a predição da rede é de **$19.67\text{ mm}$**.
  * A taxa de violação física é de **$0.00\%$** ($0 / 33.810$ passos). Essa taxa de zero decorre da distância constante entre o patamar de saturação do teto e a faixa de amplitude limitada da rede neural, e não de um corte ativo aplicado à saída durante a inferência.

---

## 8. Degeneração das Features Sinóticas de Entrada

* **Arquivos de Origem:** `src/infer_cordex_v8.py` (linhas 135–181).
* **Descrição Técnica:**
  Para atender às 138 features exigidas pelo `RobustClimateScaler V7`, variáveis dinâmicas que não constavam nas saídas dos NetCDFs CORDEX SAM-22 foram sintetizadas:
  * Pressão de superfície (`psfc`): fixada no valor constante de **$930.0\text{ hPa}$** em todos os 33.840 dias ($std = 0.00$). No histórico ERA5, essa variável apresentava variabilidade com $std = 3.57\text{ hPa}$.
  * Velocidade do vento de superfície (`wind_speed`): fixada no valor constante de **$2.20\text{ m/s}$** ($std = 0.00$). No histórico, apresentava $std = 0.86\text{ m/s}$.
  * Ventos em altitude ($u_{700}$ e $v_{700}$): expressos como senoides e cossenos harmônicos puros dependentes unicamente de $doy$. Não possuem distúrbios meteorológicos, vórtices ou transições de frentes frias.
  * Altura geopotencial $z_{500}$: construída como combinação linear direta da temperatura ($5840.0 + 1.8 \cdot tas$).
  
  Cerca de $0.87\%$ das variáveis escaladas caem fora do intervalo $[0, 1]$ do normalizador histórico (atingindo $4.5\%$ de extrapolação em $tmean$ e $4.9\%$ em $tmin$). A rede opera em regime de extrapolação semivariável sobre um espaço sinótico estático.

---

## 9. Vazamento de Alvo no Dataset de Treinamento Original

* **Arquivos de Origem:** `data/ground_truth_guarulhos_daily_v3.csv`.
* **Descrição Técnica:**
  No arquivo histórico utilizado para treinar e calibrar a rede neural:
  * A feature preditora `pr_grid_max` apresenta correlação de Pearson de **$0.9215$** com a variável dependente `pr_target`.
  * Em **$84.9\%$** dos dias da série histórica, `pr_grid_max >= pr_target`.
  * Essa alta dependência ocorreu porque `pr_grid_max` foi extraída do valor máximo espacial da grade CHIRPS que continha a estação no mesmo dia da medição.
  * No pipeline de projeção futura, como a grade observada CHIRPS não existe, o código substituiu a variável pelo máximo da subgrade CORDEX (`sub_pr.max()`). Como os eventos convectivos do modelo regional CORDEX não coincidem dia a dia com a estação, a rede perde o preditor que sustentava sua pontuação métrica original, degradando o desempenho temporal diário.

---

## 10. Descompasso de Calendário e Duração das Épocas

* **Arquivos de Origem:** `data/cordex/`, `src/cordex_preprocessor.py` e `src/calc_bioclimatic_indices.py`.
* **Descrição Técnica:**
  * Os NetCDFs CORDEX do modelo HadGEM2-ES são formulados sob calendário contínuo idealizado **`360_day`** (todos os 12 meses possuem exatamente 30 dias, totalizando 360 dias por ano).
  * O arquivo diário gerado termina em **`2099-12-30`**.
  * A época final de projeção (nominalmente denominada "2076–2100") compreende 24 anos inteiros (2076 a 2099) e não 25 anos.
  * O ano de 360 dias contém aproximadamente 5.25 dias a menos do que o ano gregoriano real (365.25 dias). Acumulados anuais brutos e contagens de dias com chuva apresentam uma discrepância sistemática intrínseca de $\approx -1.4\%$ em relação aos dados históricos observados gregorianos se não forem homogeneizados.

---

## 11. Plano de Ajuste Executado

### 11.1 Correções operacionais aplicadas

1. **Isolamento do dry-run:** `run_semcitec_pipeline.py` agora envia inferência, tabelas e figuras de teste para `tests/artifacts/dryrun/`. O modo de teste não escreve em `results/`, que fica reservado aos artefatos canônicos.
2. **Proteção contra anos incompletos:** `compute_series_indices()` descarta anos com menos de 300 dias. O dry-run continua validando a cadeia de execução, mas não publica BIO/ETCCDI anuais artificialmente completos.
3. **Horizonte publicado corrigido:** `src/infer_cordex_v8.py` usa 30 dias de lookback internamente e publica somente previsões de `2026-01-01` a `2099-12-30`.
4. **Época final corrigida:** a época passou de `Final do Século (2076-2100)` para `Final do Século (2076-2099)`, refletindo o fim real dos NetCDFs disponíveis.
5. **QC alinhado ao repositório atual:** `src/validate_datasets.py` usa Ground Truth V3 e Scaler V7, valida os 138 limites do scaler mesmo com `feature_names=null`, verifica colunas antes de acessá-las e valida o horizonte completo das projeções.
6. **Teste de integração fortalecido:** `test_13_master_pipeline_dry_run` verifica diretórios isolados, 90 janelas diagnósticas, ausência de indicadores THOR para ano parcial e preservação byte a byte da projeção oficial.
7. **Relatório executivo corrigido:** o texto não afirma automaticamente eliminação de drizzle bias, recuperação de extremos ou melhoria do modelo; apresenta os índices como estatísticas que ainda exigem validação independente.

### 11.2 Artefatos oficiais regenerados

- `results/projections/cordex_ssp245_downscaled_2026_2099.csv`: **26.640 dias**, 2026-01-01 a 2099-12-30.
- `results/projections/cordex_ssp585_downscaled_2026_2099.csv`: **26.640 dias**, 2026-01-01 a 2099-12-30.
- Tabelas BIO/ETCCDI e relatório executivo regenerados em `results/tables/`.
- Cinco figuras oficiais regeneradas em `results/figures/` com 300 DPI.

### 11.3 Verificações finais

- `python -m unittest tests.test_semcitec_pipeline -v`: **15/15 testes passaram**.
- `python src/validate_datasets.py --all --verbose --fail-if-mock`: **QC aprovado**.
- Ground Truth V3: proveniência real, faixas físicas válidas e `pr_target` comparável.
- Scaler V7: `minmax`, ajustado, **138 features**.
- Projeções: nenhuma violação registrada da barreira `W_max = 4.0 * TCWV`.

## 12. Problemas ainda abertos e plano científico

Os itens 3 a 9 do diagnóstico original continuam sendo limitações do checkpoint e da formulação científica, não bugs operacionais resolvidos pelos ajustes desta auditoria:

- **Drizzle bias e hurdle:** exigem novo treinamento/calibração com ocorrência realmente binária ou uma cabeça de ocorrência capaz de produzir zero.
- **Cauda de extremos:** exige loss/treinamento calibrado para extremos e validação contra observações independentes; não deve ser resolvido por clipping arbitrário.
- **Desacoplamento CORDEX-THOR e inversão de tendência:** exigem revisão das features futuras, teste de sensibilidade e comparação multiescalar antes de qualquer conclusão climática.
- **Features sinóticas sintéticas:** devem ser substituídas por preditores atmosféricos observacionais ou RCM compatíveis, mantendo o contrato de 138 features.
- **Leakage de `pr_grid_max`:** requer novo protocolo de treino sem preditor que contenha informação espacialmente coincidente com o alvo.
- **Calendário 360_day:** os resultados futuros devem ser comparados com o baseline explicitando a diferença de calendário; a época final não deve ser chamada de 2100 enquanto não houver dados de 2100.

Até que essas etapas sejam concluídas, os resultados regenerados são válidos como saída reproduzível do checkpoint V8, mas não como prova de melhoria física ou de desempenho preditivo futuro.
