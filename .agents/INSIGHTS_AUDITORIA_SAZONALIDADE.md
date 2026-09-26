# Insights da Auditoria de Sazonalidade e Transferência Climática

**Data:** 2026-09-05  
**Projeto:** THOR-PIML SEMCITEC 2026  
**Contexto:** conversa de revisão crítica dos resultados antes da redação/apresentação.

## 1. Insight principal

O resultado não deve ser interpretado como “THOR está errado porque não segue o CORDEX”. Downscaling pode corrigir vieses sazonais do RCM. A comparação mostrou que o CORDEX SSP5-8.5 termina com uma sazonalidade local estranha para Guarulhos, enquanto o THOR preserva melhor o padrão histórico de verão chuvoso e inverno seco.

Valores médios mensais agregados no SSP5-8.5:

| Série | Verão DJF | Inverno JJA | Razão DJF/JJA |
|---|---:|---:|---:|
| Baseline observado 1981–2014 | 719 mm | 133 mm | 5,4 |
| CORDEX 2026–2050 | 387 mm | 270 mm | 1,4 |
| CORDEX 2076–2099 | 195 mm | 322 mm | 0,6 |
| THOR 2026–2050 | 687 mm | 108 mm | 6,4 |
| THOR 2076–2099 | 888 mm | 264 mm | 3,4 |

O baseline da SEMCITEC é **1981–2014**, 34 anos completos. A figura histórica do CONIC `results/figures/fig2_seasonal_climatology_narrative.png` usa **2019–2026** e já mostra que o THOR-V8 acompanha bem o formato mensal observado. Essa evidência deve ser recuperada na nova narrativa.

## 2. O que Clausius-Clapeyron realmente sustenta

O aquecimento aumenta a capacidade de retenção de vapor d'água em aproximadamente 6,5% por Kelvin em condições próximas da saturação. Isso é uma relação termodinâmica, não uma garantia de aumento proporcional da precipitação.

Para transformar umidade em chuva, também importam circulação, convergência de umidade, convecção, levantamento, frentes e instabilidade. Logo:

- o aumento de chuva de verão projetado pelo THOR é uma hipótese física plausível;
- a magnitude dessa amplificação ainda não está validada;
- não se deve dizer que Clausius-Clapeyron sozinho “prevê” a chuva futura.

## 3. Limitação real dos dados futuros

Nos NetCDFs CORDEX SAM-22 usados no projeto, foram disponibilizadas apenas `pr` e `tas` para os cenários baixados. A inferência atual recebe essas duas fontes diretas e, em `engineer_cordex_surface_features()`, constrói as demais features necessárias ao checkpoint:

- `TCWV` sintético com ciclo sazonal fixo, fator de 6,5%/K e contribuição de `pr`;
- `psfc` e `wind_speed` constantes;
- `u700` e `v700` como harmônicos do dia do ano;
- `z500`, `q700`, `w500`, `cape` e outras variáveis como proxies derivados de `tas`, `pr` e fórmulas simplificadas.

Portanto, o modelo não está recebendo circulação, convergência e convecção do RCM de forma independente. Ele está realizando uma transferência com informação limitada e preenchendo o espaço de 138 features com parametrizações/proxies.

Essa é uma limitação de dados, não necessariamente uma falha conceitual do THOR. Porém, ela restringe a conclusão: o resultado deve ser chamado de **projeção exploratória termodinamicamente parametrizada**, e não de downscaling atmosférico completo.

## 4. Problemas que permanecem separados

Mesmo que o THOR esteja corrigindo um viés sazonal do CORDEX, continuam válidos os diagnósticos independentes:

- extremos `RX1day`, `R95p` e `R99p` comprimidos;
- persistência chuvosa irreal, com `CWD` muito alto;
- dificuldade de produzir zeros reais por `sigmoid × softplus`;
- amplitude anual futura ainda sem referência externa independente;
- features sinóticas futuras sintéticas;
- possível leakage histórico associado a `pr_grid_max`.

Não misturar “sazonalidade plausível” com “modelo correto em todos os aspectos”. O THOR pode acertar o regime médio mensal e falhar nos extremos diários ao mesmo tempo.

### 4.1 Diagnóstico Matemático e Físico da Garoa Crônica (CONIC vs. CORDEX)

A reincidência da "garoa crônica" (drizzle bias) e a elevação de $CWD$ (dias chuvosos consecutivos) possuem fundamentação matemática na formulação da rede e física na qualidade dos preditores de entrada:

1. **A Raiz Estrutural na Formulação Dual-Head (Hurdle Contínuo):**
   * Em `src/v7/model_v8.py`, a predição é calculada pelo valor esperado incondicional diferenciável:
     $$\hat{y} = p_{\text{occ}} \times \mu_{\text{int}} = \text{Sigmoid}(z_{\text{occ}}) \times \text{Softplus}(z_{\text{int}})$$
   * Como $\text{Sigmoid}(z) \in (0, 1)$ e $\text{Softplus}(z) \in (0, +\infty)$, $\hat{y}$ é **estritamente positivo em 100% dos dias**. Matematicamente, a rede não tem suporte para gerar zero exato.

2. **Evidência no Artigo do CONIC (Tabela 2, pág. 6):**
   * Na Tabela 2 do artigo do CONIC (avaliação histórica independente de 7 anos no ERA5-Land, 2019–2026), o modelo já exibia sinais desse comportamento:
     * Observado (CHIRPS): $CWD = 29\text{ dias}$.
     * ResLSTM univariada: $CWD = 14\text{ dias}$.
     * TCN univariada: $CWD = 19\text{ dias}$.
     * THOR-V7 (sem CNN espacial): $CWD = 11\text{ dias}$.
     * **THOR-V8 (PIML Espacial): $CWD = 71\text{ dias}$!**
   * A afirmação no CONIC de que o modelo "eliminou o viés de garoa" referia-se à capacidade de capturar eventos extremos de tempestade ($95\%$ de acerto em $R20\text{mm}$ contra $0\%$ de modelos puramente baseados em MSE, que esmagam toda a chuva em garoa fraca). No entanto, no limiar inferior de chuva fraca ($0\text{ a }2\text{ mm/dia}$), o THOR-V8 já mantinha uma inércia úmida residual elevada ($CWD = 71\text{ dias}$).

3. **Por que o Problema Inflou no CORDEX ($CWD = 71\text{ d} \to 125\text{ d}$)?**
   * No teste histórico do CONIC, o modelo recebia 84 preditores reais de reanálise horária/diária do ERA5-Land (quedas abruptas de temperatura de orvalho, subsidência barométrica pós-frontal, ventos secos do quadrante oeste/sudoeste). Esses sinais derrubavam $p_{\text{occ}}$ para $10^{-4}$, tornando $\hat{y} \approx 0.001\text{ mm}$, facilmente podado pelo limiar WMO ($< 1.0\text{ mm}$).
   * No CORDEX SAM-22, os NetCDFs futuros fornecem **apenas duas variáveis**: precipitação (`pr`) e temperatura a 2m (`tas`).
   * As demais 136 variáveis foram sintetizadas como médias harmônicas sazonais ou proxies estáticos em `infer_cordex_v8.py`.
   * Na ausência dos campos dinâmicos reais de alta pressão fria/seca, a rede opera permanentemente em um "regime basal úmido". Quando a entrada do CORDEX é $0.0\text{ mm/dia}$, a rede infere $p_{\text{occ}} \approx 0.1789$ e $\mu_{\text{int}} \approx 15.15\text{ mm}$, gerando:
     $$\hat{y} = 0.1789 \times 15.15\text{ mm} = 2.71\text{ mm/dia}$$
   * Como $2.71\text{ mm} > 1.0\text{ mm}$, o pós-processamento simples (`np.where(pred >= 1.0, pred, 0.0)`) falha em zerar o dia, elevando a frequência de chuva para $75.2\%$ e o $CWD$ para $125\text{ dias}$.

4. **Conclusão Científica e Decisão Metodológica:**
   * O fenômeno não é um bug acidental de código, mas sim uma consequência demonstrável da transferência de um modelo treinado com atmosfera rica em variáveis (ERA5) para um cenário de forçamento com variáveis dinâmicas ausentes (CORDEX SAM-22 reduzido a `pr` e `tas`).
   * Tentar aplicar filtros artificiais *post-hoc* (como forçar $\hat{y} = 0$ sempre que $p_{\text{occ}} < 0.35$ ou que a chuva CORDEX for zero) sem retreinar o modelo constituiria ajuste cosmético ad-hoc e violaria o protocolo experimental.
   * A conduta metodologicamente correta para o artigo da SEMCITEC é reportar essa descoberta com precisão: o THOR-PIML é excelente na reconstrução sazonal (DJF/JJA) e na preservação da média climatológica, mas a escassez de preditores sinóticos no RCM acentua o viés de persistência úmida.

## 5. Título Oficial e Linha de Pesquisa Definida

Título canônico aprovado:

> **THOR-PIML: Modelagem Híbrida com Física Informada para Projeção de Indicadores Bioclimáticos e Correção de Viés Sazonal em Guarulhos-SP sob Cenários SSP (2026–2100)**

Pergunta central:

> Um modelo com desempenho histórico competitivo mantém a sazonalidade observada quando aplicado a um RCM com viés regional, sem destruir amplitude, extremos e persistência?

Comparar:

1. CORDEX bruto, como sinal do RCM;
2. THOR-V8, como downscaling termodinâmico/espacial-temporal;
3. EQM, apenas como baseline estatístico de controle, não como método vencedor;
4. baseline observado histórico;
5. uma referência externa independente para a anomalia futura.

O resultado esperado não precisa declarar um vencedor absoluto. Pode mostrar que:

- THOR melhora a estrutura sazonal;
- CORDEX pode subestimar o verão local;
- THOR ainda falha nos extremos e na persistência;
- métricas diárias isoladas não bastam para avaliar projeções climáticas.

## 6. Próximo passo quando retomarmos

Antes de alterar o modelo, buscar uma referência independente de projeção de precipitação para São Paulo/Serra do Mar/Região Sudeste, com baseline comparável. Verificar se a anomalia futura de verão no SSP5-8.5 está na mesma ordem de grandeza do THOR.

Depois:

1. conferir se existem outros campos CORDEX ou outro ensemble/modelo com `tas`, umidade, vento, pressão e circulação;
2. se não existirem, declarar formalmente a limitação de dados;
3. construir uma figura mensal comparando baseline, CORDEX, THOR e referência externa;
4. separar as conclusões de sazonalidade, amplitude, extremos e persistência;
5. não corrigir extremos por clipping cosmético: isso exigiria novo treinamento/calibração.

## 7. Frase-guia para a apresentação

> O THOR não deve ser julgado apenas por reproduzir o RCM bruto. Neste estudo, ele recupera uma sazonalidade local mais compatível com o histórico, mas a limitação dos campos futuros disponíveis impede afirmar que a amplitude e os extremos projetados estejam fisicamente calibrados.

## 8. Pesquisa independente iniciada em 2026-09-06

### 8.1 Fonte operacional principal: Climate Change Knowledge Portal

O **World Bank Climate Change Knowledge Portal (CCKP)** oferece uma referência independente adequada para a próxima etapa:

- [Brazil — CMIP6 mean projections](https://climateknowledgeportal.worldbank.org/country/brazil/climate-data-projections)
- [Brazil — historical climatology](https://climateknowledgeportal.worldbank.org/country/brazil/climate-data-historical)

O portal informa que as projeções são um **ensemble multimodelo CMIP6**, com cenários SSP, períodos climatológicos e visualização de anomalia por estação. Também fornece dispersão P10–P90 e permite examinar o ciclo mensal. A resolução apresentada é de 0,25°.

O portal ressalta duas cautelas importantes:

1. precipitação é mais variável e incerta que temperatura;
2. a análise correta deve considerar a dispersão entre modelos, não apenas a média do ensemble.

O CCKP é adequado para testar se os `+169 mm` do THOR no verão tardio (888 mm contra 719 mm do baseline) estão dentro da ordem de grandeza do ensemble CMIP6. A página acessível não deve ser tratada como evidência numérica final até que a tabela/download seja extraída com o mesmo domínio, estação e período do projeto.

### 8.2 Literatura regional independente

As seguintes referências foram localizadas por metadados Crossref e devem entrar na revisão bibliográfica:

- **Junquas et al. (2012)**, *Summer precipitation variability over Southeastern South America in a global warming scenario*, DOI [10.1007/s00382-011-1141-y](https://doi.org/10.1007/s00382-011-1141-y). É diretamente relevante para a sazonalidade de verão no Sudeste da América do Sul e para a relação entre aquecimento e variabilidade do verão.
- **Solman & Blázquez (2019)**, *Multiscale precipitation variability over South America: Analysis of the added value of CORDEX RCM simulations*, DOI [10.1007/s00382-019-04689-1](https://doi.org/10.1007/s00382-019-04689-1). Sustenta que o valor adicional do RCM depende da escala e da métrica avaliada, não sendo automático em todo indicador.
- **Blázquez & Solman (2020)**, *Multiscale precipitation variability and extremes over South America: analysis of future changes from a set of CORDEX regional climate model simulations*, DOI [10.1007/s00382-020-05370-8](https://doi.org/10.1007/s00382-020-05370-8). Relevante para separar médias sazonais de extremos e para a incerteza entre RCMs.
- **Llopart, Reboita & da Rocha (2020)**, *Assessment of multi-model climate projections of water resources over South America CORDEX domain*, DOI [10.1007/s00382-019-04990-z](https://doi.org/10.1007/s00382-019-04990-z). Relevante como ensemble regional sul-americano, embora não seja uma validação pontual de Guarulhos.
- **Blázquez & Solman (2023)**, *Temperature and precipitation biases in CORDEX RCM simulations over South America: possible origin and impacts on the regional climate change signal*, DOI [10.1007/s00382-023-06727-5](https://doi.org/10.1007/s00382-023-06727-5). Relevante porque mostra que vieses presentes podem contaminar o sinal de mudança futura e que a origem física do viés importa.

### 8.3 Conclusão provisória da pesquisa

A literatura encontrada apoia a hipótese de que um RCM pode ter viés sazonal regional e que o sinal futuro de precipitação na América do Sul possui incerteza relevante. Ela **não autoriza ainda** afirmar que o THOR está quantitativamente correto ao projetar 888 mm de DJF no final do SSP5-8.5.

O teste correto será:

1. extrair do CCKP/CMIP6 uma série ou climatologia para a área mais próxima de Guarulhos;
2. usar baseline e períodos comparáveis;
3. calcular DJF, JJA, anual e anomalias absolutas/percentuais;
4. comparar THOR com a mediana e P10–P90 do ensemble;
5. reportar concordância, divergência e incerteza, sem selecionar uma fonte que apenas confirme o THOR.

Após a extração, a formulação adequada é **“THOR sazonalmente plausível e compatível com uma hipótese de correção de viés; sua amplitude futura está acima da faixa central do CCKP e ainda não foi validada”**. Continua pendente uma comparação observacional/reanálise no mesmo quadro, pois os P10/P90 históricos do CCKP representam apenas dispersão entre modelos.

### 8.4 Extração efetiva da API do CCKP

Em 2026-09-06, os dados foram baixados pela API oficial do CCKP para São Paulo (`BRA.37689`), unidade ADM1 mais próxima do domínio de Guarulhos. Foram extraídos os cenários SSP2-4.5 e SSP5-8.5, os percentis P10, mediana e P90, e as agregações anual e mensal para 2080-2099.

Arquivos locais:

- `.agents/cckp_data/sp_pr_anomaly_2080-2099_all.csv` — 78 registros consolidados;
- `.agents/cckp_data/sp_pr_anomaly_seasonal_approx_2080-2099.csv` — soma sazonal aproximada;
- arquivos JSON individuais preservando cada resposta da API.

Para SSP5-8.5, a API retornou a seguinte anomalia anual de precipitação para São Paulo:

| Percentil | Anomalia anual (mm) |
|---|---:|
| P10 | -217,01 |
| Mediana | +39,21 |
| P90 | +187,29 |

Uma auditoria adicional confirmou que a linha mensal da Figura 6 está correta: ela usa diretamente o produto `pr_anomaly_monthly` da API CCKP. Para SSP5-8.5, os valores medianos de DJF são aproximadamente **+3,88 mm em janeiro**, **+1,90 mm em fevereiro** e **+0,28 mm em dezembro**. Portanto, o CCKP realmente indica um aumento de verão pequeno em relação ao THOR.

Essa diferença não é um erro de plotagem, mas as anomalias não usam o mesmo baseline: CCKP calcula anomalia relativa à climatologia histórica do próprio CMIP6 (1995-2014), enquanto THOR/CORDEX na Figura 6 são calculados contra CHIRPS local (1981-2014). A figura foi rotulada explicitamente para deixar essa assimetria visível. A comparação é evidência externa de divergência de amplitude, não uma validação perfeitamente pareada.

### 8.5 Sensibilidade da resposta termodinâmica

A origem operacional mais provável da superestimativa do THOR foi localizada em `src/infer_cordex_v8.py`: a engenharia de features calculava `TCWV` com o termo fixo `1 + 0.065 * (T - 20)`. Esse fator não é uma previsão direta de precipitação; ele representa uma hipótese de capacidade de vapor e depois influencia as features usadas pela rede e o teto físico `W_max = 4 * TCWV`.

O código agora aceita três parâmetros de inferência:

- `--cc-response-per-kelvin`: resposta média, padrão `0.065`;
- `--cc-response-jitter`: desvio padrão opcional por dia;
- `--random-seed`: seed para tornar o ensemble reproduzível.

O procedimento recomendado é rodar membros separados, sem sobrescrever a projeção oficial: `0.000`, `0.030`, `0.050`, `0.065` e, opcionalmente, `0.065` com jitter `0.010` ou `0.020`. A análise deve comparar DJF, JJA, BIO12, extremos e a distância ao envelope CCKP. A aleatoriedade não deve ser usada para “corrigir” o resultado depois de observá-lo; ela deve representar uma incerteza definida antes da comparação.

Esse teste pode confirmar a hipótese causal: se reduzir a resposta de TCWV reduzir progressivamente a anomalia de DJF e aproximar o THOR do CCKP sem destruir a sazonalidade histórica, a parametrização é uma explicação forte para a amplitude excessiva. Isso seria uma melhoria de análise, não uma nova validação física do modelo.

### 8.6 Resultado do ensemble de sensibilidade SSP5-8.5

O experimento foi executado para cinco membros independentes em `.agents/sensitivity/`, sem sobrescrever `results/projections/`: resposta `0%/K`, `3%/K`, `5%/K`, `6,5%/K` e `6,5%/K` com jitter diário de `1%`.

| Caso | Resposta | DJF final (mm) | Anomalia DJF vs CHIRPS (mm) | JJA final (mm) | Razão DJF/JJA |
|---|---:|---:|---:|---:|---:|
| cc000 | 0%/K | 852,18 | +133,27 | 252,64 | 3,37 |
| cc030 | 3%/K | 869,11 | +150,20 | 257,94 | 3,37 |
| cc050 | 5%/K | 880,09 | +161,18 | 261,54 | 3,37 |
| cc065 | 6,5%/K | 888,18 | +169,27 | 264,25 | 3,36 |
| cc065_jitter010 | 6,5%/K + jitter 1% | 888,40 | +169,49 | 264,23 | 3,36 |

O resultado é decisivo para a interpretação: retirar completamente a resposta de Clausius-Clapeyron reduz a anomalia DJF em aproximadamente **36 mm**, mas ainda deixa o THOR em **+133 mm**, acima do P90 aproximado do CCKP (`+129,38 mm`). Portanto, o termo de 6,5%/K explica parte da amplitude, mas não é a causa única. O restante é produzido pelo mapeamento aprendido, pelas features sintéticas e pela transferência do domínio CORDEX para Guarulhos.

O caso `0%/K` não deve ser interpretado como fisicamente realista ou como “nenhuma mudança climática”. Ele é um controle contrafactual: remove apenas a resposta termodinâmica explicitamente parametrizada no TCWV. Mesmo nesse controle, a rede ainda responde às mudanças de temperatura, precipitação, calendário e demais proxies do CORDEX. Por isso, o fato de a anomalia continuar em `+133,27 mm` é evidência de que a superestimação não está confinada às três linhas do fator de 6,5%/K.

O jitter de 1% quase não alterou a média (`+169,27` para `+169,49 mm`), portanto aleatoriedade diária não resolve o viés de amplitude. A conclusão correta não é substituir 6,5% por um número aleatório, e sim tratar a resposta termodinâmica como uma incerteza de sensibilidade e investigar também a saída da rede e os proxies sinóticos.

Arquivos principais:

- `.agents/sensitivity/cc_sensitivity_summary_ssp585.csv`;
- `.agents/sensitivity/cc_sensitivity_ssp585_300dpi.png`;
- cinco subdiretórios com CSVs completos de 26.640 dias e logs de execução.

### 8.7 Historical model spread

The historical CMIP6 envelope was also downloaded from CCKP for Sao Paulo, using 1995-2014 as reference. For annual precipitation, the values were P10 = **1408.01 mm**, median = **1442.83 mm**, and P90 = **1489.62 mm**. The historical ensemble spread around the median is therefore approximately **-34.82/+46.79 mm**.

This spread should be reported, but it should not be mechanically subtracted from the future anomaly. It describes disagreement among models in the historical absolute climatology; it is not the standard deviation of the future anomaly and does not represent a correctable THOR error. Subtracting 15 mm from the 40 mm difference would be a post-hoc adjustment. The defensible presentation is to show THOR alongside the CCKP range and state that its signal is above the ensemble central range, while historical uncertainty prevents a binary right/wrong judgment.

Downloaded historical file: `.agents/cckp_data/sp_pr_historical_1995-2014_all.csv`.

Somando os três meses de DJF individualmente, a referência mensal produz uma faixa aproximada de **-123,92 a +129,38 mm**, com mediana de **+6,06 mm**. Essa soma dos percentis não é um percentil sazonal conjunto e deve ser tratada como aproximação, mas é suficiente para mostrar a ordem de grandeza da dispersão independente.

O THOR projeta aproximadamente **+169 mm em DJF** no período final do SSP5-8.5. Portanto, o valor do THOR fica acima do P90 aproximado do CCKP por cerca de 40 mm. A conclusão correta é que a amplitude do THOR é **mais alta que a referência multimodelo disponível**, não que esteja definitivamente errada: há diferença de domínio espacial, período (`2076-2099` versus `2080-2099`), definição de baseline e dependência entre os meses. A evidência atual apoia a classificação “sinal de verão acima da faixa central do ensemble, exigindo investigação”, e não “amplitude validada”.

### 8.8 Dispersão histórica dos modelos

Também foi baixado o envelope histórico CMIP6 do CCKP para São Paulo, usando 1995-2014 como referência. Para a precipitação anual, os valores foram P10 = **1408,01 mm**, mediana = **1442,83 mm** e P90 = **1489,62 mm**. Assim, a dispersão histórica do ensemble em torno da mediana é aproximadamente **-34,82/+46,79 mm**.

Essa dispersão deve ser reportada, mas não deve ser subtraída mecanicamente da anomalia futura. Ela descreve a divergência entre modelos na climatologia histórica absoluta; não é o desvio padrão da anomalia futura e não representa um erro corrigível do THOR. Subtrair arbitrariamente 15 mm dos 40 mm de diferença seria ajuste pós-hoc. A forma defensável é apresentar o THOR junto com a faixa do CCKP e dizer que seu sinal está acima da faixa central, enquanto a incerteza histórica reforça que a comparação não é uma sentença binária de certo/errado.

Arquivo histórico baixado: `.agents/cckp_data/sp_pr_historical_1995-2014_all.csv`.
