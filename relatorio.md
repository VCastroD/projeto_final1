# Relatório — Suíte de testes do Sentinela (Trilha 1, ML clássico)

Projeto Final — *Testes Automatizados para Modelos de IA* — IEC PUC Minas
Sistema sob teste: [`felipehp/sentinela-nortemec`](https://github.com/felipehp/sentinela-nortemec) @ `c9020d7` (versão publicada, não editada)
Autoria: VCastroD

---

## 1. Resumo

A suíte tem **179 testes**. Contra a versão publicada do Sentinela, **133 passam e 46 falham**. As 46 falhas são intencionais: cada uma documenta um de **15 defeitos reais (D01–D15)**. Todo teste que documenta defeito leva o marcador `@pytest.mark.defeito("Dxx")`, e o `conftest.py` imprime no fim do log um resumo por defeito.

**Recomendação para a Nortemec: não promover a v2.** O ganho de +3,5 pp de acurácia que motivou a promoção existe, mas foi medido com o rótulo vazando para as features (D01). No caminho real de produção, a v2 deixa passar **198 falhas contra 41 da v1** (recall 0,70 × 0,94). Entre as falhas reais do teste, há 157 que a v1 pega e a v2 perde, e **nenhuma** no sentido contrário (McNemar p = 1,1·10⁻⁴⁷). Além disso, as duas versões dependem de uma variável que só existe porque alguém já sabia da falha (D03) e de uma média de temperatura que lê o futuro (D02).

| ID | Defeito | Onde | Severidade |
|---|---|---|---|
| D01 | `maquina_risco` recalculado com o rótulo do próprio lote: vazamento do alvo | `features._risco_por_maquina` | **Crítica** |
| D02 | `temp_media_6h` é janela **centrada**: lê 2 h do futuro | `features.construir` | **Crítica** |
| D03 | `operador_senior` (OP-07) é proxy do rótulo; operador e turno mudam a decisão | `features.construir` / modelo | **Crítica** |
| D07 | v2 perde recall (0,94 → 0,70) e aumenta o custo; a promoção seria uma regressão | modelo v2 | **Crítica** |
| D05 | Dropout de vibração vira 0,0 mm/s (impossível); sensor morto derruba o recall para 0,10 | `preprocessamento.limpar` | Alta |
| D06 | `prever_registro` ignora os nomes das chaves: a ordem do dicionário muda 38% das decisões | `Modelo.prever_registro` | Alta |
| D10 | Ruído do sensor de temperatura triplicou em produção (σ 1,08 → 3,25 °C), com a mesma média | dados de produção | Alta |
| D08 | "Probabilidade" da v1 não é probabilidade (ECE 0,19; prevê 45% onde ocorre 2%) | modelo v1 | Média |
| D04 | Pressão em psi nunca é convertida para bar (32% das linhas) | `preprocessamento.limpar` | Média |
| D09 | Limiar 0,5 inadequado para a v2 (custo 75% acima do ótimo) | `LIMIAR_PADRAO` | Média |
| D11 | Perturbação menor que o ruído do sensor muda de 1,8% a 4,5% das decisões | modelo | Média |
| D14 | Leituras fisicamente impossíveis (−40 °C, 300 rpm) passam caladas e viram decisão | `preprocessamento.limpar` | Média |
| D15 | Manifest de integridade depende do fim de linha do SO: falha em Linux/Colab | `artefatos/manifest.json` | Média |
| D12 | Janelas "de 6 h/24 h" contam linhas, não horas | `features.construir` | Baixa |
| D13 | Índice de entrada repetido duplica as linhas da saída | `features.construir` | Baixa |

---

## 2. Método

**Estrutura da suíte** (`tests/`):

| Arquivo | Bloco | Testes | O que cobre |
|---|---|---|---|
| `test_contrato_dados.py` | A | 36 | `dados/` contra a ficha técnica: colunas, grade horária, faixas, unidades, faltantes, ruído do sensor, integridade |
| `test_preprocessamento.py` | A | 14 | `limpar`: tipos, faltantes, normalização, eixo parado, unidades, faixas |
| `test_features.py` | A | 31 | `construir`: ordem, embaralhamento, oráculo de cada janela, causalidade, recorte × lote, rótulo, índice |
| `test_modelo.py` | A | 16 | contrato do modelo e de `prever_registro` |
| `test_pipeline.py` | A | 15 | ponta a ponta, lote vazio/um motor/motor novo, `avaliacao` |
| `test_estatistico.py` | B | 28 | métrica para classe rara, v1 × v2 com incerteza, limiar, calibração, teste × produção |
| `test_adversarial.py` | C | 30 | ruído do sensor, importância por permutação, contrafactuais, casos-limite |
| `test_apoio.py` | — | 9 | os próprios instrumentos estatísticos, contra respostas conhecidas |

**Decisões de método**

- **Caminho de produção.** Toda métrica do bloco B é calculada com as features construídas **sem** a coluna `falha_72h`, porque é assim que o sistema roda de verdade. O caminho "rotulado" (como a equipe mediu) aparece só para mostrar a diferença (D01).
- **Incerteza com dependência temporal.** Leituras horárias do mesmo motor não são independentes. Os intervalos de confiança usam **bootstrap por blocos motor × dia** (175 blocos no teste, 1.000 reamostras). O McNemar (exato, binomial) é usado como teste pareado complementar.
- **Premissa de custo.** Um falso negativo (motor queimado, parada não programada) custa **10×** um falso positivo (equipe deslocada à toa). A premissa está declarada em `tests/test_estatistico.py` (`CUSTO_FN`, `CUSTO_FP`). As conclusões de D07 não dependem dela: o recall cai com qualquer custo.
- **Oráculo e perturbação.** As janelas móveis são verificadas de duas formas independentes: contra uma reimplementação direta da definição ("média das 6 últimas horas") e por perturbação ("muda tudo depois de t; nada até t pode mudar").
- **Instrumentos testados.** PR-AUC, ECE, McNemar, bootstrap, KS e PSI foram escritos em numpy (`apoio/estatistica.py`). Cada um tem teste com resposta conhecida (`test_apoio.py`), por exemplo o McNemar contra o valor tabelado p = 0,109375 para b = 2, c = 8.
- **Controles.** Os testes de defeito vêm acompanhados de testes que passam e delimitam o problema. A janela sem lacuna bate com a definição, as outras 12 features respeitam a causalidade, e o ruído não vira decisões confiantes. Assim a falha aponta o defeito, não o instrumento.

---

## 3. Linha de base

Limiar 0,5; *acurácia reportada* = `pipeline.executar` sobre o lote com o rótulo (como a equipe mediu). As demais colunas usam o caminho de produção. Tabela completa: [`evidencias/linha_de_base.md`](evidencias/linha_de_base.md), gerada por `python scripts/linha_de_base.py`.

| conjunto | versão | acurácia reportada | acurácia | precisão | recall | F1 | PR-AUC | ECE | FN | FP |
|---|---|---|---|---|---|---|---|---|---|---|
| teste | v1 | 0,886 | 0,765 | 0,394 | **0,937** | 0,555 | 0,581 | 0,191 | **41** | 945 |
| teste | v2 | 0,921 | 0,850 | 0,513 | **0,698** | 0,592 | 0,672 | 0,058 | **198** | 433 |
| produção | v1 | 0,858 | 0,817 | 0,435 | 0,945 | 0,596 | 0,643 | 0,169 | 33 | 734 |
| produção | v2 | 0,866 | 0,857 | 0,499 | 0,771 | 0,606 | 0,638 | 0,072 | 137 | 464 |

A prevalência de falha no teste é 15,6%. O preditor "nunca falha" tem **84,4% de acurácia**, mais que a v1 (76,5%), e recall zero. A acurácia ordena os dois ao contrário do que a manutenção precisa. As métricas usadas daqui em diante são **recall** (quantos motores que vão falhar são pegos), **PR-AUC** (qualidade da ordenação para classe rara) e **custo** (10·FN + FP).

---

## 4. Defeitos encontrados

Cada defeito tem as quatro partes pedidas: **teste que falha**, **evidência medida**, **causa raiz** e **impacto na planta**. A mensagem de cada `assert` carrega o número medido, que também aparece no log ([`evidencias/pytest_log.txt`](evidencias/pytest_log.txt)).

### D01 — O rótulo vaza para a feature `maquina_risco` (crítica)

- **Testes:** `test_features.py::test_features_nao_dependem_da_presenca_do_rotulo`, `::test_risco_do_motor_nao_e_o_proprio_rotulo_da_linha`, `test_pipeline.py::test_decisao_nao_depende_da_presenca_do_rotulo[v1,v2]`, `::test_probabilidade_de_uma_leitura_nao_depende_do_proprio_rotulo`.
- **Evidência:** o mesmo lote de teste, com e sem a coluna `falha_72h`, dá decisões diferentes em **795 de 4.200 leituras (18,9%) na v1** e 436 (10,4%) na v2. Num lote de uma linha só, `maquina_risco` vale exatamente o rótulo da linha (0,0 ou 1,0) e a probabilidade vai de 0,130 para 0,271 só por trocar o rótulo. A acurácia da v1 cai de 0,886 (reportada) para 0,765 (produção).
- **Causa raiz:** `_risco_por_maquina` faz `groupby("id_maquina")["falha_72h"].transform("mean")` quando a coluna existe. Em qualquer lote rotulado (teste, produção rotulada depois), a feature passa a ser a taxa de falha **observada no próprio período avaliado**. É informação do futuro, que em produção não existe.
- **Impacto:** todo número que a equipe usou para avaliar o sistema, incluindo o "+3,5 pp" da v2, foi medido num sistema diferente do que roda na planta. A decisão de promoção se apoia numa métrica inflada.

### D02 — `temp_media_6h` lê o futuro (crítica)

- **Testes:** `test_features.py::test_janela_movel_bate_com_a_definicao[temp_media_6h]`, `::test_feature_no_instante_t_so_usa_leituras_ate_t[temp_media_6h]`, `::test_recorte_e_lote_inteiro_concordam_nas_linhas_em_comum`.
- **Evidência:** 4.193 de 4.200 linhas divergem da definição "média das 6 últimas horas", em até 5,59 °C. Mudando só leituras posteriores a um corte, 50 linhas **anteriores** ao corte mudam. As outras 12 features passam no mesmo teste (controle). Pontuando hora a hora, só com o passado disponível (como na planta), 36 decisões da v1 (0,9%) e 64 da v2 (1,5%) diferem da pontuação em lote.
- **Causa raiz:** `rolling(JANELA_CURTA, center=True, ...)`: a janela de 6 posições cobre t−3 … t+2.
- **Impacto:** o modelo foi treinado e avaliado vendo 2 h à frente, mas em tempo real essas 2 h não existem. Ela também está entre as duas features mais importantes das duas versões por permutação (queda de PR-AUC de 0,16 na v1 e 0,22 na v2), então o desempenho medido superestima o real. Corrigir exige retreinar.

### D03 — O modelo aprendeu quem é escalado quando a falha já é conhecida (crítica)

- **Testes:** `test_adversarial.py::test_escala_do_op07_nao_reage_ao_rotulo`, `::test_nenhum_campo_nao_fisico_entre_as_5_features_mais_usadas[v1,v2]`, `::test_contrafactual_de_campo_nao_fisico_nao_muda_decisao[...]`.
- **Hipótese e o teste que a mata:** o comentário em `features.py` diz que o OP-07 "é escalado para os motores mais críticos". Se fosse escala por criticidade do motor, a presença do OP-07 não teria por que mudar no instante em que `falha_72h` vira 1, porque esse rótulo é definido olhando 72 h **à frente**. Medido no treino: o OP-07 está em **23% dos turnos nas 24 h antes** do início da janela de falha e em **57% nas 24 h depois**. No treino, P(OP-07 | falha) = 58,5% contra P(OP-07 | sem falha) = 17,8%.
- **Evidência de uso pelo modelo:** `operador_senior` é a 4ª feature mais importante nas duas versões (queda de PR-AUC 0,094 na v1 e 0,080 na v2). Trocar só o operador (OP-07 ↔ OP-01), com as mesmas leituras de sensor, muda **351 decisões (8,4%) na v1 e 462 (11,0%) na v2**. Trocar só o turno muda 4 (0,10%) e 10 (0,24%).
- **Causa raiz:** a variável é consequência da falha anunciada (alguém viu o problema e chamou o sênior), não causa nem sintoma físico. É um vazamento do alvo por via operacional.
- **Impacto:** o modelo "acerta" em parte porque o sênior já está lá. No motor em que ninguém percebeu nada, e é para esse que o sistema existe, o sinal não existe. Se a escala mudar (férias, troca de equipe), o desempenho muda sem nenhum motor mudar.

### D07 — A v2 não deve ser promovida (crítica)

- **Testes:** `test_estatistico.py::test_v2_nao_perde_recall_em_relacao_a_v1`, `::test_promover_a_v2_nao_aumenta_o_custo_esperado`, `::test_vantagem_da_v2_em_pr_auc_se_mantem_em_producao`.
- **Evidência (teste, caminho de produção, bootstrap por blocos):**
  - recall v2 − v1: **IC95% [−0,346; −0,150]**. Não-inferioridade (margem de 2 pp) rejeitada.
  - Entre as 655 falhas reais, a v1 pega e a v2 perde **157**; o contrário, **0** (McNemar exato p = 1,1·10⁻⁴⁷).
  - custo por leitura v2 − v1: IC95% [+0,072; +0,474], ou seja, 2.413 contra 1.355 no teste.
  - PR-AUC: a v2 ordena melhor **no teste** (teste que passa), mas em produção a diferença é IC95% [−0,066; +0,056]. A vantagem não se sustenta no lote seguinte.
  - Controles que passam: o ganho de acurácia da v2 é real e significativo (McNemar p < 0,001), e o "+3,5 pp" se reproduz com o pipeline rotulado.
- **Causa raiz:** a v2 foi treinada **sem reponderação de classe**. Ela troca falsos positivos por falsos negativos, o que melhora a acurácia numa classe com 15% de prevalência e piora exatamente o erro caro.
- **Impacto:** promover a v2 como está multiplicaria por 4,8 os falsos negativos (41 → 198 leituras-hora na semana de teste). Cada uma delas é um motor que vai falhar em até 72 h sem ordem preventiva aberta.

### D05 — Dropout de vibração vira 0,0 mm/s (alta)

- **Testes:** `test_preprocessamento.py::test_dropout_de_vibracao_nao_vira_valor_fisicamente_impossivel`, `test_adversarial.py::test_sensor_de_vibracao_morto_nao_esconde_falhas[v1,v2]`.
- **Evidência:** 408 leituras do teste (9,7%) saem de `limpar` com vibração 0,0, fora da faixa física de 1,2 a 8,0 mm/s. Com o sensor de vibração sem sinal na semana, o recall cai de **0,94 para 0,10 na v1** e de 0,70 para 0,24 na v2, e o sistema não emite nenhum aviso.
- **Causa raiz:** `fillna(VIBRACAO_PADRAO)` com `VIBRACAO_PADRAO = 0.0`. Para o modelo, "sem leitura" vira "motor sem nenhuma vibração", ou seja, saudável.
- **Impacto:** a falha de um sensor esconde a falha do motor. O sistema fica cego justamente quando perde instrumentação.

### D06 — `prever_registro` ignora os nomes das chaves (alta)

- **Testes:** `test_modelo.py::test_prever_registro_independe_da_ordem_das_chaves[v1,v2]`, `::test_prever_registro_recusa_chaves_que_nao_sao_features`.
- **Evidência:** as mesmas 13 features, com as chaves em ordem alfabética, mudam **115 de 300 decisões (38,3%) na v1** e 67 (22,3%) na v2. Um dicionário com 13 chaves inventadas (`sensor_0` …) é aceito sem erro. Controle: na ordem canônica, o resultado bate com a predição em lote.
- **Causa raiz:** `np.array([float(v) for v in registro.values()])`, que usa a posição e descarta o nome.
- **Impacto:** este é o caminho do **painel da sala de controle**. Qualquer mudança no código que monta o dicionário (um JSON reordenado, um campo novo) troca decisões em silêncio.

### D10 — O sensor de temperatura da produção ficou 3× mais ruidoso (alta)

- **Testes:** `test_contrato_dados.py::test_ruido_do_sensor_de_temperatura_respeita_a_ficha[producao]` (as versões `[treino]` e `[teste]` passam e servem de controle), `test_estatistico.py::test_distribuicao_das_variacoes_hora_a_hora_estavel[temperatura_c]`, `::test_mesma_media_nao_significa_mesma_distribuicao`.
- **Evidência:** σ do sensor, estimado pelas diferenças hora a hora de cada motor, é 1,04 °C no treino, 1,08 °C no teste e **3,25 °C na produção**. O aumento aparece nos 25 motores (2,9 a 3,8 °C) e em todos os 7 dias. KS sobre as variações hora a hora: D = 0,259 na temperatura contra D ≤ 0,021 nos outros sensores. A média da temperatura de treino e produção é **idêntica (67,23 °C)**, mas o desvio-padrão sobe 52%. Os testes de média passam para todos os sensores, e é por isso que eles não bastam.
- **Causa raiz (dado):** mudança no instrumento ou na aquisição entre o teste e a produção, fora da ficha (±1,0 °C). O Sentinela não monitora a entrada.
- **Impacto:** as features de temperatura são as mais importantes do modelo, e com ruído de 3 °C as decisões oscilam (ver D11). Sem monitoramento de drift, a degradação passaria despercebida.

### D08 — A "probabilidade" da v1 não é probabilidade (média)

- **Teste:** `test_estatistico.py::test_probabilidade_e_calibrada[v1]` (`[v2]` passa).
- **Evidência:** ECE da v1 = **0,191**; probabilidade média 0,347 contra prevalência 0,156. Na faixa [0,4; 0,5) a v1 prevê 45% e a frequência observada é **2%**. A v2 tem ECE 0,058 e passa no critério, mas também superestima no meio da escala (faixa [0,6; 0,7): prevê 0,65, observa 0,34).
- **Causa raiz:** `class_weight="balanced_subsample"` desloca as probabilidades para cima, e não houve recalibração depois.
- **Impacto:** o README diz que a saída "é uma probabilidade". Quem ler "70% de chance de falha" no painel está lendo um número que, na prática, corresponde a cerca de 35%.

### D04 — Pressão em psi nunca é convertida (média)

- **Testes:** `test_preprocessamento.py::test_pressao_sai_de_limpar_em_uma_unidade_so`, `::test_mesma_pressao_fisica_em_bar_ou_psi_da_o_mesmo_valor_limpo`, `test_adversarial.py::test_contrafactual_mesma_pressao_na_outra_unidade_nao_muda_decisao[v1,v2]`.
- **Evidência:** 1.344 linhas do teste (motores M18–M25, 32%) saem de `limpar` em psi (~57) ao lado das em bar (~3,9); só 68% da coluna está na faixa de 3,0 a 4,5 bar. O dado em si está certo: convertendo pela `unidade_pressao`, 99,9% cai na faixa (teste que passa). Expressar a **mesma** pressão física na outra unidade muda 41 decisões (0,98%) na v1 e 161 (3,83%) na v2.
- **Causa raiz:** `limpar` normaliza o **rótulo** `unidade_pressao` e nunca o **valor**.
- **Impacto:** a feature `pressao` vira, na prática, um indicador de "qual CLP", não de pressão. Trocar o CLP de um motor muda a decisão.

### D09 — O limiar 0,5 é arbitrário para a v2 (média)

- **Teste:** `test_estatistico.py::test_limiar_05_esta_perto_do_custo_minimo[v2]` (`[v1]` passa: para a v1, 0,5 é o ótimo da varredura).
- **Evidência:** custo da v2 = 2.413 em 0,5 (recall 0,70) contra **1.378 em 0,15** (recall 0,94). O limiar escolhido no teste também reduz o custo no lote de produção (teste que passa), então não é sobreajuste.
- **Causa raiz:** um limiar fixo para dois modelos com calibrações diferentes.
- **Impacto:** se a v2 for promovida, o limiar tem de ser reajustado. Mesmo assim, no ótimo ela empata com a v1 (1.378 × 1.355), sem ganho.

### D11 — Decisões mudam com perturbações menores que o ruído do sensor (média)

- **Teste:** `test_adversarial.py::test_perturbacao_dentro_do_ruido_do_sensor_nao_muda_decisao[...]` (critério: ≤ 1% das decisões).
- **Evidência:**

  | perturbação na temperatura | v1 | v2 |
  |---|---|---|
  | ruído uniforme ±1,0 °C (o próprio ruído da ficha) | 1,81% | 2,48% |
  | desvio de calibração +0,5 °C | 2,93% | 4,45% |
  | desvio de calibração −0,5 °C | 3,62% | 3,98% |

  Controles que passam: decisões com |p − 0,5| ≥ 0,25 não viram, e perturbação nula muda zero.
- **Causa raiz:** as árvores cortam a temperatura em limiares finos. Um deslocamento de 0,5 °C chega a mover a probabilidade em até 0,4. A instabilidade se concentra perto do limiar.
- **Impacto:** a planta toma cerca de 600 decisões por dia (25 motores × 24 h). Com 2 a 4% delas decididas pelo ruído, são 12 a 24 ordens por dia que abrem ou fecham sem que o motor tenha mudado. Com o ruído real da produção (D10), é pior. Mitigação: histerese ou exigir N alertas consecutivos.

### D14 — Leituras impossíveis passam caladas (média)

- **Teste:** `test_preprocessamento.py::test_leitura_fisicamente_impossivel_e_barrada_ou_sinalizada[...]`.
- **Evidência:** −40 °C (valor típico de termopar desconectado), 250 °C, 300 rpm e 0,5 A são aceitos por `limpar` nas 4.200 linhas e viram decisão. Com −40 °C em todas as leituras, a v1 decide "não abrir ordem" em **99,2%** delas.
- **Causa raiz:** o único filtro é `rpm >= 1`, e a ficha técnica não é aplicada em lugar nenhum.
- **Impacto:** um sensor desconectado produz "motor saudável" em vez de alarme de instrumentação.

### D15 — O manifest de integridade depende do sistema operacional (média)

- **Teste:** `test_contrato_dados.py::test_manifest_bate_com_o_conteudo_versionado_no_git`.
- **Evidência:** o git guarda os 3 CSV e o `risco_maquina.json` com fim de linha LF, mas o `manifest.json` tem o sha256 de cópias **CRLF**. Num checkout em Linux, macOS ou **Colab** (ambiente recomendado no README), `sn.artefatos.verificar_manifest()` devolve `['sha256 diferente: dados/treino.csv', 'sha256 diferente: dados/teste.csv', 'sha256 diferente: dados/producao.csv', 'sha256 diferente: artefatos/risco_maquina.json']`. No Windows com `core.autocrlf=true`, devolve `[]`. Reproduzido clonando com `core.autocrlf=false`.
- **Causa raiz:** o manifest foi gerado sobre a cópia de trabalho do Windows, e o repositório não tem `.gitattributes`.
- **Impacto:** o primeiro teste de exemplo do próprio repositório falha em metade das máquinas. Isso ensina a equipe a ignorar o verificador, e aí uma adulteração real passaria. A suíte compara o conteúdo ignorando o fim de linha (`test_dados_e_modelos_sao_os_publicados`), para que o resultado não dependa do SO de quem roda.

### D12 — Janelas contam linhas, não horas (baixa)

- **Teste:** `test_features.py::test_janela_de_6h_cobre_6_horas_mesmo_com_leitura_removida` (controle sem lacuna passa).
- **Evidência:** motor sintético com a leitura das 20 h descartada (eixo parado). Às 24 h, `vib_media_6h` = 4,117, média de 6 **linhas** que cobrem 7 h; o esperado é 4,180, média das 5 leituras das últimas 6 horas.
- **Causa raiz:** `rolling(6)` e `shift(24)` são posicionais, e `limpar` remove linhas.
- **Impacto:** nulo nos dados atuais, que têm grade completa, mas real em qualquer semana com paradas de motor.

### D13 — Índice repetido duplica linhas (baixa)

- **Teste:** `test_features.py::test_indice_repetido_nao_duplica_linhas`.
- **Evidência:** dois motores concatenados sem `reset_index` (336 linhas) produzem 672 linhas de features.
- **Causa raiz:** `saida.loc[df.index]` com rótulos repetidos.
- **Impacto:** o `pipeline` está protegido, porque `limpar` reseta o índice. Quem chama `construir` diretamente recebe linhas duplicadas e desalinhadas do alvo.

---

## 5. O que está correto (testes que passam)

Parte do sistema está certa, e a suíte mostra isso:

- **Dados:** colunas, grade horária completa (25 motores × 24 h × dias), janelas temporais consecutivas e disjuntas (28/7/7 dias), temperatura, vibração, corrente e rpm em faixa (≥ 99,9%), pressão em faixa **depois** de respeitar a unidade, operadores e turnos válidos, turno coerente com a hora, idade constante e horas de operação +1 por hora, dropout só na vibração (5 a 15%).
- **`limpar`:** não modifica a entrada, entrega tipos corretos, normaliza espaço e caixa, remove só o eixo parado, preserva a ordem e falha alto com faltante em sensor sem regra (temperatura).
- **`construir`:** 13 colunas na ordem canônica, invariante a embaralhamento, janelas isoladas por motor, determinístico, e 5 das 6 janelas batem com a definição. **12 das 13 features respeitam a causalidade.**
- **Modelo:** probabilidades em [0, 1], decisão coerente com o limiar, colunas selecionadas por nome no DataFrame, recusa NaN e forma errada.
- **Pipeline:** colunas prometidas, uma decisão por leitura na ordem de entrada, lote vazio, lote de um motor igual ao mesmo motor no lote inteiro, motor novo pontuado com o risco padrão, sensor travado sem NaN, motor superaquecido (110 °C) abre ordem.
- **Estatística:** o ganho de acurácia da v2 é significativo, a v2 ordena melhor no teste e está razoavelmente calibrada (ECE 0,058), a prevalência é estável entre teste e produção (z < 1,96), o PR-AUC não degrada em produção, e vibração, corrente e rpm não têm drift.

---

## 6. Correções (opcional): `correcoes/`

Seguindo a regra do guia, o teste veio primeiro: a suíte foi versionada com os testes falhando antes do patch. As correções ficam **fora** do pacote original, em [`correcoes/0001-corrige-D01-D06-D12-D13-D15.patch`](correcoes/0001-corrige-D01-D06-D12-D13-D15.patch). Esse patch se aplica com `git am` sobre `c9020d7`; o passo a passo está em [`correcoes/README.md`](correcoes/README.md).

O patch corrige apenas o que **não exige retreinar** os modelos publicados:

| Defeito | Correção | Resultado |
|---|---|---|
| D01 | `maquina_risco` usa sempre a tabela congelada do treino (no treino as duas coincidem, então o modelo não muda) | 5 testes passam a verde |
| D06 | `prever_registro` lê por nome e recusa chaves faltando ou desconhecidas | 3 testes passam |
| D12 | janelas e delta medidos em horas (`rolling("6h")`) | 1 teste passa |
| D13 | `construir` trabalha por posição e devolve o índice original | 1 teste passa |
| D15 | manifest regenerado sobre o conteúdo LF + `.gitattributes` com `eol=lf` | 1 teste passa |

Log com o patch: [`correcoes/pytest_log_corrigido.txt`](correcoes/pytest_log_corrigido.txt), com **143 passando e 36 falhando**. Os outros 10 defeitos continuam falhando, como esperado. Um teste que não é de defeito passa a falhar, como esperado: `test_reproduz_o_ganho_de_acuracia_reportado_pela_equipe`. Sem o vazamento, o ganho de acurácia da v2 é de 8,5 pp, não de 3,5 pp, e o número do README deixa de se reproduzir. Essa é a confirmação de que o "+3,5 pp" era um artefato de D01.

D02, D03, D04 e D05 mudam as features com que as florestas foram treinadas, e corrigi-los exige retreino (`scripts/treinar.py`). D07 a D11 são decisões de modelagem e operação: recalibrar, reajustar o limiar, monitorar drift, aplicar histerese. Todos ficam como recomendação.

---

## 7. Limitações

- O custo 10:1 é uma premissa. A conclusão de D07 (perda de recall) não depende dela; a de D09 (limiar ótimo) depende.
- O critério de 1% em D11 é um julgamento operacional, não uma norma. Os números medidos estão no relatório para quem quiser outro critério.
- O bootstrap por blocos motor × dia assume independência entre dias. Com autocorrelação maior que um dia, os intervalos ainda estariam um pouco otimistas. As conclusões de D07 têm folga grande (o limite superior do IC do recall é −0,15).
- Os dados são sintéticos. A interpretação causal de D03 (escala reativa do OP-07) é a explicação mais simples para o salto medido exatamente no início da janela de falha, mas só a operação da planta pode confirmá-la.

---

## 8. Evidência de execução

Log completo: [`evidencias/pytest_log.txt`](evidencias/pytest_log.txt) (`python -m pytest -v --tb=line`, 27/09/2026, Python 3.11.9, pytest 9.1.1, Windows 11). Resumo impresso pela própria suíte no fim do log:

```
================ defeitos do Sentinela documentados por testes ================
D01  CONFIRMADO       5 teste(s) falhando, 0 passando
D02  CONFIRMADO       3 teste(s) falhando, 0 passando
D03  CONFIRMADO       7 teste(s) falhando, 0 passando
D04  CONFIRMADO       4 teste(s) falhando, 0 passando
D05  CONFIRMADO       3 teste(s) falhando, 0 passando
D06  CONFIRMADO       3 teste(s) falhando, 0 passando
D07  CONFIRMADO       3 teste(s) falhando, 0 passando
D08  CONFIRMADO       1 teste(s) falhando, 0 passando
D09  CONFIRMADO       1 teste(s) falhando, 0 passando
D10  CONFIRMADO       3 teste(s) falhando, 2 passando
D11  CONFIRMADO       6 teste(s) falhando, 0 passando
D12  CONFIRMADO       1 teste(s) falhando, 0 passando
D13  CONFIRMADO       1 teste(s) falhando, 0 passando
D14  CONFIRMADO       4 teste(s) falhando, 0 passando
D15  CONFIRMADO       1 teste(s) falhando, 0 passando
...
======================= 46 failed, 133 passed in 37.93s =======================
```

Trechos das mensagens de falha, com os números medidos:

```
test_pipeline.py: AssertionError: 795 de 4200 decisões (18.9%) mudam com o rótulo presente
test_features.py: AssertionError: temp_media_6h: 50 linhas do passado mudaram quando só o futuro mudou (ex.: 2026-06-03 11:00:00)
test_adversarial.py: AssertionError: OP-07 presente em 23% das 24 h antes e 57% das 24 h depois
test_estatistico.py: AssertionError: recall v2 - v1: IC95% [-0.346, -0.150]; entre as falhas reais, v1 pega e v2 perde 157, o contrário 0 (McNemar p=1.1e-47)
test_modelo.py: AssertionError: 115 de 300 decisões (38.3%) mudam só pela ordem das chaves
test_contrato_dados.py: AssertionError: producao: sigma estimado = 3.25 °C
test_adversarial.py: AssertionError: v1: recall 0.94 -> 0.10 com o sensor morto
```
