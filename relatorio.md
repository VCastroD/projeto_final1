# Relatório: testes do Sentinela (Trilha 1)

Projeto Final, Testes Automatizados para Modelos de IA, IEC PUC Minas.
Sistema testado: [felipehp/sentinela-nortemec](https://github.com/felipehp/sentinela-nortemec), commit `c9020d7`, sem alterações.
Autoria: VCastroD

## 1. Resultado geral

A suíte tem 179 testes. Rodando contra o Sentinela publicado, 133 passam e 46 falham. As 46 falhas são de propósito: cada uma documenta um defeito do sistema e está marcada com `@pytest.mark.defeito("Dxx")`. No total foram 15 defeitos (D01 a D15).

Conclusão principal: **a v2 não deveria ser promovida.** O ganho de 3,5 pontos de acurácia só aparece porque, na avaliação, o rótulo vaza para uma das features (D01). Medindo do jeito que o sistema roda na planta, a v2 deixa passar 198 falhas no conjunto de teste, contra 41 da v1.

Lista dos defeitos:

| ID | Defeito | Gravidade |
|---|---|---|
| D01 | `maquina_risco` é calculado com o próprio rótulo do lote | crítica |
| D02 | `temp_media_6h` usa leituras do futuro (janela centrada) | crítica |
| D03 | o operador OP-07 é escalado quando a falha já é conhecida, e o modelo usa isso | crítica |
| D07 | a v2 perde recall (0,94 → 0,70) e custa mais que a v1 | crítica |
| D05 | vibração vazia vira 0,0; com o sensor morto o recall cai para 0,10 | alta |
| D06 | `prever_registro` ignora o nome das chaves do dicionário | alta |
| D10 | em produção o ruído do sensor de temperatura triplicou | alta |
| D08 | as probabilidades da v1 não são calibradas | média |
| D04 | pressão em psi não é convertida para bar | média |
| D09 | limiar 0,5 é ruim para a v2 | média |
| D11 | perturbações menores que o ruído do sensor mudam decisões | média |
| D14 | leituras impossíveis (−40 °C, 300 rpm) são aceitas | média |
| D15 | o manifest de integridade falha em Linux/Colab | média |
| D12 | janelas de "6 h" contam linhas, não horas | baixa |
| D13 | índice repetido duplica linhas em `construir` | baixa |

## 2. Termos usados

- **Recall:** das falhas que aconteceram, quantas o sistema avisou. É a métrica mais importante aqui, porque falha não avisada é motor queimado.
- **Precisão:** dos avisos que o sistema deu, quantos eram falha de verdade.
- **Falso negativo (FN):** motor falhou e o sistema não avisou. **Falso positivo (FP):** sistema avisou e o motor não falhou.
- **PR-AUC:** nota de 0 a 1 de quão bem o modelo ordena as leituras (falhas no topo). Um chute aleatório tira mais ou menos a prevalência, 0,16.
- **Calibração / ECE:** se o modelo diz "70%", deveria falhar mais ou menos 70% das vezes. O ECE mede a distância média entre o previsto e o que aconteceu.
- **Bootstrap:** sortear os dados de novo muitas vezes para ver quanto um número varia. Daí sai o intervalo de confiança (IC 95%).
- **McNemar:** teste que compara dois modelos nas mesmas leituras, olhando só onde eles discordam.
- **KS (Kolmogorov-Smirnov):** compara duas distribuições inteiras, não só a média. D vai de 0 (iguais) a 1.
- **Contrafactual:** mudar um único campo que não deveria importar e ver se a decisão muda.

## 3. Como os testes foram feitos

Organização dos arquivos em `tests/`:

| Arquivo | Bloco | Testes | Conteúdo |
|---|---|---|---|
| `test_contrato_dados.py` | A | 36 | dados contra a ficha técnica (faixas, unidades, faltantes, ruído, integridade) |
| `test_preprocessamento.py` | A | 14 | `limpar` |
| `test_features.py` | A | 31 | `construir` |
| `test_modelo.py` | A | 16 | modelo e `prever_registro` |
| `test_pipeline.py` | A | 15 | ponta a ponta e `avaliacao` |
| `test_estatistico.py` | B | 28 | v1 × v2, classe rara, limiar, calibração, teste × produção |
| `test_adversarial.py` | C | 30 | ruído, contrafactuais, casos-limite |
| `test_apoio.py` | – | 9 | confere as contas de `apoio/estatistica.py` |

Algumas escolhas:

- Os testes estatísticos calculam as features **sem** a coluna `falha_72h`, porque na planta ela não existe (ninguém sabe ainda se o motor vai falhar).
- O bootstrap sorteia blocos de "um motor em um dia", não linhas soltas. Leituras seguidas do mesmo motor são quase iguais, e sortear linha por linha daria intervalos estreitos demais.
- Premissa de custo: um falso negativo custa 10 vezes um falso positivo (motor queimado × equipe deslocada à toa).
- As janelas móveis foram testadas de dois jeitos: comparando com uma implementação própria ("média das 6 últimas horas") e mudando só o futuro para ver se o passado muda.
- Junto com os testes de defeito há testes que passam e servem de controle. Por exemplo, as outras 12 features passam no teste de causalidade e só `temp_media_6h` falha.

## 4. Linha de base

Limiar 0,5. A coluna "acurácia reportada" é como a equipe mediu (lote com o rótulo); as outras colunas usam as features sem o rótulo. A tabela completa está em [evidencias/linha_de_base.md](evidencias/linha_de_base.md) (`python scripts/linha_de_base.py`).

| conjunto | versão | acurácia reportada | acurácia | precisão | recall | PR-AUC | ECE | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| teste | v1 | 0,886 | 0,765 | 0,394 | 0,937 | 0,581 | 0,191 | 41 | 945 |
| teste | v2 | 0,921 | 0,850 | 0,513 | 0,698 | 0,672 | 0,058 | 198 | 433 |
| produção | v1 | 0,858 | 0,817 | 0,435 | 0,945 | 0,643 | 0,169 | 33 | 734 |
| produção | v2 | 0,866 | 0,857 | 0,499 | 0,771 | 0,638 | 0,072 | 137 | 464 |

No teste, 15,6% das leituras são falha. Um "modelo" que sempre responde "não vai falhar" acerta 84,4%, mais que a v1 (76,5%), e não pega nenhuma falha. Por isso a acurácia não serve para esta comparação, e o relatório usa recall, PR-AUC e custo.

## 5. Defeitos

Para cada defeito: o teste que falha, o que foi medido, a causa e o impacto para a manutenção. As mensagens de erro no [log](evidencias/pytest_log.txt) trazem os mesmos números.

### D01: o rótulo vaza para `maquina_risco`

- **Testes:** `test_features.py::test_features_nao_dependem_da_presenca_do_rotulo`, `::test_risco_do_motor_nao_e_o_proprio_rotulo_da_linha`, `test_pipeline.py::test_decisao_nao_depende_da_presenca_do_rotulo`, `::test_probabilidade_de_uma_leitura_nao_depende_do_proprio_rotulo`
- **Medido:** o mesmo lote de teste, com e sem a coluna `falha_72h`, gera decisões diferentes em 795 de 4.200 leituras na v1 (18,9%) e em 436 na v2 (10,4%). Num lote de uma linha só, `maquina_risco` fica igual ao rótulo da linha (0 ou 1). A acurácia da v1 cai de 0,886 para 0,765 quando o rótulo não está presente.
- **Causa:** `_risco_por_maquina` usa `groupby("id_maquina")["falha_72h"].transform("mean")` sempre que a coluna existe. Ou seja, a feature passa a ser a taxa de falha do próprio período que está sendo avaliado.
- **Impacto:** os números que a equipe usou para avaliar o sistema, inclusive o +3,5 pp da v2, não são os números da planta.

### D02: `temp_media_6h` lê o futuro

- **Testes:** `test_features.py::test_janela_movel_bate_com_a_definicao[temp_media_6h]`, `::test_feature_no_instante_t_so_usa_leituras_ate_t[temp_media_6h]`, `::test_recorte_e_lote_inteiro_concordam_nas_linhas_em_comum`
- **Medido:** 4.193 de 4.200 linhas diferem da "média das últimas 6 horas", com diferença de até 5,59 °C. Mudando só leituras depois de um horário de corte, 50 linhas antes do corte mudam. As outras 12 features não mudam. Pontuando hora a hora, só com o passado (como na planta), 36 decisões da v1 (0,9%) e 64 da v2 (1,5%) ficam diferentes da pontuação em lote.
- **Causa:** `rolling(6, center=True)`: a janela vai de t−3 até t+2.
- **Impacto:** é uma das duas features mais importantes do modelo, e o modelo foi treinado e avaliado vendo 2 horas à frente, que não existem em tempo real. Corrigir exige retreinar.

### D03: o modelo usa quem foi escalado para o turno

- **Testes:** `test_adversarial.py::test_escala_do_op07_nao_reage_ao_rotulo`, `::test_nenhum_campo_nao_fisico_entre_as_5_features_mais_usadas`, `::test_contrafactual_de_campo_nao_fisico_nao_muda_decisao`
- **Hipótese:** o código diz que o OP-07 é escalado para os "motores mais críticos". Se fosse isso, a presença dele não mudaria justo quando `falha_72h` vira 1, porque esse rótulo olha 72 horas para frente.
- **Medido:** no treino, o OP-07 está em 23% das horas antes do início da janela de falha e em 57% das horas depois. `operador_senior` é a 4ª feature mais importante nas duas versões. Trocar só o operador (OP-07 ↔ OP-01), com as mesmas leituras de sensor, muda 351 decisões na v1 (8,4%) e 462 na v2 (11,0%). Trocar só o turno muda 4 e 10 decisões.
- **Causa:** o OP-07 é chamado quando alguém já percebeu o problema. A feature é consequência da falha, não sinal dela.
- **Impacto:** no motor em que ninguém percebeu nada, justamente o caso para o qual o sistema existe, esse sinal não existe. E se a escala mudar (férias, troca de equipe), o modelo muda de comportamento sem nenhum motor ter mudado.

### D07: a v2 não é melhor que a v1

- **Testes:** `test_estatistico.py::test_v2_nao_perde_recall_em_relacao_a_v1`, `::test_promover_a_v2_nao_aumenta_o_custo_esperado`, `::test_vantagem_da_v2_em_pr_auc_se_mantem_em_producao`
- **Medido (teste, bootstrap):**
  - diferença de recall v2 − v1: IC 95% de −0,346 a −0,150.
  - das 655 falhas reais, 157 são pegas pela v1 e perdidas pela v2; o contrário não acontece nenhuma vez (McNemar p = 1,1·10⁻⁴⁷).
  - custo (10·FN + FP): 2.413 na v2 contra 1.355 na v1.
  - PR-AUC: a v2 é melhor no teste (esse teste passa), mas na produção a diferença some (IC de −0,066 a +0,056).
  - O ganho de acurácia da v2 é real (McNemar p < 0,001), só que ele vem de avisar menos.
- **Causa:** a v2 foi treinada sem reponderação de classe. Ela troca falsos positivos por falsos negativos, o que melhora a acurácia e piora justamente o erro caro.
- **Impacto:** promover a v2 multiplicaria por 4,8 os falsos negativos (41 → 198 leituras na semana de teste).

### D05: vibração vazia vira 0,0

- **Testes:** `test_preprocessamento.py::test_dropout_de_vibracao_nao_vira_valor_fisicamente_impossivel`, `test_adversarial.py::test_sensor_de_vibracao_morto_nao_esconde_falhas`
- **Medido:** 408 leituras do teste (9,7%) saem de `limpar` com vibração 0,0, fora da faixa física (1,2 a 8,0 mm/s). Com o sensor de vibração sem sinal a semana toda, o recall cai de 0,94 para 0,10 na v1 e de 0,70 para 0,24 na v2, sem nenhum aviso.
- **Causa:** `fillna(0.0)`. Para o modelo, "sem leitura" vira "motor sem vibração", ou seja, saudável.
- **Impacto:** quando um sensor quebra, o sistema para de enxergar as falhas.

### D06: `prever_registro` ignora os nomes das chaves

- **Testes:** `test_modelo.py::test_prever_registro_independe_da_ordem_das_chaves`, `::test_prever_registro_recusa_chaves_que_nao_sao_features`
- **Medido:** as mesmas 13 features com as chaves em ordem alfabética mudam 115 de 300 decisões na v1 (38,3%) e 67 na v2 (22,3%). Um dicionário com chaves inventadas (`sensor_0`, `sensor_1`, ...) é aceito sem erro.
- **Causa:** o método usa `registro.values()`, pela posição, e não pelo nome.
- **Impacto:** é o caminho do painel da sala de controle. Se o código que monta o dicionário mudar a ordem das chaves, as decisões mudam sem nenhum erro.

### D10: o sensor de temperatura ficou mais ruidoso em produção

- **Testes:** `test_contrato_dados.py::test_ruido_do_sensor_de_temperatura_respeita_a_ficha[producao]` (treino e teste passam), `test_estatistico.py::test_distribuicao_das_variacoes_hora_a_hora_estavel[temperatura_c]`, `::test_mesma_media_nao_significa_mesma_distribuicao`
- **Medido:** o ruído estimado do sensor é 1,04 °C no treino, 1,08 °C no teste e 3,25 °C na produção, em todos os 25 motores e em todos os dias. No KS das variações hora a hora, a temperatura dá D = 0,259; os outros sensores ficam em até 0,021. A média de temperatura do treino e da produção é a mesma (67,23 °C), mas o desvio-padrão sobe 52%. Os testes que comparam só médias passam para todos os sensores.
- **Causa:** mudança no sensor ou na aquisição entre o teste e a produção. A ficha diz ±1,0 °C. O Sentinela não monitora os dados de entrada.
- **Impacto:** as features de temperatura estão entre as mais importantes do modelo, e com esse ruído as decisões oscilam mais (ver D11).

### D08: as probabilidades da v1 não são calibradas

- **Teste:** `test_estatistico.py::test_probabilidade_e_calibrada[v1]` (a v2 passa)
- **Medido:** ECE da v1 = 0,191. A probabilidade média prevista é 0,347, mas só 15,6% das leituras falham. Na faixa de 40% a 50%, a v1 prevê 45% e a frequência real é 2%. A v2 tem ECE 0,058, mas também exagera no meio da escala (prevê 65% onde acontece 34%).
- **Causa:** o treino com `class_weight="balanced_subsample"` puxa as probabilidades para cima, e não houve recalibração depois.
- **Impacto:** um "70% de chance de falha" no painel da v1 corresponde, na prática, a uns 35%.

### D04: pressão em psi não é convertida

- **Testes:** `test_preprocessamento.py::test_pressao_sai_de_limpar_em_uma_unidade_so`, `::test_mesma_pressao_fisica_em_bar_ou_psi_da_o_mesmo_valor_limpo`, `test_adversarial.py::test_contrafactual_mesma_pressao_na_outra_unidade_nao_muda_decisao`
- **Medido:** 1.344 linhas do teste (motores M18 a M25) saem de `limpar` em psi (~57) misturadas com as em bar (~3,9). O dado em si está certo: convertendo pela coluna `unidade_pressao`, 99,9% cai na faixa. Mandar a mesma pressão na outra unidade muda 41 decisões na v1 e 161 na v2.
- **Causa:** `limpar` padroniza o texto da coluna `unidade_pressao`, mas nunca converte o valor.
- **Impacto:** a feature `pressao` vira, na prática, um indicador de qual CLP enviou a leitura.

### D09: limiar 0,5 para a v2

- **Teste:** `test_estatistico.py::test_limiar_05_esta_perto_do_custo_minimo[v2]` (para a v1, 0,5 já é o melhor limiar e o teste passa)
- **Medido:** custo da v2 = 2.413 no limiar 0,5 (recall 0,70) e 1.378 no limiar 0,15 (recall 0,94). O limiar escolhido no teste também é melhor na produção, então não é sobreajuste.
- **Causa:** o mesmo limiar fixo para dois modelos com calibrações diferentes.
- **Impacto:** mesmo no melhor limiar, a v2 só empata com a v1 (1.378 contra 1.355).

### D11: ruído menor que o do sensor muda decisões

- **Teste:** `test_adversarial.py::test_perturbacao_dentro_do_ruido_do_sensor_nao_muda_decisao` (critério: no máximo 1% das decisões)
- **Medido:**

  | perturbação na temperatura | v1 | v2 |
  |---|---|---|
  | ruído aleatório de ±1,0 °C | 1,81% | 2,48% |
  | +0,5 °C em tudo | 2,93% | 4,45% |
  | −0,5 °C em tudo | 3,62% | 3,98% |

  Decisões com probabilidade longe de 0,5 (|p − 0,5| ≥ 0,25) não mudam (teste de controle passa).
- **Causa:** as árvores cortam a temperatura em limiares muito próximos; meio grau chega a mudar a probabilidade em até 0,4.
- **Impacto:** a planta toma cerca de 600 decisões por dia (25 motores × 24 h). Entre 2 e 4% delas mudam por causa de ruído, o que dá umas 12 a 24 ordens por dia abrindo ou fechando sem que o motor tenha mudado.

### D14: leituras impossíveis são aceitas

- **Teste:** `test_preprocessamento.py::test_leitura_fisicamente_impossivel_e_barrada_ou_sinalizada`
- **Medido:** −40 °C (valor típico de termopar desconectado), 250 °C, 300 rpm e 0,5 A passam por `limpar` e viram decisão. Com −40 °C em todas as leituras, a v1 decide "não abrir ordem" em 99,2% dos casos.
- **Causa:** o único filtro de `limpar` é `rpm >= 1`.
- **Impacto:** um sensor desconectado aparece como "motor saudável", em vez de gerar um alarme.

### D15: o manifest só confere em Windows

- **Teste:** `test_contrato_dados.py::test_manifest_bate_com_o_conteudo_versionado_no_git`
- **Medido:** o git guarda os CSV e o `risco_maquina.json` com quebra de linha LF, mas o `manifest.json` tem o sha256 dessas mesmas cópias com CRLF (Windows). Clonando com `core.autocrlf=false`, que é o que acontece em Linux, macOS e Colab, `verificar_manifest()` aponta 4 arquivos como diferentes.
- **Causa:** o manifest foi gerado em Windows e o repositório não tem `.gitattributes`.
- **Impacto:** o primeiro teste de exemplo do próprio repositório falha no Colab, que é o ambiente recomendado. A suíte compara o conteúdo ignorando a quebra de linha, para não depender do sistema operacional.

### D12: janela de "6 h" conta linhas

- **Teste:** `test_features.py::test_janela_de_6h_cobre_6_horas_mesmo_com_leitura_removida` (sem leitura removida, passa)
- **Medido:** num motor de teste com a leitura das 20 h descartada (eixo parado), `vib_media_6h` às 24 h dá 4,117, que é a média de 6 linhas cobrindo 7 horas. O certo seria 4,180, a média das leituras das últimas 6 horas.
- **Causa:** `rolling(6)` e `shift(24)` contam posições; `limpar` remove linhas.
- **Impacto:** nenhum nos dados atuais, que não têm buracos, mas acontece em qualquer semana com parada de motor.

### D13: índice repetido duplica linhas

- **Teste:** `test_features.py::test_indice_repetido_nao_duplica_linhas`
- **Medido:** dois motores concatenados sem `reset_index` (336 linhas) geram 672 linhas de features.
- **Causa:** `saida.loc[df.index]` com índice repetido.
- **Impacto:** o `pipeline` não é afetado, porque `limpar` reseta o índice. Quem chama `construir` direto recebe linhas duplicadas.

## 6. O que funciona

- **Dados:** colunas certas, uma leitura por motor por hora sem buracos, conjuntos em sequência (28/7/7 dias), sensores dentro da faixa, pressão dentro da faixa depois de converter a unidade, turno coerente com a hora, faltante só na vibração.
- **`limpar`:** não altera a entrada, converte os tipos, padroniza texto, remove só as leituras de eixo parado e mantém a ordem.
- **`construir`:** 13 colunas na ordem certa, mesmo resultado com as linhas embaralhadas, janelas separadas por motor, 5 das 6 janelas batem com a definição e 12 das 13 features só usam o passado.
- **Modelo e pipeline:** probabilidades entre 0 e 1, decisão coerente com o limiar, recusa NaN, funciona com lote vazio, com um motor só e com motor novo, e um motor a 110 °C gera ordem.
- **Estatística:** o ganho de acurácia da v2 é significativo, a v2 está razoavelmente calibrada, a taxa de falhas não muda entre teste e produção, e vibração, corrente e rpm não mudaram de distribuição.

## 7. Correções (opcional)

Primeiro os testes foram versionados falhando; a correção veio depois, em um patch separado, sem mexer no submódulo: [correcoes/0001-corrige-D01-D06-D12-D13-D15.patch](correcoes/0001-corrige-D01-D06-D12-D13-D15.patch). O passo a passo está em [correcoes/README.md](correcoes/README.md).

O patch corrige só o que não obriga a retreinar o modelo:

| Defeito | Correção |
|---|---|
| D01 | `maquina_risco` sempre usa a tabela congelada do treino |
| D06 | `prever_registro` lê pelo nome e recusa chaves erradas |
| D12 | janelas medidas em horas (`rolling("6h")`) |
| D13 | `construir` trabalha por posição e devolve o índice original |
| D15 | manifest refeito sobre os arquivos LF e `.gitattributes` com `eol=lf` |

Com o patch ([log](correcoes/pytest_log_corrigido.txt)): 143 passam e 36 falham. Os 11 testes desses 5 defeitos passam. Um teste que antes passava agora falha, e isso era esperado: `test_reproduz_o_ganho_de_acuracia_reportado_pela_equipe`. Sem o vazamento, a diferença de acurácia entre v2 e v1 é de 8,5 pp, e não mais de 3,5 pp, o que confirma que o número do README vinha do D01.

D02, D03, D04 e D05 mudam as features usadas no treino e só podem ser corrigidos retreinando o modelo (`scripts/treinar.py`). D07 a D11 e D14 ficam como recomendação: recalibrar, ajustar o limiar, monitorar a entrada, exigir alertas seguidos antes de abrir ordem e validar as faixas da ficha.

## 8. Limitações

- O custo 10:1 é uma premissa. A conclusão de D07 (perda de recall) não depende dela; a de D09 (melhor limiar) depende.
- O limite de 1% em D11 é um critério escolhido por mim. Os valores medidos estão no relatório.
- O bootstrap supõe que dias diferentes são independentes. Mesmo que os intervalos estejam um pouco estreitos, a conclusão de D07 tem folga grande.
- Os dados são sintéticos. A explicação de D03 é a mais simples para o salto do OP-07 no início da janela de falha, mas só quem opera a planta pode confirmar.

## 9. Execução

Log completo em [evidencias/pytest_log.txt](evidencias/pytest_log.txt) (`python -m pytest -v --tb=line`, Python 3.11.9, pytest 9.1.1, Windows 11). Final do log:

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
======================= 46 failed, 133 passed in 15.75s =======================
```

Algumas mensagens de falha:

```
test_pipeline.py: AssertionError: 795 de 4200 decisões (18.9%) mudam com o rótulo presente
test_features.py: AssertionError: temp_media_6h: 50 linhas do passado mudaram quando só o futuro mudou (ex.: 2026-06-03 11:00:00)
test_adversarial.py: AssertionError: OP-07 presente em 23% das 24 h antes e 57% das 24 h depois
test_estatistico.py: AssertionError: recall v2 - v1: IC95% [-0.346, -0.150]; entre as falhas reais, v1 pega e v2 perde 157, o contrário 0 (McNemar p=1.1e-47)
test_modelo.py: AssertionError: 115 de 300 decisões (38.3%) mudam só pela ordem das chaves
test_contrato_dados.py: AssertionError: producao: sigma estimado = 3.25 °C
test_adversarial.py: AssertionError: v1: recall 0.94 -> 0.10 com o sensor morto
```
