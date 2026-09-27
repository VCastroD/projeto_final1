# Suíte de testes do Sentinela: Projeto Final, Trilha 1 (ML clássico)

Testes automatizados para o [Sentinela](https://github.com/felipehp/sentinela-nortemec), sistema de manutenção preditiva que prevê se um motor elétrico vai falhar nas próximas 72 h.

A suíte testa o sistema **de fora**. O pacote `sentinela` não foi copiado nem editado: ele entra como **submódulo git** fixado na versão publicada (`c9020d7`).

- **Relatório:** [`relatorio.md`](relatorio.md), com o que foi testado, por quê, e os 15 defeitos encontrados, cada um com teste, evidência, causa raiz e impacto.
- **Log da execução:** [`evidencias/pytest_log.txt`](evidencias/pytest_log.txt).
- **Linha de base numérica:** [`evidencias/linha_de_base.md`](evidencias/linha_de_base.md).
- **Correções opcionais (patch separado):** [`correcoes/`](correcoes/).

---

## Como rodar

Precisa de Python 3.10+ e git.

```bash
git clone --recurse-submodules https://github.com/VCastroD/projeto_final1.git
cd projeto_final1
pip install -r requirements.txt
python -m pytest
```

Se o repositório já foi clonado sem `--recurse-submodules`, rode `git submodule update --init` antes.

No **Google Colab**:

```python
!git clone --recurse-submodules https://github.com/VCastroD/projeto_final1.git
%cd projeto_final1
!pip install -q -r requirements.txt
!python -m pytest
```

Comandos úteis:

```bash
python -m pytest -m "not defeito"     # só o que o sistema faz certo (131 testes, todos passam)
python -m pytest -m defeito           # só os testes que documentam defeitos (falham de propósito)
python -m pytest tests/test_estatistico.py -v
python scripts/linha_de_base.py       # regenera evidencias/linha_de_base.md
```

Para rodar a suíte contra outra cópia do Sentinela (por exemplo, com o patch de `correcoes/`), defina `SENTINELA_DIR` apontando para a raiz dela. O cabeçalho do pytest mostra de onde o pacote foi importado.

### O que esperar

**179 testes: 133 passam e 46 falham.** As falhas são intencionais. O guia define que *"defeito encontrado se documenta com um teste que falha"*, então cada teste vermelho tem o marcador `@pytest.mark.defeito("Dxx")`. O `conftest.py` imprime no fim da execução um resumo por defeito. Qualquer falha **sem** esse marcador seria um problema da suíte, e não há nenhuma.

---

## Estrutura

```
projeto_final1/
  README.md                 este arquivo
  relatorio.md              relatório (achados, método, evidências)
  requirements.txt          numpy, pandas, pytest
  pytest.ini
  conftest.py               localiza o Sentinela, fixtures, marcador `defeito` e resumo
  apoio/
    estatistica.py          PR-AUC, ECE, McNemar exato, bootstrap por blocos, KS, PSI (numpy)
    ficha.py                ficha técnica dos sensores como dados (faixas, unidades, ruído)
    sistema.py              de onde importar o `sentinela` (submódulo ou SENTINELA_DIR)
  tests/
    test_contrato_dados.py  Bloco A: dados/ contra a ficha técnica + integridade
    test_preprocessamento.py  Bloco A: preprocessamento.limpar
    test_features.py        Bloco A: features.construir (oráculo, causalidade, rótulo)
    test_modelo.py          contrato do modelo e de prever_registro
    test_pipeline.py        ponta a ponta e sentinela.avaliacao
    test_estatistico.py     Bloco B: v1 x v2 com IC/McNemar, classe rara, limiar, calibração, drift
    test_adversarial.py     Bloco C: ruído do sensor, contrafactuais, importância, casos-limite
    test_apoio.py           testa os instrumentos estatísticos contra respostas conhecidas
  scripts/linha_de_base.py  tabela de métricas usada no relatório
  evidencias/               log do pytest e linha de base
  correcoes/                patch opcional (D01, D06, D12, D13, D15) + log com o patch
  sentinela-nortemec/       submódulo: sistema sob teste, intocado
```

---

## Principais achados

| ID | Defeito | Severidade |
|---|---|---|
| D01 | `maquina_risco` é recalculado com o **rótulo** do lote: 18,9% das decisões da v1 mudam só por a coluna `falha_72h` estar presente | Crítica |
| D02 | `temp_media_6h` usa janela **centrada** e lê 2 h do futuro | Crítica |
| D03 | O OP-07 é escalado quando a falha **já é conhecida** (23% → 57% no início da janela de falha); trocar só o operador muda 8 a 11% das decisões | Crítica |
| D07 | A v2 ganha acurácia mas perde recall (0,94 → 0,70; IC95% da diferença [−0,35; −0,15]); 157 falhas pegas pela v1 e perdidas pela v2, 0 no sentido inverso | Crítica |
| D05 | Dropout de vibração vira 0,0 mm/s; com o sensor morto o recall cai para 0,10 | Alta |
| D06 | `prever_registro` usa a ordem do dicionário e ignora os nomes: 38% das decisões mudam | Alta |
| D10 | Ruído do sensor de temperatura triplicou em produção (σ 1,08 → 3,25 °C), com a mesma média | Alta |
| D08 | v1 descalibrada (ECE 0,19; prevê 45% onde ocorre 2%) | Média |
| D04 | Pressão em psi nunca é convertida para bar | Média |
| D09 | Limiar 0,5 custa 75% acima do ótimo para a v2 | Média |
| D11 | Perturbação menor que o ruído do sensor muda 1,8 a 4,5% das decisões | Média |
| D14 | Leituras impossíveis (−40 °C, 300 rpm) viram decisão sem aviso | Média |
| D15 | O manifest de integridade falha em Linux/Colab (sha256 gerado sobre CRLF) | Média |
| D12 | Janelas "de 6 h" contam linhas, não horas | Baixa |
| D13 | Índice repetido duplica linhas em `construir` | Baixa |

**Recomendação:** não promover a v2. O "+3,5 pp de acurácia" foi medido com o rótulo vazando (D01), e no caminho real de produção a v2 multiplica por 4,8 os motores que falham sem ordem preventiva.

---

## Execução da suíte (saída do pytest)

Trecho final de [`evidencias/pytest_log.txt`](evidencias/pytest_log.txt) (27/09/2026, Python 3.11.9, pytest 9.1.1):

```
================ defeitos do Sentinela documentados por testes ================
D01  CONFIRMADO       5 teste(s) falhando, 0 passando
       - test_features_nao_dependem_da_presenca_do_rotulo
       - test_risco_do_motor_nao_e_o_proprio_rotulo_da_linha
       - test_decisao_nao_depende_da_presenca_do_rotulo[v1]
       - test_decisao_nao_depende_da_presenca_do_rotulo[v2]
       - test_probabilidade_de_uma_leitura_nao_depende_do_proprio_rotulo
D02  CONFIRMADO       3 teste(s) falhando, 0 passando
       - test_janela_movel_bate_com_a_definicao[temp_media_6h]
       - test_feature_no_instante_t_so_usa_leituras_ate_t[temp_media_6h]
       - test_recorte_e_lote_inteiro_concordam_nas_linhas_em_comum
D03  CONFIRMADO       7 teste(s) falhando, 0 passando
       - test_nenhum_campo_nao_fisico_entre_as_5_features_mais_usadas[v1]
       - test_nenhum_campo_nao_fisico_entre_as_5_features_mais_usadas[v2]
       - test_escala_do_op07_nao_reage_ao_rotulo
       - test_contrafactual_de_campo_nao_fisico_nao_muda_decisao[v1-operador_OP07_trocado]
       - test_contrafactual_de_campo_nao_fisico_nao_muda_decisao[v1-turno_trocado]
       - test_contrafactual_de_campo_nao_fisico_nao_muda_decisao[v2-operador_OP07_trocado]
       - test_contrafactual_de_campo_nao_fisico_nao_muda_decisao[v2-turno_trocado]
D04  CONFIRMADO       4 teste(s) falhando, 0 passando
       - test_contrafactual_mesma_pressao_na_outra_unidade_nao_muda_decisao[v1]
       - test_contrafactual_mesma_pressao_na_outra_unidade_nao_muda_decisao[v2]
       - test_pressao_sai_de_limpar_em_uma_unidade_so
       - test_mesma_pressao_fisica_em_bar_ou_psi_da_o_mesmo_valor_limpo
D05  CONFIRMADO       3 teste(s) falhando, 0 passando
       - test_sensor_de_vibracao_morto_nao_esconde_falhas[v1]
       - test_sensor_de_vibracao_morto_nao_esconde_falhas[v2]
       - test_dropout_de_vibracao_nao_vira_valor_fisicamente_impossivel
D06  CONFIRMADO       3 teste(s) falhando, 0 passando
       - test_prever_registro_independe_da_ordem_das_chaves[v1]
       - test_prever_registro_independe_da_ordem_das_chaves[v2]
       - test_prever_registro_recusa_chaves_que_nao_sao_features
D07  CONFIRMADO       3 teste(s) falhando, 0 passando
       - test_v2_nao_perde_recall_em_relacao_a_v1
       - test_promover_a_v2_nao_aumenta_o_custo_esperado
       - test_vantagem_da_v2_em_pr_auc_se_mantem_em_producao
D08  CONFIRMADO       1 teste(s) falhando, 0 passando
       - test_probabilidade_e_calibrada[v1]
D09  CONFIRMADO       1 teste(s) falhando, 0 passando
       - test_limiar_05_esta_perto_do_custo_minimo[v2]
D10  CONFIRMADO       3 teste(s) falhando, 2 passando
       - test_ruido_do_sensor_de_temperatura_respeita_a_ficha[producao]
       - test_distribuicao_das_variacoes_hora_a_hora_estavel[temperatura_c]
       - test_mesma_media_nao_significa_mesma_distribuicao
D11  CONFIRMADO       6 teste(s) falhando, 0 passando
       - test_perturbacao_dentro_do_ruido_do_sensor_nao_muda_decisao[v1-ruido_uniforme_1C]
       - test_perturbacao_dentro_do_ruido_do_sensor_nao_muda_decisao[v1-desvio_+0.5C]
       - test_perturbacao_dentro_do_ruido_do_sensor_nao_muda_decisao[v1-desvio_-0.5C]
       - test_perturbacao_dentro_do_ruido_do_sensor_nao_muda_decisao[v2-ruido_uniforme_1C]
       - test_perturbacao_dentro_do_ruido_do_sensor_nao_muda_decisao[v2-desvio_+0.5C]
       - test_perturbacao_dentro_do_ruido_do_sensor_nao_muda_decisao[v2-desvio_-0.5C]
D12  CONFIRMADO       1 teste(s) falhando, 0 passando
       - test_janela_de_6h_cobre_6_horas_mesmo_com_leitura_removida
D13  CONFIRMADO       1 teste(s) falhando, 0 passando
       - test_indice_repetido_nao_duplica_linhas
D14  CONFIRMADO       4 teste(s) falhando, 0 passando
       - test_leitura_fisicamente_impossivel_e_barrada_ou_sinalizada[temperatura_c--40.0]
       - test_leitura_fisicamente_impossivel_e_barrada_ou_sinalizada[temperatura_c-250.0]
       - test_leitura_fisicamente_impossivel_e_barrada_ou_sinalizada[rpm-300.0]
       - test_leitura_fisicamente_impossivel_e_barrada_ou_sinalizada[corrente_a-0.5]
D15  CONFIRMADO       1 teste(s) falhando, 0 passando
       - test_manifest_bate_com_o_conteudo_versionado_no_git
======================= 46 failed, 133 passed in 37.93s =======================
```

As duas execuções "passando" de D10 são os controles `[treino]` e `[teste]`: o ruído do sensor está dentro da ficha nesses conjuntos e só estoura na produção.
