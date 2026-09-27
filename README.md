# Testes do Sentinela: Projeto Final, Trilha 1

Suíte de testes para o [Sentinela](https://github.com/felipehp/sentinela-nortemec), sistema que prevê se um motor elétrico vai falhar nas próximas 72 horas.

O Sentinela não foi copiado nem alterado. Ele entra como submódulo git (pasta `sentinela-nortemec/`), fixado no commit publicado `c9020d7`.

O relatório com os defeitos encontrados está em [relatorio.md](relatorio.md).

## Como rodar

Precisa de Python 3.10 ou mais novo e do git.

```bash
git clone --recurse-submodules https://github.com/VCastroD/projeto_final1.git
cd projeto_final1
pip install -r requirements.txt
python -m pytest
```

Se o repositório já foi clonado sem `--recurse-submodules`, rode antes `git submodule update --init`.

No Google Colab:

```python
!git clone --recurse-submodules https://github.com/VCastroD/projeto_final1.git
%cd projeto_final1
!pip install -q -r requirements.txt
!python -m pytest
```

Outros comandos:

```bash
python -m pytest -v                          # lista cada teste
python -m pytest -m "not defeito"            # só o que o sistema faz certo (131 testes, todos passam)
python -m pytest -m defeito                  # só os testes que documentam defeitos
python -m pytest tests/test_estatistico.py   # um arquivo só
python scripts/linha_de_base.py              # gera evidencias/linha_de_base.md
```

## Resultado esperado

179 testes: 133 passam e 46 falham.

As falhas são de propósito. O enunciado pede que cada defeito seja documentado com um teste que falha, então esses testes estão marcados com `@pytest.mark.defeito("Dxx")`. No fim da execução, o pytest imprime a lista de defeitos confirmados.

## Pastas

```
projeto_final1/
  README.md               este arquivo
  relatorio.md            relatório: método, defeitos encontrados e evidências
  requirements.txt        dependências (numpy, pandas, pytest)
  pytest.ini              configuração do pytest
  conftest.py             acha o Sentinela, cria o marcador `defeito` e as fixtures de dados
  apoio/
    estatistica.py        PR-AUC, calibração, McNemar, bootstrap, KS, PSI
    ficha.py              ficha técnica dos sensores (faixas, unidades, ruído)
    sistema.py            decide de onde importar o Sentinela
  tests/
    test_contrato_dados.py    dados contra a ficha técnica
    test_preprocessamento.py  preprocessamento.limpar
    test_features.py          features.construir
    test_modelo.py            modelo e prever_registro
    test_pipeline.py          pipeline completo e sentinela.avaliacao
    test_estatistico.py       v1 x v2, limiar, calibração, teste x produção
    test_adversarial.py       ruído, contrafactuais, casos-limite
    test_apoio.py             confere as contas de apoio/estatistica.py
  scripts/
    linha_de_base.py      gera a tabela de métricas usada no relatório
  evidencias/
    pytest_log.txt        log completo da execução
    linha_de_base.md      métricas de v1 e v2 nos três conjuntos
  correcoes/              patch opcional que corrige D01, D06, D12, D13 e D15
  sentinela-nortemec/     o sistema testado (submódulo, sem alterações)
```

## Defeitos encontrados

| ID | Defeito |
|---|---|
| D01 | `maquina_risco` é calculado com o próprio rótulo do lote |
| D02 | `temp_media_6h` usa leituras do futuro |
| D03 | o operador OP-07 é escalado quando a falha já é conhecida, e o modelo usa isso |
| D04 | pressão em psi não é convertida para bar |
| D05 | vibração vazia vira 0,0 |
| D06 | `prever_registro` ignora o nome das chaves |
| D07 | a v2 perde recall em relação à v1 (0,94 → 0,70) |
| D08 | as probabilidades da v1 não são calibradas |
| D09 | limiar 0,5 é ruim para a v2 |
| D10 | em produção o ruído do sensor de temperatura triplicou |
| D11 | ruído menor que o do sensor muda decisões |
| D12 | janelas de "6 h" contam linhas, não horas |
| D13 | índice repetido duplica linhas |
| D14 | leituras impossíveis são aceitas |
| D15 | o manifest de integridade falha em Linux/Colab |

Detalhes, números e causa de cada um estão no [relatório](relatorio.md).

## Saída da execução

Final de [evidencias/pytest_log.txt](evidencias/pytest_log.txt):

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
======================= 46 failed, 133 passed in 15.75s =======================
```

As duas execuções que passam em D10 são o treino e o teste: o ruído do sensor só passa do limite na produção.
