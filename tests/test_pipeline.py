"""Ponta a ponta (`pipeline.executar`) e o módulo `avaliacao`."""
import numpy as np
import pandas as pd
import pytest

import sentinela as sn

VERSOES = ("v1", "v2")


def test_saida_tem_as_colunas_prometidas(lote_teste):
    rotulado = sn.pipeline.executar(lote_teste)
    sem_rotulo = sn.pipeline.executar(lote_teste.drop(columns="falha_72h"))
    assert list(rotulado.columns) == ["id_maquina", "timestamp", "probabilidade", "predicao", "falha_72h"]
    assert list(sem_rotulo.columns) == ["id_maquina", "timestamp", "probabilidade", "predicao"]


def test_uma_decisao_por_leitura_valida_na_ordem_de_entrada(lote_teste):
    embaralhado = lote_teste.sample(frac=1.0, random_state=3)
    saida = sn.pipeline.executar(embaralhado)
    assert len(saida) == len(embaralhado)
    assert list(saida["id_maquina"]) == list(embaralhado["id_maquina"])
    assert list(saida["timestamp"]) == list(pd.to_datetime(embaralhado["timestamp"]))


@pytest.mark.parametrize("limiar", [0.3, 0.5, 0.7])
def test_decisao_segue_o_limiar(lote_teste, limiar):
    saida = sn.pipeline.executar(lote_teste, limiar=limiar)
    assert (saida["predicao"] == (saida["probabilidade"] >= limiar)).all()


def test_lote_so_com_eixo_parado_devolve_saida_vazia(lote_teste):
    parado = lote_teste.head(10).copy()
    parado["rpm"] = 0.0
    saida = sn.pipeline.executar(parado)
    assert len(saida) == 0


@pytest.mark.parametrize("versao", VERSOES)
def test_lote_de_um_motor_so_da_o_mesmo_que_no_lote_inteiro(lote_teste, versao):
    """Sem rótulo, as features de um motor só dependem dele mesmo."""
    sem_rotulo = lote_teste.drop(columns="falha_72h")
    inteiro = sn.pipeline.executar(sem_rotulo, versao=versao)
    so_m05 = sn.pipeline.executar(sem_rotulo[sem_rotulo["id_maquina"] == "M05"], versao=versao)
    esperado = inteiro[inteiro["id_maquina"] == "M05"].reset_index(drop=True)
    pd.testing.assert_frame_equal(so_m05.reset_index(drop=True), esperado)


def test_motor_novo_fora_da_tabela_de_risco_e_pontuado(lote_teste):
    novo = lote_teste.drop(columns="falha_72h")
    novo = novo[novo["id_maquina"] == "M05"].assign(id_maquina="M26")
    saida = sn.pipeline.executar(novo)
    assert len(saida) == len(novo) and saida["probabilidade"].notna().all()


@pytest.mark.defeito("D01")
@pytest.mark.parametrize("versao", VERSOES)
def test_decisao_nao_depende_da_presenca_do_rotulo(lote_teste, versao):
    """O mesmo lote, com e sem a coluna `falha_72h`. Na produção real o
    rótulo não existe; na avaliação da equipe, existe. As decisões têm de
    ser as mesmas — senão a métrica reportada não é a da produção."""
    com = sn.pipeline.executar(lote_teste, versao=versao)
    sem = sn.pipeline.executar(lote_teste.drop(columns="falha_72h"), versao=versao)
    mudaram = int((com["predicao"] != sem["predicao"]).sum())
    assert mudaram == 0, f"{mudaram} de {len(com)} decisões ({mudaram / len(com):.1%}) mudam com o rótulo presente"


@pytest.mark.defeito("D01")
def test_probabilidade_de_uma_leitura_nao_depende_do_proprio_rotulo(lote_teste):
    linha = lote_teste.iloc[[0]].copy()
    probs = {}
    for rotulo in (0, 1):
        linha["falha_72h"] = rotulo
        probs[rotulo] = float(sn.pipeline.executar(linha)["probabilidade"].iloc[0])
    assert probs[0] == pytest.approx(probs[1]), f"probabilidade por rótulo: {probs}"


# --------------------------------------------------------------------------
# avaliacao
# --------------------------------------------------------------------------
def test_metricas_conferem_com_contagem_manual():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    assert sn.avaliacao.matriz_confusao(y, p) == {"vp": 2, "vn": 6, "fp": 1, "fn": 1}
    m = sn.avaliacao.metricas(y, p)
    assert m["acuracia"] == pytest.approx(0.8)
    assert m["precisao"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_metricas_sem_positivos_nao_dividem_por_zero():
    m = sn.avaliacao.metricas([0, 0, 0], [0, 0, 0])
    assert m == {"acuracia": 1.0, "precisao": 0.0, "recall": 0.0, "f1": 0.0}


def test_matriz_confusao_recusa_tamanhos_diferentes():
    with pytest.raises(ValueError):
        sn.avaliacao.matriz_confusao([0, 1], [0, 1, 1])
