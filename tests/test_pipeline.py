"""pipeline.executar (ponta a ponta) e sentinela.avaliacao."""
import numpy as np
import pandas as pd
import pytest

import sentinela as sn

VERSOES = ("v1", "v2")


def test_saida_tem_as_colunas_prometidas(lote_teste):
    com_rotulo = sn.pipeline.executar(lote_teste)
    sem_rotulo = sn.pipeline.executar(lote_teste.drop(columns="falha_72h"))
    assert list(com_rotulo.columns) == ["id_maquina", "timestamp", "probabilidade", "predicao", "falha_72h"]
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
    assert len(sn.pipeline.executar(parado)) == 0


@pytest.mark.parametrize("versao", VERSOES)
def test_lote_de_um_motor_so_da_o_mesmo_que_no_lote_inteiro(lote_teste, versao):
    df = lote_teste.drop(columns="falha_72h")
    lote_inteiro = sn.pipeline.executar(df, versao=versao)
    so_m05 = sn.pipeline.executar(df[df["id_maquina"] == "M05"], versao=versao)

    m05_no_lote_inteiro = lote_inteiro[lote_inteiro["id_maquina"] == "M05"].reset_index(drop=True)
    pd.testing.assert_frame_equal(so_m05.reset_index(drop=True), m05_no_lote_inteiro)


def test_motor_novo_fora_da_tabela_de_risco_e_pontuado(lote_teste):
    df = lote_teste.drop(columns="falha_72h")
    motor_novo = df[df["id_maquina"] == "M05"].copy()
    motor_novo["id_maquina"] = "M26"
    saida = sn.pipeline.executar(motor_novo)
    assert len(saida) == len(motor_novo)
    assert saida["probabilidade"].notna().all()


@pytest.mark.defeito("D01")
@pytest.mark.parametrize("versao", VERSOES)
def test_decisao_nao_depende_da_presenca_do_rotulo(lote_teste, versao):
    """Em produção a coluna falha_72h não existe; a decisão não pode depender dela."""
    com_rotulo = sn.pipeline.executar(lote_teste, versao=versao)
    sem_rotulo = sn.pipeline.executar(lote_teste.drop(columns="falha_72h"), versao=versao)
    mudaram = int((com_rotulo["predicao"] != sem_rotulo["predicao"]).sum())
    assert mudaram == 0, f"{mudaram} de {len(com_rotulo)} decisões ({mudaram / len(com_rotulo):.1%}) mudam com o rótulo presente"


@pytest.mark.defeito("D01")
def test_probabilidade_de_uma_leitura_nao_depende_do_proprio_rotulo(lote_teste):
    linha = lote_teste.iloc[[0]].copy()

    linha["falha_72h"] = 0
    proba_rotulo_0 = float(sn.pipeline.executar(linha)["probabilidade"].iloc[0])
    linha["falha_72h"] = 1
    proba_rotulo_1 = float(sn.pipeline.executar(linha)["probabilidade"].iloc[0])

    assert proba_rotulo_0 == pytest.approx(proba_rotulo_1), (
        f"probabilidade = {proba_rotulo_0:.3f} com rótulo 0 e {proba_rotulo_1:.3f} com rótulo 1"
    )


def test_metricas_conferem_com_contagem_manual():
    y_real = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    assert sn.avaliacao.matriz_confusao(y_real, y_pred) == {"vp": 2, "vn": 6, "fp": 1, "fn": 1}

    m = sn.avaliacao.metricas(y_real, y_pred)
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
