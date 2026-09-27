"""Bloco C: testes adversariais (ruído do sensor, contrafactuais, casos-limite).

Tudo roda com as features calculadas sem o rótulo, sobre o conjunto de teste.
"""
import numpy as np
import pandas as pd
import pytest

import sentinela as sn
from apoio import estatistica, ficha

VERSOES = ("v1", "v2")


def decidir(leituras, versao):
    X = sn.features.construir(leituras)
    return sn.modelo.carregar(versao).prever(X)


def probabilidade(leituras, versao):
    X = sn.features.construir(leituras)
    return sn.modelo.carregar(versao).prever_proba(X)


# --- perturbações menores que o ruído do sensor (±1,0 °C) ------------------

def perturbar_temperatura(leituras, tipo, semente=0):
    alterado = leituras.copy()
    if tipo == "ruido_uniforme_1C":
        rng = np.random.default_rng(semente)
        alterado["temperatura_c"] += rng.uniform(-1.0, 1.0, len(alterado))
    elif tipo == "desvio_+0.5C":
        alterado["temperatura_c"] += 0.5
    elif tipo == "desvio_-0.5C":
        alterado["temperatura_c"] -= 0.5
    return alterado


@pytest.mark.defeito("D11")
@pytest.mark.parametrize("perturbacao", ["ruido_uniforme_1C", "desvio_+0.5C", "desvio_-0.5C"])
@pytest.mark.parametrize("versao", VERSOES)
def test_perturbacao_dentro_do_ruido_do_sensor_nao_muda_decisao(teste_avaliado, versao, perturbacao):
    """No máximo 1% das decisões pode mudar com uma perturbação que o sensor nem distingue."""
    perturbado = perturbar_temperatura(teste_avaliado.leituras, perturbacao)
    mudou = decidir(perturbado, versao) != teste_avaliado.pred(versao)
    assert mudou.mean() <= 0.01, f"{versao}/{perturbacao}: {mudou.sum()} decisões ({mudou.mean():.2%}) mudaram"


@pytest.mark.parametrize("versao", VERSOES)
def test_ruido_do_sensor_nao_vira_decisao_confiante(teste_avaliado, versao):
    # controle do teste acima: o problema está perto do limiar; decisões
    # com probabilidade longe de 0,5 (|p - 0,5| >= 0,25) praticamente não viram
    proba_original = teste_avaliado.proba[versao]
    confiante = np.abs(proba_original - 0.5) >= 0.25

    for semente in range(3):
        perturbado = perturbar_temperatura(teste_avaliado.leituras, "ruido_uniforme_1C", semente)
        virou = (probabilidade(perturbado, versao) >= 0.5) != (proba_original >= 0.5)
        assert virou[confiante].mean() <= 0.001


def test_perturbacao_nula_nao_muda_nada(teste_avaliado):
    assert (decidir(teste_avaliado.leituras.copy(), "v1") == teste_avaliado.pred("v1")).all()


# --- de que o modelo depende ------------------------------------------------

@pytest.mark.defeito("D03")
@pytest.mark.parametrize("versao", VERSOES)
def test_nenhum_campo_nao_fisico_entre_as_5_features_mais_usadas(teste_avaliado, versao):
    """Importância por permutação: quanto a PR-AUC cai ao embaralhar cada feature."""
    modelo = sn.modelo.carregar(versao)
    y_real = teste_avaliado.y_real
    pr_auc_original = estatistica.pr_auc(y_real, teste_avaliado.proba[versao])
    rng = np.random.default_rng(0)

    queda = {}
    for feature in teste_avaliado.features.columns:
        X = teste_avaliado.features.copy()
        X[feature] = rng.permutation(X[feature].to_numpy())
        queda[feature] = pr_auc_original - estatistica.pr_auc(y_real, modelo.prever_proba(X))

    top5 = sorted(queda, key=queda.get, reverse=True)[:5]
    descricao = ", ".join(f"{f} ({queda[f]:.3f})" for f in top5)
    assert "operador_senior" not in top5 and "turno" not in top5, f"{versao} top-5: {descricao}"


def presenca_do_op07_antes_e_depois_da_falha(df, horas=24):
    """Fração de horas com OP-07 nas `horas` antes e depois de falha_72h virar 1."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["id_maquina", "timestamp"])

    antes, depois = [], []
    for _, motor in df.groupby("id_maquina"):
        op07 = (motor["id_operador"] == "OP-07").to_numpy()
        falha = motor["falha_72h"].to_numpy()
        for i in range(1, len(falha)):
            if falha[i - 1] == 0 and falha[i] == 1:  # começo de uma janela de falha
                antes.extend(op07[max(0, i - horas):i])
                depois.extend(op07[i:i + horas])
    return np.mean(antes), np.mean(depois)


@pytest.mark.defeito("D03")
def test_escala_do_op07_nao_reage_ao_rotulo(carregar):
    """Se o OP-07 fosse escalado por criticidade do motor, a presença dele não
    saltaria justo quando falha_72h vira 1, porque esse rótulo olha 72 h para frente."""
    antes, depois = presenca_do_op07_antes_e_depois_da_falha(carregar("treino"))
    assert depois - antes <= 0.10, f"OP-07 presente em {antes:.0%} das 24 h antes e {depois:.0%} das 24 h depois"


def trocar_campo(leituras, campo):
    alterado = leituras.copy()
    if campo == "operador_OP07_trocado":
        era_op07 = alterado["id_operador"] == "OP-07"
        alterado.loc[era_op07, "id_operador"] = "OP-01"
        alterado.loc[~era_op07, "id_operador"] = "OP-07"
    elif campo == "turno_trocado":
        alterado["turno"] = alterado["turno"] % 3 + 1  # 1->2, 2->3, 3->1
    return alterado


@pytest.mark.defeito("D03")
@pytest.mark.parametrize("campo", ["operador_OP07_trocado", "turno_trocado"])
@pytest.mark.parametrize("versao", VERSOES)
def test_contrafactual_de_campo_nao_fisico_nao_muda_decisao(teste_avaliado, versao, campo):
    """Mesmas leituras de sensor, outro operador ou outro turno: a decisão tem de ser a mesma."""
    mudou = decidir(trocar_campo(teste_avaliado.leituras, campo), versao) != teste_avaliado.pred(versao)
    assert mudou.sum() == 0, f"{versao}/{campo}: {mudou.sum()} decisões ({mudou.mean():.2%}) mudaram"


@pytest.mark.defeito("D04")
@pytest.mark.parametrize("versao", VERSOES)
def test_contrafactual_mesma_pressao_na_outra_unidade_nao_muda_decisao(carregar, versao):
    """Quem reportava em bar passa a reportar em psi e vice-versa; a pressão física é a mesma."""
    df = carregar("teste").drop(columns="falha_72h")
    original = sn.pipeline.executar(df, versao=versao)["predicao"]

    era_psi = df["unidade_pressao"] == "psi"
    trocado = df.copy()
    trocado.loc[era_psi, "pressao"] = df.loc[era_psi, "pressao"] / ficha.PSI_POR_BAR
    trocado.loc[era_psi, "unidade_pressao"] = "bar"
    trocado.loc[~era_psi, "pressao"] = df.loc[~era_psi, "pressao"] * ficha.PSI_POR_BAR
    trocado.loc[~era_psi, "unidade_pressao"] = "psi"

    mudou = sn.pipeline.executar(trocado, versao=versao)["predicao"] != original
    assert mudou.sum() == 0, f"{versao}: {mudou.sum()} decisões ({mudou.mean():.2%}) mudaram só pela unidade"


@pytest.mark.parametrize("versao", VERSOES)
def test_contrafactual_formatacao_dos_identificadores_nao_muda_decisao(carregar, versao):
    df = carregar("teste").drop(columns="falha_72h")
    original = sn.pipeline.executar(df, versao=versao)

    sujo = df.copy()
    sujo["id_maquina"] = " " + df["id_maquina"].str.lower() + " "
    sujo["id_operador"] = df["id_operador"].str.lower() + " "
    sujo["unidade_pressao"] = " " + df["unidade_pressao"].str.upper()

    pd.testing.assert_series_equal(sn.pipeline.executar(sujo, versao=versao)["predicao"], original["predicao"])


# --- casos-limite -----------------------------------------------------------

@pytest.mark.defeito("D05")
@pytest.mark.parametrize("versao", VERSOES)
def test_sensor_de_vibracao_morto_nao_esconde_falhas(carregar, teste_avaliado, versao):
    """Sensor de vibração sem sinal a semana toda. Esperado: recusar o lote ou manter
    pelo menos metade do recall. Hoje limpar() troca o vazio por 0,0 (= motor saudável)."""
    df = carregar("teste").drop(columns="falha_72h")
    df["vibracao_rms"] = np.nan
    try:
        saida = sn.pipeline.executar(df, versao=versao)
    except ValueError:
        return

    falhas = teste_avaliado.y_real == 1
    recall_normal = teste_avaliado.pred(versao)[falhas].mean()
    recall_sensor_morto = saida["predicao"].to_numpy()[falhas].mean()
    assert recall_sensor_morto >= 0.5 * recall_normal, (
        f"{versao}: recall {recall_normal:.2f} -> {recall_sensor_morto:.2f} com o sensor morto"
    )


@pytest.mark.parametrize("versao", VERSOES)
def test_motor_superaquecido_abre_ordem(carregar, versao):
    df = carregar("teste").drop(columns="falha_72h")
    motor = df[df["id_maquina"] == "M11"].copy()
    motor["temperatura_c"] = 110.0
    assert sn.pipeline.executar(motor, versao=versao)["predicao"].all()


@pytest.mark.parametrize("versao", VERSOES)
def test_sensor_travado_em_valor_repetido(carregar, versao):
    df = carregar("teste").drop(columns="falha_72h")
    motor = df[df["id_maquina"] == "M05"].copy()
    for coluna in ["temperatura_c", "vibracao_rms", "corrente_a", "pressao", "rpm"]:
        motor[coluna] = motor[coluna].iloc[0]

    saida = sn.pipeline.executar(motor, versao=versao)
    assert len(saida) == len(motor)
    assert saida["probabilidade"].between(0.0, 1.0).all()


@pytest.mark.parametrize("versao", VERSOES)
def test_lote_de_uma_linha_so(carregar, versao):
    linha = carregar("teste").drop(columns="falha_72h").iloc[[100]]
    saida = sn.pipeline.executar(linha, versao=versao)
    assert len(saida) == 1
    assert 0.0 <= saida["probabilidade"].iloc[0] <= 1.0


@pytest.mark.parametrize("versao", VERSOES)
def test_limiares_extremos(teste_avaliado, versao):
    modelo = sn.modelo.carregar(versao)
    assert modelo.prever(teste_avaliado.features, limiar=0.0).all()
    assert not modelo.prever(teste_avaliado.features, limiar=1.0 + 1e-9).any()
