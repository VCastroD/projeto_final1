"""Bloco C — testes adversariais: ruído, contrafactuais e casos-limite.

Tudo roda no caminho de produção (features sem o rótulo), sobre o conjunto
de teste.
"""
import numpy as np
import pandas as pd
import pytest

import sentinela as sn
from apoio import estatistica as est
from apoio import ficha

VERSOES = ("v1", "v2")
NAO_FISICAS = {"operador_senior", "turno"}


def _decidir(limpo, versao):
    return sn.modelo.carregar(versao).prever(sn.features.construir(limpo))


def _proba(limpo, versao):
    return sn.modelo.carregar(versao).prever_proba(sn.features.construir(limpo))


# --------------------------------------------------------------------------
# Perturbações menores que o ruído do sensor
# --------------------------------------------------------------------------
PERTURBACOES = {
    "ruido_uniforme_1C": lambda n, rng: rng.uniform(-1.0, 1.0, n),
    "desvio_+0.5C": lambda n, rng: np.full(n, 0.5),
    "desvio_-0.5C": lambda n, rng: np.full(n, -0.5),
}


@pytest.mark.defeito("D11")
@pytest.mark.parametrize("perturbacao", list(PERTURBACOES))
@pytest.mark.parametrize("versao", VERSOES)
def test_perturbacao_dentro_do_ruido_do_sensor_nao_muda_decisao(aval_teste, versao, perturbacao):
    """Ficha: ruído de +-1,0 °C. Duas leituras do MESMO estado do motor
    podem diferir disso. Critério: no máximo 1% das decisões mudam."""
    rng = np.random.default_rng(0)
    alterado = aval_teste.limpo.copy()
    alterado["temperatura_c"] += PERTURBACOES[perturbacao](len(alterado), rng)
    mudou = _decidir(alterado, versao) != aval_teste.pred(versao)
    assert mudou.mean() <= 0.01, f"{versao}/{perturbacao}: {mudou.sum()} decisões ({mudou.mean():.2%}) mudaram"


@pytest.mark.parametrize("versao", VERSOES)
def test_ruido_do_sensor_nao_vira_decisao_confiante(aval_teste, versao):
    """Controle do teste acima: a instabilidade mora perto do limiar. Decisões
    com |p - 0,5| >= 0,25 (praticamente) não viram com ruído de +-1 °C."""
    p = aval_teste.proba[versao]
    confiante = np.abs(p - 0.5) >= 0.25
    for semente in range(3):
        rng = np.random.default_rng(semente)
        alterado = aval_teste.limpo.copy()
        alterado["temperatura_c"] += rng.uniform(-1.0, 1.0, len(alterado))
        virou = (_proba(alterado, versao) >= 0.5) != (p >= 0.5)
        assert virou[confiante].mean() <= 0.001


def test_perturbacao_nula_nao_muda_nada(aval_teste):
    """Controle do instrumento: sem perturbação, zero mudanças."""
    assert (_decidir(aval_teste.limpo.copy(), "v1") == aval_teste.pred("v1")).all()


# --------------------------------------------------------------------------
# De que o modelo depende — e se isso é legítimo
# --------------------------------------------------------------------------
@pytest.mark.defeito("D03")
@pytest.mark.parametrize("versao", VERSOES)
def test_nenhum_campo_nao_fisico_entre_as_5_features_mais_usadas(aval_teste, versao):
    """Importância por permutação (queda de PR-AUC ao embaralhar a coluna)."""
    rng = np.random.default_rng(0)
    modelo = sn.modelo.carregar(versao)
    base = est.precisao_media(aval_teste.y, aval_teste.proba[versao])
    queda = {}
    for feature in aval_teste.X.columns:
        X = aval_teste.X.copy()
        X[feature] = rng.permutation(X[feature].to_numpy())
        queda[feature] = base - est.precisao_media(aval_teste.y, modelo.prever_proba(X))
    top5 = sorted(queda, key=queda.get, reverse=True)[:5]
    assert not (set(top5) & NAO_FISICAS), (
        f"{versao} top-5: " + ", ".join(f"{f} ({queda[f]:.3f})" for f in top5)
    )


def _presenca_op07_em_torno_do_inicio_da_falha(df, horas=24):
    """Fração de turnos com OP-07 nas `horas` antes e depois de falha_72h
    passar de 0 para 1 em cada motor."""
    ordenado = df.assign(timestamp=pd.to_datetime(df["timestamp"])).sort_values(["id_maquina", "timestamp"])
    antes, depois = [], []
    for _, motor in ordenado.groupby("id_maquina"):
        motor = motor.reset_index(drop=True)
        op07 = (motor["id_operador"] == "OP-07").to_numpy()
        inicios = np.flatnonzero(np.diff(motor["falha_72h"].to_numpy()) == 1) + 1
        for i in inicios:
            antes.extend(op07[max(0, i - horas):i])
            depois.extend(op07[i:i + horas])
    return float(np.mean(antes)), float(np.mean(depois))


@pytest.mark.defeito("D03")
def test_escala_do_op07_nao_reage_ao_rotulo(carregar):
    """Hipótese: OP-07 é escalado para motores que ALGUÉM JÁ SABE que vão
    falhar. Se a escala fosse por criticidade do motor (como diz o
    comentário em features.py), a presença do OP-07 não saltaria exatamente
    na hora em que `falha_72h` vira 1 — o rótulo é definido olhando 72 h
    para frente, algo que a escala não poderia saber."""
    antes, depois = _presenca_op07_em_torno_do_inicio_da_falha(carregar("treino"))
    assert depois - antes <= 0.10, f"OP-07 presente em {antes:.0%} das 24 h antes e {depois:.0%} das 24 h depois"


CONTRAFACTUAIS = {
    # campo que não altera o estado físico do motor -> transformação
    "operador_OP07_trocado": lambda l: l.assign(
        id_operador=np.where(l["id_operador"] == "OP-07", "OP-01", "OP-07")
    ),
    "turno_trocado": lambda l: l.assign(turno=l["turno"] % 3 + 1),
}


@pytest.mark.defeito("D03")
@pytest.mark.parametrize("campo", list(CONTRAFACTUAIS))
@pytest.mark.parametrize("versao", VERSOES)
def test_contrafactual_de_campo_nao_fisico_nao_muda_decisao(aval_teste, versao, campo):
    """Mesmas leituras de sensor, outro operador de turno / outro turno."""
    alterado = CONTRAFACTUAIS[campo](aval_teste.limpo.copy())
    mudou = _decidir(alterado, versao) != aval_teste.pred(versao)
    assert mudou.sum() == 0, f"{versao}/{campo}: {mudou.sum()} decisões ({mudou.mean():.2%}) mudaram"


@pytest.mark.defeito("D04")
@pytest.mark.parametrize("versao", VERSOES)
def test_contrafactual_mesma_pressao_na_outra_unidade_nao_muda_decisao(carregar, versao):
    """Troca o CLP: quem reportava em bar passa a reportar em psi e vice-versa.
    A pressão física é idêntica."""
    bruto = carregar("teste").drop(columns="falha_72h")
    original = sn.pipeline.executar(bruto, versao=versao)["predicao"]
    psi = bruto["unidade_pressao"] == "psi"
    trocado = bruto.copy()
    trocado.loc[psi, "pressao"] = bruto.loc[psi, "pressao"] / ficha.PSI_POR_BAR
    trocado.loc[~psi, "pressao"] = bruto.loc[~psi, "pressao"] * ficha.PSI_POR_BAR
    trocado["unidade_pressao"] = np.where(psi, "bar", "psi")
    mudou = sn.pipeline.executar(trocado, versao=versao)["predicao"] != original
    assert mudou.sum() == 0, f"{versao}: {mudou.sum()} decisões ({mudou.mean():.2%}) mudaram só pela unidade"


@pytest.mark.parametrize("versao", VERSOES)
def test_contrafactual_formatacao_dos_identificadores_nao_muda_decisao(carregar, versao):
    """' m05 ', 'op-07 ', ' BAR' — `limpar` normaliza, e a decisão se mantém."""
    bruto = carregar("teste").drop(columns="falha_72h")
    original = sn.pipeline.executar(bruto, versao=versao)
    sujo = bruto.assign(
        id_maquina=" " + bruto["id_maquina"].str.lower() + " ",
        id_operador=bruto["id_operador"].str.lower() + " ",
        unidade_pressao=" " + bruto["unidade_pressao"].str.upper(),
    )
    pd.testing.assert_series_equal(sn.pipeline.executar(sujo, versao=versao)["predicao"], original["predicao"])


# --------------------------------------------------------------------------
# Casos-limite
# --------------------------------------------------------------------------
@pytest.mark.defeito("D05")
@pytest.mark.parametrize("versao", VERSOES)
def test_sensor_de_vibracao_morto_nao_esconde_falhas(carregar, aval_teste, versao):
    """Sensor de vibração sem sinal na semana inteira. `limpar` troca o
    vazio por 0,0 e o modelo lê 'motor sem vibração nenhuma' = saudável.
    Esperado: recusar o lote ou manter pelo menos metade do recall."""
    bruto = carregar("teste").drop(columns="falha_72h").assign(vibracao_rms=np.nan)
    try:
        saida = sn.pipeline.executar(bruto, versao=versao)
    except ValueError:
        return
    y = aval_teste.y
    recall_base = aval_teste.pred(versao)[y == 1].mean()
    recall_morto = saida["predicao"].to_numpy()[y == 1].mean()
    assert recall_morto >= 0.5 * recall_base, f"{versao}: recall {recall_base:.2f} -> {recall_morto:.2f} com o sensor morto"


@pytest.mark.parametrize("versao", VERSOES)
def test_motor_superaquecido_abre_ordem(carregar, versao):
    """110 °C sustentados (acima da faixa de operação) num motor saudável."""
    bruto = carregar("teste").drop(columns="falha_72h")
    motor = bruto[bruto["id_maquina"] == "M11"].assign(temperatura_c=110.0)
    assert sn.pipeline.executar(motor, versao=versao)["predicao"].all()


@pytest.mark.parametrize("versao", VERSOES)
def test_sensor_travado_em_valor_repetido(carregar, versao):
    """Todas as leituras iguais por uma semana: nada de NaN, divisão por
    zero ou exceção — o pipeline devolve uma probabilidade válida por hora."""
    bruto = carregar("teste").drop(columns="falha_72h")
    motor = bruto[bruto["id_maquina"] == "M05"].copy()
    for coluna in ["temperatura_c", "vibracao_rms", "corrente_a", "pressao", "rpm"]:
        motor[coluna] = motor[coluna].iloc[0]
    saida = sn.pipeline.executar(motor, versao=versao)
    assert len(saida) == len(motor)
    assert saida["probabilidade"].between(0.0, 1.0).all()


@pytest.mark.parametrize("versao", VERSOES)
def test_lote_de_uma_linha_so(carregar, versao):
    linha = carregar("teste").drop(columns="falha_72h").iloc[[100]]
    saida = sn.pipeline.executar(linha, versao=versao)
    assert len(saida) == 1 and 0.0 <= saida["probabilidade"].iloc[0] <= 1.0


@pytest.mark.parametrize("versao", VERSOES)
def test_limiares_extremos(aval_teste, versao):
    modelo = sn.modelo.carregar(versao)
    assert modelo.prever(aval_teste.X, limiar=0.0).all()
    assert not modelo.prever(aval_teste.X, limiar=1.0 + 1e-9).any()
