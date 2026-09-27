"""Bloco A: preprocessamento.limpar (tipos, faltantes, normalização)."""
import numpy as np
import pandas as pd
import pytest

import sentinela as sn
from apoio import ficha

limpar = sn.preprocessamento.limpar


def leitura(**alteracoes):
    """Uma leitura válida qualquer; `alteracoes` sobrescreve campos."""
    base = {
        "timestamp": "2026-06-01 00:00:00",
        "id_maquina": "M01",
        "idade_equipamento_meses": 120,
        "id_operador": "OP-03",
        "turno": 1,
        "temperatura_c": 68.0,
        "vibracao_rms": 3.0,
        "pressao": 3.9,
        "unidade_pressao": "bar",
        "corrente_a": 19.5,
        "rpm": 1755.0,
        "horas_operacao": 12000.0,
        "falha_72h": 0,
    }
    base.update(alteracoes)
    return base


def test_nao_modifica_o_quadro_de_entrada(lote_teste):
    original = lote_teste.copy()
    limpar(lote_teste)
    pd.testing.assert_frame_equal(lote_teste, original)


def test_tipos_de_saida(lote_teste):
    limpo = limpar(lote_teste)
    assert pd.api.types.is_datetime64_any_dtype(limpo["timestamp"])
    for coluna in ["turno", "idade_equipamento_meses", "operador_cod"]:
        assert pd.api.types.is_integer_dtype(limpo[coluna]), coluna
    for coluna in ["temperatura_c", "vibracao_rms", "pressao", "corrente_a", "rpm"]:
        assert pd.api.types.is_float_dtype(limpo[coluna]), coluna


def test_normaliza_rotulos_com_espaco_e_caixa():
    df = pd.DataFrame([leitura(id_maquina=" m07 ", id_operador="op-07 ", unidade_pressao=" PSI ")])
    limpo = limpar(df)
    assert limpo.loc[0, "id_maquina"] == "M07"
    assert limpo.loc[0, "id_operador"] == "OP-07"
    assert limpo.loc[0, "unidade_pressao"] == "psi"
    assert limpo.loc[0, "operador_cod"] == 7


def test_remove_leitura_com_eixo_parado_e_so_ela(lote_teste):
    lote_teste.loc[[3, 10, 20], "rpm"] = 0.0
    limpo = limpar(lote_teste)
    assert len(limpo) == len(lote_teste) - 3
    assert (limpo["rpm"] > 0).all()


def test_preserva_a_ordem_das_linhas(lote_teste):
    embaralhado = lote_teste.sample(frac=1.0, random_state=7)
    limpo = limpar(embaralhado)
    assert list(limpo["id_maquina"]) == list(embaralhado["id_maquina"])
    assert list(limpo["timestamp"]) == list(pd.to_datetime(embaralhado["timestamp"]))
    assert list(limpo.index) == list(range(len(limpo)))


def test_sem_faltantes_na_saida_do_conjunto_real(lote_teste):
    assert limpar(lote_teste).notna().all().all()


def test_faltante_sem_tratamento_definido_falha_alto_no_pipeline(lote_teste):
    # temperatura vazia não tem regra na ficha; recusar o lote é aceitável
    df = lote_teste.drop(columns="falha_72h")
    df.loc[5, "temperatura_c"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sn.pipeline.executar(df, versao="v1")


@pytest.mark.defeito("D05")
def test_dropout_de_vibracao_nao_vira_valor_fisicamente_impossivel(lote_teste):
    """limpar() preenche o dropout com 0,0; a ficha diz que a vibração fica entre 1,2 e 8,0."""
    limpo = limpar(lote_teste)
    minimo = ficha.FAIXAS["vibracao_rms"][0]
    abaixo = int((limpo["vibracao_rms"] < minimo).sum())
    assert abaixo == 0, f"{abaixo} leituras de vibração abaixo de {minimo} mm/s após limpar()"


@pytest.mark.defeito("D04")
def test_pressao_sai_de_limpar_em_uma_unidade_so(lote_teste):
    """limpar() normaliza o texto de unidade_pressao, mas não converte o valor."""
    limpo = limpar(lote_teste)
    em_psi = int((limpo["unidade_pressao"] == "psi").sum())
    na_faixa = limpo["pressao"].between(*ficha.FAIXAS["pressao_bar"]).mean()
    assert em_psi == 0 and na_faixa >= ficha.FRACAO_MINIMA_NA_FAIXA, (
        f"{em_psi} linhas continuam em psi; só {na_faixa:.1%} da pressão está em 3,0-4,5 bar"
    )


@pytest.mark.defeito("D04")
def test_mesma_pressao_fisica_em_bar_ou_psi_da_o_mesmo_valor_limpo():
    df = pd.DataFrame([
        leitura(pressao=3.9, unidade_pressao="bar"),
        leitura(pressao=3.9 * ficha.PSI_POR_BAR, unidade_pressao="psi"),
    ])
    limpo = limpar(df)
    assert limpo.loc[0, "pressao"] == pytest.approx(limpo.loc[1, "pressao"], abs=1e-3)


@pytest.mark.defeito("D14")
@pytest.mark.parametrize(
    "campo, valor",
    [("temperatura_c", -40.0), ("temperatura_c", 250.0), ("rpm", 300.0), ("corrente_a", 0.5)],
)
def test_leitura_fisicamente_impossivel_e_barrada_ou_sinalizada(lote_teste, campo, valor):
    """-40 °C é o valor típico de termopar desconectado; 300 rpm é eixo parando.

    Esperado: limpar() remove a leitura ou levanta erro. Hoje só barra rpm < 1.
    """
    df = lote_teste.drop(columns="falha_72h")
    df[campo] = valor
    try:
        limpo = limpar(df)
    except ValueError:
        return
    assert len(limpo) == 0, f"{len(limpo)} leituras com {campo}={valor} aceitas por limpar()"
