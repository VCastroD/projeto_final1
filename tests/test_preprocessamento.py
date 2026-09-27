"""Bloco A — testes unitários de `preprocessamento.limpar`.

Promessa do docstring: "tipos, faltantes e normalização de rótulos".
"""
import numpy as np
import pandas as pd
import pytest

import sentinela as sn
from apoio import ficha

limpar = sn.preprocessamento.limpar


def _linha(**campos):
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
    base.update(campos)
    return base


# --------------------------------------------------------------------------
# O que funciona
# --------------------------------------------------------------------------
def test_nao_modifica_o_quadro_de_entrada(lote_teste):
    copia = lote_teste.copy()
    limpar(lote_teste)
    pd.testing.assert_frame_equal(lote_teste, copia)


def test_tipos_de_saida(lote_teste):
    limpo = limpar(lote_teste)
    assert pd.api.types.is_datetime64_any_dtype(limpo["timestamp"])
    for coluna in ["turno", "idade_equipamento_meses", "operador_cod"]:
        assert pd.api.types.is_integer_dtype(limpo[coluna]), coluna
    for coluna in ["temperatura_c", "vibracao_rms", "pressao", "corrente_a", "rpm"]:
        assert pd.api.types.is_float_dtype(limpo[coluna]), coluna


def test_normaliza_rotulos_com_espaco_e_caixa():
    bruto = pd.DataFrame([_linha(id_maquina=" m07 ", id_operador="op-07 ", unidade_pressao=" PSI ")])
    limpo = limpar(bruto)
    assert limpo.loc[0, "id_maquina"] == "M07"
    assert limpo.loc[0, "id_operador"] == "OP-07"
    assert limpo.loc[0, "unidade_pressao"] == "psi"
    assert limpo.loc[0, "operador_cod"] == 7


def test_remove_leitura_com_eixo_parado_e_so_ela(lote_teste):
    bruto = lote_teste.copy()
    bruto.loc[[3, 10, 20], "rpm"] = 0.0
    limpo = limpar(bruto)
    assert len(limpo) == len(bruto) - 3
    assert (limpo["rpm"] > 0).all()


def test_preserva_a_ordem_das_linhas(lote_teste):
    embaralhado = lote_teste.sample(frac=1.0, random_state=7)
    limpo = limpar(embaralhado)
    assert list(limpo["id_maquina"]) == list(embaralhado["id_maquina"].str.upper())
    assert list(limpo["timestamp"]) == list(pd.to_datetime(embaralhado["timestamp"]))
    assert list(limpo.index) == list(range(len(limpo)))


def test_sem_faltantes_na_saida_do_conjunto_real(lote_teste):
    assert limpar(lote_teste).notna().all().all()


def test_faltante_sem_tratamento_definido_falha_alto_no_pipeline(lote_teste):
    """Temperatura vazia não tem regra na ficha. O pipeline recusa o lote em
    vez de inventar um valor — comportamento aceitável (falha visível)."""
    bruto = lote_teste.drop(columns="falha_72h")
    bruto.loc[5, "temperatura_c"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sn.pipeline.executar(bruto, versao="v1")


# --------------------------------------------------------------------------
# Defeitos
# --------------------------------------------------------------------------
@pytest.mark.defeito("D05")
def test_dropout_de_vibracao_nao_vira_valor_fisicamente_impossivel(lote_teste):
    """Ficha: vibração opera entre 1,2 e 8,0 mm/s. `limpar` preenche o
    dropout com 0,0 — um motor girando a 1.750 rpm com vibração zero não
    existe. 408 leituras do teste (9,7%) saem assim."""
    limpo = limpar(lote_teste)
    minimo, _ = ficha.FAIXAS["vibracao_rms"]
    fora = int((limpo["vibracao_rms"] < minimo).sum())
    assert fora == 0, f"{fora} leituras de vibração abaixo de {minimo} mm/s após limpar()"


@pytest.mark.defeito("D04")
def test_pressao_sai_de_limpar_em_uma_unidade_so(lote_teste):
    """Ficha: 'alguns CLPs reportam em psi'. `limpar` normaliza o rótulo
    `unidade_pressao` mas nunca converte o valor: 1.344 leituras do teste
    seguem em psi (~57) ao lado das em bar (~3,9)."""
    limpo = limpar(lote_teste)
    em_psi = int((limpo["unidade_pressao"] == "psi").sum())
    na_faixa = limpo["pressao"].between(*ficha.FAIXAS["pressao_bar"]).mean()
    assert em_psi == 0 and na_faixa >= ficha.FRACAO_MINIMA_NA_FAIXA, (
        f"{em_psi} linhas continuam em psi; só {na_faixa:.1%} da pressão está em 3,0-4,5 bar"
    )


@pytest.mark.defeito("D04")
def test_mesma_pressao_fisica_em_bar_ou_psi_da_o_mesmo_valor_limpo():
    """Contrafactual de unidade: 3,9 bar e 56,56 psi são a mesma pressão."""
    bruto = pd.DataFrame(
        [
            _linha(pressao=3.9, unidade_pressao="bar"),
            _linha(pressao=3.9 * ficha.PSI_POR_BAR, unidade_pressao="psi"),
        ]
    )
    limpo = limpar(bruto)
    assert limpo.loc[0, "pressao"] == pytest.approx(limpo.loc[1, "pressao"], abs=1e-3)


@pytest.mark.defeito("D14")
@pytest.mark.parametrize(
    "campo, valor",
    [("temperatura_c", -40.0), ("temperatura_c", 250.0), ("rpm", 300.0), ("corrente_a", 0.5)],
)
def test_leitura_fisicamente_impossivel_e_barrada_ou_sinalizada(lote_teste, campo, valor):
    """-40 °C é o valor típico de termopar desconectado; 300 rpm é eixo
    desacelerando, não motor em operação. `limpar` só barra rpm < 1; o resto
    passa calado e vira decisão. Esperado: remover a leitura ou falhar alto."""
    bruto = lote_teste.drop(columns="falha_72h")
    bruto[campo] = valor
    try:
        limpo = limpar(bruto)
    except ValueError:
        return  # recusar o lote é uma resposta aceitável
    assert len(limpo) == 0, f"{len(limpo)} leituras com {campo}={valor} aceitas por limpar()"
