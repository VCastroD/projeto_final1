"""Bloco A (parte 1) — os dados em `dados/` respeitam a ficha técnica?

A dica do guia é validar os dados, não só o código. Cada regra da ficha que
dá para checar vira um teste, nos três conjuntos.
"""
import hashlib

import numpy as np
import pandas as pd
import pytest

import sentinela as sn
from apoio import ficha

CONJUNTOS = sn.dados.CONJUNTOS


def _sha256(relativo: str, fim_de_linha: bytes) -> str:
    """sha256 do arquivo com o fim de linha forçado para LF ou CRLF."""
    conteudo = (sn.artefatos.RAIZ / relativo).read_bytes().replace(b"\r\n", b"\n")
    if fim_de_linha == b"\r\n":
        conteudo = conteudo.replace(b"\n", b"\r\n")
    return hashlib.sha256(conteudo).hexdigest()


def test_dados_e_modelos_sao_os_publicados():
    """A suíte roda contra o conteúdo publicado, não uma cópia alterada.

    Compara o conteúdo ignorando o fim de linha (ver D15): o resultado deste
    teste não pode depender do sistema operacional de quem roda.
    """
    manifest = sn.artefatos.ler_manifest()["arquivos"]
    divergentes = [
        rel for rel, esperado in manifest.items()
        if esperado not in {_sha256(rel, b"\n"), _sha256(rel, b"\r\n")}
    ]
    assert divergentes == []


@pytest.mark.defeito("D15")
def test_manifest_bate_com_o_conteudo_versionado_no_git():
    """O git guarda os CSV e o risco_maquina.json com fim de linha LF, mas o
    manifest foi gerado sobre cópias CRLF (Windows com core.autocrlf=true).
    Em Linux, macOS e Colab — o ambiente que o README recomenda —
    `verificar_manifest()` acusa 4 arquivos 'adulterados' que estão
    íntegros; no Windows, passa. Um verificador de integridade que depende
    do sistema operacional ensina a equipe a ignorá-lo."""
    manifest = sn.artefatos.ler_manifest()["arquivos"]
    divergentes = [rel for rel, esperado in manifest.items() if _sha256(rel, b"\n") != esperado]
    assert divergentes == [], f"com fim de linha LF (checkout em Linux/Colab) não batem: {divergentes}"


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_colunas_do_historiador(carregar, nome):
    assert tuple(carregar(nome).columns) == sn.dados.COLUNAS


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_grade_horaria_completa_sem_duplicatas(carregar, nome):
    """25 motores, uma leitura por motor por hora, sem buracos nem repetições."""
    df = carregar(nome)
    ts = pd.to_datetime(df["timestamp"])
    assert not df.duplicated(["id_maquina", "timestamp"]).any()
    assert df["id_maquina"].nunique() == 25
    horas = int((ts.max() - ts.min()) / pd.Timedelta(hours=1)) + 1
    assert len(df) == 25 * horas


def test_conjuntos_sao_janelas_consecutivas_e_disjuntas(carregar):
    """treino (dias 1-28) < teste (29-35) < producao (36-42), sem sobreposição."""
    limites = {n: pd.to_datetime(carregar(n)["timestamp"]).agg(["min", "max"]) for n in CONJUNTOS}
    assert limites["treino"]["max"] + pd.Timedelta(hours=1) == limites["teste"]["min"]
    assert limites["teste"]["max"] + pd.Timedelta(hours=1) == limites["producao"]["min"]
    dias = {n: (lim["max"] - lim["min"]).days + 1 for n, lim in limites.items()}
    assert dias == {"treino": 28, "teste": 7, "producao": 7}


def _fracao_na_faixa(valores, faixa):
    valores = pd.Series(valores).dropna()
    return float(valores.between(*faixa).mean())


@pytest.mark.parametrize("nome", CONJUNTOS)
@pytest.mark.parametrize("coluna", ["temperatura_c", "vibracao_rms", "corrente_a", "rpm"])
def test_sensores_dentro_da_faixa_de_operacao(carregar, nome, coluna):
    df = carregar(nome)
    assert _fracao_na_faixa(df[coluna], ficha.FAIXAS[coluna]) >= ficha.FRACAO_MINIMA_NA_FAIXA


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_pressao_dentro_da_faixa_depois_de_respeitar_a_unidade(carregar, nome):
    """O dado está certo — desde que se leia `unidade_pressao`.

    Convertendo psi para bar, >= 99,9% das leituras caem em 3,0-4,5 bar. Sem
    converter, só ~68% caem. A falha, portanto, não é do dado: é de quem não
    converte (ver D04 em test_preprocessamento.py).
    """
    df = carregar(nome)
    assert set(df["unidade_pressao"].str.strip().str.lower()) <= ficha.UNIDADES_PRESSAO
    em_bar = ficha.pressao_em_bar(df["pressao"], df["unidade_pressao"])
    assert _fracao_na_faixa(em_bar, ficha.FAIXAS["pressao_bar"]) >= ficha.FRACAO_MINIMA_NA_FAIXA
    assert _fracao_na_faixa(df["pressao"], ficha.FAIXAS["pressao_bar"]) < 0.75


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_campos_categoricos_e_estaticos(carregar, nome):
    df = carregar(nome)
    assert set(df["id_operador"]) <= ficha.OPERADORES
    assert set(df["turno"]) <= ficha.TURNOS
    assert set(df["falha_72h"]) <= {0, 1}
    assert df["idade_equipamento_meses"].between(*ficha.FAIXAS["idade_equipamento_meses"]).all()
    # idade não muda dentro de uma semana; horas de operação sobem 1 por hora
    assert (df.groupby("id_maquina")["idade_equipamento_meses"].nunique() == 1).all()
    ordenado = df.sort_values(["id_maquina", "timestamp"])
    incremento = ordenado.groupby("id_maquina")["horas_operacao"].diff().dropna()
    assert (incremento == 1.0).all()


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_turno_coerente_com_a_hora(carregar, nome):
    df = carregar(nome)
    hora = pd.to_datetime(df["timestamp"]).dt.hour
    assert ((hora // 8 + 1) == df["turno"]).all()


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_dropout_de_vibracao_e_o_unico_faltante(carregar, nome):
    """A ficha só prevê faltante na vibração (dropout). Nenhum outro campo vem vazio."""
    df = carregar(nome)
    faltantes = df.isna().sum()
    assert set(faltantes[faltantes > 0].index) == {"vibracao_rms"}
    assert 0.05 < df["vibracao_rms"].isna().mean() < 0.15


def _ruido_estimado(df, coluna):
    """sigma do sensor estimado pelas diferenças hora a hora de cada motor.

    Var(x_t - x_{t-1}) = 2 sigma^2 + variação real do processo, logo
    std(diff)/sqrt(2) é um limite superior para o ruído do sensor.
    """
    ordenado = df.sort_values(["id_maquina", "timestamp"])
    diferenca = ordenado.groupby("id_maquina")[coluna].diff().dropna()
    return float(diferenca.std() / np.sqrt(2))


@pytest.mark.defeito("D10")
@pytest.mark.parametrize("nome", CONJUNTOS)
def test_ruido_do_sensor_de_temperatura_respeita_a_ficha(carregar, nome):
    """Ficha: ruído do sensor de temperatura +-1,0 °C.

    treino e teste: sigma estimado ~1,05 °C (consistente com a ficha).
    producao: ~3,2 °C em TODOS os 25 motores, TODOS os dias — o sensor
    passou a medir com o triplo do ruído, e a média nem se mexeu.
    """
    sigma = _ruido_estimado(carregar(nome), "temperatura_c")
    assert sigma <= 1.25 * ficha.RUIDO_TEMPERATURA_C, f"{nome}: sigma estimado = {sigma:.2f} °C"
