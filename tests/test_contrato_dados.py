"""Bloco A: os arquivos em dados/ respeitam a ficha técnica do README?"""
import hashlib

import numpy as np
import pandas as pd
import pytest

import sentinela as sn
from apoio import ficha

CONJUNTOS = ("treino", "teste", "producao")


def sha256_normalizado(arquivo, fim_de_linha):
    conteudo = (sn.artefatos.RAIZ / arquivo).read_bytes().replace(b"\r\n", b"\n")
    if fim_de_linha == "crlf":
        conteudo = conteudo.replace(b"\n", b"\r\n")
    return hashlib.sha256(conteudo).hexdigest()


def test_dados_e_modelos_sao_os_publicados():
    # aceita LF ou CRLF para o resultado não depender do sistema operacional (ver D15)
    manifest = sn.artefatos.ler_manifest()["arquivos"]
    diferentes = []
    for arquivo, esperado in manifest.items():
        if esperado not in (sha256_normalizado(arquivo, "lf"), sha256_normalizado(arquivo, "crlf")):
            diferentes.append(arquivo)
    assert diferentes == []


@pytest.mark.defeito("D15")
def test_manifest_bate_com_o_conteudo_versionado_no_git():
    """O git guarda os arquivos em LF, mas o manifest foi gerado sobre cópias CRLF (Windows).

    Em Linux/macOS/Colab, verificar_manifest() acusa 4 arquivos que estão íntegros.
    """
    manifest = sn.artefatos.ler_manifest()["arquivos"]
    diferentes = []
    for arquivo, esperado in manifest.items():
        if sha256_normalizado(arquivo, "lf") != esperado:
            diferentes.append(arquivo)
    assert diferentes == [], f"com fim de linha LF não batem: {diferentes}"


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_colunas_do_historiador(carregar, nome):
    assert tuple(carregar(nome).columns) == sn.dados.COLUNAS


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_grade_horaria_completa_sem_duplicatas(carregar, nome):
    df = carregar(nome)
    horarios = pd.to_datetime(df["timestamp"])
    total_horas = int((horarios.max() - horarios.min()) / pd.Timedelta(hours=1)) + 1

    assert not df.duplicated(["id_maquina", "timestamp"]).any()
    assert df["id_maquina"].nunique() == 25
    assert len(df) == 25 * total_horas


def test_conjuntos_sao_janelas_consecutivas_e_disjuntas(carregar):
    inicio, fim = {}, {}
    for nome in CONJUNTOS:
        horarios = pd.to_datetime(carregar(nome)["timestamp"])
        inicio[nome], fim[nome] = horarios.min(), horarios.max()

    assert fim["treino"] + pd.Timedelta(hours=1) == inicio["teste"]
    assert fim["teste"] + pd.Timedelta(hours=1) == inicio["producao"]
    assert (fim["treino"] - inicio["treino"]).days + 1 == 28
    assert (fim["teste"] - inicio["teste"]).days + 1 == 7
    assert (fim["producao"] - inicio["producao"]).days + 1 == 7


def fracao_na_faixa(valores, faixa):
    return float(pd.Series(valores).dropna().between(*faixa).mean())


# Exigimos 99,9% e não 100%: motor perto de falhar pode sair da faixa.
# Um erro de unidade, por outro lado, tira ~30% das leituras.
@pytest.mark.parametrize("nome", CONJUNTOS)
@pytest.mark.parametrize("coluna", ["temperatura_c", "vibracao_rms", "corrente_a", "rpm"])
def test_sensores_dentro_da_faixa_de_operacao(carregar, nome, coluna):
    df = carregar(nome)
    assert fracao_na_faixa(df[coluna], ficha.FAIXAS[coluna]) >= ficha.FRACAO_MINIMA_NA_FAIXA


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_pressao_dentro_da_faixa_depois_de_respeitar_a_unidade(carregar, nome):
    """O dado está certo se a unidade for respeitada; quem não converte é o limpar() (D04)."""
    df = carregar(nome)
    assert set(df["unidade_pressao"].str.strip().str.lower()) <= {"bar", "psi"}

    em_bar = ficha.pressao_em_bar(df["pressao"], df["unidade_pressao"])
    assert fracao_na_faixa(em_bar, ficha.FAIXAS["pressao_bar"]) >= ficha.FRACAO_MINIMA_NA_FAIXA
    assert fracao_na_faixa(df["pressao"], ficha.FAIXAS["pressao_bar"]) < 0.75  # sem converter


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_campos_categoricos_e_estaticos(carregar, nome):
    df = carregar(nome)
    assert set(df["id_operador"]) <= ficha.OPERADORES
    assert set(df["turno"]) <= {1, 2, 3}
    assert set(df["falha_72h"]) <= {0, 1}
    assert df["idade_equipamento_meses"].between(6, 180).all()
    assert (df.groupby("id_maquina")["idade_equipamento_meses"].nunique() == 1).all()

    ordenado = df.sort_values(["id_maquina", "timestamp"])
    incremento = ordenado.groupby("id_maquina")["horas_operacao"].diff().dropna()
    assert (incremento == 1.0).all()


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_turno_coerente_com_a_hora(carregar, nome):
    df = carregar(nome)
    hora = pd.to_datetime(df["timestamp"]).dt.hour
    assert ((hora // 8 + 1) == df["turno"]).all()  # 0-7h -> 1, 8-15h -> 2, 16-23h -> 3


@pytest.mark.parametrize("nome", CONJUNTOS)
def test_dropout_de_vibracao_e_o_unico_faltante(carregar, nome):
    df = carregar(nome)
    vazios = df.isna().sum()
    assert set(vazios[vazios > 0].index) == {"vibracao_rms"}
    assert 0.05 < df["vibracao_rms"].isna().mean() < 0.15


def ruido_do_sensor(df, coluna):
    # De uma hora para a outra o valor real quase não muda, então a diferença
    # entre leituras seguidas é basicamente ruído. std(diff) / sqrt(2) porque
    # cada diferença soma o ruído de duas leituras.
    ordenado = df.sort_values(["id_maquina", "timestamp"])
    diferencas = ordenado.groupby("id_maquina")[coluna].diff().dropna()
    return float(diferencas.std() / np.sqrt(2))


@pytest.mark.defeito("D10")
@pytest.mark.parametrize("nome", CONJUNTOS)
def test_ruido_do_sensor_de_temperatura_respeita_a_ficha(carregar, nome):
    """Ficha: ±1,0 °C. Treino e teste dão ~1,05; produção dá ~3,2 em todos os motores."""
    ruido = ruido_do_sensor(carregar(nome), "temperatura_c")
    assert ruido <= 1.25 * ficha.RUIDO_TEMPERATURA_C, f"{nome}: sigma estimado = {ruido:.2f} °C"
