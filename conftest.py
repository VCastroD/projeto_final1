"""Localiza o sentinela, registra o marcador `defeito` e define as fixtures."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from apoio import sistema

PASTA_SENTINELA = sistema.localizar()

try:
    import sentinela as sn
except ImportError:
    pytest.exit(
        "Pacote `sentinela` não encontrado. Rode `git submodule update --init` "
        "ou defina SENTINELA_DIR (ver README).",
        returncode=4,
    )

VERSOES = ("v1", "v2")


def pytest_report_header(config):
    return f"sentinela {sn.__version__} importado de: {PASTA_SENTINELA or 'pacote instalado'}"


# --- marcador de defeito ---------------------------------------------------
# Testes marcados com @pytest.mark.defeito("Dxx") falham de propósito:
# documentam um defeito do Sentinela. No fim da execução imprimimos um resumo.

def pytest_configure(config):
    config.addinivalue_line("markers", "defeito(id): teste que documenta um defeito do Sentinela")
    config.defeitos = {}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    resultado = yield
    relatorio = resultado.get_result()
    marcador = item.get_closest_marker("defeito")
    if marcador is None or relatorio.when != "call":
        return

    id_defeito = marcador.args[0]
    nome_teste = item.nodeid.split("::", 1)[1]
    registro = item.config.defeitos.setdefault(id_defeito, {"falhou": [], "passou": []})
    if relatorio.failed:
        registro["falhou"].append(nome_teste)
    else:
        registro["passou"].append(nome_teste)


def pytest_terminal_summary(terminalreporter, config):
    if not config.defeitos:
        return
    terminalreporter.section("defeitos do Sentinela documentados por testes")
    for id_defeito in sorted(config.defeitos):
        falhou = config.defeitos[id_defeito]["falhou"]
        passou = config.defeitos[id_defeito]["passou"]
        situacao = "CONFIRMADO" if falhou else "nao reproduzido"
        terminalreporter.write_line(
            f"{id_defeito:<4} {situacao:<16} {len(falhou)} teste(s) falhando, {len(passou)} passando"
        )
        for nome in falhou:
            terminalreporter.write_line(f"       - {nome}")


# --- dados -----------------------------------------------------------------
_cache = {}


def ler_conjunto(nome):
    """Lê o CSV uma vez e devolve sempre uma cópia (testes podem alterar à vontade)."""
    if nome not in _cache:
        _cache[nome] = sn.dados.carregar(nome)
    return _cache[nome].copy()


@pytest.fixture(scope="session")
def carregar():
    return ler_conjunto


@pytest.fixture
def lote_teste():
    return ler_conjunto("teste")


@dataclass
class ConjuntoAvaliado:
    """Conjunto processado como na produção: features calculadas sem o rótulo (ver D01)."""

    nome: str
    leituras: pd.DataFrame  # saída de limpar(), sem falha_72h
    y_real: np.ndarray
    features: pd.DataFrame
    proba: dict = field(default_factory=dict)  # {"v1": array, "v2": array}
    motor_dia: np.ndarray = None  # blocos do bootstrap, ex.: "M01|2026-06-01"

    def pred(self, versao, limiar=0.5):
        return (self.proba[versao] >= limiar).astype(int)


def avaliar_conjunto(nome):
    leituras = sn.preprocessamento.limpar(ler_conjunto(nome))
    y_real = leituras.pop("falha_72h").to_numpy().astype(int)
    features = sn.features.construir(leituras)
    motor_dia = (leituras["id_maquina"] + "|" + leituras["timestamp"].dt.strftime("%Y-%m-%d")).to_numpy()

    conjunto = ConjuntoAvaliado(nome, leituras, y_real, features, motor_dia=motor_dia)
    for versao in VERSOES:
        conjunto.proba[versao] = sn.modelo.carregar(versao).prever_proba(features)
    return conjunto


@pytest.fixture(scope="session")
def teste_avaliado():
    return avaliar_conjunto("teste")


@pytest.fixture(scope="session")
def producao_avaliada():
    return avaliar_conjunto("producao")
