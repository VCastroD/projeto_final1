"""Configuração da suíte: localiza o Sentinela, fixtures e resumo de defeitos.

O sistema sob teste NÃO é copiado nem editado aqui; `apoio.sistema`
explica de onde ele é importado.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from apoio import sistema

_DIR_SENTINELA = sistema.localizar()

try:
    import sentinela as sn  # noqa: E402
except ImportError:  # pragma: no cover - só dispara com instalação incompleta
    pytest.exit(
        "Pacote `sentinela` não encontrado. Rode `git submodule update --init` "
        "ou defina SENTINELA_DIR (ver README).",
        returncode=4,
    )

VERSOES = ("v1", "v2")


# --------------------------------------------------------------------------
# Marcador de defeito e resumo no fim do log
# --------------------------------------------------------------------------
def pytest_report_header(config):
    origem = _DIR_SENTINELA or "pacote instalado"
    return f"sentinela {sn.__version__} importado de: {origem}"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "defeito(id): o teste documenta um defeito do Sentinela (ver relatorio.md). "
        "Ele FALHA de propósito enquanto o defeito existir.",
    )
    config._defeitos = {}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    resultado = yield
    relatorio = resultado.get_result()
    marca = item.get_closest_marker("defeito")
    if marca and relatorio.when == "call":
        estado = item.config._defeitos.setdefault(marca.args[0], {"falhou": [], "passou": []})
        estado["falhou" if relatorio.failed else "passou"].append(item.nodeid.split("::", 1)[1])


def pytest_terminal_summary(terminalreporter, config):
    defeitos = getattr(config, "_defeitos", {})
    if not defeitos:
        return
    terminalreporter.section("defeitos do Sentinela documentados por testes")
    for ident in sorted(defeitos):
        estado = defeitos[ident]
        situacao = "CONFIRMADO" if estado["falhou"] else "nao reproduzido"
        terminalreporter.write_line(
            f"{ident:<4} {situacao:<16} {len(estado['falhou'])} teste(s) falhando, "
            f"{len(estado['passou'])} passando"
        )
        for nome in estado["falhou"]:
            terminalreporter.write_line(f"       - {nome}")


# --------------------------------------------------------------------------
# Dados e avaliação
# --------------------------------------------------------------------------
_BRUTO: dict[str, pd.DataFrame] = {}


def _bruto(nome: str) -> pd.DataFrame:
    if nome not in _BRUTO:
        _BRUTO[nome] = sn.dados.carregar(nome)
    return _BRUTO[nome].copy()


@pytest.fixture(scope="session")
def carregar():
    """Devolve uma cópia nova do conjunto cru a cada chamada."""
    return _bruto


@pytest.fixture
def lote_teste():
    return _bruto("teste")


@dataclass
class Avaliacao:
    """Um conjunto avaliado pelo caminho de produção (sem o rótulo nas features)."""

    nome: str
    limpo: pd.DataFrame  # sem a coluna falha_72h
    y: np.ndarray
    X: pd.DataFrame
    proba: dict[str, np.ndarray] = field(default_factory=dict)
    grupos: np.ndarray | None = None  # motor x dia, para bootstrap por blocos

    def pred(self, versao: str, limiar: float = 0.5) -> np.ndarray:
        return (self.proba[versao] >= limiar).astype(int)


def avaliar(nome: str) -> Avaliacao:
    limpo = sn.preprocessamento.limpar(_bruto(nome))
    y = limpo.pop(sn.dados.ALVO).to_numpy().astype(int)
    X = sn.features.construir(limpo)
    grupos = (limpo["id_maquina"] + "|" + limpo["timestamp"].dt.strftime("%Y-%m-%d")).to_numpy()
    av = Avaliacao(nome=nome, limpo=limpo, y=y, X=X, grupos=grupos)
    for versao in VERSOES:
        av.proba[versao] = sn.modelo.carregar(versao).prever_proba(X)
    return av


@pytest.fixture(scope="session")
def aval_teste() -> Avaliacao:
    return avaliar("teste")


@pytest.fixture(scope="session")
def aval_producao() -> Avaliacao:
    return avaliar("producao")
