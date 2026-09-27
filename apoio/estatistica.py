"""Instrumentos estatísticos da suíte, em numpy puro.

O pacote `sentinela.avaliacao` expõe só acurácia/precisão/recall/F1. Tudo o
que a suíte precisa além disso (PR-AUC, calibração, McNemar, bootstrap,
KS, PSI) está aqui — e é testado em `tests/test_apoio.py`, porque um
instrumento de medida sem teste não prova nada.
"""
from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------
# Métricas para classe rara
# --------------------------------------------------------------------------
def precisao_media(y, p) -> float:
    """Average precision (área sob a curva precisão-recall, forma em degraus)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if y.sum() == 0:
        return float("nan")
    ordem = np.argsort(-p, kind="mergesort")
    y_ord = y[ordem]
    vp = np.cumsum(y_ord)
    precisao = vp / np.arange(1, len(y_ord) + 1)
    return float((precisao * y_ord).sum() / y_ord.sum())


def contagens(y, pred) -> dict[str, int]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    return {
        "vp": int(((y == 1) & (pred == 1)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "vn": int(((y == 0) & (pred == 0)).sum()),
    }


def custo(y, pred, custo_fn: float, custo_fp: float) -> float:
    c = contagens(y, pred)
    return custo_fn * c["fn"] + custo_fp * c["fp"]


def varrer_limiar(y, p, limiares, custo_fn: float, custo_fp: float) -> list[dict]:
    """Precisão, recall e custo para cada limiar."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    linhas = []
    for limiar in limiares:
        pred = (p >= limiar).astype(int)
        c = contagens(y, pred)
        previstos = c["vp"] + c["fp"]
        linhas.append(
            {
                "limiar": float(limiar),
                "precisao": c["vp"] / previstos if previstos else 0.0,
                "recall": c["vp"] / (c["vp"] + c["fn"]) if (c["vp"] + c["fn"]) else 0.0,
                "custo": custo_fn * c["fn"] + custo_fp * c["fp"],
                **c,
            }
        )
    return linhas


# --------------------------------------------------------------------------
# Calibração
# --------------------------------------------------------------------------
def tabela_calibracao(y, p, n_faixas: int = 10) -> list[dict]:
    """Agrupa por faixa de probabilidade: média prevista × frequência observada."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    bordas = np.linspace(0.0, 1.0, n_faixas + 1)
    faixa = np.clip(np.digitize(p, bordas[1:-1]), 0, n_faixas - 1)
    linhas = []
    for f in range(n_faixas):
        sel = faixa == f
        if sel.any():
            linhas.append(
                {
                    "faixa": f"[{bordas[f]:.1f}, {bordas[f + 1]:.1f})",
                    "n": int(sel.sum()),
                    "prevista": float(p[sel].mean()),
                    "observada": float(y[sel].mean()),
                }
            )
    return linhas


def erro_calibracao_esperado(y, p, n_faixas: int = 10) -> float:
    """ECE: média ponderada de |prevista - observada| por faixa."""
    tabela = tabela_calibracao(y, p, n_faixas)
    total = sum(linha["n"] for linha in tabela)
    return float(sum(linha["n"] * abs(linha["prevista"] - linha["observada"]) for linha in tabela) / total)


# --------------------------------------------------------------------------
# Comparação pareada v1 x v2
# --------------------------------------------------------------------------
def mcnemar_exato(acerto_a, acerto_b) -> dict:
    """Teste de McNemar exato (binomial) sobre acertos pareados.

    b = A acerta e B erra; c = A erra e B acerta. Sob H0 (mesma taxa de
    erro), b ~ Binomial(b + c, 1/2). Devolve b, c e o p-valor bilateral.
    """
    a = np.asarray(acerto_a).astype(bool)
    bb = np.asarray(acerto_b).astype(bool)
    b = int((a & ~bb).sum())
    c = int((~a & bb).sum())
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_valor": 1.0}
    k = min(b, c)
    cauda = sum(math.comb(n, i) for i in range(k + 1))
    p_valor = min(1.0, 2 * cauda / 2**n)
    return {"b": b, "c": c, "p_valor": float(p_valor)}


def bootstrap_por_grupo(grupos, estatistica, n_reamostras: int = 1000, semente: int = 0) -> np.ndarray:
    """Bootstrap por blocos: reamostra grupos inteiros (ex.: motor x dia).

    Leituras horárias do mesmo motor são autocorrelacionadas; reamostrar
    linhas soltas estreitaria o intervalo artificialmente. `estatistica`
    recebe um vetor de índices de linha e devolve um número.
    """
    grupos = np.asarray(grupos)
    rotulos, codigo = np.unique(grupos, return_inverse=True)
    linhas_por_grupo = [np.flatnonzero(codigo == g) for g in range(len(rotulos))]
    rng = np.random.default_rng(semente)
    valores = np.empty(n_reamostras)
    for r in range(n_reamostras):
        escolhidos = rng.integers(0, len(rotulos), len(rotulos))
        idx = np.concatenate([linhas_por_grupo[g] for g in escolhidos])
        valores[r] = estatistica(idx)
    return valores


def intervalo(valores, confianca: float = 0.95) -> tuple[float, float]:
    alfa = (1 - confianca) / 2
    return float(np.quantile(valores, alfa)), float(np.quantile(valores, 1 - alfa))


# --------------------------------------------------------------------------
# Comparação de distribuições
# --------------------------------------------------------------------------
def ks_duas_amostras(a, b) -> dict:
    """Estatística D de Kolmogorov-Smirnov e p-valor assintótico."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    todos = np.concatenate([a, b])
    fa = np.searchsorted(a, todos, side="right") / len(a)
    fb = np.searchsorted(b, todos, side="right") / len(b)
    d = float(np.max(np.abs(fa - fb)))
    n_ef = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(n_ef) + 0.12 + 0.11 / math.sqrt(n_ef)) * d
    p_valor = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return {"d": d, "p_valor": float(min(1.0, max(0.0, p_valor)))}


def indice_estabilidade_populacional(referencia, atual, n_faixas: int = 10) -> float:
    """PSI com faixas nos quantis da referência."""
    referencia = np.asarray(referencia, dtype=float)
    atual = np.asarray(atual, dtype=float)
    bordas = np.quantile(referencia, np.linspace(0, 1, n_faixas + 1))
    bordas[0], bordas[-1] = -np.inf, np.inf
    pr = np.clip(np.histogram(referencia, bordas)[0] / len(referencia), 1e-6, None)
    pa = np.clip(np.histogram(atual, bordas)[0] / len(atual), 1e-6, None)
    return float(np.sum((pa - pr) * np.log(pa / pr)))
