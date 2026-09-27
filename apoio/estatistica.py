"""Métricas e testes estatísticos que o sentinela.avaliacao não oferece."""
import math

import numpy as np


def matriz_confusao(y_real, y_pred):
    y_real = np.asarray(y_real).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "vp": int(np.sum((y_real == 1) & (y_pred == 1))),
        "fn": int(np.sum((y_real == 1) & (y_pred == 0))),
        "fp": int(np.sum((y_real == 0) & (y_pred == 1))),
        "vn": int(np.sum((y_real == 0) & (y_pred == 0))),
    }


def custo(y_real, y_pred, custo_fn, custo_fp):
    m = matriz_confusao(y_real, y_pred)
    return custo_fn * m["fn"] + custo_fp * m["fp"]


def varrer_limiares(y_real, proba, limiares, custo_fn, custo_fp):
    proba = np.asarray(proba, dtype=float)
    resultados = []
    for limiar in limiares:
        y_pred = (proba >= limiar).astype(int)
        m = matriz_confusao(y_real, y_pred)
        precisao = m["vp"] / (m["vp"] + m["fp"]) if m["vp"] + m["fp"] > 0 else 0.0
        recall = m["vp"] / (m["vp"] + m["fn"]) if m["vp"] + m["fn"] > 0 else 0.0
        resultados.append({
            "limiar": float(limiar),
            "precisao": precisao,
            "recall": recall,
            "custo": custo_fn * m["fn"] + custo_fp * m["fp"],
            **m,
        })
    return resultados


def pr_auc(y_real, proba):
    """Average precision (área sob a curva precisão x recall)."""
    y_real = np.asarray(y_real).astype(int)
    proba = np.asarray(proba, dtype=float)
    if y_real.sum() == 0:
        return float("nan")

    ordem = np.argsort(-proba, kind="mergesort")
    y_ordenado = y_real[ordem]
    precisao_no_topo = np.cumsum(y_ordenado) / np.arange(1, len(y_ordenado) + 1)
    return float(np.sum(precisao_no_topo * y_ordenado) / y_ordenado.sum())


def tabela_calibracao(y_real, proba, n_faixas=10):
    """Probabilidade média prevista x frequência observada, por faixa."""
    y_real = np.asarray(y_real).astype(int)
    proba = np.asarray(proba, dtype=float)
    bordas = np.linspace(0, 1, n_faixas + 1)
    faixa_de_cada_linha = np.clip(np.digitize(proba, bordas[1:-1]), 0, n_faixas - 1)

    tabela = []
    for faixa in range(n_faixas):
        dentro = faixa_de_cada_linha == faixa
        if dentro.any():
            tabela.append({
                "faixa": f"[{bordas[faixa]:.1f}, {bordas[faixa + 1]:.1f})",
                "n": int(dentro.sum()),
                "prevista": float(proba[dentro].mean()),
                "observada": float(y_real[dentro].mean()),
            })
    return tabela


def erro_calibracao(y_real, proba, n_faixas=10):
    """ECE: média ponderada de |prevista - observada| nas faixas."""
    tabela = tabela_calibracao(y_real, proba, n_faixas)
    total = sum(f["n"] for f in tabela)
    return float(sum(f["n"] * abs(f["prevista"] - f["observada"]) for f in tabela) / total)


def mcnemar(acertos_a, acertos_b):
    """McNemar exato: b = só A acertou, c = só B acertou."""
    acertos_a = np.asarray(acertos_a).astype(bool)
    acertos_b = np.asarray(acertos_b).astype(bool)
    b = int(np.sum(acertos_a & ~acertos_b))
    c = int(np.sum(~acertos_a & acertos_b))
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_valor": 1.0}

    # sob H0, b ~ Binomial(n, 0.5); p bilateral
    cauda = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    p_valor = min(1.0, 2 * cauda / 2**n)
    return {"b": b, "c": c, "p_valor": float(p_valor)}


def bootstrap_por_blocos(blocos, funcao, n_reamostras=1000, semente=0):
    """Reamostra blocos inteiros (ex.: motor x dia) em vez de linhas soltas.

    Leituras horárias do mesmo motor são correlacionadas; reamostrar linhas
    daria um intervalo mais estreito do que deveria. `funcao` recebe os
    índices das linhas sorteadas e devolve um número.
    """
    blocos = np.asarray(blocos)
    nomes = np.unique(blocos)
    linhas_por_bloco = [np.flatnonzero(blocos == nome) for nome in nomes]

    rng = np.random.default_rng(semente)
    valores = np.empty(n_reamostras)
    for i in range(n_reamostras):
        sorteados = rng.integers(0, len(nomes), len(nomes))
        linhas = np.concatenate([linhas_por_bloco[j] for j in sorteados])
        valores[i] = funcao(linhas)
    return valores


def intervalo_95(valores):
    return float(np.quantile(valores, 0.025)), float(np.quantile(valores, 0.975))


def teste_ks(a, b):
    """Kolmogorov-Smirnov de duas amostras (D e p-valor assintótico)."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    todos = np.concatenate([a, b])
    acumulada_a = np.searchsorted(a, todos, side="right") / len(a)
    acumulada_b = np.searchsorted(b, todos, side="right") / len(b)
    d = float(np.max(np.abs(acumulada_a - acumulada_b)))

    n = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p_valor = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return {"d": d, "p_valor": float(min(1.0, max(0.0, p_valor)))}


def psi(referencia, atual, n_faixas=10):
    """Population Stability Index com faixas nos quantis da referência."""
    referencia = np.asarray(referencia, dtype=float)
    atual = np.asarray(atual, dtype=float)
    bordas = np.quantile(referencia, np.linspace(0, 1, n_faixas + 1))
    bordas[0], bordas[-1] = -np.inf, np.inf

    frac_ref = np.histogram(referencia, bordas)[0] / len(referencia)
    frac_atual = np.histogram(atual, bordas)[0] / len(atual)
    frac_ref = np.clip(frac_ref, 1e-6, None)
    frac_atual = np.clip(frac_atual, 1e-6, None)
    return float(np.sum((frac_atual - frac_ref) * np.log(frac_atual / frac_ref)))
