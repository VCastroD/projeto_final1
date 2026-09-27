"""Confere as funções de apoio/estatistica.py em casos com resposta conhecida."""
import numpy as np
import pytest

from apoio import estatistica


def test_pr_auc_de_ordenacao_perfeita_e_1_e_de_ordenacao_invertida_e_baixa():
    y_real = np.array([0, 0, 0, 1, 1])
    assert estatistica.pr_auc(y_real, [0.1, 0.2, 0.3, 0.8, 0.9]) == pytest.approx(1.0)
    assert estatistica.pr_auc(y_real, [0.9, 0.8, 0.7, 0.2, 0.1]) < 0.5


def test_pr_auc_de_score_aleatorio_fica_perto_da_prevalencia():
    rng = np.random.default_rng(0)
    y_real = (rng.random(20000) < 0.15).astype(int)
    assert estatistica.pr_auc(y_real, rng.random(20000)) == pytest.approx(0.15, abs=0.01)


def test_ece_de_modelo_calibrado_e_quase_zero_e_de_modelo_inflado_nao():
    rng = np.random.default_rng(1)
    proba = rng.random(50000)
    y_real = (rng.random(50000) < proba).astype(int)
    assert estatistica.erro_calibracao(y_real, proba) < 0.01
    assert estatistica.erro_calibracao(y_real, np.clip(proba + 0.3, 0, 1)) > 0.2


def test_mcnemar_simetrico_nao_rejeita_e_assimetrico_rejeita():
    a = np.array([1] * 50 + [0] * 50 + [1] * 100, dtype=bool)
    b = np.array([0] * 50 + [1] * 50 + [1] * 100, dtype=bool)
    assert estatistica.mcnemar(a, b)["p_valor"] == pytest.approx(1.0)

    a = np.array([1] * 5 + [0] * 45, dtype=bool)
    b = np.array([0] * 5 + [1] * 45, dtype=bool)
    resultado = estatistica.mcnemar(a, b)
    assert (resultado["b"], resultado["c"]) == (5, 45)
    assert resultado["p_valor"] < 1e-6


def test_mcnemar_bate_com_valor_tabelado():
    # b=2, c=8 -> p exato bilateral = 2 * P(X <= 2 | n=10, p=1/2) = 0.109375
    a = np.array([1, 1] + [0] * 8, dtype=bool)
    b = np.array([0, 0] + [1] * 8, dtype=bool)
    assert estatistica.mcnemar(a, b)["p_valor"] == pytest.approx(0.109375)


def test_bootstrap_por_blocos_cobre_a_media_verdadeira():
    rng = np.random.default_rng(2)
    blocos = np.repeat(np.arange(200), 10)
    x = rng.normal(5.0, 1.0, 200)[blocos] + rng.normal(0, 0.1, 2000)

    def media(linhas):
        return x[linhas].mean()

    minimo, maximo = estatistica.intervalo_95(estatistica.bootstrap_por_blocos(blocos, media, 500))
    assert minimo < 5.0 < maximo


def test_ks_iguais_nao_rejeita_e_escala_diferente_rejeita():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 3000)
    assert estatistica.teste_ks(a, rng.normal(0, 1, 3000))["p_valor"] > 0.01

    resultado = estatistica.teste_ks(a, rng.normal(0, 3, 3000))  # mesma média, desvio 3x
    assert resultado["d"] > 0.2
    assert resultado["p_valor"] < 1e-10


def test_psi_zero_para_mesma_distribuicao():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 5000)
    assert estatistica.psi(a, rng.normal(0, 1, 5000)) < 0.02
    assert estatistica.psi(a, rng.normal(0, 3, 5000)) > 0.25


def test_varredura_de_limiar_conta_certo():
    y_real = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.4, 0.6, 0.1])
    resultado = estatistica.varrer_limiares(y_real, proba, [0.5], custo_fn=10, custo_fp=1)[0]
    assert (resultado["vp"], resultado["fn"], resultado["fp"], resultado["vn"]) == (1, 1, 1, 1)
    assert resultado["custo"] == 11
