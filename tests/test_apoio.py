"""Testa os instrumentos de `apoio/estatistica.py` contra casos de resposta
conhecida. Se o instrumento está errado, todo o bloco B está errado."""
import numpy as np
import pytest

from apoio import estatistica as est


def test_pr_auc_de_ordenacao_perfeita_e_1_e_de_ordenacao_invertida_e_baixa():
    y = np.array([0, 0, 0, 1, 1])
    assert est.precisao_media(y, [0.1, 0.2, 0.3, 0.8, 0.9]) == pytest.approx(1.0)
    assert est.precisao_media(y, [0.9, 0.8, 0.7, 0.2, 0.1]) < 0.5


def test_pr_auc_de_score_aleatorio_fica_perto_da_prevalencia():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.15).astype(int)
    assert est.precisao_media(y, rng.random(20000)) == pytest.approx(0.15, abs=0.01)


def test_ece_de_modelo_calibrado_e_quase_zero_e_de_modelo_inflado_nao():
    rng = np.random.default_rng(1)
    p = rng.random(50000)
    y = (rng.random(50000) < p).astype(int)
    assert est.erro_calibracao_esperado(y, p) < 0.01
    assert est.erro_calibracao_esperado(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_mcnemar_simetrico_nao_rejeita_e_assimetrico_rejeita():
    a = np.array([1] * 50 + [0] * 50 + [1] * 100, dtype=bool)
    b = np.array([0] * 50 + [1] * 50 + [1] * 100, dtype=bool)
    assert est.mcnemar_exato(a, b)["p_valor"] == pytest.approx(1.0)
    a = np.array([1] * 5 + [0] * 45, dtype=bool)
    b = np.array([0] * 5 + [1] * 45, dtype=bool)
    r = est.mcnemar_exato(a, b)
    assert (r["b"], r["c"]) == (5, 45) and r["p_valor"] < 1e-6


def test_mcnemar_bate_com_valor_tabelado():
    # b=2, c=8 -> p exato bilateral = 2 * P(X<=2 | n=10, 1/2) = 0.109375
    a = np.array([1, 1] + [0] * 8, dtype=bool)
    b = np.array([0, 0] + [1] * 8, dtype=bool)
    assert est.mcnemar_exato(a, b)["p_valor"] == pytest.approx(0.109375)


def test_bootstrap_por_grupo_cobre_a_media_verdadeira():
    rng = np.random.default_rng(2)
    grupos = np.repeat(np.arange(200), 10)
    x = rng.normal(5.0, 1.0, 200)[grupos] + rng.normal(0, 0.1, 2000)
    baixo, alto = est.intervalo(est.bootstrap_por_grupo(grupos, lambda idx: x[idx].mean(), 500))
    assert baixo < 5.0 < alto


def test_ks_iguais_nao_rejeita_e_escala_diferente_rejeita():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 3000), rng.normal(0, 1, 3000)
    assert est.ks_duas_amostras(a, b)["p_valor"] > 0.01
    c = rng.normal(0, 3, 3000)  # mesma média, desvio 3x
    r = est.ks_duas_amostras(a, c)
    assert r["d"] > 0.2 and r["p_valor"] < 1e-10


def test_psi_zero_para_mesma_distribuicao():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 5000)
    assert est.indice_estabilidade_populacional(a, rng.normal(0, 1, 5000)) < 0.02
    assert est.indice_estabilidade_populacional(a, rng.normal(0, 3, 5000)) > 0.25


def test_varredura_de_limiar_conta_certo():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    linha = est.varrer_limiar(y, p, [0.5], custo_fn=10, custo_fp=1)[0]
    assert (linha["vp"], linha["fn"], linha["fp"], linha["vn"]) == (1, 1, 1, 1)
    assert linha["custo"] == 11
