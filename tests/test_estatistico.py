"""Bloco B — testes estatísticos.

Toda avaliação aqui usa o CAMINHO DE PRODUÇÃO: features construídas sem a
coluna `falha_72h` (ver D01). A comparação v1 x v2 usa bootstrap por
blocos motor x dia (175 blocos no teste), porque leituras horárias do mesmo
motor não são independentes.

Premissa de custo (documentada no relatório): um falso negativo (motor
queimado, parada não programada) custa 10x um falso positivo (equipe
deslocada à toa).
"""
import numpy as np
import pytest

import sentinela as sn
from apoio import estatistica as est

CUSTO_FN = 10.0
CUSTO_FP = 1.0
N_BOOT = 1000
LIMIARES = np.round(np.arange(0.05, 0.96, 0.05), 2)


def _recall(y, pred):
    positivos = y == 1
    return pred[positivos].mean() if positivos.any() else np.nan


def _ic_diferenca(av, metrica):
    """IC 95% por bootstrap em blocos de metrica(v2) - metrica(v1)."""
    y = av.y

    def estatistica(idx):
        return metrica(y[idx], av, "v2", idx) - metrica(y[idx], av, "v1", idx)

    return est.intervalo(est.bootstrap_por_grupo(av.grupos, estatistica, N_BOOT, semente=42))


def _m_recall(y, av, versao, idx):
    return _recall(y, av.pred(versao)[idx])


def _m_custo(y, av, versao, idx):
    return est.custo(y, av.pred(versao)[idx], CUSTO_FN, CUSTO_FP) / len(idx)


def _m_pr_auc(y, av, versao, idx):
    return est.precisao_media(y, av.proba[versao][idx])


# --------------------------------------------------------------------------
# Métrica adequada à classe rara
# --------------------------------------------------------------------------
def test_acuracia_premia_o_preditor_que_nunca_preve_falha(aval_teste):
    """Prevalência de 15,6%: 'nunca falha' acerta 84,4% e não pega nenhum
    motor. A v1 pega 94% das falhas e tem acurácia MENOR. A acurácia ordena
    os dois ao contrário do que a manutenção precisa."""
    y = aval_teste.y
    trivial = np.zeros_like(y)
    acc_trivial = (trivial == y).mean()
    acc_v1 = (aval_teste.pred("v1") == y).mean()
    assert acc_trivial == pytest.approx(0.844, abs=0.001)
    assert acc_v1 < acc_trivial
    assert _recall(y, aval_teste.pred("v1")) > 0.9 > _recall(y, trivial)


@pytest.mark.parametrize("versao", ["v1", "v2"])
def test_pr_auc_bem_acima_da_prevalencia(aval_teste, versao):
    """PR-AUC de um classificador aleatório = prevalência (0,156)."""
    assert est.precisao_media(aval_teste.y, aval_teste.proba[versao]) > 3 * aval_teste.y.mean()


# --------------------------------------------------------------------------
# v1 x v2
# --------------------------------------------------------------------------
def test_reproduz_o_ganho_de_acuracia_reportado_pela_equipe(lote_teste):
    """+3,5 pp com o pipeline rotulado — o número do README se reproduz.
    (Ele está contaminado por D01: sem o rótulo, as duas acurácias caem.)"""
    acc = {}
    for versao in ("v1", "v2"):
        saida = sn.pipeline.executar(lote_teste, versao=versao)
        acc[versao] = sn.avaliacao.metricas(saida["falha_72h"], saida["predicao"])["acuracia"]
    assert acc["v2"] - acc["v1"] == pytest.approx(0.035, abs=0.002)


def test_ganho_de_acuracia_da_v2_e_significativo_mcnemar(aval_teste):
    y = aval_teste.y
    r = est.mcnemar_exato(aval_teste.pred("v1") == y, aval_teste.pred("v2") == y)
    assert r["c"] > r["b"] and r["p_valor"] < 0.001, r


def test_v2_ordena_melhor_que_a_v1_no_teste(aval_teste):
    """PR-AUC: IC 95% da diferença v2 - v1 inteiro acima de zero."""
    baixo, alto = _ic_diferenca(aval_teste, _m_pr_auc)
    assert baixo > 0, (baixo, alto)


@pytest.mark.defeito("D07")
def test_v2_nao_perde_recall_em_relacao_a_v1(aval_teste):
    """Não-inferioridade com margem de 2 pp: promover a v2 não pode deixar
    passar mais motores que vão falhar."""
    y = aval_teste.y
    positivos = y == 1
    r = est.mcnemar_exato(aval_teste.pred("v1")[positivos] == 1, aval_teste.pred("v2")[positivos] == 1)
    baixo, alto = _ic_diferenca(aval_teste, _m_recall)
    assert baixo >= -0.02, (
        f"recall v2 - v1: IC95% [{baixo:+.3f}, {alto:+.3f}]; entre as falhas reais, "
        f"v1 pega e v2 perde {r['b']}, o contrário {r['c']} (McNemar p={r['p_valor']:.1e})"
    )


@pytest.mark.defeito("D07")
def test_promover_a_v2_nao_aumenta_o_custo_esperado(aval_teste):
    baixo, alto = _ic_diferenca(aval_teste, _m_custo)
    assert baixo <= 0, f"custo/leitura v2 - v1 (FN=10, FP=1): IC95% [{baixo:+.3f}, {alto:+.3f}]"


@pytest.mark.defeito("D07")
def test_vantagem_da_v2_em_pr_auc_se_mantem_em_producao(aval_producao):
    """O ganho de ordenação visto no teste precisa sobreviver ao lote seguinte."""
    baixo, alto = _ic_diferenca(aval_producao, _m_pr_auc)
    assert baixo > 0, f"PR-AUC v2 - v1 em produção: IC95% [{baixo:+.3f}, {alto:+.3f}]"


# --------------------------------------------------------------------------
# Limiar de decisão
# --------------------------------------------------------------------------
@pytest.mark.parametrize("versao", ["v1", "v2"])
def test_recall_cai_e_precisao_sobe_com_o_limiar(aval_teste, versao):
    varredura = est.varrer_limiar(aval_teste.y, aval_teste.proba[versao], LIMIARES, CUSTO_FN, CUSTO_FP)
    recalls = [linha["recall"] for linha in varredura]
    assert all(a >= b for a, b in zip(recalls, recalls[1:]))
    assert varredura[-1]["precisao"] > varredura[0]["precisao"]


@pytest.mark.parametrize("versao", ["v1", pytest.param("v2", marks=pytest.mark.defeito("D09"))])
def test_limiar_05_esta_perto_do_custo_minimo(aval_teste, versao):
    """Custo no limiar 0,5 no máximo 10% acima do melhor limiar da varredura."""
    varredura = est.varrer_limiar(aval_teste.y, aval_teste.proba[versao], LIMIARES, CUSTO_FN, CUSTO_FP)
    melhor = min(varredura, key=lambda linha: linha["custo"])
    padrao = next(linha for linha in varredura if linha["limiar"] == 0.5)
    assert padrao["custo"] <= 1.10 * melhor["custo"], (
        f"{versao}: custo {padrao['custo']:.0f} em 0,5 (recall {padrao['recall']:.2f}) contra "
        f"{melhor['custo']:.0f} em {melhor['limiar']} (recall {melhor['recall']:.2f})"
    )


def test_limiar_escolhido_no_teste_reduz_o_custo_da_v2_em_producao(aval_teste, aval_producao):
    """Controle fora da amostra: o limiar ótimo do teste também é melhor que
    0,5 no lote de produção — o ajuste não é sobreajuste ao teste."""
    varredura = est.varrer_limiar(aval_teste.y, aval_teste.proba["v2"], LIMIARES, CUSTO_FN, CUSTO_FP)
    melhor = min(varredura, key=lambda linha: linha["custo"])["limiar"]
    y, p = aval_producao.y, aval_producao.proba["v2"]
    assert est.custo(y, p >= melhor, CUSTO_FN, CUSTO_FP) < est.custo(y, p >= 0.5, CUSTO_FN, CUSTO_FP)


# --------------------------------------------------------------------------
# Calibração
# --------------------------------------------------------------------------
@pytest.mark.parametrize("versao", [pytest.param("v1", marks=pytest.mark.defeito("D08")), "v2"])
def test_probabilidade_e_calibrada(aval_teste, versao):
    """Por faixa de 0,1: probabilidade média prevista x frequência observada.
    Critério: erro de calibração esperado (ECE) <= 0,10."""
    y, p = aval_teste.y, aval_teste.proba[versao]
    ece = est.erro_calibracao_esperado(y, p)
    tabela = est.tabela_calibracao(y, p)
    pior = max(tabela, key=lambda linha: abs(linha["prevista"] - linha["observada"]))
    assert ece <= 0.10, (
        f"{versao}: ECE={ece:.3f}; prob. média {p.mean():.3f} x prevalência {y.mean():.3f}; "
        f"pior faixa {pior['faixa']}: prevista {pior['prevista']:.2f}, observada {pior['observada']:.2f}"
    )


# --------------------------------------------------------------------------
# Teste x produção
# --------------------------------------------------------------------------
SENSORES = ["temperatura_c", "vibracao_rms", "corrente_a", "rpm"]


@pytest.mark.parametrize("sensor", SENSORES)
def test_media_dos_sensores_estavel_entre_teste_e_producao(carregar, sensor):
    """Passa para TODOS os sensores — e é justamente por isso que não basta."""
    teste, producao = carregar("teste")[sensor], carregar("producao")[sensor]
    assert abs(producao.mean() - teste.mean()) <= 0.5 * teste.std()


def _variacoes(df, coluna):
    ordenado = df.sort_values(["id_maquina", "timestamp"])
    return ordenado.groupby("id_maquina")[coluna].diff().dropna().to_numpy()


@pytest.mark.parametrize(
    "sensor",
    [pytest.param(s, marks=pytest.mark.defeito("D10")) if s == "temperatura_c" else s for s in SENSORES],
)
def test_distribuicao_das_variacoes_hora_a_hora_estavel(carregar, sensor):
    """KS sobre x_t - x_{t-1} de cada motor. Diferenciar remove o nível de
    cada motor (que muda conforme quais motores estão falhando) e isola o
    comportamento do sensor. Critério: D <= 0,10."""
    ks = est.ks_duas_amostras(_variacoes(carregar("teste"), sensor), _variacoes(carregar("producao"), sensor))
    assert ks["d"] <= 0.10, f"{sensor}: KS D={ks['d']:.3f}, p={ks['p_valor']:.1e}"


@pytest.mark.defeito("D10")
def test_mesma_media_nao_significa_mesma_distribuicao(carregar):
    """Temperatura: treino e produção têm a MESMA média (67,23 °C). Quem
    compara médias conclui 'sem drift'. O desvio-padrão sobe 52%."""
    treino, producao = carregar("treino")["temperatura_c"], carregar("producao")["temperatura_c"]
    assert treino.mean() == pytest.approx(producao.mean(), abs=0.01)  # a média engana...
    razao = producao.std() / treino.std()
    ks = est.ks_duas_amostras(treino, producao)
    assert razao <= 1.2, f"desvio-padrão x{razao:.2f}; KS D={ks['d']:.3f}, p={ks['p_valor']:.1e}"


def test_prevalencia_estavel_entre_teste_e_producao(aval_teste, aval_producao):
    """Teste z de duas proporções: 15,6% x 14,3%."""
    p1, p2 = aval_teste.y.mean(), aval_producao.y.mean()
    n1, n2 = len(aval_teste.y), len(aval_producao.y)
    comum = (p1 * n1 + p2 * n2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(comum * (1 - comum) * (1 / n1 + 1 / n2))
    assert abs(z) < 1.96


@pytest.mark.parametrize("versao", ["v1", "v2"])
def test_pr_auc_nao_degrada_em_producao(aval_teste, aval_producao, versao):
    ap_teste = est.precisao_media(aval_teste.y, aval_teste.proba[versao])
    ap_prod = est.precisao_media(aval_producao.y, aval_producao.proba[versao])
    assert ap_prod >= ap_teste - 0.05
