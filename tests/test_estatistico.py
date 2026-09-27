"""Bloco B: testes estatísticos.

Tudo aqui usa as features calculadas SEM a coluna falha_72h, que é como o
sistema roda na planta (ver D01).

Premissa de custo: um falso negativo (motor queima sem aviso) custa 10 vezes
um falso positivo (equipe deslocada à toa).
"""
import numpy as np
import pytest

import sentinela as sn
from apoio import estatistica

CUSTO_FN = 10.0
CUSTO_FP = 1.0
N_REAMOSTRAS = 1000
LIMIARES = np.round(np.arange(0.05, 0.96, 0.05), 2)


def recall(conjunto, versao, linhas):
    y_real = conjunto.y_real[linhas]
    y_pred = conjunto.pred(versao)[linhas]
    return y_pred[y_real == 1].mean()


def custo_por_leitura(conjunto, versao, linhas):
    y_real = conjunto.y_real[linhas]
    y_pred = conjunto.pred(versao)[linhas]
    return estatistica.custo(y_real, y_pred, CUSTO_FN, CUSTO_FP) / len(linhas)


def pr_auc(conjunto, versao, linhas):
    return estatistica.pr_auc(conjunto.y_real[linhas], conjunto.proba[versao][linhas])


def intervalo_v2_menos_v1(conjunto, metrica):
    """IC 95% de metrica(v2) - metrica(v1), por bootstrap em blocos motor x dia."""
    def diferenca(linhas):
        return metrica(conjunto, "v2", linhas) - metrica(conjunto, "v1", linhas)

    valores = estatistica.bootstrap_por_blocos(conjunto.motor_dia, diferenca, N_REAMOSTRAS, semente=42)
    return estatistica.intervalo_95(valores)


# --- métrica certa para classe rara -------------------------------------------

def test_acuracia_premia_o_preditor_que_nunca_preve_falha(teste_avaliado):
    """Com 15,6% de falhas, "nunca falha" acerta 84,4% e tem acurácia maior que a v1."""
    y_real = teste_avaliado.y_real
    nunca_falha = np.zeros_like(y_real)
    y_pred_v1 = teste_avaliado.pred("v1")

    acuracia_nunca_falha = (nunca_falha == y_real).mean()
    acuracia_v1 = (y_pred_v1 == y_real).mean()

    assert acuracia_nunca_falha == pytest.approx(0.844, abs=0.001)
    assert acuracia_v1 < acuracia_nunca_falha
    assert y_pred_v1[y_real == 1].mean() > 0.9  # mas a v1 pega mais de 90% das falhas


@pytest.mark.parametrize("versao", ["v1", "v2"])
def test_pr_auc_bem_acima_da_prevalencia(teste_avaliado, versao):
    # um modelo que chuta tem PR-AUC ~ prevalência (0,156)
    valor = estatistica.pr_auc(teste_avaliado.y_real, teste_avaliado.proba[versao])
    assert valor > 3 * teste_avaliado.y_real.mean()


# --- v1 x v2 ------------------------------------------------------------------

def test_reproduz_o_ganho_de_acuracia_reportado_pela_equipe(lote_teste):
    """O +3,5 pp do README se reproduz, mas só com o rótulo no lote (D01)."""
    acuracia = {}
    for versao in ("v1", "v2"):
        saida = sn.pipeline.executar(lote_teste, versao=versao)
        acuracia[versao] = sn.avaliacao.metricas(saida["falha_72h"], saida["predicao"])["acuracia"]
    assert acuracia["v2"] - acuracia["v1"] == pytest.approx(0.035, abs=0.002)


def test_ganho_de_acuracia_da_v2_e_significativo_mcnemar(teste_avaliado):
    y_real = teste_avaliado.y_real
    resultado = estatistica.mcnemar(teste_avaliado.pred("v1") == y_real, teste_avaliado.pred("v2") == y_real)
    assert resultado["c"] > resultado["b"]
    assert resultado["p_valor"] < 0.001


def test_v2_ordena_melhor_que_a_v1_no_teste(teste_avaliado):
    minimo, maximo = intervalo_v2_menos_v1(teste_avaliado, pr_auc)
    assert minimo > 0, f"PR-AUC v2 - v1: IC95% [{minimo:+.3f}, {maximo:+.3f}]"


@pytest.mark.defeito("D07")
def test_v2_nao_perde_recall_em_relacao_a_v1(teste_avaliado):
    """Não-inferioridade: a v2 pode perder no máximo 2 pp de recall."""
    minimo, maximo = intervalo_v2_menos_v1(teste_avaliado, recall)

    falhas = teste_avaliado.y_real == 1
    v1_avisou = teste_avaliado.pred("v1")[falhas] == 1
    v2_avisou = teste_avaliado.pred("v2")[falhas] == 1
    resultado = estatistica.mcnemar(v1_avisou, v2_avisou)

    assert minimo >= -0.02, (
        f"recall v2 - v1: IC95% [{minimo:+.3f}, {maximo:+.3f}]; entre as falhas reais, "
        f"v1 pega e v2 perde {resultado['b']}, o contrário {resultado['c']} (McNemar p={resultado['p_valor']:.1e})"
    )


@pytest.mark.defeito("D07")
def test_promover_a_v2_nao_aumenta_o_custo_esperado(teste_avaliado):
    minimo, maximo = intervalo_v2_menos_v1(teste_avaliado, custo_por_leitura)
    assert minimo <= 0, f"custo/leitura v2 - v1 (FN=10, FP=1): IC95% [{minimo:+.3f}, {maximo:+.3f}]"


@pytest.mark.defeito("D07")
def test_vantagem_da_v2_em_pr_auc_se_mantem_em_producao(producao_avaliada):
    minimo, maximo = intervalo_v2_menos_v1(producao_avaliada, pr_auc)
    assert minimo > 0, f"PR-AUC v2 - v1 em produção: IC95% [{minimo:+.3f}, {maximo:+.3f}]"


# --- limiar de decisão --------------------------------------------------------

@pytest.mark.parametrize("versao", ["v1", "v2"])
def test_recall_cai_e_precisao_sobe_com_o_limiar(teste_avaliado, versao):
    resultados = estatistica.varrer_limiares(
        teste_avaliado.y_real, teste_avaliado.proba[versao], LIMIARES, CUSTO_FN, CUSTO_FP
    )
    recalls = [r["recall"] for r in resultados]
    assert recalls == sorted(recalls, reverse=True)
    assert resultados[-1]["precisao"] > resultados[0]["precisao"]


@pytest.mark.parametrize("versao", ["v1", pytest.param("v2", marks=pytest.mark.defeito("D09"))])
def test_limiar_05_esta_perto_do_custo_minimo(teste_avaliado, versao):
    """O custo no limiar 0,5 pode ser no máximo 10% maior que o do melhor limiar."""
    resultados = estatistica.varrer_limiares(
        teste_avaliado.y_real, teste_avaliado.proba[versao], LIMIARES, CUSTO_FN, CUSTO_FP
    )
    melhor = min(resultados, key=lambda r: r["custo"])
    em_05 = [r for r in resultados if r["limiar"] == 0.5][0]

    assert em_05["custo"] <= 1.10 * melhor["custo"], (
        f"{versao}: custo {em_05['custo']:.0f} em 0,5 (recall {em_05['recall']:.2f}) contra "
        f"{melhor['custo']:.0f} em {melhor['limiar']} (recall {melhor['recall']:.2f})"
    )


def test_limiar_escolhido_no_teste_reduz_o_custo_da_v2_em_producao(teste_avaliado, producao_avaliada):
    """O limiar ótimo do teste também é melhor que 0,5 na produção (não é sobreajuste)."""
    resultados = estatistica.varrer_limiares(
        teste_avaliado.y_real, teste_avaliado.proba["v2"], LIMIARES, CUSTO_FN, CUSTO_FP
    )
    melhor_limiar = min(resultados, key=lambda r: r["custo"])["limiar"]

    y_real = producao_avaliada.y_real
    proba = producao_avaliada.proba["v2"]
    custo_melhor = estatistica.custo(y_real, proba >= melhor_limiar, CUSTO_FN, CUSTO_FP)
    custo_05 = estatistica.custo(y_real, proba >= 0.5, CUSTO_FN, CUSTO_FP)
    assert custo_melhor < custo_05


# --- calibração ---------------------------------------------------------------

@pytest.mark.parametrize("versao", [pytest.param("v1", marks=pytest.mark.defeito("D08")), "v2"])
def test_probabilidade_e_calibrada(teste_avaliado, versao):
    """Erro de calibração (ECE) de no máximo 0,10."""
    y_real = teste_avaliado.y_real
    proba = teste_avaliado.proba[versao]
    ece = estatistica.erro_calibracao(y_real, proba)

    tabela = estatistica.tabela_calibracao(y_real, proba)
    pior = max(tabela, key=lambda f: abs(f["prevista"] - f["observada"]))
    assert ece <= 0.10, (
        f"{versao}: ECE={ece:.3f}; prob. média {proba.mean():.3f} x prevalência {y_real.mean():.3f}; "
        f"pior faixa {pior['faixa']}: prevista {pior['prevista']:.2f}, observada {pior['observada']:.2f}"
    )


# --- teste x produção ---------------------------------------------------------

SENSORES = ["temperatura_c", "vibracao_rms", "corrente_a", "rpm"]


@pytest.mark.parametrize("sensor", SENSORES)
def test_media_dos_sensores_estavel_entre_teste_e_producao(carregar, sensor):
    """Passa para todos os sensores, e por isso comparar médias não basta."""
    teste = carregar("teste")[sensor]
    producao = carregar("producao")[sensor]
    assert abs(producao.mean() - teste.mean()) <= 0.5 * teste.std()


def variacao_hora_a_hora(df, coluna):
    ordenado = df.sort_values(["id_maquina", "timestamp"])
    return ordenado.groupby("id_maquina")[coluna].diff().dropna().to_numpy()


# Comparamos a variação de uma hora para a outra, e não o valor em si: o
# valor depende de quais motores estão falhando naquela semana; a variação
# isola o comportamento do sensor.
@pytest.mark.parametrize(
    "sensor",
    [
        pytest.param("temperatura_c", marks=pytest.mark.defeito("D10")),
        "vibracao_rms",
        "corrente_a",
        "rpm",
    ],
)
def test_distribuicao_das_variacoes_hora_a_hora_estavel(carregar, sensor):
    resultado = estatistica.teste_ks(
        variacao_hora_a_hora(carregar("teste"), sensor),
        variacao_hora_a_hora(carregar("producao"), sensor),
    )
    assert resultado["d"] <= 0.10, f"{sensor}: KS D={resultado['d']:.3f}, p={resultado['p_valor']:.1e}"


@pytest.mark.defeito("D10")
def test_mesma_media_nao_significa_mesma_distribuicao(carregar):
    """Temperatura: treino e produção têm a mesma média (67,23 °C), mas o desvio sobe 52%."""
    treino = carregar("treino")["temperatura_c"]
    producao = carregar("producao")["temperatura_c"]
    assert treino.mean() == pytest.approx(producao.mean(), abs=0.01)

    razao_desvio = producao.std() / treino.std()
    resultado = estatistica.teste_ks(treino, producao)
    assert razao_desvio <= 1.2, (
        f"desvio-padrão x{razao_desvio:.2f}; KS D={resultado['d']:.3f}, p={resultado['p_valor']:.1e}"
    )


def test_prevalencia_estavel_entre_teste_e_producao(teste_avaliado, producao_avaliada):
    """Teste z de duas proporções (15,6% x 14,3%)."""
    p1, n1 = teste_avaliado.y_real.mean(), len(teste_avaliado.y_real)
    p2, n2 = producao_avaliada.y_real.mean(), len(producao_avaliada.y_real)
    p_comum = (p1 * n1 + p2 * n2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(p_comum * (1 - p_comum) * (1 / n1 + 1 / n2))
    assert abs(z) < 1.96


@pytest.mark.parametrize("versao", ["v1", "v2"])
def test_pr_auc_nao_degrada_em_producao(teste_avaliado, producao_avaliada, versao):
    no_teste = estatistica.pr_auc(teste_avaliado.y_real, teste_avaliado.proba[versao])
    em_producao = estatistica.pr_auc(producao_avaliada.y_real, producao_avaliada.proba[versao])
    assert em_producao >= no_teste - 0.05
