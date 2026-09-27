"""Bloco A — testes unitários de `features.construir`.

A propriedade central de uma janela móvel: o valor no instante t só pode
depender de leituras até t. Os testes aqui verificam isso por perturbação
(muda o futuro, o passado não pode mudar) e por oráculo (reimplementação
independente, direta da definição).
"""
import numpy as np
import pandas as pd
import pytest

import sentinela as sn

construir = sn.features.construir
ORDEM = sn.features.ORDEM_FEATURES


@pytest.fixture(scope="module")
def limpo_rotulado():
    return sn.preprocessamento.limpar(sn.dados.carregar("teste"))


@pytest.fixture(scope="module")
def limpo(limpo_rotulado):
    """Quadro limpo SEM o rótulo — é o que chega na produção de verdade."""
    return limpo_rotulado.drop(columns=sn.dados.ALVO)


# --------------------------------------------------------------------------
# Forma e ordem
# --------------------------------------------------------------------------
def test_devolve_as_13_features_na_ordem_canonica(limpo):
    X = construir(limpo)
    assert tuple(X.columns) == ORDEM
    assert len(X) == len(limpo)
    assert X.index.equals(limpo.index)
    assert np.isfinite(X.to_numpy()).all()


@pytest.mark.parametrize("semente", [0, 1, 2])
def test_embaralhar_as_linhas_nao_muda_nenhuma_feature(limpo, semente):
    """Promessa do docstring: 'o quadro de entrada pode vir em qualquer ordem'."""
    X = construir(limpo)
    embaralhado = limpo.sample(frac=1.0, random_state=semente)
    X_emb = construir(embaralhado)
    assert list(X_emb.index) == list(embaralhado.index)  # volta na ordem de entrada
    pd.testing.assert_frame_equal(X_emb.loc[X.index], X)


def test_janela_de_um_motor_nao_enxerga_outro_motor(limpo):
    X = construir(limpo)
    alterado = limpo.copy()
    outro = alterado["id_maquina"] == "M02"
    alterado.loc[outro, ["temperatura_c", "vibracao_rms", "corrente_a"]] += [20.0, 3.0, 5.0]
    X_alt = construir(alterado)
    pd.testing.assert_frame_equal(X_alt.loc[~outro], X.loc[~outro])


def test_features_sao_deterministicas(limpo):
    pd.testing.assert_frame_equal(construir(limpo), construir(limpo))


# --------------------------------------------------------------------------
# Oráculo: cada janela contra a sua definição
# --------------------------------------------------------------------------
def _oraculo(limpo, feature):
    """Reimplementação direta: janela de N horas terminando em t, por motor."""
    ordenado = limpo.sort_values(["id_maquina", "timestamp"])
    grupo = ordenado.groupby("id_maquina")
    definicoes = {
        "temp_media_6h": lambda: grupo["temperatura_c"].transform(lambda s: s.rolling(6, min_periods=1).mean()),
        "temp_max_24h": lambda: grupo["temperatura_c"].transform(lambda s: s.rolling(24, min_periods=1).max()),
        "delta_temp_24h": lambda: grupo["temperatura_c"].transform(lambda s: s - s.shift(24)).fillna(0.0),
        "vib_media_6h": lambda: grupo["vibracao_rms"].transform(lambda s: s.rolling(6, min_periods=1).mean()),
        "vib_max_24h": lambda: grupo["vibracao_rms"].transform(lambda s: s.rolling(24, min_periods=1).max()),
        "corrente_media_6h": lambda: grupo["corrente_a"].transform(lambda s: s.rolling(6, min_periods=1).mean()),
    }
    return definicoes[feature]().loc[limpo.index]


@pytest.mark.parametrize(
    "feature",
    [
        pytest.param("temp_media_6h", marks=pytest.mark.defeito("D02")),
        "temp_max_24h",
        "delta_temp_24h",
        "vib_media_6h",
        "vib_max_24h",
        "corrente_media_6h",
    ],
)
def test_janela_movel_bate_com_a_definicao(limpo, feature):
    """'média 6h' no instante t = média das 6 leituras terminando em t."""
    obtido = construir(limpo)[feature]
    esperado = _oraculo(limpo, feature)
    diferenca = (obtido - esperado).abs()
    assert diferenca.max() < 1e-9, f"{feature}: {int((diferenca > 1e-9).sum())} linhas divergem, até {diferenca.max():.2f}"


# --------------------------------------------------------------------------
# Causalidade: o futuro não pode vazar para o passado
# --------------------------------------------------------------------------
def _perturbar_futuro(limpo, corte):
    alterado = limpo.copy()
    futuro = alterado["timestamp"] > corte
    alterado.loc[futuro, "temperatura_c"] += 15.0
    alterado.loc[futuro, "vibracao_rms"] += 2.0
    alterado.loc[futuro, "corrente_a"] += 4.0
    alterado.loc[futuro, "pressao"] += 0.3
    alterado.loc[futuro, "rpm"] -= 20.0
    alterado.loc[futuro, "id_operador"] = "OP-07"
    return alterado


@pytest.mark.parametrize(
    "feature",
    [pytest.param(f, marks=pytest.mark.defeito("D02")) if f == "temp_media_6h" else f for f in ORDEM],
)
def test_feature_no_instante_t_so_usa_leituras_ate_t(limpo, feature):
    """Muda TUDO depois de um corte; nenhuma feature até o corte pode mudar."""
    corte = pd.Timestamp("2026-06-03 12:00:00")
    passado = limpo["timestamp"] <= corte
    antes = construir(limpo).loc[passado, feature]
    depois = construir(_perturbar_futuro(limpo, corte)).loc[passado, feature]
    mudou = (antes - depois).abs() > 1e-9
    assert not mudou.any(), (
        f"{feature}: {int(mudou.sum())} linhas do passado mudaram quando só o futuro mudou "
        f"(ex.: {limpo.loc[mudou[mudou].index[0], 'timestamp']})"
    )


@pytest.mark.defeito("D02")
def test_recorte_e_lote_inteiro_concordam_nas_linhas_em_comum(limpo):
    """Rodar sobre os 3 primeiros dias ou sobre a semana inteira deve dar o
    mesmo valor para as linhas dos 3 primeiros dias. Em tempo real, a linha
    t é pontuada quando só existe o passado."""
    recorte = limpo[limpo["timestamp"] < pd.Timestamp("2026-06-04")]
    X_recorte = construir(recorte)
    X_inteiro = construir(limpo).loc[recorte.index]
    diferentes = ~np.isclose(X_recorte.to_numpy(), X_inteiro.to_numpy()).all(axis=1)
    assert not diferentes.any(), f"{int(diferentes.sum())} de {len(recorte)} linhas mudam conforme o fim do lote"


# --------------------------------------------------------------------------
# Vazamento do rótulo
# --------------------------------------------------------------------------
@pytest.mark.defeito("D01")
def test_features_nao_dependem_da_presenca_do_rotulo(limpo_rotulado, limpo):
    """A coluna-alvo não pode influenciar as features. `_risco_por_maquina`
    recalcula o risco com `falha_72h` do próprio lote quando ela existe."""
    com_rotulo = construir(limpo_rotulado)
    sem_rotulo = construir(limpo)
    divergentes = [f for f in ORDEM if not np.allclose(com_rotulo[f], sem_rotulo[f])]
    assert divergentes == [], f"features que mudam com o rótulo presente: {divergentes}"


@pytest.mark.defeito("D01")
def test_risco_do_motor_nao_e_o_proprio_rotulo_da_linha(limpo_rotulado):
    """Lote de uma linha só: se o risco é o rótulo, a feature É a resposta."""
    linha = limpo_rotulado.iloc[[0]].copy()
    riscos = {}
    for rotulo in (0, 1):
        linha[sn.dados.ALVO] = rotulo
        riscos[rotulo] = float(construir(linha)["maquina_risco"].iloc[0])
    assert riscos[0] == riscos[1], f"maquina_risco = {riscos} conforme o rótulo da própria linha"


# --------------------------------------------------------------------------
# Janela em horas x janela em linhas; índice
# --------------------------------------------------------------------------
def _motor_sintetico(horas=30):
    ts = pd.date_range("2026-06-01", periods=horas, freq="h")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "id_maquina": "M01",
            "idade_equipamento_meses": 120,
            "id_operador": "OP-03",
            "turno": ts.hour // 8 + 1,
            "temperatura_c": 60.0 + 0.5 * np.arange(horas),
            "vibracao_rms": 2.0 + 0.1 * np.arange(horas),
            "pressao": 3.9,
            "unidade_pressao": "bar",
            "corrente_a": 19.0,
            "rpm": 1755.0,
            "horas_operacao": 12000.0 + np.arange(horas),
        }
    )


@pytest.mark.defeito("D12")
def test_janela_de_6h_cobre_6_horas_mesmo_com_leitura_removida():
    """`limpar` remove leituras de eixo parado; a janela de '6 h' passa a
    cobrir 7 h porque conta linhas, não horas."""
    bruto = _motor_sintetico()
    bruto.loc[20, "rpm"] = 0.0  # eixo parou às 20h: leitura descartada
    limpo = sn.preprocessamento.limpar(bruto)
    X = construir(limpo)
    t = limpo.index[limpo["timestamp"] == pd.Timestamp("2026-06-02 00:00:00")][0]
    janela = limpo[(limpo["timestamp"] > limpo.loc[t, "timestamp"] - pd.Timedelta(hours=6))
                   & (limpo["timestamp"] <= limpo.loc[t, "timestamp"])]
    obtido, esperado = X.loc[t, "vib_media_6h"], janela["vibracao_rms"].mean()
    assert obtido == pytest.approx(esperado), (
        f"vib_media_6h = {obtido:.3f} (6 linhas, 7 h) x {esperado:.3f} ({len(janela)} leituras nas últimas 6 h)"
    )


def test_janela_de_6h_sem_lacuna_bate_com_a_media_das_6_horas():
    """Controle do teste acima: sem lacuna, linha = hora e a janela está certa."""
    limpo = sn.preprocessamento.limpar(_motor_sintetico())
    X = construir(limpo)
    assert X.loc[24, "vib_media_6h"] == pytest.approx(limpo.loc[19:24, "vibracao_rms"].mean())


@pytest.mark.defeito("D13")
def test_indice_repetido_nao_duplica_linhas(limpo):
    """Concatenar dois quadros sem `reset_index` é comum. `construir` faz
    `.loc[df.index]` e devolve o dobro de linhas."""
    a = limpo[limpo["id_maquina"] == "M01"].reset_index(drop=True)
    b = limpo[limpo["id_maquina"] == "M02"].reset_index(drop=True)
    juntos = pd.concat([a, b])
    assert len(construir(juntos)) == len(juntos)
