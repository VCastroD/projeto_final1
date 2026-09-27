"""Bloco A: features.construir.

A regra principal de uma janela móvel: o valor no instante t só pode usar
leituras até t. Testamos isso de dois jeitos: comparando com uma
implementação de referência e alterando o futuro para ver se o passado muda.
"""
import numpy as np
import pandas as pd
import pytest

import sentinela as sn

construir = sn.features.construir
FEATURES = sn.features.ORDEM_FEATURES


@pytest.fixture(scope="module")
def limpo_com_rotulo():
    return sn.preprocessamento.limpar(sn.dados.carregar("teste"))


@pytest.fixture(scope="module")
def limpo(limpo_com_rotulo):
    # na produção a coluna falha_72h não existe
    return limpo_com_rotulo.drop(columns="falha_72h")


def test_devolve_as_13_features_na_ordem_canonica(limpo):
    X = construir(limpo)
    assert tuple(X.columns) == FEATURES
    assert len(X) == len(limpo)
    assert X.index.equals(limpo.index)
    assert np.isfinite(X.to_numpy()).all()


@pytest.mark.parametrize("semente", [0, 1, 2])
def test_embaralhar_as_linhas_nao_muda_nenhuma_feature(limpo, semente):
    X = construir(limpo)
    embaralhado = limpo.sample(frac=1.0, random_state=semente)
    X_embaralhado = construir(embaralhado)

    assert list(X_embaralhado.index) == list(embaralhado.index)  # volta na ordem de entrada
    pd.testing.assert_frame_equal(X_embaralhado.loc[X.index], X)


def test_janela_de_um_motor_nao_enxerga_outro_motor(limpo):
    X = construir(limpo)

    alterado = limpo.copy()
    m02 = alterado["id_maquina"] == "M02"
    alterado.loc[m02, "temperatura_c"] += 20.0
    alterado.loc[m02, "vibracao_rms"] += 3.0
    alterado.loc[m02, "corrente_a"] += 5.0
    X_alterado = construir(alterado)

    pd.testing.assert_frame_equal(X_alterado[~m02], X[~m02])


def test_features_sao_deterministicas(limpo):
    pd.testing.assert_frame_equal(construir(limpo), construir(limpo))


# --- comparação com implementação de referência ------------------------------

def janela_de_referencia(limpo, feature):
    """Calcula a feature pela definição: janela terminando no instante t, por motor."""
    ordenado = limpo.sort_values(["id_maquina", "timestamp"])
    por_motor = ordenado.groupby("id_maquina")

    if feature == "temp_media_6h":
        valores = por_motor["temperatura_c"].rolling(6, min_periods=1).mean()
    elif feature == "temp_max_24h":
        valores = por_motor["temperatura_c"].rolling(24, min_periods=1).max()
    elif feature == "vib_media_6h":
        valores = por_motor["vibracao_rms"].rolling(6, min_periods=1).mean()
    elif feature == "vib_max_24h":
        valores = por_motor["vibracao_rms"].rolling(24, min_periods=1).max()
    elif feature == "corrente_media_6h":
        valores = por_motor["corrente_a"].rolling(6, min_periods=1).mean()
    elif feature == "delta_temp_24h":
        temperatura_24h_antes = por_motor["temperatura_c"].shift(24)
        return (ordenado["temperatura_c"] - temperatura_24h_antes).fillna(0.0).loc[limpo.index]
    else:
        raise ValueError(feature)

    # groupby().rolling() devolve índice (motor, linha); tiramos o motor
    return valores.reset_index(level=0, drop=True).loc[limpo.index]


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
    obtido = construir(limpo)[feature]
    esperado = janela_de_referencia(limpo, feature)
    diferenca = (obtido - esperado).abs()
    linhas_diferentes = int((diferenca > 1e-9).sum())
    assert linhas_diferentes == 0, f"{feature}: {linhas_diferentes} linhas divergem, até {diferenca.max():.2f}"


# --- causalidade: mudar o futuro não pode mudar o passado --------------------

def alterar_depois_de(limpo, corte):
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
    [
        pytest.param("temp_media_6h", marks=pytest.mark.defeito("D02")),
        "temp_max_24h",
        "delta_temp_24h",
        "vib_media_6h",
        "vib_max_24h",
        "corrente_media_6h",
        "pressao",
        "rpm",
        "idade_equipamento_meses",
        "horas_operacao",
        "maquina_risco",
        "operador_senior",
        "turno",
    ],
)
def test_feature_no_instante_t_so_usa_leituras_ate_t(limpo, feature):
    corte = pd.Timestamp("2026-06-03 12:00:00")
    passado = limpo["timestamp"] <= corte

    antes = construir(limpo).loc[passado, feature]
    depois = construir(alterar_depois_de(limpo, corte)).loc[passado, feature]

    mudou = (antes - depois).abs() > 1e-9
    assert not mudou.any(), (
        f"{feature}: {int(mudou.sum())} linhas do passado mudaram quando só o futuro mudou "
        f"(ex.: {limpo.loc[mudou[mudou].index[0], 'timestamp']})"
    )


@pytest.mark.defeito("D02")
def test_recorte_e_lote_inteiro_concordam_nas_linhas_em_comum(limpo):
    """Rodar só os 3 primeiros dias ou a semana toda deve dar o mesmo valor nesses 3 dias."""
    recorte = limpo[limpo["timestamp"] < pd.Timestamp("2026-06-04")]
    X_recorte = construir(recorte)
    X_semana = construir(limpo).loc[recorte.index]

    linha_igual = np.isclose(X_recorte.to_numpy(), X_semana.to_numpy()).all(axis=1)
    diferentes = int((~linha_igual).sum())
    assert diferentes == 0, f"{diferentes} de {len(recorte)} linhas mudam conforme o fim do lote"


# --- vazamento do rótulo -----------------------------------------------------

@pytest.mark.defeito("D01")
def test_features_nao_dependem_da_presenca_do_rotulo(limpo_com_rotulo, limpo):
    """_risco_por_maquina recalcula o risco com falha_72h quando a coluna existe."""
    com_rotulo = construir(limpo_com_rotulo)
    sem_rotulo = construir(limpo)
    mudaram = [f for f in FEATURES if not np.allclose(com_rotulo[f], sem_rotulo[f])]
    assert mudaram == [], f"features que mudam com o rótulo presente: {mudaram}"


@pytest.mark.defeito("D01")
def test_risco_do_motor_nao_e_o_proprio_rotulo_da_linha(limpo_com_rotulo):
    linha = limpo_com_rotulo.iloc[[0]].copy()

    linha["falha_72h"] = 0
    risco_se_nao_falhou = float(construir(linha)["maquina_risco"].iloc[0])
    linha["falha_72h"] = 1
    risco_se_falhou = float(construir(linha)["maquina_risco"].iloc[0])

    assert risco_se_nao_falhou == risco_se_falhou, (
        f"maquina_risco = {risco_se_nao_falhou} com rótulo 0 e {risco_se_falhou} com rótulo 1"
    )


# --- janela em horas e índice ------------------------------------------------

def motor_sintetico(horas=30):
    """Um motor com leituras de hora em hora e vibração subindo 0,1 por hora."""
    horarios = pd.date_range("2026-06-01", periods=horas, freq="h")
    return pd.DataFrame({
        "timestamp": horarios,
        "id_maquina": "M01",
        "idade_equipamento_meses": 120,
        "id_operador": "OP-03",
        "turno": horarios.hour // 8 + 1,
        "temperatura_c": 60.0 + 0.5 * np.arange(horas),
        "vibracao_rms": 2.0 + 0.1 * np.arange(horas),
        "pressao": 3.9,
        "unidade_pressao": "bar",
        "corrente_a": 19.0,
        "rpm": 1755.0,
        "horas_operacao": 12000.0 + np.arange(horas),
    })


@pytest.mark.defeito("D12")
def test_janela_de_6h_cobre_6_horas_mesmo_com_leitura_removida():
    """Se limpar() descarta uma leitura (eixo parado), a janela de "6 h" passa a cobrir 7 h."""
    df = motor_sintetico()
    df.loc[20, "rpm"] = 0.0  # a leitura das 20h é descartada
    limpo = sn.preprocessamento.limpar(df)
    X = construir(limpo)

    instante = pd.Timestamp("2026-06-02 00:00:00")
    linha = limpo.index[limpo["timestamp"] == instante][0]
    ultimas_6h = limpo[(limpo["timestamp"] > instante - pd.Timedelta(hours=6)) & (limpo["timestamp"] <= instante)]

    obtido = X.loc[linha, "vib_media_6h"]
    esperado = ultimas_6h["vibracao_rms"].mean()
    assert obtido == pytest.approx(esperado), (
        f"vib_media_6h = {obtido:.3f} (6 linhas, 7 h) x {esperado:.3f} ({len(ultimas_6h)} leituras nas últimas 6 h)"
    )


def test_janela_de_6h_sem_lacuna_bate_com_a_media_das_6_horas():
    # controle do teste anterior: sem lacuna, 6 linhas = 6 horas
    limpo = sn.preprocessamento.limpar(motor_sintetico())
    X = construir(limpo)
    assert X.loc[24, "vib_media_6h"] == pytest.approx(limpo.loc[19:24, "vibracao_rms"].mean())


@pytest.mark.defeito("D13")
def test_indice_repetido_nao_duplica_linhas(limpo):
    """Concatenar dois quadros sem reset_index é comum; construir() devolve o dobro de linhas."""
    m01 = limpo[limpo["id_maquina"] == "M01"].reset_index(drop=True)
    m02 = limpo[limpo["id_maquina"] == "M02"].reset_index(drop=True)
    juntos = pd.concat([m01, m02])
    assert len(construir(juntos)) == len(juntos)
