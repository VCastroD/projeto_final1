"""sentinela.modelo: entrada, saída e prever_registro (usado pelo painel)."""
import numpy as np
import pytest

import sentinela as sn

VERSOES = ("v1", "v2")


@pytest.fixture(scope="module")
def X():
    limpo = sn.preprocessamento.limpar(sn.dados.carregar("teste")).drop(columns="falha_72h")
    return sn.features.construir(limpo)


@pytest.fixture(scope="module")
def amostra(X):
    return X.iloc[::14]  # 300 linhas, para prever_registro não demorar


def test_versao_desconhecida_e_recusada():
    with pytest.raises(ValueError):
        sn.modelo.carregar("v3")


@pytest.mark.parametrize("versao", VERSOES)
def test_artefato_declara_a_ordem_canonica(versao):
    assert sn.modelo.carregar(versao).ordem_features == sn.features.ORDEM_FEATURES


@pytest.mark.parametrize("versao", VERSOES)
def test_probabilidade_valida_e_decisao_coerente_com_o_limiar(X, versao):
    modelo = sn.modelo.carregar(versao)
    proba = modelo.prever_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    for limiar in (0.2, 0.5, 0.8):
        assert (modelo.prever(X, limiar=limiar) == (proba >= limiar)).all()


@pytest.mark.parametrize("versao", VERSOES)
def test_ordem_das_colunas_do_dataframe_nao_importa(X, versao):
    # com DataFrame o modelo seleciona as colunas pelo nome
    modelo = sn.modelo.carregar(versao)
    colunas_invertidas = X[list(reversed(X.columns))]
    np.testing.assert_array_equal(modelo.prever_proba(colunas_invertidas), modelo.prever_proba(X))


@pytest.mark.parametrize("versao", VERSOES)
def test_recusa_nan_infinito_e_forma_errada(X, versao):
    modelo = sn.modelo.carregar(versao)

    com_nan = X.head(3).copy()
    com_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        modelo.prever_proba(com_nan)

    with pytest.raises(ValueError):
        modelo.prever_proba(X.to_numpy()[:, :12])  # uma coluna a menos

    with pytest.raises(ValueError):
        modelo.prever_proba(X.drop(columns="rpm"))


@pytest.mark.parametrize("versao", VERSOES)
def test_prever_registro_na_ordem_canonica_bate_com_o_lote(amostra, versao):
    modelo = sn.modelo.carregar(versao)
    em_lote = list(modelo.prever(amostra))
    um_por_um = [modelo.prever_registro(registro) for registro in amostra.to_dict("records")]
    assert um_por_um == em_lote


@pytest.mark.parametrize("versao", VERSOES)
def test_prever_registro_recusa_numero_errado_de_features(X, versao):
    registro = X.iloc[0].to_dict()
    del registro["turno"]
    with pytest.raises(ValueError):
        sn.modelo.carregar(versao).prever_registro(registro)


@pytest.mark.defeito("D06")
@pytest.mark.parametrize("versao", VERSOES)
def test_prever_registro_independe_da_ordem_das_chaves(amostra, versao):
    """prever_registro usa registro.values() e ignora os nomes das chaves."""
    modelo = sn.modelo.carregar(versao)
    trocadas = 0
    for registro in amostra.to_dict("records"):
        em_ordem_alfabetica = dict(sorted(registro.items()))
        if modelo.prever_registro(registro) != modelo.prever_registro(em_ordem_alfabetica):
            trocadas += 1
    assert trocadas == 0, (
        f"{trocadas} de {len(amostra)} decisões ({trocadas / len(amostra):.1%}) mudam só pela ordem das chaves"
    )


@pytest.mark.defeito("D06")
def test_prever_registro_recusa_chaves_que_nao_sao_features(X):
    valores = X.iloc[0].to_numpy()
    registro = {}
    for i, valor in enumerate(valores):
        registro[f"sensor_{i}"] = valor
    with pytest.raises((KeyError, ValueError)):
        sn.modelo.carregar("v1").prever_registro(registro)
