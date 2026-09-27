"""Contrato de `sentinela.modelo`: entrada, saída e o atalho do painel."""
import numpy as np
import pytest

import sentinela as sn

ORDEM = sn.features.ORDEM_FEATURES
VERSOES = ("v1", "v2")


@pytest.fixture(scope="module")
def X():
    limpo = sn.preprocessamento.limpar(sn.dados.carregar("teste")).drop(columns=sn.dados.ALVO)
    return sn.features.construir(limpo)


def test_versao_desconhecida_e_recusada():
    with pytest.raises(ValueError):
        sn.modelo.carregar("v3")


@pytest.mark.parametrize("versao", VERSOES)
def test_artefato_declara_a_ordem_canonica(versao):
    assert sn.modelo.carregar(versao).ordem_features == ORDEM


@pytest.mark.parametrize("versao", VERSOES)
def test_probabilidade_valida_e_decisao_coerente_com_o_limiar(X, versao):
    modelo = sn.modelo.carregar(versao)
    p = modelo.prever_proba(X)
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()
    for limiar in (0.2, 0.5, 0.8):
        assert (modelo.prever(X, limiar=limiar) == (p >= limiar)).all()


@pytest.mark.parametrize("versao", VERSOES)
def test_ordem_das_colunas_do_dataframe_nao_importa(X, versao):
    """Com DataFrame o modelo seleciona colunas pelo nome — correto."""
    modelo = sn.modelo.carregar(versao)
    invertido = X[list(reversed(X.columns))]
    np.testing.assert_array_equal(modelo.prever_proba(invertido), modelo.prever_proba(X))


@pytest.mark.parametrize("versao", VERSOES)
def test_recusa_nan_infinito_e_forma_errada(X, versao):
    modelo = sn.modelo.carregar(versao)
    com_nan = X.head(3).copy()
    com_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        modelo.prever_proba(com_nan)
    with pytest.raises(ValueError):
        modelo.prever_proba(X.to_numpy()[:, :12])
    with pytest.raises(ValueError):
        modelo.prever_proba(X.drop(columns="rpm"))


@pytest.mark.parametrize("versao", VERSOES)
def test_prever_registro_na_ordem_canonica_bate_com_o_lote(X, versao):
    modelo = sn.modelo.carregar(versao)
    amostra = X.iloc[::14]
    lote = modelo.prever(amostra)
    unitario = [modelo.prever_registro(r) for r in amostra.to_dict("records")]
    assert list(lote) == unitario


@pytest.mark.parametrize("versao", VERSOES)
def test_prever_registro_recusa_numero_errado_de_features(X, versao):
    registro = X.iloc[0].to_dict()
    registro.pop("turno")
    with pytest.raises(ValueError):
        sn.modelo.carregar(versao).prever_registro(registro)


@pytest.mark.defeito("D06")
@pytest.mark.parametrize("versao", VERSOES)
def test_prever_registro_independe_da_ordem_das_chaves(X, versao):
    """O painel monta o dicionário; nada garante a ordem das chaves. O
    método usa `registro.values()` e ignora os nomes — um dicionário com as
    mesmas 13 features em ordem alfabética muda a decisão."""
    modelo = sn.modelo.carregar(versao)
    amostra = X.iloc[::14]
    canonico = [modelo.prever_registro(r) for r in amostra.to_dict("records")]
    alfabetico = [modelo.prever_registro(dict(sorted(r.items()))) for r in amostra.to_dict("records")]
    trocadas = sum(a != b for a, b in zip(canonico, alfabetico))
    assert trocadas == 0, f"{trocadas} de {len(amostra)} decisões ({trocadas / len(amostra):.1%}) mudam só pela ordem das chaves"


@pytest.mark.defeito("D06")
def test_prever_registro_recusa_chaves_que_nao_sao_features(X):
    """13 valores com nomes errados passam sem erro — nada é validado."""
    registro = {f"sensor_{i}": v for i, v in enumerate(X.iloc[0].to_numpy())}
    with pytest.raises((KeyError, ValueError)):
        sn.modelo.carregar("v1").prever_registro(registro)
