"""Gera a linha de base numérica usada no relatório.

    python scripts/linha_de_base.py            # imprime e grava evidencias/linha_de_base.md

Não é teste: é a medição. Os testes em tests/ transformam cada número
relevante daqui em um critério de aprovação.
"""
from __future__ import annotations

import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from apoio import sistema  # noqa: E402

sistema.localizar()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sentinela as sn  # noqa: E402
from apoio import estatistica as est  # noqa: E402

CUSTO_FN, CUSTO_FP = 10.0, 1.0
LIMIARES = np.round(np.arange(0.05, 0.96, 0.05), 2)


def tabela(linhas: list[dict], colunas: list[str]) -> str:
    cab = "| " + " | ".join(colunas) + " |\n|" + "---|" * len(colunas) + "\n"
    corpo = ""
    for linha in linhas:
        valores = []
        for c in colunas:
            v = linha[c]
            valores.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        corpo += "| " + " | ".join(valores) + " |\n"
    return cab + corpo


def avaliar(nome: str):
    bruto = sn.dados.carregar(nome)
    limpo = sn.preprocessamento.limpar(bruto)
    y = limpo.pop(sn.dados.ALVO).to_numpy()
    X = sn.features.construir(limpo)
    return bruto, limpo, y, X


def main() -> str:
    saida = ["# Linha de base do Sentinela\n"]

    # 1. Métricas: caminho reportado (lote rotulado) x caminho de produção
    linhas = []
    for nome in sn.dados.CONJUNTOS:
        bruto, _, y, X = avaliar(nome)
        for versao in ("v1", "v2"):
            reportado = sn.pipeline.executar(bruto, versao=versao)
            m_rep = sn.avaliacao.metricas(reportado["falha_72h"], reportado["predicao"])
            p = sn.modelo.carregar(versao).prever_proba(X)
            pred = (p >= 0.5).astype(int)
            m = sn.avaliacao.metricas(y, pred)
            c = est.contagens(y, pred)
            linhas.append(
                {
                    "conjunto": nome,
                    "versão": versao,
                    "acurácia (reportada)": m_rep["acuracia"],
                    "acurácia": m["acuracia"],
                    "precisão": m["precisao"],
                    "recall": m["recall"],
                    "F1": m["f1"],
                    "PR-AUC": est.precisao_media(y, p),
                    "ECE": est.erro_calibracao_esperado(y, p),
                    "FN": c["fn"],
                    "FP": c["fp"],
                    "custo (10·FN+FP)": int(est.custo(y, pred, CUSTO_FN, CUSTO_FP)),
                }
            )
    saida.append("## 1. Métricas no limiar 0,5\n")
    saida.append(
        "*acurácia (reportada)*: `pipeline.executar` sobre o lote **com** `falha_72h` "
        "(como a equipe mediu). Demais colunas: caminho de produção, features **sem** o rótulo.\n"
    )
    saida.append(tabela(linhas, list(linhas[0])))

    # 2. Varredura de limiar no teste
    _, _, y, X = avaliar("teste")
    saida.append("\n## 2. Varredura de limiar (teste, custo = 10·FN + FP)\n")
    for versao in ("v1", "v2"):
        p = sn.modelo.carregar(versao).prever_proba(X)
        v = est.varrer_limiar(y, p, LIMIARES, CUSTO_FN, CUSTO_FP)
        for linha in v:
            linha["limiar"] = f"{linha['limiar']:.2f}"
            linha["custo"] = int(linha["custo"])
        saida.append(f"\n**{versao}**\n\n" + tabela(v, ["limiar", "precisao", "recall", "fp", "fn", "custo"]))

    # 3. Calibração no teste
    saida.append("\n## 3. Calibração (teste)\n")
    for versao in ("v1", "v2"):
        p = sn.modelo.carregar(versao).prever_proba(X)
        saida.append(
            f"\n**{versao}** — ECE = {est.erro_calibracao_esperado(y, p):.3f}; "
            f"prob. média {p.mean():.3f} x prevalência {y.mean():.3f}\n\n"
            + tabela(est.tabela_calibracao(y, p), ["faixa", "n", "prevista", "observada"])
        )

    # 4. Distribuições teste x produção
    teste, producao = sn.dados.carregar("teste"), sn.dados.carregar("producao")
    treino = sn.dados.carregar("treino")

    def variacoes(df, c):
        o = df.sort_values(["id_maquina", "timestamp"])
        return o.groupby("id_maquina")[c].diff().dropna().to_numpy()

    linhas = []
    for c in ["temperatura_c", "vibracao_rms", "corrente_a", "rpm"]:
        a, b = teste[c].dropna(), producao[c].dropna()
        ks_n = est.ks_duas_amostras(a, b)
        ks_d = est.ks_duas_amostras(variacoes(teste, c), variacoes(producao, c))
        linhas.append(
            {
                "sensor": c,
                "média treino": float(treino[c].mean()),
                "média teste": float(a.mean()),
                "média produção": float(b.mean()),
                "dp treino": float(treino[c].std()),
                "dp teste": float(a.std()),
                "dp produção": float(b.std()),
                "KS nível D": ks_n["d"],
                "PSI nível": est.indice_estabilidade_populacional(a, b),
                "KS variação hora a hora D": ks_d["d"],
                "σ sensor teste": float(variacoes(teste, c).std() / np.sqrt(2)),
                "σ sensor produção": float(variacoes(producao, c).std() / np.sqrt(2)),
            }
        )
    saida.append("\n## 4. Teste x produção (sensores)\n")
    saida.append(tabela(linhas, list(linhas[0])))

    # 5. Vazamento: decisões que mudam com o rótulo presente
    saida.append("\n## 5. Decisões que mudam só por a coluna `falha_72h` estar no lote\n")
    linhas = []
    for nome in ("teste", "producao"):
        bruto = sn.dados.carregar(nome)
        for versao in ("v1", "v2"):
            com = sn.pipeline.executar(bruto, versao=versao)["predicao"]
            sem = sn.pipeline.executar(bruto.drop(columns="falha_72h"), versao=versao)["predicao"]
            linhas.append({"conjunto": nome, "versão": versao, "decisões alteradas": f"{(com != sem).mean():.1%}"})
    saida.append(tabela(linhas, ["conjunto", "versão", "decisões alteradas"]))

    # 6. Pontuação em tempo real x em lote (janela centrada)
    saida.append("\n## 6. Pontuação hora a hora (só passado) x lote da semana (teste)\n")
    limpo = avaliar("teste")[1]
    X_lote = sn.features.construir(limpo)
    linhas = []
    for versao in ("v1", "v2"):
        modelo = sn.modelo.carregar(versao)
        lote = modelo.prever(X_lote)
        mudou = 0
        for t in sorted(limpo["timestamp"].unique()):
            passado = limpo[limpo["timestamp"] <= t]
            agora = passado.index[passado["timestamp"] == t]
            mudou += int((modelo.prever(sn.features.construir(passado).loc[agora]) != lote[agora]).sum())
        linhas.append({"versão": versao, "decisões diferentes": f"{mudou} ({mudou / len(limpo):.1%})"})
    saida.append(tabela(linhas, ["versão", "decisões diferentes"]))

    return "\n".join(saida)


if __name__ == "__main__":
    texto = main()
    destino = RAIZ / "evidencias" / "linha_de_base.md"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    print(texto)
