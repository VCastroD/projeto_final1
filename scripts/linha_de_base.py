"""Gera evidencias/linha_de_base.md com as métricas usadas no relatório.

    python scripts/linha_de_base.py
"""
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from apoio import sistema  # noqa: E402

sistema.localizar()

import numpy as np  # noqa: E402

import sentinela as sn  # noqa: E402
from apoio import estatistica  # noqa: E402

CUSTO_FN = 10.0
CUSTO_FP = 1.0
LIMIARES = np.round(np.arange(0.05, 0.96, 0.05), 2)
SENSORES = ["temperatura_c", "vibracao_rms", "corrente_a", "rpm"]


def tabela_markdown(linhas, colunas):
    texto = "| " + " | ".join(colunas) + " |\n"
    texto += "|" + "---|" * len(colunas) + "\n"
    for linha in linhas:
        celulas = []
        for coluna in colunas:
            valor = linha[coluna]
            if isinstance(valor, float):
                celulas.append(f"{valor:.3f}")
            else:
                celulas.append(str(valor))
        texto += "| " + " | ".join(celulas) + " |\n"
    return texto


def preparar(nome):
    """Devolve (bruto, leituras sem rótulo, y_real, X)."""
    bruto = sn.dados.carregar(nome)
    leituras = sn.preprocessamento.limpar(bruto)
    y_real = leituras.pop("falha_72h").to_numpy()
    X = sn.features.construir(leituras)
    return bruto, leituras, y_real, X


def variacao_hora_a_hora(df, coluna):
    ordenado = df.sort_values(["id_maquina", "timestamp"])
    return ordenado.groupby("id_maquina")[coluna].diff().dropna().to_numpy()


def secao_metricas():
    linhas = []
    for nome in ("treino", "teste", "producao"):
        bruto, _, y_real, X = preparar(nome)
        for versao in ("v1", "v2"):
            # como a equipe mediu: lote com a coluna falha_72h
            saida_com_rotulo = sn.pipeline.executar(bruto, versao=versao)
            acuracia_reportada = sn.avaliacao.metricas(saida_com_rotulo["falha_72h"], saida_com_rotulo["predicao"])["acuracia"]

            proba = sn.modelo.carregar(versao).prever_proba(X)
            y_pred = (proba >= 0.5).astype(int)
            m = sn.avaliacao.metricas(y_real, y_pred)
            erros = estatistica.matriz_confusao(y_real, y_pred)

            linhas.append({
                "conjunto": nome,
                "versão": versao,
                "acurácia (reportada)": acuracia_reportada,
                "acurácia": m["acuracia"],
                "precisão": m["precisao"],
                "recall": m["recall"],
                "F1": m["f1"],
                "PR-AUC": estatistica.pr_auc(y_real, proba),
                "ECE": estatistica.erro_calibracao(y_real, proba),
                "FN": erros["fn"],
                "FP": erros["fp"],
                "custo (10·FN+FP)": int(estatistica.custo(y_real, y_pred, CUSTO_FN, CUSTO_FP)),
            })

    texto = "## 1. Métricas no limiar 0,5\n\n"
    texto += ("*acurácia (reportada)*: `pipeline.executar` sobre o lote **com** `falha_72h` "
              "(como a equipe mediu). Demais colunas: features calculadas **sem** o rótulo.\n\n")
    return texto + tabela_markdown(linhas, list(linhas[0]))


def secao_limiar(y_real, X):
    texto = "## 2. Varredura de limiar (teste, custo = 10·FN + FP)\n"
    for versao in ("v1", "v2"):
        proba = sn.modelo.carregar(versao).prever_proba(X)
        resultados = estatistica.varrer_limiares(y_real, proba, LIMIARES, CUSTO_FN, CUSTO_FP)
        for r in resultados:
            r["limiar"] = f"{r['limiar']:.2f}"
            r["custo"] = int(r["custo"])
        texto += f"\n**{versao}**\n\n" + tabela_markdown(resultados, ["limiar", "precisao", "recall", "fp", "fn", "custo"])
    return texto


def secao_calibracao(y_real, X):
    texto = "## 3. Calibração (teste)\n"
    for versao in ("v1", "v2"):
        proba = sn.modelo.carregar(versao).prever_proba(X)
        ece = estatistica.erro_calibracao(y_real, proba)
        texto += f"\n**{versao}** — ECE = {ece:.3f}; prob. média {proba.mean():.3f} x prevalência {y_real.mean():.3f}\n\n"
        texto += tabela_markdown(estatistica.tabela_calibracao(y_real, proba), ["faixa", "n", "prevista", "observada"])
    return texto


def secao_drift():
    treino = sn.dados.carregar("treino")
    teste = sn.dados.carregar("teste")
    producao = sn.dados.carregar("producao")

    linhas = []
    for sensor in SENSORES:
        no_teste = teste[sensor].dropna()
        na_producao = producao[sensor].dropna()
        variacao_teste = variacao_hora_a_hora(teste, sensor)
        variacao_producao = variacao_hora_a_hora(producao, sensor)
        linhas.append({
            "sensor": sensor,
            "média treino": float(treino[sensor].mean()),
            "média teste": float(no_teste.mean()),
            "média produção": float(na_producao.mean()),
            "dp treino": float(treino[sensor].std()),
            "dp teste": float(no_teste.std()),
            "dp produção": float(na_producao.std()),
            "KS nível D": estatistica.teste_ks(no_teste, na_producao)["d"],
            "PSI nível": estatistica.psi(no_teste, na_producao),
            "KS variação hora a hora D": estatistica.teste_ks(variacao_teste, variacao_producao)["d"],
            "σ sensor teste": float(variacao_teste.std() / np.sqrt(2)),
            "σ sensor produção": float(variacao_producao.std() / np.sqrt(2)),
        })
    return "## 4. Teste x produção (sensores)\n\n" + tabela_markdown(linhas, list(linhas[0]))


def secao_vazamento():
    linhas = []
    for nome in ("teste", "producao"):
        bruto = sn.dados.carregar(nome)
        for versao in ("v1", "v2"):
            com_rotulo = sn.pipeline.executar(bruto, versao=versao)["predicao"]
            sem_rotulo = sn.pipeline.executar(bruto.drop(columns="falha_72h"), versao=versao)["predicao"]
            alteradas = (com_rotulo != sem_rotulo).mean()
            linhas.append({"conjunto": nome, "versão": versao, "decisões alteradas": f"{alteradas:.1%}"})
    texto = "## 5. Decisões que mudam só por a coluna `falha_72h` estar no lote\n\n"
    return texto + tabela_markdown(linhas, ["conjunto", "versão", "decisões alteradas"])


def secao_tempo_real(leituras):
    """Pontua cada hora usando só o passado e compara com a pontuação do lote inteiro."""
    X_semana = sn.features.construir(leituras)
    linhas = []
    for versao in ("v1", "v2"):
        modelo = sn.modelo.carregar(versao)
        decisao_semana = modelo.prever(X_semana)
        diferentes = 0
        for instante in sorted(leituras["timestamp"].unique()):
            ate_agora = leituras[leituras["timestamp"] <= instante]
            linhas_agora = ate_agora.index[ate_agora["timestamp"] == instante]
            decisao_agora = modelo.prever(sn.features.construir(ate_agora).loc[linhas_agora])
            diferentes += int((decisao_agora != decisao_semana[linhas_agora]).sum())
        linhas.append({"versão": versao, "decisões diferentes": f"{diferentes} ({diferentes / len(leituras):.1%})"})
    texto = "## 6. Pontuação hora a hora (só passado) x lote da semana (teste)\n\n"
    return texto + tabela_markdown(linhas, ["versão", "decisões diferentes"])


def main():
    _, leituras_teste, y_teste, X_teste = preparar("teste")
    secoes = [
        "# Linha de base do Sentinela\n",
        secao_metricas(),
        secao_limiar(y_teste, X_teste),
        secao_calibracao(y_teste, X_teste),
        secao_drift(),
        secao_vazamento(),
        secao_tempo_real(leituras_teste),
    ]
    return "\n\n".join(secoes)


if __name__ == "__main__":
    texto = main()
    destino = RAIZ / "evidencias" / "linha_de_base.md"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    print(texto)
