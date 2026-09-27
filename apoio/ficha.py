"""Ficha técnica dos sensores (README do Sentinela)."""
import numpy as np
import pandas as pd

PSI_POR_BAR = 14.5038
RUIDO_TEMPERATURA_C = 1.0

FAIXAS = {
    "temperatura_c": (45.0, 95.0),
    "vibracao_rms": (1.2, 8.0),
    "pressao_bar": (3.0, 4.5),
    "corrente_a": (12.0, 30.0),
    "rpm": (1650.0, 1800.0),
    "idade_equipamento_meses": (6, 180),
}

OPERADORES = {"OP-01", "OP-02", "OP-03", "OP-04", "OP-05", "OP-06",
              "OP-07", "OP-08", "OP-09", "OP-10", "OP-11", "OP-12"}

# fração mínima de leituras dentro da faixa de operação
FRACAO_MINIMA_NA_FAIXA = 0.999


def pressao_em_bar(pressao, unidade):
    pressao = np.asarray(pressao, dtype=float)
    em_psi = (pd.Series(unidade).str.strip().str.lower() == "psi").to_numpy()
    return np.where(em_psi, pressao / PSI_POR_BAR, pressao)
