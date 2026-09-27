"""Ficha técnica dos sensores (README do Sentinela) transcrita como dados.

É o contrato contra o qual a suíte valida `dados/` e a saída de
`preprocessamento.limpar`.
"""
from __future__ import annotations

PSI_POR_BAR = 14.5038

FAIXAS = {
    "temperatura_c": (45.0, 95.0),
    "vibracao_rms": (1.2, 8.0),
    "pressao_bar": (3.0, 4.5),
    "corrente_a": (12.0, 30.0),
    "rpm": (1650.0, 1800.0),
    "idade_equipamento_meses": (6, 180),
}

RUIDO_TEMPERATURA_C = 1.0

OPERADORES = {f"OP-{i:02d}" for i in range(1, 13)}
TURNOS = {1, 2, 3}
UNIDADES_PRESSAO = {"bar", "psi"}

# Uma leitura fora da faixa de operação não é necessariamente erro (um motor
# prestes a falhar sai da faixa). O contrato de qualidade de dados exige que
# pelo menos 99,9% das leituras estejam dentro dela — o `mostly` do Great
# Expectations. Erro de unidade derruba essa taxa para ~68%.
FRACAO_MINIMA_NA_FAIXA = 0.999


def pressao_em_bar(pressao, unidade):
    """Converte a coluna de pressão para bar conforme `unidade_pressao`."""
    import numpy as np

    unidade = np.asarray(unidade).astype(str)
    pressao = np.asarray(pressao, dtype=float)
    return np.where(np.char.lower(np.char.strip(unidade)) == "psi", pressao / PSI_POR_BAR, pressao)
