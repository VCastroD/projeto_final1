"""Localiza o pacote `sentinela` sem copiá-lo nem editá-lo.

Ordem de busca:
1. `$SENTINELA_DIR` (se definida) — para rodar a suíte contra um patch;
2. o submódulo `sentinela-nortemec/` (versão publicada, commit fixado);
3. um `sentinela` já instalado com `pip install -e`.
"""
from __future__ import annotations

import os
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]


def localizar() -> pathlib.Path | None:
    for candidato in (os.environ.get("SENTINELA_DIR"), RAIZ / "sentinela-nortemec"):
        if candidato and (pathlib.Path(candidato) / "src" / "sentinela").is_dir():
            src = str(pathlib.Path(candidato).resolve() / "src")
            if src not in sys.path:
                sys.path.insert(0, src)
            return pathlib.Path(candidato).resolve()
    return None
