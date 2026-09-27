# Correções (opcional)

O guia permite corrigir, desde que em patch ou branch separado e **depois** do teste que falha. Nesta entrega, os testes foram versionados primeiro (falhando) e as correções vieram depois, neste patch. O pacote dentro do submódulo `sentinela-nortemec/` continua intocado.

`0001-corrige-D01-D06-D12-D13-D15.patch` corrige só o que **não exige retreinar** os modelos publicados:

| Defeito | Arquivo | Correção |
|---|---|---|
| D01 | `src/sentinela/features.py` | `maquina_risco` usa sempre a tabela congelada do treino. No treino, ela é idêntica à média do lote, então as florestas continuam válidas. |
| D06 | `src/sentinela/modelo.py` | `prever_registro` lê os valores pelo nome da feature e recusa chaves faltando ou desconhecidas. |
| D12 | `src/sentinela/features.py` | janelas `rolling("6h")`/`rolling("24h")` e delta de 24 h medidos no tempo. Com a grade horária completa, o resultado é idêntico ao anterior. |
| D13 | `src/sentinela/features.py` | `construir` trabalha por posição e devolve o índice original. |
| D15 | `artefatos/manifest.json`, `.gitattributes` | sha256 regenerado sobre o conteúdo LF versionado; `eol=lf` para `dados/` e `artefatos/` em qualquer SO. |

Ficam fora do patch, porque mudam as features com que os modelos foram treinados e exigem retreino: D02 (janela centrada), D03 (operador/turno como feature), D04 (pressão em psi) e D05 (dropout = 0,0). D07 a D11 e D14 são decisões de modelagem e operação e aparecem como recomendações no relatório.

## Como aplicar e rodar a suíte contra o sistema corrigido

A partir da pasta que contém `projeto_final1/`:

```bash
git clone -c core.autocrlf=false https://github.com/felipehp/sentinela-nortemec.git sentinela-corrigido
cd sentinela-corrigido
git am ../projeto_final1/correcoes/0001-corrige-D01-D06-D12-D13-D15.patch
cd ../projeto_final1
SENTINELA_DIR=../sentinela-corrigido python -m pytest      # PowerShell: $env:SENTINELA_DIR="..\sentinela-corrigido"; python -m pytest
```

## Resultado

[`pytest_log_corrigido.txt`](pytest_log_corrigido.txt): **143 passando, 36 falhando** (antes: 133 e 46).

- D01, D06, D12, D13 e D15 aparecem como `nao reproduzido`: os 11 testes que falhavam passam.
- Os outros 10 defeitos seguem confirmados.
- `test_reproduz_o_ganho_de_acuracia_reportado_pela_equipe` passa a falhar, como esperado. Sem o vazamento, o ganho de acurácia da v2 é de 8,5 pp, não de 3,5 pp. O número do README do Sentinela só existia por causa de D01.
