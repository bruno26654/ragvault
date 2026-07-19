# ADR 0010 — Filters and query planner

## Decisão
Filtro JSON DSL parseado para AST tipada; semântica definida para campo ausente/null/tipos incompatíveis/NaN (documentada e testada). Operador desconhecido é erro (não igualdade literal silenciosa). Planner v1 baseado em regras explicáveis: Flat abaixo de `flat_threshold` ou quando HNSW filtrado ficar faminto; HNSW com travessia filtrada caso contrário; toda decisão registrada em `plan.reason`. Índices tipados (roaring bitmaps, ranges) e planner por custo estatístico ficam para v0.2 — o predicado por candidato é correto, apenas não indexado.

## Validação
Testes de filtro em todos os sinais; explain testado.
