# ADR 0008 — CPU index strategy

## Decisão
Ordem: Flat (baseline/ground truth/fallback) → HNSW (M=16, ef_construction=200, ef_search=64, heurística de diversidade, inserção incremental) → filtros integrados à travessia (nós reprovados servem de ponte, nunca aparecem) com retry de ef ampliado e fallback exato para filtros restritivos. IVF/SQ8/PQ/OPQ ficam para Gate D com gate de benchmark (compressão/recall/latência medidos antes de declarar pronto).

## Kernels
Loops desenrolados 4-acc auto-vetorizados pelo LLVM; sem `unsafe`, sem `target-cpu=native` em wheels; referência escalar + teste diferencial por kernel; dims prioritárias e não alinhadas testadas.

## Validação
Recall ≥ 0.9 vs Flat em teste unitário; recall/QPS reais em benchmarks/RESULTS.md.
