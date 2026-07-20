# ADR 0008 — CPU index strategy

## Decisão
Ordem: Flat (baseline/ground truth/fallback) → HNSW (M=16, ef_construction=200, ef_search=64, heurística de diversidade, inserção incremental) → filtros integrados à travessia (nós reprovados servem de ponte, nunca aparecem) com retry de ef ampliado e fallback exato para filtros restritivos. SQ8 está implementado como backend `sq8_flat` (quantização int8 por vetor com escala própria, varredura 4x menor, oversampling 4x e rescore f32; substitui o grafo quando `quantization="sq8"` — ingestão mais rápida, recall quase exato, custo O(n) por query; cosine/dot apenas, L2 rejeitado). Compressão/recall/latência medidos em benchmarks/RESULTS.md antes de declarar pronto. IVF-Flat e IVF-PQ estão implementados como estruturas de aceleração reconstruíveis (k-means determinístico em amostra ≤20k, nprobe configurável, delta scan para escritas pós-treino; PQ com ADC 8-bit + oversample 8x + rescore f32) — números em benchmarks/RESULTS.md. OPQ/binary permanecem em Gate D com o mesmo gate de benchmark.

## Kernels
Loops desenrolados 4-acc auto-vetorizados pelo LLVM; sem `unsafe`, sem `target-cpu=native` em wheels; referência escalar + teste diferencial por kernel; dims prioritárias e não alinhadas testadas.

## Validação
Recall ≥ 0.9 vs Flat em teste unitário; recall/QPS reais em benchmarks/RESULTS.md.
