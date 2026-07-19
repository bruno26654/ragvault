# ADR 0007 — Vector memory layout

## Decisão
Arena contígua row-major `Vec<f32>` com ids internos u32 densos; tombstones em `Vec<bool>`; metadados frios materializados apenas após seleção. Cosine normaliza no insert (busca vira dot). AoS/SoA blocked layouts e quantização intercalada ficam para o trabalho de quantização (Gate D), decididos por benchmark.

## Validação
Testes de arena; benchmark de QPS em benchmarks/RESULTS.md.
