# ADR 0007 — Vector memory layout

## Decisão
Arena contígua row-major `Vec<f32>` com ids internos u32 densos; tombstones em `Vec<bool>`; metadados frios materializados apenas após seleção. Cosine normaliza no insert (busca vira dot). A arena suporta um prefixo mmap somente-leitura (`storage="mmap"`, base servida por page cache via bytemuck cast seguro, cauda de escritas em RAM) com paridade byte-a-byte testada. AoS/SoA blocked layouts e quantização intercalada permanecem decididos por benchmark futuro.

## Validação
Testes de arena; benchmark de QPS em benchmarks/RESULTS.md.
