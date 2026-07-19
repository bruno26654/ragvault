# ADR 0003 — Crate and layer architecture

## Contexto
O prompt de produto previa 6 crates (incl. ragvault-context e ragvault-ingestion em Rust).

## Decisão
Workspace com 5 crates: core (tipos/erros/filtros), vector (kernels/flat/hnsw), retrieval (bm25/sparse/fusão), engine (WAL/snapshots/planner/vault) e python (PyO3). **Ingestão e montagem de contexto vivem no pacote Python**: são dominados por strings, parsers e tokenizers plugáveis do ecossistema Python e não são hot path vetorial. Isso evita modularização artificial e FFI desnecessária.

## Consequências
- Parsers/chunkers extensíveis com plugins Python simples.
- Se profiling futuro mostrar contexto como gargalo, a lógica migra para um crate `ragvault-context` sem mudança de API pública.

## Validação
Sem dependências circulares; `cargo test --workspace` verde; camada Python coberta por pytest.
