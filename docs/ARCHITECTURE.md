# Arquitetura

```
python/ragvault           API pública: KnowledgeBase, ingestão, contexto, avaliação, CLI
crates/ragvault-python    Bindings PyO3 (abi3, GIL liberado nos hot paths)
crates/ragvault-engine    Vault: WAL, snapshots, lock, planner, busca híbrida
crates/ragvault-retrieval BM25 incremental, sparse index, fusão RRF
crates/ragvault-vector    Arena row-major, kernels, Flat, HNSW
crates/ragvault-core      Tipos, erros, modelo documental, DSL de filtros
```

## Fluxo de uma query

```
retrieve(query)
  → embed query (camada Python, cacheável)
  → engine.search (Rust, GIL liberado)
      → planner: flat | hnsw | flat_filtered_fallback (razões no plan)
      → dense + bm25 (+ sparse) com filtro integrado
      → fusão RRF ponderada
  → montagem de contexto (cap/dedup/MMR/expansão/budget)
  → citações estáveis
```

## Fluxo de uma escrita

```
upsert_document
  → WAL append (crc32, seq) — durável conforme política
  → aplicar sob write lock: arena.push + hnsw.insert + bm25.add (+sparse)
  → publicar doc/rows; tombstonar versão anterior
```

`flush()` publica uma generation de snapshot (arquivos com checksum, manifest por rename atômico) e trunca o WAL. Reopen = snapshot + replay de WAL com seq maior que o do manifest.

## Decisões

Ver `docs/adr/` — 15 ADRs cobrindo limites de produto, modelo de dados, WAL, garantias transacionais, layout de memória, estratégia de índices CPU, planner, fusão, contexto, API Python, compatibilidade de formato e metodologia de avaliação.
