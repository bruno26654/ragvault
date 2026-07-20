# Storage e durabilidade

## Layout em disco

```
<kb>/
├── ragvault.json          configuração (preset, embedding, chunking...)
├── embedding-cache.db     cache sqlite de embeddings
└── vault/
    ├── LOCK               lock exclusivo de writer (fs2)
    ├── wal.log            write-ahead log
    ├── manifest.json      aponta a generation atual (publish atômico)
    └── gen-N/
        ├── state.json     docs, chunks, bm25, grafo hnsw (DTO v1, checksums)
        └── vectors.bin    f32 LE row-major
```

## Garantias (v0.1)

- Toda operação documental é registrada no WAL antes de aplicar; crash entre WAL e apply é recuperado no reopen (replay idempotente).
- Cauda de WAL rasgada/corrompida é truncada com segurança — a operação nunca foi confirmada.
- `flush()` publica snapshot por rename atômico; crash em qualquer ponto deixa o vault legível (generation anterior preservada até o novo manifest ser durável).
- Política de sync: `batch` (default; sobrevive a crash de processo, pode perder a cauda em power loss) ou `sync` (fsync por commit) via `wal_sync`.
- Um writer por diretório (lock); leitores concorrentes no mesmo processo; leitores cross-process de vault aberto **não são suportados** em v0.1.

## Limitações conhecidas e documentadas

- Snapshot v1 serializa estado como JSON — reabertura de vaults muito grandes é mais lenta que um formato binário de segmentos (planejado como format_version 2, com migração explícita).
- Compactação é síncrona (determinística em testes); background compaction planejada.
