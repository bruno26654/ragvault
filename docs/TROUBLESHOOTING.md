# Troubleshooting e limitações conhecidas

## Erros comuns

- **VaultLockedError**: outro processo (ou outra instância no mesmo processo) tem o vault aberto para escrita. Feche-o (`kb.close()` / context manager). Um writer por diretório é regra de v0.1.
- **ConfigurationError: embedding ... was requested**: o KB foi criado com outro embedding. Reabrir com espaço vetorial diferente corromperia a busca; crie outro KB ou aguarde `migrate_embeddings` (planejado).
- **DimensionMismatchError**: seu callable de embedding retornou dimensão diferente da armazenada.
- **IngestionError ... pip install "ragvault[pdf]"**: instale o extra indicado.
- **VaultCorruptError**: checksum/formato inválido em snapshot ou WAL. `ragvault doctor ./kb` diagnostica; a generation anterior é preservada até um novo flush ser publicado.

## Limitações conhecidas (v0.1)

1. Leitores em processo separado de um vault aberto por outro processo não são suportados.
2. Sparse vectors: não persistidos no WAL (só em snapshot); `compact()` não reconstrói postings sparse.
3. Filtros avaliam predicado por candidato (correto, mas sem índices bitmap/range dedicados — custo cresce com o pool filtrado).
4. Snapshot v1 em JSON: reopen de bases com centenas de milhares de chunks é mais lento do que o formato binário planejado.
5. `wal_sync="batch"` (default) pode perder os últimos commits em queda de energia (não em crash de processo). Use `wal_sync="sync"` para fsync por commit.
6. GPU, quantização, kb.tune/compare, integrações LangChain/LlamaIndex e Studio ainda não existem (TASKS.md).
7. Wheels PyPI ainda não publicadas; instalação via maturin.
