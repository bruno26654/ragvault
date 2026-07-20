# Troubleshooting e limitações conhecidas

## Erros comuns

- **VaultLockedError**: outro processo (ou outra instância no mesmo processo) tem o vault aberto para escrita. Feche-o (`kb.close()` / context manager). Um writer por diretório é regra de v0.1.
- **ConfigurationError: embedding ... was requested**: o KB foi criado com outro embedding. Reabrir com espaço vetorial diferente corromperia a busca; crie outro KB ou aguarde `migrate_embeddings` (planejado).
- **DimensionMismatchError**: seu callable de embedding retornou dimensão diferente da armazenada.
- **IngestionError ... pip install "ragvault[pdf]"**: instale o extra indicado.
- **VaultCorruptError**: checksum/formato inválido em snapshot ou WAL. `ragvault doctor ./kb` diagnostica; a generation anterior é preservada até um novo flush ser publicado.

## Limitações conhecidas (v0.1)

1. Leitores em processo separado de um vault aberto por outro processo não são suportados.
2. Filtros avaliam predicado por candidato (correto, mas sem índices bitmap/range dedicados — custo cresce com o pool filtrado).
3. Snapshot v1 em JSON: reopen de bases com centenas de milhares de chunks é mais lento do que o formato binário planejado.
4. `wal_sync="batch"` (default) pode perder os últimos commits em queda de energia (não em crash de processo). Use `wal_sync="sync"` para fsync por commit.
5. GPU (cuVS/CAGRA) está implementado como sidecar **experimental** com testes de plumbing, mas não foi validado em hardware real — runbook em docs/GPU.md. Adaptadores LlamaIndex/Haystack/DSPy: implementados com erro acionável, não testados contra as libs reais (LangChain e Faiss interop são testados de verdade).
6. `storage="mmap"`: em Windows, a remoção da generation antiga durante `flush()` pode falhar enquanto o arquivo estiver mapeado (POSIX não tem esse problema); nesse caso a limpeza acontece no próximo flush.
7. IVF é uma estrutura de aceleração reconstruível (treinada em open/flush/compact); escritas entre rebuilds são varridas exatamente (delta scan) — nunca invisíveis, mas o custo da query cresce até o próximo flush.
6. Wheels PyPI ainda não publicadas; instalação via maturin.