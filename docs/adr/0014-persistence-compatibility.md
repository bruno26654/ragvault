# ADR 0014 — Persistence compatibility

## Decisão
DTOs versionados (PersistedManifestV1/StateV1...) com `format_version` gating; abrir versão futura falha com erro claro. v1 usa state JSON + vectors.bin com checksums CRC32 por arquivo. Migração para formato binário de segmentos entrará como format_version=2 com abertura de v1 e migração explícita.

## Validação
Checksums verificados no load; reopen testado; golden files ficam para o primeiro release público (formato ainda pode mudar pré-0.1.0 tag).
