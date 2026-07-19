# ADR 0004 — Storage layout

## Contexto
Precisamos de durabilidade real com implementação auditável em v0.1.

## Decisão
v0.1 usa uma arena mutável única + tombstones em bitmap + compactação síncrona que reescreve arena e índices. Snapshots com generations (`gen-N/`), manifest com checksums e publish por rename atômico. Segmentos imutáveis múltiplos + background compaction ficam para v0.2 (interface do engine já isola queries atrás de um snapshot lógico com RwLock).

## Alternativas
LSM completo com segmentos já em v1 — risco de bugs de merge sem cobertura de testes suficiente no prazo.

## Consequências
- Escritas grandes pausam leitores momentaneamente (write lock) — aceitável para embedded v1, documentado.
- Compaction é determinística em testes.

## Validação
`compact_drops_tombstones_and_preserves_results`, testes de reopen.
