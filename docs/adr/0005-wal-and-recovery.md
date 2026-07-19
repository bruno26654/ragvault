# ADR 0005 — WAL and recovery

## Decisão
WAL append-only por vault: frame = [header_len][payload_len][seq][header JSON][payload f32 LE][crc32]. CRC cobre seq+header+payload. Replay para na primeira moldura inválida e trunca a cauda (operação nunca confirmada ao caller). Políticas: `sync` (fsync por commit) e `batch` (flush por commit, fsync no flush/close) — default batch, documentado o trade-off de perda em power loss.

## Alternativas
- bincode binário puro: menos debugável; JSON no header mantém inspecionabilidade com payload binário para vetores.

## Validação
Testes de torn tail, corrupção no meio, replay idempotente, truncamento pós-snapshot.
