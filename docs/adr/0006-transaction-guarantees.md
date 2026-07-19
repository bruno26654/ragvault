# ADR 0006 — Transaction guarantees (v0.1)

## Garantias
- Atomicidade por operação documental (upsert/replace/delete): WAL antes de aplicar; aplicação sob write lock.
- Queries com snapshot consistente (read lock): sem dirty reads, sem mistura de versões.
- Replay idempotente; durabilidade conforme política do WAL.
- Um writer por diretório (file lock exclusivo, fs2); leitores concorrentes no mesmo processo.

## Não prometido em v0.1
- Serializabilidade entre múltiplas operações, transações multi-documento, multi-writer entre processos, leitores cross-process de vault aberto.

## Validação
Testes de engine + testes de concorrência Python.
