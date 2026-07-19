# Performance

Números reais medidos neste repositório: **benchmarks/RESULTS.md** (gerado por `python benchmarks/bench_retrieval.py`; toda linha vem de execução real, com hardware e metodologia registrados no próprio arquivo).

## O que já está otimizado

- Kernels f32 desenrolados (4 acumuladores) auto-vetorizados pelo LLVM — portáteis, sem `unsafe`, com referência escalar e teste diferencial (dims 384/512/768/1024/1536/3072 e não alinhadas).
- Top-k por heap limitado com merge por shard (nunca ordena o conjunto todo); Flat paralelo com Rayon acima de 8192 vetores.
- Arena contígua row-major; cosine normalizado no insert (busca = dot).
- GIL liberado em busca/ingestão/flush/compact (testado).

## O que ainda não está (honesto)

- SQ8/IVF/PQ/OPQ, mmap, prefetch explícito, intrinsics por arquitetura, autotuning: planejados (Gate D em TASKS.md). Não há números para eles porque não existem.
- Snapshot JSON v1 torna reopen de vaults muito grandes mais lento que o necessário — medido em RESULTS.md, correção planejada.

## Comparações

A comparação com faiss-cpu em RESULTS.md usa mesmo dataset, mesmas queries, mesmo k, mesma máquina e 1 thread, e descreve as diferenças de fronteira (RagVault mede através do binding Python + checagem de tombstones/filtros; faiss mede apenas `index.search`). Nenhuma afirmação universal de superioridade é feita — apenas os cenários executados.
