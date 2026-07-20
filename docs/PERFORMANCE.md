# Performance

Números reais medidos neste repositório: **benchmarks/RESULTS.md** (gerado por `python benchmarks/bench_retrieval.py`; toda linha vem de execução real, com hardware e metodologia registrados no próprio arquivo).

## O que já está otimizado

- Kernels f32 desenrolados (4 acumuladores) auto-vetorizados pelo LLVM — portáteis, sem `unsafe`, com referência escalar e teste diferencial (dims 384/512/768/1024/1536/3072 e não alinhadas).
- Top-k por heap limitado com merge por shard (nunca ordena o conjunto todo); Flat paralelo com Rayon acima de 8192 vetores.
- Arena contígua row-major; cosine normalizado no insert (busca = dot).
- GIL liberado em busca/ingestão/flush/compact (testado).

## Quantização SQ8 (`quantization="sq8"` / preset `low_memory`)

Backend `sq8_flat`: cada vetor é quantizado para int8 com escala própria; a busca varre os códigos int8 (4x menos banda de memória), sobreamostra 4x e refina os sobreviventes contra os vetores f32 originais — recall quase exato sem construir grafo (ingestão muito mais rápida, escolha certa para coleções médias com muita escrita ou filtro). Números reais em benchmarks/RESULTS.md. Limitações: métricas cosine/dot (L2 é rejeitada com erro claro); os f32 são mantidos para rescoring, então a economia é no custo de varredura e no grafo, não no total residente.

## IVF-Flat / IVF-PQ (`index="ivf_flat"|"ivf_pq"`)

k-means determinístico por seed em amostra ≤20k; recall cresce monotonicamente
com `nprobe` (testado); sondagem completa == exato (testado); PQ com rescore
f32 mantém recall alto com códigos ~m bytes/vetor. Números reais na seção IVF
de benchmarks/RESULTS.md.

## mmap (`storage="mmap"`)

Vetores do snapshot servidos por page cache; paridade byte-a-byte com o modo
memory testada. Ganho principal: abrir bases grandes sem materializar os f32
na heap do processo.

## O que ainda não está (honesto)

- OPQ/binary quantization, prefetch explícito, intrinsics por arquitetura,
  índices bitmap de metadados: planejados (TASKS.md). Não há números para
  eles porque não existem.
- Snapshot JSON v1 torna reopen de vaults muito grandes mais lento que o necessário — medido em RESULTS.md, correção planejada.

## Comparações

A comparação com faiss-cpu em RESULTS.md usa mesmo dataset, mesmas queries, mesmo k, mesma máquina e 1 thread, e descreve as diferenças de fronteira (RagVault mede através do binding Python + checagem de tombstones/filtros; faiss mede apenas `index.search`). Nenhuma afirmação universal de superioridade é feita — apenas os cenários executados.
