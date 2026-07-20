# SESSION HANDOFF

Atualizado ao final de cada sessão de trabalho (disciplina de continuidade).

```
branch:        claude/ragvault-python-library-xeltxs
commit-base:   f69fc5e (merge da main pós-PR#1/#2; baseline auditado nesta sessão)
último commit: dd1ce16 (context v2) — todos pushed
push:          em dia — working tree limpo
PR:            #1/#2 mergeadas; PR dos 5 commits de hardening (5aec3b9..dd1ce16) aberta nesta sessão
```

## Baseline auditado (esta sessão, neste hardware: 4 vCPUs, sem GPU)

- `cargo fmt --check`: limpo
- `cargo clippy --workspace --all-targets -- -D warnings`: limpo
- `cargo test --workspace --exclude ragvault-python`: 95 unit/lib + 6 differential + 5 proptest (ver contagens por crate no CI)
- `python -m pytest`: 79 passed, 1 deselected (marcador `gpu` — sem hardware)
- Smoke: quickstart, CLI init/sync/query/doctor OK
- Não validável aqui: CUDA/cuVS (runbook em docs/GPU.md), wheels macOS/Windows/ARM64

## Concluído nesta sessão

1. **P0 atomicidade (fix(engine) 5aec3b9)** — reproduzido por teste antes do fix:
   escrita rejeitada deixava mutação parcial + WAL envenenado; replace rejeitado
   destruía a versão antiga. Corrigido com prepared-write (validação integral
   pré-WAL), apply publish-at-end e replay que trata falha de batch commitado
   como corrupção explícita.
2. **Suíte diferencial (04dc12a)** — before-compact == after-compact ==
   after-compact+reopen para flat/hnsw/sq8/ivf_flat/ivf_pq/mmap com workload
   misto (multi-chunk, sparse, replaces, deletes, 2 tenants).
3. **P1 identidade de ingestão + presets honestos (5d996c7)** —
   sha256(raw_bytes) como identidade; fingerprints separados
   (source/parsed/processing) com invalidação por mudança de pipeline;
   preset `quality` exige decisão explícita de embedding (erro acionável);
   novo preset `offline-lite`.

## Em andamento / próxima ação exata

Nenhuma alteração local pendente. Próximas tarefas na ordem da diretiva:

1. **P1 Context Builder v2** (docs/../python/ragvault/context.py):
   fusão de chunks sobrepostos/adjacentes por offsets (não cruzar documento/
   versão), flag de truncamento explícita no resultado, testes adversariais
   (sobreposição, budget mínimo, ACL). O budget pós-expansão já é respeitado.
2. **P1 avaliação com texto real**: dataset reproduzível (ex.: subset BEIR
   scifact/nfcorpus baixado explicitamente) + sentence-transformers instalado
   explicitamente; comparar bm25 / hashed-ngram / dense / hybrid / +MMR /
   +expansion em benchmarks/RESULTS-RAG.md.
3. **P2 filtros tipados** (crates/ragvault-engine): bitmaps por keyword/bool +
   índice ordenado para ranges; planner escolhe prefilter vs traversal por
   seletividade; benchmarks 100%..0.1%.
4. **P2 batch nativo**: `search_many` no binding (matriz de queries, GIL
   liberado, rayon), `retrieve_many` usa; documentar async como thread-offload.
5. **P2 profiling HNSW** (visited pool geracional, scratch por thread) — só
   otimizar com perfil medido.
6. **P2 integrações reais** (instalar llama-index-core/haystack-ai/dspy no CI
   opcional) e wheels multiplataforma (exige runners CI).

## Regras de continuidade

- Commit + push após cada unidade coerente (feito nesta sessão: 3 commits).
- Não usar force push; não mergear em main sem autorização (usuário já faz
  merge via PRs #1/#2 — abrir PR quando um conjunto estiver testado).
