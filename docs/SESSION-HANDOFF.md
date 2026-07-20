# SESSION HANDOFF

Atualizado ao final de cada sessão de trabalho (disciplina de continuidade).

```
branch:        claude/ragvault-python-library-xeltxs
commit-base:   f69fc5e (merge da main pós-PR#1/#2)
último commit: 221ef7e (CI multiplataforma + integrações + changelog/release) — todos pushed
push:          em dia — working tree limpo
PR:            #1/#2 mergeadas; PR #3 aberta para main com os commits de hardening
               (5aec3b9..221ef7e). Merge em main pendente de autorização humana.
```

## Baseline auditado (neste hardware: CPU, sem GPU)

- `cargo fmt --check`: limpo
- `cargo clippy --workspace --all-targets -- -D warnings`: limpo
- `cargo test --workspace --exclude ragvault-python`: unit/lib + differential + proptest verdes
- `python -m pytest`: 87 passed, 6 skipped, 1 deselected
  (skips = roundtrips de LangChain/LlamaIndex/Haystack/DSPy sem as libs +
  teste real de cuVS; deselect = marcador `gpu`)
- Smoke: quickstart, CLI init/sync/query/doctor OK
- **Não validável neste ambiente:** CUDA/cuVS (runbook em docs/GPU.md);
  execução de wheels macOS/Windows/aarch64 (exige runners CI — a matriz existe);
  linhas semânticas do eval RAG (política de rede nega huggingface.co, CONNECT 403).

## Concluído (histórico de hardening, do mais antigo ao mais recente)

1. **P0 atomicidade (5aec3b9)** — reproduzido por teste antes do fix: escrita
   rejeitada deixava mutação parcial + WAL envenenado; replace rejeitado
   destruía a versão antiga. Corrigido com prepared-write (validação integral
   pré-WAL), apply publish-at-end e replay que trata batch commitado corrompido
   como corrupção explícita.
2. **Suíte diferencial (04dc12a)** — before-compact == after-compact ==
   after-reopen para flat/hnsw/sq8/ivf_flat/ivf_pq/mmap, workload misto, 2 tenants.
3. **Identidade de ingestão + presets honestos (5d996c7)** — sha256(raw_bytes)
   como identidade; fingerprints source/parsed/processing; `quality` exige
   embedding explícito; novo preset `offline-lite`.
4. **Context Builder v2 (dd1ce16)** — fusão de runs adjacentes e flag
   `result.truncated` explícita; testes adversariais.
5. **Avaliação texto real (ae8fe9f)** — dataset commitado (30 passagens / 24
   queries) + harness com Recall@k/MRR/nDCG/precisão/tokens/p50/p95 para
   bm25/lexical/hybrid/+MMR/+expansion (RESULTS-RAG.md). Linhas semânticas
   `blocked` por rede — comando exato documentado, sem números inventados.
6. **Filtros tipados (3b7fddf)** — posting lists keyword/bool + BTreeMap
   numérico para ranges, interseção AND com predicado residual, visível no
   plano; benchmark 12×/172×/484× (RESULTS-FILTERS.md).
7. **Batch nativo (cd95482)** — `search_many`/`retrieve_many` em uma chamada
   Rust com GIL liberado e paralelismo por query; equivalência com sequencial
   (`TestNativeBatch`).
8. **CI multiplataforma + integrações + release docs (221ef7e)** — matriz de
   wheels (Linux x86-64/aarch64, macOS arm64, Windows x86-64) com smoke de
   clean-install; job `integrations` com versões fixadas de
   LangChain/LlamaIndex/Haystack/DSPy rodando roundtrips reais (`TestIntegrations`);
   `CHANGELOG.md` e `docs/RELEASE.md`.

## Próximas tarefas (ordem da diretiva v1.0 CPU single-node)

Restam apenas itens de performance/formato de armazenamento — nenhum P0/P1 aberto.

1. **Profiling HNSW** — medir antes de otimizar. Primeiro alvo comprovado:
   alocação O(N) de `visited`/scratch por busca em `hnsw.rs`. Substituir por
   visited pool geracional + scratch por thread; refazer curvas
   recall / p50/p95/p99 / QPS / memória vs Faiss no mesmo recall.
   Só aplicar mudanças que o perfil justificar.
2. **Storage v2 binário segmentado** — segmentos imutáveis + mutable segment,
   manifest atômico versionado, checksums em streaming, query multi-segmento,
   compactação segura, compatibilidade/migração de formato, testes de
   crash/reopen/read-durante-compactação. (Hoje: snapshot JSON v1 +
   `vectors.bin` binário mmap-able.)
3. **Roaring bitmaps / histogramas / OPQ / binary quantization** — deferred;
   só substituir posting lists / quantizadores atuais quando profiling ou
   benchmark demonstrar necessidade.

## Regras de continuidade

- Commit + push após cada unidade coerente.
- **Sem force push.** Preservar trabalho existente.
- **Não mergear em main nem publicar release/PyPI sem autorização humana
  explícita.** Merge de PR #3 e qualquer publicação são decisões do usuário.
- CAGRA/cuVS permanece experimental enquanto não houver hardware GPU real.
- Committer `noreply@anthropic.com`; não colocar identificador de modelo em
  commits/PRs.
