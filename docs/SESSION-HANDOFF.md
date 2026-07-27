# SESSION HANDOFF

Atualizado ao final de cada sessão de trabalho (disciplina de continuidade).

```
branch:        claude/ragvault-python-library-xeltxs (rebased em main pós-PR#3)
main:          7426e3c (merge da PR #3 — hardening P0/P1+P2 completo)
último commit: storage v2 completo (ADR 0016) — ver git log; todos pushed
PR:            #1/#2/#3 mergeadas; PR #4 (storage v2) aberta para main,
               CI verde nas 4 plataformas. Merge pendente de autorização humana.
```

## Baseline auditado (neste hardware: CPU, sem GPU)

- `cargo fmt --check` / `clippy --workspace -D warnings`: limpos
- `cargo test --workspace --exclude ragvault-python`: 46 engine + 7 differential
  (inclui multi-segmento) + 3 proptest_filter + 37 vector + demais — verdes
- `python -m pytest`: 89 passed, 6 skipped, 1 deselected
- CI (matriz completa): rust, python 3.9/3.11/3.12, integrations (LangChain/
  LlamaIndex/Haystack/DSPy fixados), wheels Linux x86-64/aarch64 + macOS arm64 +
  Windows x86-64 com smoke de clean-install — **tudo verde em 3d3a62b+**
- **Não validável neste ambiente:** CUDA/cuVS (runbook em docs/GPU.md);
  linhas semânticas do eval RAG (rede nega huggingface.co, CONNECT 403).

## Estado do produto (v1.0 CPU single-node)

**Nenhum P0/P1 aberto. Storage v2 (ADR 0016) completo:**

1. Container de segmento binário (`crate::segment`) — CRC por registro +
   CRC de stream, verificação incremental.
2. Base binária v2 (`gen-N/state.rvseg`, `format_version = 2`) + migração
   v1→v2 transparente.
3. Flush O(delta): segmentos de delta imutáveis; orçamento `MAX_DELTA_SEGMENTS`;
   `compact()` colapsa deltas; open aplica base→deltas→WAL.
4. Compactação read-friendly: rebuild fora do lock (leitores nunca bloqueiam
   durante o rebuild), swap seq-checked com retry e fallback sob lock.

Trabalho anterior já em main (PR #3): P0 atomicidade prepared-write, suíte
diferencial, identidade por raw-bytes, presets honestos, context v2, eval de
texto real, filtros tipados, batch nativo, HNSW `VisitedSet` geracional
(p99 10.4× em N=200k), correções Windows (WAL sem O_APPEND, sqlite fechado),
CI multiplataforma + integrações reais, CHANGELOG + docs/RELEASE.

## Próximas ações possíveis (nada obrigatório para v1.0)

1. **Merge da PR #4** — aguarda autorização humana explícita.
2. Release candidate: seguir `docs/RELEASE.md` §6 (bump de versão, tag) —
   **não publicar PyPI/release sem autorização**.
3. Eval semântico real — requer ambiente com acesso a huggingface.co
   (comando documentado em benchmarks/RESULTS-RAG.md; não inventar números).
4. GPU real (cuVS/CAGRA) — requer hardware; runbook em docs/GPU.md.
5. Otimizações futuras gated em benchmark: roaring bitmaps, OPQ/binary
   quant, compaction em background agendado, SIMD por arch.

## Regras de continuidade

- Commit + push após cada unidade coerente. **Sem force push.**
- **Não mergear em main nem publicar release/PyPI sem autorização humana
  explícita.**
- CAGRA/cuVS permanece experimental enquanto não houver hardware GPU real.
- Committer `noreply@anthropic.com`; não colocar identificador de modelo em
  commits/PRs.
