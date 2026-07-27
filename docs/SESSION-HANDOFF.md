# SESSION HANDOFF

Atualizado ao final de cada sessão de trabalho (disciplina de continuidade).

```
branch:        claude/ragvault-python-library-xeltxs (reiniciada em main pós-PR#5)
main:          22e880d (merge da PR #5 — release 1.0.0-rc1)
versão:        1.0.0-rc1 (Rust) / 1.0.0rc1 (Python, PEP 440)
PR:            #1..#5 mergeadas. Nenhuma PR aberta.
CI:            matriz completa verde em 7e96bcc (rust, py 3.9/3.11/3.12,
               integrations fixadas, wheels 4 plataformas com smoke).
tag:           v1.0.0-rc1 criada localmente em 22e880d, mas o proxy git desta
               sessão só autoriza push da branch designada (403 para tags) e o
               MCP do GitHub não tem create-tag/release. Criar a tag/release
               pela UI do GitHub (target: 22e880d) ou com git local.
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

## Próximas ações possíveis

1. **Tag/release v1.0.0-rc1** — autorizados pelo usuário ("autorizado daqui
   para frente"), mas inexequíveis deste ambiente: criar a tag `v1.0.0-rc1`
   (target `22e880d`) e a release (marcada pre-release) pela UI do GitHub ou
   com git local. Wheels das 4 plataformas ficam como artefatos do run de CI
   verde na main.
2. **PyPI** — autorizado, porém sem credenciais neste ambiente (sem token,
   sem ~/.pypirc). Publicar de uma máquina com token: baixar os artefatos do
   CI e `twine upload dist/*.whl` (ou `maturin publish`).
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
