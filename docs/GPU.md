# GPU (cuVS/CAGRA) — experimental, não validado em hardware real

## Status honesto

O suporte GPU está **implementado como sidecar e coberto por testes de
plumbing com um cuVS falso** (`tests/python/test_advanced_features.py::TestGpuPlumbing`),
mas **este ambiente de desenvolvimento não tem GPU**: as chamadas reais ao
cuVS seguem a API Python documentada do cuVS e precisam ser validadas com o
runbook abaixo antes de uso em produção. Nenhum benchmark GPU foi executado
e nenhum número é publicado — números só entram em RESULTS.md por execução real.

## Arquitetura

- O pacote CPU nunca importa CUDA. GPU é opt-in: `pip install cuvs-cu12`
  (extra `ragvault[gpu-cu12]`).
- `ragvault.gpu.CagraDenseSearcher(kb)` é um **sidecar**: exporta os vetores
  vivos do KB (`kb.export_dense()`), constrói um grafo CAGRA na GPU e serve a
  geração de candidatos densos. BM25, fusão, contexto e citações permanecem
  na CPU, inalterados.
- `kb.retrieve(query, dense_searcher=searcher)` usa o sidecar; em caso de
  falha do sidecar a query cai automaticamente para o caminho CPU e o motivo
  fica registrado em `result.plan.reason`.
- **Filtros neste caminho são pós-filtro com oversampling 4×** (avaliados
  pela mesma DSL Rust via `filter_chunks`) — diferente dos backends CPU, que
  têm filtro integrado à travessia. Isso é declarado no plan de toda query.
- Documentos inseridos após a construção exigem `searcher.rebuild()`.

## Runbook de validação (máquina com CUDA 12+)

```bash
# 1. instalar
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release
pip install cuvs-cu12          # RAPIDS/NVIDIA index; ver docs do cuVS

# 2. teste real (marcado, roda apenas com -m gpu)
python -m pytest -m gpu -v
# esperado: TestGpuPlumbing::test_real_cuvs_end_to_end PASSED

# 3. smoke manual
python examples/gpu_rag.py

# 4. benchmark (gera números reais; só então eles podem ser citados)
python - <<'PY'
import time, numpy as np, ragvault, tempfile
from ragvault.gpu import CagraDenseSearcher
kb = ragvault.open(tempfile.mkdtemp() + "/kb")
kb.add([{"id": f"d{i}", "text": f"doc {i} " + "filler " * 20} for i in range(50_000)])
s = CagraDenseSearcher(kb); print("build s:", s.build_seconds)
q = kb.embedder.embed_queries(["doc 42"])[0]
t0 = time.monotonic()
for _ in range(100): s.search(q, 10)
print("p_mean ms:", (time.monotonic()-t0)*10)
PY
```

## Critérios para remover o rótulo "experimental"

1. `pytest -m gpu` verde em runner CUDA (CI job dedicado, não bloqueante do pacote CPU).
2. Recall CAGRA vs Flat CPU medido no mesmo dataset.
3. Custo de transferência CPU↔GPU reportado separadamente do tempo de busca.
4. Números publicados em benchmarks/RESULTS.md com hardware identificado.

## Planejado (não iniciado)

Multi-GPU (replicação/sharding), IVF-Flat/IVF-PQ GPU, DLPack/CUDA Array
Interface para ingestão zero-round-trip, build GPU + serving CPU
(`kb.optimize(build_device="cuda", serving_device="cpu")`).
