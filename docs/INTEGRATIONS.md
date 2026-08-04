# Integrações

| Integração | API | Status | Dependência |
|---|---|---|---|
| LangChain | `kb.as_langchain_retriever(k=8)` | **validated** (testado contra langchain-core real) | `pip install langchain-core` |
| LlamaIndex | `kb.as_llamaindex_retriever(k=8)` | implemented, untested (erro acionável sem a lib) | `pip install llama-index-core` |
| Haystack 2.x | `kb.as_haystack_retriever(k=8)` | implemented, untested (erro acionável sem a lib) | `pip install haystack-ai` |
| DSPy | `kb.as_dspy_retriever(k=8)` | implemented, untested (erro acionável sem a lib) | `pip install dspy` |
| Faiss (interop) | `ragvault.compat.faiss` | **validated** (round-trip testado com faiss-cpu real) | `pip install "ragvault[faiss]"` |
| GPU cuVS/CAGRA | `ragvault.gpu.CagraDenseSearcher` | experimental — ver docs/GPU.md | `pip install "ragvault[gpu-cu12]"` |
| NLI (verificação offline) | `ragvault.nli.nli_verifier()` | implemented, **não medido** — ver `benchmarks/RESULTS-VERIFICATION.md` | `pip install "ragvault[nli]"` |
| Segmentador de sentença | `kb.ask(..., segmenter=...)` | callable seu (PySBD/PyICU/spaCy) — sem dependência no núcleo | opcional |
| NumPy | nativo em toda a API | validated | — |

Todos os retrievers carregam proveniência completa (document_id, versão,
chunk_id, título, URI, seção, score) nos metadados dos documentos devolvidos.

## Faiss: nível de compatibilidade

`ragvault.compat.faiss` é **convertible**: exporta/importa vetores e ids por
valor (`export_to_faiss`, `import_vectors`, `reconstruct_from_faiss`).
Não há compatibilidade binária de formato nem de API — ver ANALISE-FAISS.md.

## Validação das integrações não testadas

```bash
pip install llama-index-core && python -m pytest tests/python -q -k llamaindex
pip install haystack-ai      && python - -c "…ver examples/"
pip install dspy             && python - -c "…"
```
O padrão dos adaptadores é idêntico ao do LangChain (testado); o risco
residual está em mudanças de API das libs externas.
