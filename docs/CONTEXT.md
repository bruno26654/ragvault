# Montagem de contexto e citações

`kb.retrieve()` devolve contexto pronto para o modelo:

1. **Cap por documento** (`max_chunks_per_document`) e dedup exata de conteúdo.
2. **Diversidade MMR** (`mmr_lambda`): penaliza chunks com alta sobreposição de tokens com os já escolhidos.
3. **Token budget** (`token_budget`): seleção respeita o orçamento; se o melhor chunk sozinho excede, ele é truncado (nunca contexto vazio). Tokenizer plugável via `ChunkingConfig.tokenizer`.
4. **Expansão de vizinhos** (`context_window={"before": 1, "after": 1}`): adiciona chunks adjacentes do mesmo documento/versão, marcados `expanded=True`, sem estourar o budget.
5. **Ordenação**: documentos pelo melhor score; chunks em ordem de leitura dentro do documento.
6. **Citações**: blocos `[n]` no contexto mapeiam para `result.citations[n-1]` com document_id, versão, chunk_ids, título, URI, seção e página — sempre vinculadas a chunks realmente armazenados.

```python
RetrievalResult(
  context: str, chunks: list[RetrievedChunk], citations: list[Citation],
  token_count: int, plan: dict, trace: dict | None,
)
```
