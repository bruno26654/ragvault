# Avaliação

Dataset JSONL (ou iterável de dicts):

```json
{"query": "prazo de reembolso", "relevant_ids": ["policies/cancelamento.md"]}
```

`relevant_ids` aceita ids de documento ou de chunk.

```python
report = kb.evaluate("evaluation.jsonl", k=10)
print(report.to_markdown())
report.to_json("report.json"); report.to_csv("report.csv")
```

Métricas: Recall@k, Precision@k, hit rate, MRR, nDCG@k, document recall, duplicate rate, tokens médios de contexto, latência p50/p95, além de detalhe por query em `report.per_query`.

Métricas de resposta gerada (faithfulness, answer relevance) dependem de LLM e prompt — ficam fora do core por decisão registrada (ADR 0015).
