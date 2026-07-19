# ADR 0015 — Evaluation methodology

## Decisão
Avaliação nativa focada em retrieval (Recall@k, Precision@k, MRR, nDCG, hit rate, duplicate rate, tokens de contexto, latência p50/p95) sobre datasets JSONL com ids relevantes. Métricas de resposta gerada (faithfulness etc.) ficam fora do core por dependerem de LLM/prompt. Benchmarks publicados somente com números executados; comparações externas exigem mesmo hardware/dataset/k/recall e afirmações restritas ao cenário medido.

## Validação
`kb.evaluate` testado end-to-end; benchmarks/RESULTS.md gerado por execução real.
