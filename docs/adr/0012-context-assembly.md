# ADR 0012 — Context assembly

## Decisão
`retrieve()` entrega contexto, não vizinhos: cap por documento → dedup exata → diversidade MMR usando sobreposição de tokens (não requer vetores brutos; documentado) → seleção por token budget (truncagem quando o melhor chunk sozinho excede o budget) → expansão de vizinhos dentro do mesmo documento/versão respeitando budget → ordenação por documento → citações estáveis [n] apontando para chunks reais.

## Validação
Testes de budget, dedup, expansão, k-limite e validade de citações (get_chunk existe para todo chunk citado).
