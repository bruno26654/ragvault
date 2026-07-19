# ADR 0011 — Hybrid retrieval

## Decisão
Sinais: dense (Flat/HNSW), BM25 (k1=1.2, b=0.75, tokenizer Unicode), sparse opcional fornecido pelo usuário. Fusão por RRF ponderado (k0=60): rank-based, nunca soma escalas incompatíveis; semântica do score documentada; empates determinísticos por id. Pesos configuráveis por preset/config.

## Alternativas
Normalized weighted sum — exige calibração por corpus; DBSF — deferido até validação própria.

## Validação
Testes de fusão; `test_hybrid_beats_single_signal_on_lexical_query`.
