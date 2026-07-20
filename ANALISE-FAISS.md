# Análise — relação do RagVault com o Faiss

## Papel do Faiss no projeto

O RagVault tem engine CPU própria (Flat, HNSW, SQ8) — não é clone nem wrapper do Faiss. O Faiss é usado exclusivamente como **referência de comparação em benchmark** (`pip install faiss-cpu`, opcional; o pacote RagVault não depende dele).

## Comparação executada (benchmarks/RESULTS.md)

Mesmo dataset, mesmas queries, mesmo k, mesma máquina, 1 thread — resultados reais no cenário 50k×384 (gaussianos normalizados, caso adversarial para recall absoluto):

- **Recall em `ef` igual**: RagVault HNSW acompanha o Faiss de perto (diferença ≤ 0.04 em todos os `ef` medidos) — as duas implementações do algoritmo se comportam de forma equivalente.
- **QPS**: Faiss é ~3× mais rápido no cenário medido. Causas conhecidas e registradas: overhead do binding Python/JSON por query (incluído de propósito — é o que o usuário observa), kernels sem intrinsics dedicados por arquitetura, e checagem de tombstones/filtros no hot path.
- **Build**: Faiss constrói mais rápido; a ingestão do RagVault inclui durabilidade WAL, que o Faiss não oferece.

**Não afirmamos superioridade universal sobre o Faiss.** O critério para qualquer afirmação futura de vantagem está no ADR 0015 (≥3 datasets, múltiplos tamanhos/dimensões/seeds, recall equivalente, mesmo hardware).

## Onde o RagVault se diferencia do Faiss (por design, não por benchmark)

Faiss é uma biblioteca de índices; RagVault é um fluxo RAG completo: modelo documental com versões, WAL/recovery, BM25+híbrido, filtros integrados à travessia, montagem de contexto com citações, avaliação nativa e API Python de alto nível. Ver COMPETITIVE-ANALYSIS.md.

## Compatibilidade

Nenhuma compatibilidade binária com formatos Faiss é afirmada. Um namespace `ragvault.compat.faiss` (migração/importação lógica de vetores) está planejado e será rotulado como *convertible*, nunca *binary-format-compatible*, conforme as definições do projeto.
