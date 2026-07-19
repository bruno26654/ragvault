# ADR 0001 — Product boundaries

## Contexto
RagVault poderia ser posicionado como vector DB genérico, clone do Faiss ou framework RAG completo com LLMs embutidos.

## Requisitos
Qualidade de contexto > praticidade Python > time-to-first-RAG > correção > durabilidade (hierarquia do produto).

## Alternativas
1. Vector DB genérico com API de índices — máxima flexibilidade, péssimo time-to-first-RAG.
2. Framework de agentes com LLM embutido — viola neutralidade de provedor e cria chamadas de rede silenciosas.
3. **Biblioteca Python-first local-first cobrindo documentos → contexto citável, com engine vetorial como infraestrutura interna.**

## Decisão
Alternativa 3. A abstração pública é `KnowledgeBase`; índices (Flat/HNSW/BM25) são detalhes internos expostos apenas via configuração e `explain()`. LLMs entram só via `ask(llm=...)` fornecido pelo usuário. Nenhuma chamada externa ou download silencioso.

## Consequências
- O caminho comum não expõe `HnswIndex`, `Bm25Index` etc.
- Recursos de geração (answer eval, rerankers remotos) ficam em integrações opcionais.

## Riscos
Usuários avançados podem querer acesso de baixo nível → mitigado com `kb._vault` (não estável) e planos de API `Database/Collection` futura.

## Validação
Testes end-to-end exercitam exclusivamente a API pública.
