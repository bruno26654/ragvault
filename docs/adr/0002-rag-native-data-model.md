# ADR 0002 — RAG-native data model

## Contexto
Vector DBs genéricos armazenam (id, vetor, payload). RAG precisa de proveniência, versionamento e vizinhança.

## Decisão
Modelo Source → Document → DocumentVersion → Chunk (ragvault-core/types.rs). Chunks carregam document_id, versão, índice, offsets, section_path, página, links previous/next e metadados. Vetores ficam em arena separada (ragvault-vector), nunca dentro das estruturas frias.

## Alternativas consideradas
- Payload JSON livre (Qdrant-style): flexível, mas não garante proveniência para citações.
- Grafo completo de nós/relacionamentos: complexidade desnecessária para v1.

## Consequências
- Citações sempre têm vínculo real com documento+versão.
- `replace_document` publica versão nova atomicamente; versões antigas ficam invisíveis (testado).

## Validação
`replace_publishes_atomically_and_hides_old_version`, `test_document_versioning_and_replace`.
