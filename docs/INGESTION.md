# Ingestão

## kb.add / kb.add_documents

```python
kb.add("um texto")
kb.add(["texto 1", "texto 2"], metadata={"lang": "pt"})
kb.add_documents([{"id": "doc-1", "text": "...", "title": "...", "metadata": {...}}])
```

IDs omitidos derivam do hash do conteúdo (idempotente). Re-adicionar o mesmo id substitui o documento atomicamente com nova versão.

## kb.sync — espelhamento idempotente de diretório

Descobre arquivos (globs `include`/`exclude`), calcula hash, ignora inalterados, substitui modificados, remove ausentes (`delete_missing=True`), reporta erros por arquivo (`on_error="continue"|"raise"`), devolve `SyncReport`.

## Parsers embutidos

txt, md (título + seções), html (tags/scripts removidos, título), json, jsonl (campo `text`), csv (linhas viram `coluna: valor`), código-fonte (30+ extensões). PDF (`ragvault[pdf]`) e DOCX (`ragvault[office]`) como extras — sem o extra, o erro diz exatamente o que instalar. Extensível:

```python
ragvault.register_parser(".xyz", my_parser)
```

## Chunking

Estratégias `auto/markdown/recursive/paragraph/sentence/fixed`; markdown preserva hierarquia de headings como `section_path`; chunks têm offsets, contagem de tokens e links de vizinhança para expansão de contexto. Parâmetros: `target_tokens`, `max_tokens`, `overlap_tokens`.

## Cache de embeddings

Chave = hash do conteúdo + model_id + configuração; SQLite em `embedding-cache.db`; sobrevive a reopen; estatísticas em `kb.stats()["embedding_cache"]`. Re-sincronizar conteúdo inalterado nunca reprocessa embeddings.
