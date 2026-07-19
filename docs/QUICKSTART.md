# Quickstart

## Instalação (a partir do código-fonte, enquanto wheels não estão no PyPI)

```bash
git clone https://github.com/bruno26654/ragvault.git && cd ragvault
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release
```

## Primeiro RAG em 5 linhas

```python
import ragvault

kb = ragvault.open("./knowledge", preset="quality")
kb.sync("./documents")           # txt, md, html, json, jsonl, csv, código; pdf/docx com extras
result = kb.retrieve("Quais são as regras de cancelamento?", token_budget=6000)
print(result.context)
for c in result.citations:
    print(f"[{c.index}] {c.title or c.document_id} (v{c.document_version})")
```

O default funciona 100% offline com o embedder lexical embutido. Para qualidade semântica:

```bash
pip install "ragvault[local-models]"
```

```python
kb = ragvault.open("./knowledge", preset="quality",
                   embedding="sentence-transformers:all-MiniLM-L6-v2")
```

## Sincronização contínua

`kb.sync()` é idempotente: rode quantas vezes quiser.

```python
report = kb.sync("./documents", include=["**/*.md"], exclude=["**/archive/**"])
print(report)   # SyncReport(discovered=840, added=17, updated=4, unchanged=816, deleted=3, failed=0)
```

## Perguntas com o seu LLM

```python
def my_llm(prompt: str) -> str:
    ...  # chame o provedor que você quiser

answer = kb.ask("Quais são as regras de cancelamento?", llm=my_llm, citations=True)
print(answer.text)
```

## Avaliação

Crie `evaluation.jsonl`:

```json
{"query": "prazo de reembolso", "relevant_ids": ["policies/cancelamento.md"]}
```

```python
report = kb.evaluate("evaluation.jsonl", k=5)
print(report.to_markdown())
```

## CLI

```bash
ragvault init ./knowledge --preset quality
ragvault sync ./knowledge ./documents
ragvault query ./knowledge "minha pergunta" --explain
ragvault doctor ./knowledge
```
