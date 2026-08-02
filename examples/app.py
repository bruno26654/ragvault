#!/usr/bin/env python3
"""RAG de alta qualidade sobre base grande — o máximo delegado ao RagVault.

Fora do RagVault, só o estritamente necessário: Groq como provedor de
inferência (resposta, decomposição e verificação) e python-dotenv para a
chave. Tudo o mais — parsing, chunking, embeddings + cache, indexação,
busca híbrida, multi-query, fusão, precedência de versões, MMR, montagem de
contexto, citações e persistência durável — é RagVault.

    pip install ragvault[local-models] groq python-dotenv
    echo 'GROQ_API_KEY=...' > .env
    python app.py "sua pergunta"

.env:
    GROQ_API_KEY=...
    GROQ_MODEL=llama-3.3-70b-versatile     # opcional
    DOCS_DIR=./documents                   # opcional
    KB_DIR=./knowledge                     # opcional
    EMBEDDING=local:multilingual           # opcional (o padrão)
"""

from __future__ import annotations

import json
import os
import re
import sys

import ragvault
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
_groq = Groq(api_key=os.environ["GROQ_API_KEY"])
_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def complete(prompt: str) -> str:
    """Único ponto de contato com o provedor. RagVault nunca chama a rede."""
    response = _groq.chat.completions.create(
        model=_MODEL, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _json_list(raw: str) -> list:
    """Extrai o array JSON da resposta do modelo. Se falhar, levanta — o
    RagVault trata a exceção e cai para o fallback seguro da etapa."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    return json.loads(match.group(0) if match else raw)


def decompose(question: str) -> list[str]:
    """Pergunta composta → subconsultas. Falha aqui = consulta única."""
    return _json_list(complete(
        "Divida a pergunta no conjunto mínimo de buscas independentes "
        "necessárias para respondê-la por completo. Responda apenas com um "
        f"array JSON de strings.\n\nPergunta: {question}"
    ))


def verify(payload: dict) -> list:
    """Julga cada afirmação contra a fonte que ela cita. Falha aqui =
    resposta original preservada."""
    claims = "\n".join(
        f"{i}. {c['claim']}   (cita: {c['citations'] or 'nada'})"
        for i, c in enumerate(payload["claims"], 1)
    )
    return _json_list(complete(
        "Verifique se cada afirmação é sustentada pelos blocos de contexto "
        f"que ela cita.\n\n# Contexto\n{payload['context']}\n\n"
        f"# Pergunta do usuário\n{payload['question']}\n\n# Afirmações\n{claims}\n\n"
        f"Para cada afirmação responda um objeto JSON com 'verdict' (um de "
        f"{payload['verdicts']}) e 'rationale'. Use 'question_fact' quando a "
        "afirmação apenas repete algo que o usuário afirmou na pergunta, e "
        "'contradicted' quando o bloco citado diz o contrário. Responda "
        "apenas com um array JSON."
    ))


# Base persistida: preset de qualidade, embeddings multilíngues locais
# carregados pelo RagVault, e mmap para base grande (ativa na reabertura,
# quando já existe snapshot em disco — mantém a RAM constante).
kb = ragvault.open(
    os.getenv("KB_DIR", "./knowledge"),
    preset="quality",
    embedding=os.getenv("EMBEDDING", "local:multilingual"),
    storage="mmap",
)

# Ingestão idempotente: só o que mudou é reprocessado (hash de conteúdo +
# cache de embeddings). Rodar de novo em base grande é barato.
report = kb.sync(os.getenv("DOCS_DIR", "./documents"))
kb.flush()
print(f"base: {report} | {kb.stats()['live_chunks']} chunks", file=sys.stderr)

question = " ".join(sys.argv[1:]) or "Resuma as regras principais dos documentos."

answer = kb.ask_multi(
    question,
    llm=complete,                  # responde
    decompose=decompose,           # decompõe perguntas compostas
    verify=verify,                 # valida cada afirmação contra a fonte
    verification_mode="repair",    # remove/corrige o que não se sustenta
    resolve_versions=True,         # versão vigente vence a revogada
    citations=True,                # [n] inexistente é removido
    k=8,
    token_budget=6000,
)

print(answer.text)

# Só as fontes que a resposta final realmente citou: o contexto costuma
# trazer mais documentos do que a resposta usa, e listar os não usados
# como "fonte" sugeriria um respaldo que não existe.
usados = {int(n) for n in re.findall(r"\[(\d+)\]", answer.text)}
print("\nFontes:")
for c in answer.citations:
    if c.index in usados:
        print(f"  [{c.index}] {c.title or c.document_id} — {c.uri or c.document_id}")

if not answer.verification.ok:
    print("\nAfirmações sem suporte documental (removidas/corrigidas):")
    for claim in answer.unverified_claims:
        print(f"  - {claim.claim}\n    ↳ {claim.rationale}")

kb.close()
