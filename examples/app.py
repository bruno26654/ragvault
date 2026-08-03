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


def _json(raw: str):
    """Extrai o JSON (array ou objeto) da resposta do modelo. Se falhar,
    levanta — o RagVault trata a exceção e cai para o fallback seguro da
    etapa."""
    match = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
    return json.loads(match.group(0) if match else raw)


def decompose(question: str) -> list[str]:
    """Pergunta composta → subconsultas. Falha aqui = consulta única.

    Cada subconsulta deve tratar uma única obrigação de resposta: subconsultas
    compostas gastam uma vaga de contexto para duas evidências e costumam
    trazer só uma delas."""
    return _json(complete(
        "Divida a pergunta no conjunto mínimo de buscas independentes "
        "necessárias para respondê-la por completo. Cada busca deve cobrir "
        "exatamente um fato pedido — nunca combine dois numa só. Responda "
        f"apenas com um array JSON de strings.\n\nPergunta: {question}"
    ))


def verify(payload: dict) -> dict:
    """Segmenta a resposta em proposições e julga cada uma contra a fonte que
    cita, dizendo também quais facetas da pergunta ficaram cobertas.

    O juiz não reescreve nada: um verificador que propõe a correção e depois a
    aprova está avaliando o próprio texto. Falha aqui = resposta original
    preservada e `ok=False` — nunca um passe silencioso.
    """
    # Cada claim vem com as fontes que *ela* citou: o suporte não pode ser
    # emprestado de um bloco que a afirmação nunca nomeou.
    cited = "\n\n".join(
        f"## Afirmação {i}: {c['claim']}\n" + (
            "\n".join(f"[{e['index']}] ({e['metadata']}) {e['text']}"
                      for e in c["evidence"])
            or "(sem fonte citada)"
        )
        for i, c in enumerate(payload["claims"], 1)
    )
    return _json(complete(
        "Você verifica uma resposta de RAG. Não reescreva nada: apenas "
        "segmente e classifique.\n\n"
        f"# Contexto completo\n{payload['context']}\n\n"
        f"# Fontes que cada afirmação citou\n{cited}\n\n"
        f"# Pergunta do usuário\n{payload['question']}\n\n"
        f"# Resposta\n{payload['answer']}\n\n"
        f"# Facetas que a resposta deveria cobrir\n{payload['facets']}\n\n"
        "1. Segmente a resposta em TODAS as proposições materiais: trechos "
        "literais da resposta, em ordem, sem sobreposição, cada um com uma "
        "única proposição (uma frase com dois fatos vira dois itens).\n"
        f"2. Classifique cada um com 'verdict' (um de {payload['verdicts']}) "
        "e 'rationale', julgando contra os fatos explícitos da pergunta, os "
        "blocos citados e os metadados das fontes:\n"
        "   - 'contradicted' também quando contraria um fato dito na "
        "pergunta, mesmo sem documento envolvido;\n"
        "   - 'question_fact' só para repetir algo que a pergunta afirmou; um "
        "fato dado na pergunta nunca é ausente nem indeterminável. Conclusões "
        "e deduções são 'inference';\n"
        "   - afirmação sobre regra passada ou revogada só é 'supported' se "
        "algum bloco citado tiver metadados que mostrem esse estado antigo — "
        "divergir da regra atual não prova que a antiga existiu.\n"
        "3. Em 'supported', inclua 'quote': as palavras exatas da fonte citada "
        "que sustentam a afirmação, copiadas literalmente. O RagVault confere "
        "a citação contra a fonte — uma citação inventada reprova a "
        "afirmação.\n"
        "4. Para cada faceta, 'covered' só é verdadeiro quando TODOS os seus "
        "componentes foram respondidos corretamente.\n"
        "Omitir uma proposição ou uma faceta não é aprovação. Responda apenas "
        'com JSON: {"claims": [{"claim", "verdict", "rationale", "quote"}], '
        '"facets": [{"facet", "covered", "rationale"}]}.'
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

# Fidelidade e completude são eixos separados: toda afirmação pode estar
# sustentada e a resposta ainda deixar uma faceta da pergunta sem resposta.
if not answer.verification.ok:
    print("\nAfirmações sem suporte documental (removidas):")
    for claim in answer.unverified_claims:
        print(f"  - {claim.claim}\n    ↳ {claim.rationale}")

if answer.verification.complete is False:
    print("\nFacetas da pergunta não cobertas:")
    for facet in answer.verification.uncovered_facets:
        print(f"  - {facet['facet']}\n    ↳ {facet['rationale']}")

kb.close()
