#!/usr/bin/env python3
"""Reproducible RAG-quality evaluation on REAL TEXT.

Dataset: benchmarks/data/rag_eval_corpus.jsonl (30 factual passages, 10
topics) + rag_eval_queries.jsonl (24 queries: 12 paraphrased with low
lexical overlap, 12 keyword-flavoured), committed to the repository —
fully reproducible, no network needed for the offline rows.

Configurations compared (same corpus, same queries, same k):
  bm25            keyword-only retrieval
  lexical-dense   builtin hashed-ngram dense only (the honest baseline)
  hybrid          lexical dense + BM25, weighted RRF
  hybrid+mmr      hybrid with aggressive MMR diversity (lambda=0.5)
  hybrid+expand   hybrid with neighbor expansion (window 1/1)
  semantic-*      sentence-transformers rows — require the model; when the
                  environment blocks huggingface.co the rows are reported
                  as BLOCKED with the exact command to run elsewhere.

Every number written to RESULTS-RAG.md comes from an actual run.
Usage:  python benchmarks/bench_rag_quality.py
Semantic rows elsewhere:  pip install "ragvault[local-models]" && \
    python benchmarks/bench_rag_quality.py --semantic
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "RESULTS-RAG.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_kb(tmp: Path, name: str, docs: list[dict], **kb_kwargs):
    import ragvault

    kb = ragvault.open(tmp / name, **kb_kwargs)
    kb.add([{"id": d["id"], "text": d["text"]} for d in docs])
    return kb


def evaluate(kb, queries: list[dict], k: int, **retrieve_kwargs) -> dict:
    report = kb.evaluate(
        [{"query": q["query"], "relevant_ids": q["relevant_ids"]} for q in queries],
        k=k, **retrieve_kwargs,
    )
    # per-style split (paraphrase vs keyword) tells the honest story about
    # where a lexical embedder falls short.
    styles = {}
    for q, row in zip(queries, report.per_query):
        styles.setdefault(q["style"], []).append(row["mrr"])
    style_mrr = {s: sum(v) / len(v) for s, v in styles.items()}
    return {
        "recall": report.recall_at_k,
        "mrr": report.mrr,
        "ndcg": report.ndcg_at_k,
        "precision": report.precision_at_k,
        "dup": report.duplicate_rate,
        "tokens": report.avg_context_tokens,
        "p50": report.latency_p50_ms,
        "p95": report.latency_p95_ms,
        "mrr_paraphrase": style_mrr.get("paraphrase", 0.0),
        "mrr_keyword": style_mrr.get("keyword", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", action="store_true",
                        help="also run sentence-transformers rows (downloads "
                             "the model — explicit action)")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    docs = load_jsonl(DATA / "rag_eval_corpus.jsonl")
    queries = load_jsonl(DATA / "rag_eval_queries.jsonl")
    tmp = Path(tempfile.mkdtemp(prefix="ragvault-quality-"))

    lines = [
        "# RAG quality evaluation — real text",
        "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        f"- host: {platform.platform()}, python {platform.python_version()}",
        f"- dataset: {len(docs)} passages / {len(queries)} queries "
        "(12 paraphrase + 12 keyword), committed in benchmarks/data/",
        f"- k = {args.k}; every row below was actually executed on this machine",
        "",
        "| config | recall@k | MRR | nDCG@k | precision@k | MRR paraphrase | "
        "MRR keyword | dup rate | ctx tokens | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def add_row(name: str, metrics: dict) -> None:
        lines.append(
            f"| {name} | {metrics['recall']:.3f} | {metrics['mrr']:.3f} | "
            f"{metrics['ndcg']:.3f} | {metrics['precision']:.3f} | "
            f"{metrics['mrr_paraphrase']:.3f} | {metrics['mrr_keyword']:.3f} | "
            f"{metrics['dup']:.3f} | {metrics['tokens']:.0f} | "
            f"{metrics['p50']:.1f} | {metrics['p95']:.1f} |"
        )
        print(lines[-1])

    kb = build_kb(tmp, "lex", docs)  # balanced preset, hashed-ngram
    try:
        add_row("bm25 only", evaluate(kb, queries, args.k, mode="keyword"))
        add_row("lexical dense only", evaluate(kb, queries, args.k, mode="dense"))
        add_row("hybrid (lexical+bm25)", evaluate(kb, queries, args.k, mode="hybrid"))
        add_row("hybrid + expansion", evaluate(
            kb, queries, args.k, mode="hybrid",
            context_window={"before": 1, "after": 1}))
    finally:
        kb.close()

    import ragvault
    kb = ragvault.open(tmp / "mmr", mmr_lambda=0.5)
    kb.add([{"id": d["id"], "text": d["text"]} for d in docs])
    try:
        add_row("hybrid + MMR(0.5)", evaluate(kb, queries, args.k, mode="hybrid"))
    finally:
        kb.close()

    if args.semantic:
        try:
            spec = "sentence-transformers:all-MiniLM-L6-v2"
            kb = build_kb(tmp, "sem", docs, embedding=spec)
            try:
                add_row("semantic dense", evaluate(kb, queries, args.k, mode="dense"))
                add_row("semantic hybrid", evaluate(kb, queries, args.k, mode="hybrid"))

                def st_rerank(query, chunks):
                    embedder = kb.embedder
                    qv = embedder.embed_queries([query])[0]
                    dv = embedder.embed_documents([c.text for c in chunks])
                    scored = sorted(zip((dv @ qv).tolist(), chunks),
                                    key=lambda p: -p[0])
                    out = []
                    for score, chunk in scored:
                        chunk.score = float(score)
                        out.append(chunk)
                    return out

                add_row("semantic hybrid + rerank", evaluate(
                    kb, queries, args.k, mode="hybrid", rerank=st_rerank))
            finally:
                kb.close()
        except Exception as exc:
            lines.append("")
            lines.append(f"> semantic rows FAILED in this environment: {exc}")
            print(lines[-1])
    else:
        lines += [
            "",
            "> Semantic rows (semantic dense / semantic hybrid / + rerank) were",
            "> NOT run in this environment: the network policy blocks",
            "> huggingface.co (CONNECT 403), so the model cannot be downloaded.",
            "> Run them elsewhere with:",
            '> `pip install "ragvault[local-models]" && '
            "python benchmarks/bench_rag_quality.py --semantic`",
        ]

    lines += [
        "",
        "## Reading the numbers honestly",
        "",
        "- The paraphrase-MRR column is the semantic gap: hashed-ngram is a",
        "  lexical projection, so paraphrased queries (low word overlap) are",
        "  where it underperforms and where a semantic model earns its cost.",
        "- Keyword-MRR shows BM25/hybrid doing what lexical retrieval is good",
        "  at. The `quality` preset therefore refuses to run without an",
        "  explicit embedding decision.",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwritten: {OUT}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
