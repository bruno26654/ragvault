from __future__ import annotations

import numpy as np
import pytest

import ragvault


def test_filters_restrict_results(kb):
    kb.add([
        {"id": "a", "text": "contract renewal terms for enterprise",
         "metadata": {"department": "legal", "year": 2024}},
        {"id": "b", "text": "contract renewal terms for consumers",
         "metadata": {"department": "sales", "year": 2023}},
    ])
    result = kb.retrieve("contract renewal", filters={"department": "legal"}, k=5)
    assert {c.document_id for c in result.chunks} == {"a"}
    result = kb.retrieve("contract renewal", filters={"year": {"gte": 2024}}, k=5)
    assert {c.document_id for c in result.chunks} == {"a"}
    result = kb.retrieve("contract renewal", k=5)
    assert {c.document_id for c in result.chunks} == {"a", "b"}


def test_invalid_filter_raises_early(kb):
    kb.add("something")
    with pytest.raises(ValueError):
        kb.retrieve("q", filters={"a": {"unknown_op": 1}})


def test_token_budget_is_respected(kb):
    paragraphs = [f"Paragraph {i}: " + ("verylongword " * 120) for i in range(10)]
    kb.add([{"id": f"d{i}", "text": p} for i, p in enumerate(paragraphs)])
    result = kb.retrieve("paragraph verylongword", token_budget=200, k=10)
    assert result.token_count <= 200
    assert len(result.chunks) >= 1


def test_k_limits_context_blocks(kb):
    kb.add([{"id": f"d{i}", "text": f"shared topic variant {i}"} for i in range(10)])
    result = kb.retrieve("shared topic", k=2)
    non_expanded = [c for c in result.chunks if not c.expanded]
    assert len(non_expanded) <= 2
    assert result.context.count("[") <= 4  # citation markers stay consistent


def test_context_window_expansion(kb):
    text = "\n\n".join(f"Section paragraph number {i} about topic-{i}." for i in range(30))
    kb.add([{"id": "long", "text": text}])
    result = kb.retrieve(
        "topic-7", k=1, mode="keyword",
        context_window={"before": 1, "after": 1},
    )
    expanded = [c for c in result.chunks if c.expanded]
    retrieved = [c for c in result.chunks if not c.expanded]
    assert retrieved, "must retrieve at least one chunk"
    if len(kb.inspect("long")["chunks"]) > 1:
        assert expanded, "neighbor expansion should add adjacent chunks"
        for chunk in expanded:
            assert chunk.document_id == "long"


def test_duplicate_content_is_deduplicated(kb):
    kb.add([
        {"id": "x", "text": "identical duplicate paragraph"},
        {"id": "y", "text": "identical duplicate paragraph"},
        {"id": "z", "text": "a different paragraph about ducks"},
    ])
    result = kb.retrieve("duplicate paragraph ducks", k=3)
    texts = [c.text for c in result.chunks]
    assert len(texts) == len(set(texts)), "exact duplicates must not repeat in context"


def test_custom_embedder_callable(tmp_path):
    def tiny_embedder(texts):
        rng = np.random.default_rng(42)
        out = []
        for t in texts:
            rs = np.random.default_rng(abs(hash(t)) % (2**32))
            out.append(rs.normal(size=32))
        return np.array(out, dtype=np.float32)

    kb = ragvault.open(tmp_path / "kb", embedding=tiny_embedder)
    try:
        assert kb.embedder.dimension == 32
        kb.add(["alpha text", "beta text"])
        result = kb.retrieve("alpha text", k=1, mode="dense")
        assert result.chunks
    finally:
        kb.close()


def test_embedding_mismatch_on_reopen_rejected(tmp_path):
    path = tmp_path / "kb"
    kb = ragvault.open(path, embedding="builtin:hashed-ngram:256")
    kb.add("content")
    kb.close()
    with pytest.raises(ragvault.ConfigurationError):
        ragvault.open(path, embedding="builtin:hashed-ngram:512")


def test_hybrid_beats_single_signal_on_lexical_query(kb):
    # A query with an exact rare token: BM25 must contribute.
    kb.add([
        {"id": "code", "text": "Error XK-4211 means the payment gateway timed out."},
        {"id": "other", "text": "General information about payments and gateways."},
    ])
    result = kb.retrieve("what does XK-4211 mean?", k=1)
    assert result.chunks[0].document_id == "code"


def test_retrieve_many_and_async(kb):
    import asyncio

    kb.add([{"id": "a", "text": "first topic"}, {"id": "b", "text": "second topic"}])
    results = kb.retrieve_many(["first topic", "second topic"], k=1)
    assert len(results) == 2

    async def run():
        return await kb.aretrieve_many(["first topic", "second topic"], k=1)

    async_results = asyncio.run(run())
    assert len(async_results) == 2
    assert async_results[0].chunks[0].document_id == "a"


def test_ask_uses_context_and_llm(kb):
    kb.add([{"id": "policy", "text": "Refunds are issued within 30 days of cancellation."}])
    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return "Refunds take 30 days. [1]"

    answer = kb.ask("how long do refunds take?", llm=fake_llm)
    assert "30 days" in answer.text
    assert answer.citations
    assert "Refunds are issued" in prompts[0], "context must be in the prompt"


def test_tenant_isolation(kb):
    kb.for_tenant("acme").add([{"id": "acme-doc", "text": "acme secret plan"}])
    kb.for_tenant("globex").add([{"id": "globex-doc", "text": "globex secret plan"}])
    acme = kb.for_tenant("acme").retrieve("secret plan", k=10)
    assert {c.document_id for c in acme.chunks} == {"acme-doc"}
    globex = kb.for_tenant("globex").retrieve("secret plan", k=10)
    assert {c.document_id for c in globex.chunks} == {"globex-doc"}
    # adversarial: filter injection cannot escape the tenant scope
    sneaky = kb.for_tenant("acme").retrieve(
        "secret plan", k=10, filters={"tenant_id": "globex"}
    )
    assert not sneaky.chunks


def test_explain_and_trace(kb):
    kb.add("some content for planning")
    result = kb.retrieve("planning content", explain=True, trace=True)
    assert result.plan["dense_backend"] in ("flat", "hnsw", "flat_filtered_fallback")
    assert "reason" in result.plan
    assert result.trace is not None
    assert "context" in result.trace
    assert result.trace["context"]["token_budget"] > 0


def test_rerank_callback_and_tolerant_failure(kb):
    kb.add([{"id": "a", "text": "alpha beta"}, {"id": "b", "text": "alpha gamma"}])

    def reverse_rerank(query, chunks):
        return list(reversed(chunks))

    result = kb.retrieve("alpha", k=2, rerank=reverse_rerank)
    assert len(result.chunks) == 2

    def broken_rerank(query, chunks):
        raise RuntimeError("model exploded")

    result = kb.retrieve("alpha", k=2, rerank=broken_rerank, trace=True)
    assert result.chunks, "tolerant mode keeps pre-rerank ranking"
    assert "rerank_error" in result.trace
