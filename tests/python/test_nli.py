"""The NLI verifier: offline fidelity checking without an LLM judge.

Every test here runs against a stub model — no downloads, no network. What is
being tested is the adapter's reasoning (label mapping, premise granularity,
aggregation, what happens with no citation), not a checkpoint's accuracy. That
is what `benchmarks/bench_nli_verifier.py` is for, and it is deliberately not
a unit test: it needs a real model and real labelled data.
"""

from __future__ import annotations

import numpy as np
import pytest

import ragvault
from ragvault.errors import ConfigurationError
from ragvault.nli import NLIVerifier, calibrate_threshold, nli_verifier

#: Canonical order the adapter uses internally: (entail, neutral, contradict).
ENTAIL = (6.0, 1.0, 0.0)
NEUTRAL = (0.0, 6.0, 1.0)
CONTRADICT = (0.0, 1.0, 6.0)


class StubTokenizer:
    """Records the pairs it is asked to encode, and passes them through."""

    def __init__(self):
        self.pairs = []

    def __call__(self, premises, hypotheses, **kwargs):
        self.pairs.extend(zip(premises, hypotheses))
        return {"pairs": list(zip(premises, hypotheses))}


class StubModel:
    """A model whose logits are decided by a rule, laid out in *its own*
    label order — which is the point: a real checkpoint decides that order,
    and the adapter has to read it rather than assume it."""

    def __init__(self, rule, id2label=None):
        self.rule = rule
        self.config = type("Config", (), {})()
        self.config.id2label = id2label or {
            0: "entailment", 1: "neutral", 2: "contradiction",
        }
        self.calls = 0
        positions = {}
        for idx, name in self.config.id2label.items():
            for canonical, key in enumerate(("entail", "neutral", "contradict")):
                if key in str(name).lower():
                    positions[canonical] = int(idx)
        self._positions = positions

    def __call__(self, **inputs):
        self.calls += 1
        rows = []
        for premise, hypothesis in inputs["pairs"]:
            row = [0.0, 0.0, 0.0]
            for canonical, value in enumerate(self.rule(premise, hypothesis)):
                row[self._positions[canonical]] = value
            rows.append(row)
        return type("Output", (), {"logits": np.array(rows)})()


def build(rule, id2label=None, **kwargs):
    model = StubModel(rule, id2label)
    tokenizer = StubTokenizer()
    verifier = NLIVerifier(model, tokenizer=tokenizer, **kwargs)
    return verifier, model, tokenizer


def payload(claim, *texts, question="how long?"):
    return {
        "question": question,
        "answer": claim,
        "claims": [{
            "claim": claim,
            "citations": [1],
            "evidence": [
                {"index": i + 1, "document_id": f"d{i}", "chunk_ids": [f"c{i}"],
                 "text": text, "metadata": {}, "title": "t"}
                for i, text in enumerate(texts)
            ],
        }],
    }


class TestLabelMapping:
    """Checkpoints disagree about label order. Reading it by position is the
    bug that inverts every verdict on half the models on the hub."""

    @pytest.mark.parametrize("id2label", [
        {0: "entailment", 1: "neutral", 2: "contradiction"},
        {0: "contradiction", 1: "neutral", 2: "entailment"},   # reversed
        {0: "NEUTRAL", 1: "ENTAILMENT", 2: "CONTRADICTION"},   # shuffled, cased
    ])
    def test_verdict_follows_the_declared_labels(self, id2label):
        verifier, _, _ = build(lambda p, h: ENTAIL, id2label)
        out = verifier(payload("Refunds take 30 days [1].", "Refunds take 30 days."))
        assert out["claims"][0]["verdict"] == "supported"

    def test_opaque_labels_are_refused(self):
        with pytest.raises(ConfigurationError, match="cannot map NLI labels"):
            build(lambda p, h: ENTAIL, {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})

    def test_a_model_without_labels_is_refused(self):
        model = StubModel(lambda p, h: ENTAIL)
        del model.config.id2label
        with pytest.raises(ConfigurationError, match="no config.id2label"):
            NLIVerifier(model, tokenizer=StubTokenizer())

    def test_a_preloaded_model_needs_its_tokenizer(self):
        with pytest.raises(ConfigurationError, match="matched pair"):
            NLIVerifier(StubModel(lambda p, h: ENTAIL))


class TestPremiseGranularity:
    """SummaC's finding: NLI degrades on long premises, and scoring each
    source sentence separately against the claim recovers most of it."""

    BLOCK = ("Orders ship within five days. Refunds take 30 days. "
             "Support answers within 24 hours.")

    def test_sentence_granularity_scores_each_sentence(self):
        verifier, _, tokenizer = build(lambda p, h: NEUTRAL)
        verifier(payload("Refunds take 30 days [1].", self.BLOCK))
        assert len(tokenizer.pairs) == 3
        assert [p for p, _ in tokenizer.pairs] == [
            "Orders ship within five days.",
            "Refunds take 30 days.",
            "Support answers within 24 hours.",
        ]

    def test_block_granularity_scores_the_whole_chunk(self):
        verifier, _, tokenizer = build(lambda p, h: NEUTRAL, granularity="block")
        verifier(payload("Refunds take 30 days [1].", self.BLOCK))
        assert len(tokenizer.pairs) == 1
        assert tokenizer.pairs[0][0] == self.BLOCK

    def test_one_entailing_sentence_is_enough(self):
        """Existential, which is why `max` is the operator and not a knob:
        the other two sentences being unrelated does not weaken the one that
        actually says it."""
        def rule(premise, hypothesis):
            return ENTAIL if "Refunds take 30" in premise else NEUTRAL

        verifier, _, _ = build(rule)
        out = verifier(payload("Refunds take 30 days [1].", self.BLOCK))
        assert out["claims"][0]["verdict"] == "supported"

    def test_blocks_are_never_concatenated(self):
        verifier, _, tokenizer = build(lambda p, h: NEUTRAL, granularity="block")
        verifier(payload("A claim [1][2].", "First block.", "Second block."))
        assert [p for p, _ in tokenizer.pairs] == ["First block.", "Second block."]

    def test_unsegmentable_text_falls_back_to_the_whole_block(self):
        """Thai has no sentence terminator, so the splitter returns one unit.
        That is the degraded mode, not an error — and it is why the splitter's
        script coverage matters to the NLI path too."""
        thai = "การคืนเงินใช้เวลาสามสิบวันและการจัดส่งใช้เวลาห้าวัน"
        verifier, _, tokenizer = build(lambda p, h: NEUTRAL)
        verifier(payload("Refunds take 30 days [1].", thai))
        assert [p for p, _ in tokenizer.pairs] == [thai]


class TestAggregation:
    def test_contradiction_is_reported(self):
        verifier, _, _ = build(lambda p, h: CONTRADICT)
        out = verifier(payload("Refunds take 90 days [1].", "Refunds take 30 days."))
        assert out["claims"][0]["verdict"] == "contradicted"

    def test_neutral_is_unsupported_not_contradicted(self):
        verifier, _, _ = build(lambda p, h: NEUTRAL)
        out = verifier(payload("Refunds are free [1].", "Refunds take 30 days."))
        assert out["claims"][0]["verdict"] == "unsupported"

    def test_entailment_outranks_a_stray_contradiction(self):
        """Across many premise sentences some unrelated one eventually scores
        as a contradiction. Letting that win would delete correct text in
        `repair` mode, and the two errors are not symmetric: a missed
        contradiction leaves a visibly wrong sentence, a false one silently
        removes a right sentence."""
        def rule(premise, hypothesis):
            return ENTAIL if "30 days" in premise else CONTRADICT

        verifier, _, _ = build(rule)
        out = verifier(payload(
            "Refunds take 30 days [1].",
            "Refunds take 30 days. Unrelated policy text.",
        ))
        assert out["claims"][0]["verdict"] == "supported"

    def test_the_winning_premise_becomes_the_quote(self):
        """The quote is verbatim source text, so the structural layer can
        re-check this adapter's own output instead of trusting it."""
        def rule(premise, hypothesis):
            return ENTAIL if "Refunds" in premise else NEUTRAL

        verifier, _, _ = build(rule)
        out = verifier(payload(
            "Refunds take 30 days [1].",
            "Orders ship in five days. Refunds take 30 days.",
        ))
        assert out["claims"][0]["quote"] == "Refunds take 30 days."


class TestStructuralShortcuts:
    def test_an_uncited_claim_costs_no_model_call(self):
        verifier, model, _ = build(lambda p, h: ENTAIL)
        out = verifier({"claims": [{"claim": "No citation here.",
                                    "citations": [], "evidence": []}]})
        assert out["claims"][0]["verdict"] == "uncited"
        assert model.calls == 0

    def test_evidence_without_text_is_uncited(self):
        verifier, model, _ = build(lambda p, h: ENTAIL)
        out = verifier(payload("A claim [1].", "   "))
        assert out["claims"][0]["verdict"] == "uncited"
        assert model.calls == 0

    def test_citation_markers_are_stripped_from_the_hypothesis(self):
        """`[1]` is addressing information for the reader; no premise can
        entail it, and leaving it in only adds tokens that never match."""
        verifier, _, tokenizer = build(lambda p, h: ENTAIL)
        verifier(payload("Refunds take 30 days [1].", "Refunds take 30 days."))
        assert tokenizer.pairs[0][1] == "Refunds take 30 days."


class TestThreshold:
    def test_argmax_is_the_default(self):
        verifier, _, _ = build(lambda p, h: ENTAIL)
        assert verifier.threshold is None

    def test_a_high_threshold_withholds_the_verdict(self):
        """Miscalibrated confidence is the normal state for NLI, so the knob
        exists — but the default must not embed somebody else's constant."""
        verifier, _, _ = build(lambda p, h: ENTAIL, threshold=0.999)
        out = verifier(payload("Refunds take 30 days [1].", "Refunds take 30 days."))
        assert out["claims"][0]["verdict"] == "unsupported"

    @pytest.mark.parametrize("bad", [0.0, 1.5, -0.2])
    def test_impossible_thresholds_are_refused(self, bad):
        with pytest.raises(ConfigurationError, match="threshold must be"):
            build(lambda p, h: ENTAIL, threshold=bad)

    def test_unknown_granularity_is_refused(self):
        with pytest.raises(ConfigurationError, match="unknown granularity"):
            build(lambda p, h: ENTAIL, granularity="paragraph")


class TestCalibration:
    def test_it_reports_a_threshold_and_the_sweep(self):
        def rule(premise, hypothesis):
            return ENTAIL if "30 days" in premise else CONTRADICT

        verifier, _, _ = build(rule)
        result = calibrate_threshold(verifier, [
            ("Refunds take 30 days.", "Refunds take 30 days.", "supported"),
            ("Orders ship in five.", "Refunds take 90 days.", "contradicted"),
        ])
        assert result["n"] == 2
        assert 0.0 < result["threshold"] <= 1.0
        assert result["balanced_accuracy"] == 1.0
        assert len(result["sweep"]) == 19

    def test_it_refuses_to_guess_without_data(self):
        verifier, _, _ = build(lambda p, h: ENTAIL)
        with pytest.raises(ConfigurationError, match="needs labelled examples"):
            calibrate_threshold(verifier, [])


class TestEndToEnd:
    @pytest.fixture
    def kb(self, tmp_path):
        base = ragvault.open(tmp_path / "kb")
        base.add([
            {"id": "refund", "text":
             "Refund requests must be filed within 30 days of purchase."},
            {"id": "shipping", "text":
             "Orders ship within five business days nationwide."},
        ])
        yield base
        base.close()

    def test_repair_removes_only_the_contradicted_claim(self, kb):
        def rule(premise, hypothesis):
            if "90 days" in hypothesis:
                return CONTRADICT
            return ENTAIL if "30 days" in premise else NEUTRAL

        verifier, _, _ = build(rule)
        answer = kb.ask(
            "refund",
            llm=lambda p: "Refunds take 30 days [1]. Refunds take 90 days [1].",
            verify=verifier, verification_mode="repair", k=2,
        )
        assert "Refunds take 30 days [1]." in answer.text
        assert "90 days" not in answer.text
        assert answer.verification.counts()["contradicted"] == 1

    def test_the_adapter_survives_its_own_quote_check(self, kb):
        """`supported` carries a quote, and the ground check re-verifies it
        against the cited block. An adapter whose quotes did not survive that
        would report support the library then rejects."""
        verifier, _, _ = build(lambda p, h: ENTAIL)
        answer = kb.ask("refund", llm=lambda p: "Refunds take 30 days [1].",
                        verify=verifier, k=2)
        assert answer.verification.ok
        assert answer.verification.claims[0].verdict == "supported"
        assert answer.verification.structural_issues == []

    def test_factory_returns_a_usable_verifier(self):
        verifier = nli_verifier(
            StubModel(lambda p, h: ENTAIL), tokenizer=StubTokenizer(),
        )
        assert isinstance(verifier, NLIVerifier)
        assert verifier.granularity == "sentence"


class TestBenchmarkHarness:
    """The benchmark cannot run here (huggingface.co is blocked in CI), so its
    scoring and reporting would otherwise ship untested — and a reporting bug
    would only surface on the machine that finally runs it."""

    @pytest.fixture
    def bench(self):
        import importlib.util
        from pathlib import Path

        path = (Path(__file__).resolve().parents[2]
                / "benchmarks" / "bench_nli_verifier.py")
        spec = importlib.util.spec_from_file_location("bench_nli", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_perfect_verifier_scores_perfectly(self, bench):
        pairs = bench.load_pairs(None)
        assert len({(r["claim"], r["premise"]) for r in pairs}) == len(pairs), (
            "a (claim, premise) pair must appear once, or it carries two labels"
        )

        def verifier(payload):
            claim = payload["claims"][0]
            text = claim["evidence"][0]["text"]
            for row in pairs:
                # Padding is appended, so the bare premise stays a prefix.
                if row["claim"] == claim["claim"] and text.startswith(row["premise"]):
                    return {"claims": [{"verdict": row["label"]}]}
            raise AssertionError(f"unmatched pair: {claim['claim'][:50]!r}")

        result = bench.run_config(verifier, pairs, pad=False)
        assert result["accuracy"] == 1.0
        assert result["false_contradicted"] == 0.0
        assert set(result["by_lang"]) == {"en", "es", "pt"}

    def test_false_contradicted_counts_only_deletable_claims(self, bench):
        """The gate metric: claims that are not contradictions but were called
        one. Real contradictions must not dilute it."""
        pairs = bench.load_pairs(None)
        result = bench.run_config(
            lambda payload: {"claims": [{"verdict": "contradicted"}]},
            pairs, pad=False,
        )
        assert result["false_contradicted"] == 1.0
        assert result["eligible_n"] == sum(
            1 for r in pairs if r["label"] != "contradicted"
        )

    def test_the_report_renders_from_real_results(self, bench):
        pairs = bench.load_pairs(None)
        results = {
            name: bench.run_config(
                lambda payload: {"claims": [{"verdict": "supported"}]}, pairs, pad,
            )
            for name, pad in (
                ("sentence / bare premise", False),
                ("sentence / padded premise", True),
                ("block / padded premise", True),
            )
        }
        text = bench.render(results, pairs, "stub-model")
        assert "false-contradicted" in text
        assert "sentence / padded premise" in text
        assert "Accuracy by language" in text
        assert "nan" not in text

    def test_the_dataset_is_well_formed(self, bench):
        pairs = bench.load_pairs(None)
        assert len(pairs) >= 30
        for row in pairs:
            assert row["label"] in ("supported", "contradicted", "unsupported")
            assert row["premise"].strip() and row["claim"].strip()
        # Every label represented in every language, or the per-language rows
        # would compare different tasks.
        for lang in {r["lang"] for r in pairs}:
            labels = {r["label"] for r in pairs if r["lang"] == lang}
            assert labels == {"supported", "contradicted", "unsupported"}, lang
