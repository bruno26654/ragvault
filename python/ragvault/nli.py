"""Offline fidelity checking with a Natural Language Inference model.

Every other check in RagVault is *structural*: a quote is a substring of its
source, a verdict names an admissible ground, spans are ordered and
non-overlapping. None of them can answer the question that matters most — does
the cited evidence actually **entail** the claim? Until now only a
caller-supplied LLM judge could, which left the default offline install with no
fidelity check at all.

NLI answers it offline, on CPU, and its three labels map onto the existing
verdict taxonomy without inventing anything::

    entailment    -> supported
    contradiction -> contradicted
    neutral       -> unsupported

Usage — the result is an ordinary ``verify=`` callable, so nothing in the core
changes::

    from ragvault.nli import nli_verifier

    answer = kb.ask(question, llm=my_llm, verify=nli_verifier())

What this adapter does **not** do, stated plainly because a verifier that
quietly under-reports is worse than none:

* It never returns ``question_fact`` or ``inference``. Both require reading
  the user's question as a source of truth, which NLI does not do. Claims that
  are really inferences will come back ``unsupported``.
* It reports no facet coverage, so with facets declared ``complete`` is
  ``False``. Pass ``facets=[]``, or pair this with an LLM judge for the
  completeness axis. Fidelity and completeness are separate axes on purpose.
* Its accuracy is **not uniform across languages**. A multilingual checkpoint
  inherits ~100 pretraining languages from XLM-R but only the ~15 that XNLI
  fine-tunes on. Outside those, behaviour is unmeasured — run
  ``benchmarks/bench_nli_verifier.py`` on your own corpus before trusting it.

**Long premises are where NLI degrades**, and RAG chunks are long. The fix is
not ours: SummaC (Laban et al., TACL 2022) showed across six datasets that
splitting the premise into sentences and scoring each against the claim
recovers most of the loss. That is the default here (``granularity="sentence"``)
and it is why the sentence splitter's script coverage matters so much — a
language the splitter cannot segment silently falls back to whole-block
premises, which is the degraded mode.

Measured, and it changes what this is for
-----------------------------------------

``benchmarks/RESULTS-VERIFICATION.md``, 36 labelled pairs, en/es/pt,
mDeBERTa-v3-base-xnli on CPU. On a *bare* premise the adapter is good: 0.89
accuracy, 4% false-contradicted. On a **padded premise — a realistic retrieved
chunk — it is not**: accuracy 0.78 and the false-contradicted rate rises to
**21%**, meaning roughly one correct claim in five would be deleted by
``repair``. Whole-block granularity is worse on both counts (0.72 / 25%), so
this is not a granularity that can be tuned around.

The mechanism is visible in the table: ``unsupported`` keeps precision 1.00 but
its recall collapses from 0.64 to 0.27. The model does not stop being right
when it speaks — it stops saying "neutral" once the chunk contains competing
content, and what should have been neutral is asserted as a contradiction.

So the adapter **refuses ``repair`` and ``strict`` by default**
(``allow_repair=True`` to override once you have measured your own corpus). It
reports; it does not delete. A wrong verdict in ``report``/``annotate`` is a
visible mislabel, and that is a trade worth making — a wrong verdict in
``repair`` is a correct sentence that silently disappears, which is the failure
this whole module exists to prevent.

**It is also slow.** p50 is ~10 s per claim on CPU with a bare premise and
~22 s with a realistic one, so a five-claim answer costs minutes. Block
granularity halves it and costs accuracy. Batch across claims, use a GPU, or
treat this as an offline audit pass rather than something in the request path.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .errors import ConfigurationError
from .verification import (
    CONTRADICTED,
    SUPPORTED,
    UNCITED,
    UNSUPPORTED,
    _CITATION_RE,
    split_claims,
)

#: Multilingual by default: an English-only checkpoint would quietly mislabel
#: every other language rather than fail, which is the worse failure.
DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

#: Canonical order used internally for the score matrix.
ENTAILMENT, NEUTRAL, CONTRADICTION = 0, 1, 2

_LABEL_KEYS = (
    (ENTAILMENT, "entail"),
    (NEUTRAL, "neutral"),
    (CONTRADICTION, "contradict"),
)


def _label_positions(id2label: dict) -> list[int]:
    """Map canonical (entail, neutral, contradict) onto the model's own order.

    Read by *name*, never by index. Checkpoints disagree about ordering — some
    put contradiction at 0, others at 2 — so a hardcoded position silently
    inverts every verdict on half the models available. `id2label` is the
    model's own statement about itself and costs nothing to consult.
    """
    lowered = {int(idx): str(name).lower() for idx, name in id2label.items()}
    positions: list[int] = []
    for _, key in _LABEL_KEYS:
        matches = [idx for idx, name in lowered.items() if key in name]
        if len(matches) != 1:
            raise ConfigurationError(
                f"cannot map NLI labels: expected exactly one label containing "
                f"{key!r} in id2label={id2label!r}. Models with opaque labels "
                "(LABEL_0, LABEL_1…) cannot be used safely — the mapping would "
                "be a guess, and a wrong guess inverts every verdict."
            )
        positions.append(matches[0])
    return positions


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _as_numpy(value: Any) -> np.ndarray:
    """Tensor or array-like to numpy, without importing torch to find out."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


#: Space left stranded in front of punctuation once a marker is removed.
_ORPHAN_SPACE_RE = re.compile(r"\s+(?=[.!?,;:)\]}])")


def _hypothesis(claim: str) -> str:
    """The claim as a standalone sentence: citation markers are addressing
    information for the reader, not part of what is being asserted, and
    leaving them in feeds the model tokens the premise can never entail.

    Removing "[1]" from "…30 days [1]." leaves "…30 days ." — a stranded space
    that tokenizes as its own token and makes the hypothesis differ from the
    premise for a reason that has nothing to do with meaning.
    """
    bare = " ".join(_CITATION_RE.sub("", claim).split())
    return _ORPHAN_SPACE_RE.sub("", bare)


class NLIVerifier:
    """Callable that judges claim fidelity with an NLI model. See module docs."""

    def __init__(
        self,
        model: Any = DEFAULT_MODEL,
        *,
        tokenizer: Any = None,
        granularity: str = "sentence",
        threshold: Optional[float] = None,
        batch_size: int = 16,
        max_length: int = 512,
        device: Optional[str] = None,
        allow_repair: bool = False,
    ) -> None:
        #: Refuses `repair`/`strict` by default — see the module docstring for
        #: the measurement. Opt in once you have run the benchmark on your own
        #: corpus and the false-contradicted rate is acceptable to you.
        self.destructive_modes_allowed = bool(allow_repair)
        if granularity not in ("sentence", "block"):
            raise ConfigurationError(
                f"unknown granularity {granularity!r}; expected 'sentence' or 'block'"
            )
        if threshold is not None and not 0.0 < threshold <= 1.0:
            raise ConfigurationError(
                f"threshold must be in (0, 1], got {threshold!r}"
            )
        self.granularity = granularity
        self.threshold = threshold
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        if isinstance(model, str):
            try:
                from transformers import (  # type: ignore[import-not-found]
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError(
                    f"the NLI verifier for {model!r} requires the optional "
                    'dependency: pip install "ragvault[nli]"'
                ) from exc
            self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
            self._model = AutoModelForSequenceClassification.from_pretrained(model)
            self.model_id = model
            if device is not None:  # pragma: no cover - hardware dependent
                self._model = self._model.to(device)
            with contextlib.suppress(AttributeError):  # pragma: no cover
                self._model.eval()
        else:
            if tokenizer is None:
                raise ConfigurationError(
                    "when passing a preloaded model, its tokenizer must be "
                    "passed too — they are a matched pair"
                )
            self._model, self._tokenizer = model, tokenizer
            self.model_id = getattr(model, "name_or_path", "preloaded")

        config = getattr(self._model, "config", None)
        id2label = getattr(config, "id2label", None)
        if not id2label:
            raise ConfigurationError(
                "the NLI model exposes no config.id2label, so its labels "
                "cannot be mapped onto verdicts by name"
            )
        self._positions = _label_positions(id2label)

        try:  # torch is only needed to disable gradients; absence is not fatal
            import torch  # type: ignore[import-not-found]

            self._no_grad: Callable[[], Any] = torch.no_grad
        except ImportError:  # pragma: no cover - depends on extras
            self._no_grad = contextlib.nullcontext  # type: ignore[assignment]

    # -- scoring -----------------------------------------------------------

    def score(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """Probabilities per pair, as ``[n, 3]`` in (entail, neutral, contradict)."""
        if not premises:
            return np.zeros((0, 3))
        out = []
        for start in range(0, len(premises), self.batch_size):
            chunk_p = list(premises[start:start + self.batch_size])
            chunk_h = list(hypotheses[start:start + self.batch_size])
            inputs = self._tokenizer(
                chunk_p, chunk_h, return_tensors="pt", truncation=True,
                padding=True, max_length=self.max_length,
            )
            with self._no_grad():
                logits = _as_numpy(self._model(**inputs).logits)
            out.append(_softmax(logits)[:, self._positions])
        return np.vstack(out)

    # -- verdicts ----------------------------------------------------------

    def _premises(self, text: str) -> list[str]:
        """Premise units for one cited block.

        Sentence granularity is SummaC-ZS: NLI degrades on long premises, and
        scoring each sentence separately recovers most of it. When the splitter
        cannot segment a language it returns one unit, which is the honest
        fallback — the same text, scored the degraded way, not an error.
        """
        text = (text or "").strip()
        if not text:
            return []
        if self.granularity == "block":
            return [text]
        return split_claims(text) or [text]

    def _judge(self, claim: str, evidence: Sequence[dict]) -> dict:
        hypothesis = _hypothesis(claim)
        blocks = [(ev, self._premises(ev.get("text") or "")) for ev in evidence]
        premises = [p for _, units in blocks for p in units]
        if not hypothesis or not premises:
            return {
                "verdict": UNCITED,
                "rationale": "the claim cites no source with usable text",
            }

        scores = self.score(premises, [hypothesis] * len(premises))
        if self.threshold is None:
            labels = scores.argmax(axis=1)
        else:
            labels = np.full(len(premises), NEUTRAL)
            labels[scores[:, CONTRADICTION] >= self.threshold] = CONTRADICTION
            labels[scores[:, ENTAILMENT] >= self.threshold] = ENTAILMENT

        # Aggregation is existential, which is what makes `max` the right
        # operator rather than a tuning choice: one source sentence entailing
        # the claim is enough to support it.
        #
        # Entailment is checked *before* contradiction on purpose. Scanning
        # many premise sentences, some unrelated one eventually scores as a
        # contradiction; letting that outrank a real entailment would delete
        # correct text in `repair` mode. The two errors are not symmetric —
        # a missed contradiction leaves a wrong sentence in place and visible,
        # a false one silently removes a right sentence.
        best_entail = int(np.argmax(scores[:, ENTAILMENT]))
        best_contra = int(np.argmax(scores[:, CONTRADICTION]))
        if (labels == ENTAILMENT).any():
            idx, verdict = best_entail, SUPPORTED
        elif (labels == CONTRADICTION).any():
            idx, verdict = best_contra, CONTRADICTED
        else:
            idx, verdict = best_entail, UNSUPPORTED

        probs = scores[idx]
        result = {
            "verdict": verdict,
            "rationale": (
                f"NLI ({self.granularity}) over {len(premises)} premise unit(s): "
                f"entail={probs[ENTAILMENT]:.2f} neutral={probs[NEUTRAL]:.2f} "
                f"contradict={probs[CONTRADICTION]:.2f}"
            ),
        }
        # The winning premise is verbatim source text, so it doubles as the
        # `quote`: the structural layer then re-checks this adapter's own
        # output against the cited block, and a non-existent or unspecific
        # span is rejected there rather than trusted here.
        if verdict in (SUPPORTED, CONTRADICTED):
            result["quote"] = premises[idx]
        return result

    def __call__(self, payload: dict) -> dict:
        claims = payload.get("claims", [])
        out = []
        for item in claims:
            claim = str(item.get("claim", ""))
            evidence = item.get("evidence") or []
            if not evidence:
                # Structural, and decided without paying for a model call.
                out.append({
                    "verdict": UNCITED,
                    "rationale": "the claim cites no source",
                })
                continue
            out.append(self._judge(claim, evidence))
        return {"claims": out}


def nli_verifier(model: Any = DEFAULT_MODEL, **kwargs: Any) -> NLIVerifier:
    """Build an NLI-backed ``verify=`` callable. See the module docstring."""
    return NLIVerifier(model, **kwargs)


def calibrate_threshold(
    verifier: NLIVerifier,
    labelled: Sequence[tuple[str, str, str]],
    *,
    steps: int = 19,
) -> dict:
    """Pick a decision threshold on *your* data instead of inheriting one.

    ``labelled`` is a sequence of ``(premise, claim, expected_verdict)``, where
    the verdict is ``supported`` / ``contradicted`` / ``unsupported``.

    A fixed probability cutoff is consolidated practice — SummaC tunes one per
    dataset — but the constant itself is not transferable: NLI models are
    poorly calibrated and each corpus sits somewhere different. So the default
    decision rule here stays plain ``argmax``, and this helper exists to derive
    a threshold from labelled examples rather than borrowing someone else's
    number. Returns the best threshold, its balanced accuracy, and the full
    sweep so the shape of the curve is visible rather than just its peak.
    """
    if not labelled:
        raise ConfigurationError("calibrate_threshold needs labelled examples")
    premises = [p for p, _, _ in labelled]
    hypotheses = [_hypothesis(c) for _, c, _ in labelled]
    expected = [v for _, _, v in labelled]
    scores = verifier.score(premises, hypotheses)

    def balanced_accuracy(predicted: Sequence[str]) -> float:
        recalls = []
        for verdict in (SUPPORTED, CONTRADICTED, UNSUPPORTED):
            total = sum(1 for e in expected if e == verdict)
            if not total:
                continue
            hits = sum(
                1 for e, p in zip(expected, predicted) if e == verdict and p == verdict
            )
            recalls.append(hits / total)
        return float(np.mean(recalls)) if recalls else 0.0

    sweep = []
    for threshold in np.linspace(0.05, 0.95, steps):
        predicted = []
        for row in scores:
            if row[ENTAILMENT] >= threshold:
                predicted.append(SUPPORTED)
            elif row[CONTRADICTION] >= threshold:
                predicted.append(CONTRADICTED)
            else:
                predicted.append(UNSUPPORTED)
        sweep.append({
            "threshold": round(float(threshold), 3),
            "balanced_accuracy": round(balanced_accuracy(predicted), 4),
        })

    argmax_predicted = [
        (SUPPORTED, UNSUPPORTED, CONTRADICTED)[int(row.argmax())] for row in scores
    ]
    best = max(sweep, key=lambda row: row["balanced_accuracy"])
    return {
        "threshold": best["threshold"],
        "balanced_accuracy": best["balanced_accuracy"],
        "argmax_balanced_accuracy": round(balanced_accuracy(argmax_predicted), 4),
        "n": len(labelled),
        "sweep": sweep,
    }
