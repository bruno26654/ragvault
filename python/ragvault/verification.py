"""Post-generation semantic validation for ``ask()`` and ``ask_multi()``.

Retrieval integrity (only real chunks can be cited) and citation-marker
sanity (no `[n]` beyond the context) are enforced upstream. What neither can
catch is the harder failure: a citation that *exists* but does not actually
support the claim attached to it, a claim that silently restates a premise
from the user's question as if a document said it, or a claim contradicted by
the source it cites.

This module adds an optional verification pass over the generated answer:

1. **Split** the answer into claims (sentence-level by default; a verifier may
   supply its own segmentation).
2. **Attach** the `[n]` markers found in each claim to the citations they
   refer to, resolved back to the real chunks that produced them.
3. **Judge** each claim with a caller-supplied verifier — typically an LLM, but
   any callable works. The verifier returns a verdict per claim:
   ``supported`` / ``unsupported`` / ``contradicted`` / ``uncited`` /
   ``question_fact`` / ``inference``, with a rationale.
4. **Act** according to ``verification_mode``:
   - ``"report"`` — change nothing, attach the report (default when verifying);
   - ``"annotate"`` — mark problem claims inline with a visible tag;
   - ``"repair"`` — drop unsupported/contradicted claims, or use the
     verifier's ``replacement`` text when it offers one;
   - ``"strict"`` — like repair, but also drops claims that cite nothing.

RagVault never calls a provider itself: ``verify`` is a plain callable. If the
verifier raises, the original answer is preserved unchanged and the failure is
recorded — a broken verifier must never destroy a valid answer.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .context import Citation
from .errors import ConfigurationError

#: Verdicts a verifier may return. Anything else is rejected with an
#: actionable error rather than silently treated as "fine".
SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
CONTRADICTED = "contradicted"
UNCITED = "uncited"
QUESTION_FACT = "question_fact"
INFERENCE = "inference"

VERDICTS = frozenset({
    SUPPORTED, UNSUPPORTED, CONTRADICTED, UNCITED, QUESTION_FACT, INFERENCE,
})

#: Verdicts that `repair`/`strict` remove or replace.
_PROBLEM_VERDICTS = frozenset({UNSUPPORTED, CONTRADICTED})

VERIFICATION_MODES = ("report", "annotate", "repair", "strict")

_CITATION_RE = re.compile(r"\[(\d+)\]")
#: Sentence-ish split that keeps the terminator with the sentence and does not
#: break on decimals or common abbreviations followed by a lowercase word.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


@dataclass
class ClaimVerification:
    """One claim of the answer and what the verifier concluded about it."""

    claim: str
    citations: list[int]
    verdict: str
    rationale: str = ""
    #: Chunk ids backing the cited blocks — the concrete evidence examined.
    chunk_ids: list[str] = field(default_factory=list)
    #: Text the verifier proposed instead (used by `repair`/`strict`).
    replacement: Optional[str] = None
    #: What the pipeline actually did: kept / annotated / rewritten / removed.
    action: str = "kept"

    @property
    def is_problem(self) -> bool:
        return self.verdict in _PROBLEM_VERDICTS

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "citations": list(self.citations),
            "verdict": self.verdict,
            "rationale": self.rationale,
            "chunk_ids": list(self.chunk_ids),
            "replacement": self.replacement,
            "action": self.action,
        }


@dataclass
class VerificationReport:
    """Outcome of the verification pass."""

    mode: str
    claims: list[ClaimVerification] = field(default_factory=list)
    #: Text after applying the mode (identical to the input in `report` mode).
    repaired_text: str = ""
    #: Set when the verifier raised: the answer is preserved untouched.
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """True when no claim was judged unsupported or contradicted."""
        return not any(c.is_problem for c in self.claims)

    @property
    def unsupported(self) -> list[ClaimVerification]:
        return [c for c in self.claims if c.is_problem]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "claims": [c.to_dict() for c in self.claims],
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "counts": self.counts(),
        }

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for claim in self.claims:
            out[claim.verdict] = out.get(claim.verdict, 0) + 1
        return out

    def __repr__(self) -> str:
        return (
            f"VerificationReport(mode={self.mode!r}, ok={self.ok}, "
            f"claims={len(self.claims)}, counts={self.counts()})"
        )


def split_claims(text: str) -> list[str]:
    """Split an answer into claim-sized units (sentences, terminator kept)."""
    stripped = text.strip()
    if not stripped:
        return []
    return [part.strip() for part in _SENTENCE_RE.split(stripped) if part.strip()]


def citations_in(claim: str) -> list[int]:
    """Citation indices referenced by a claim, in order, deduplicated."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(claim):
        index = int(match.group(1))
        if index not in seen:
            seen.append(index)
    return seen


def _evidence_for(indices: Sequence[int], citations: Sequence[Citation]) -> list[dict]:
    """Resolve `[n]` markers to the real citations they point at."""
    by_index = {c.index: c for c in citations}
    evidence = []
    for index in indices:
        citation = by_index.get(index)
        if citation is None:
            continue  # marker beyond the context; sanitizers strip these
        evidence.append({
            "index": citation.index,
            "document_id": citation.document_id,
            "document_version": citation.document_version,
            "chunk_ids": list(citation.chunk_ids),
            "title": citation.title,
        })
    return evidence


def _coerce_verdict(raw: object, claim: str) -> tuple[str, str, Optional[str]]:
    """Normalize one verifier result into (verdict, rationale, replacement)."""
    if isinstance(raw, str):
        verdict, rationale, replacement = raw, "", None
    elif isinstance(raw, dict):
        verdict = str(raw.get("verdict", "")).strip().lower()
        rationale = str(raw.get("rationale", "") or "")
        replacement = raw.get("replacement")
        replacement = None if replacement is None else str(replacement)
    else:
        raise ConfigurationError(
            "verifier must return a verdict string or a dict with a 'verdict' "
            f"key per claim, got {type(raw).__name__} for claim {claim[:60]!r}"
        )
    verdict = verdict.strip().lower()
    if verdict not in VERDICTS:
        raise ConfigurationError(
            f"unknown verdict {verdict!r} for claim {claim[:60]!r}; "
            f"expected one of {sorted(VERDICTS)}"
        )
    return verdict, rationale, replacement


def verify_answer(
    *,
    question: str,
    answer_text: str,
    context: str,
    citations: Sequence[Citation],
    verify: Callable,
    mode: str = "report",
) -> VerificationReport:
    """Run the verification pass. See the module docstring for the contract.

    ``verify`` is called once with a payload describing the question, the
    answer, the context and every claim with its resolved evidence, and must
    return one verdict per claim (a list, or a dict with a ``claims`` key).
    """
    if mode not in VERIFICATION_MODES:
        raise ConfigurationError(
            f"unknown verification_mode {mode!r}; available: {list(VERIFICATION_MODES)}"
        )
    if not callable(verify):
        raise ConfigurationError("verify must be a callable")

    report = VerificationReport(mode=mode, repaired_text=answer_text)
    started = time.monotonic()

    claims = split_claims(answer_text)
    if not claims:
        report.elapsed_ms = (time.monotonic() - started) * 1000
        return report

    payload = {
        "question": question,
        "answer": answer_text,
        "context": context,
        "claims": [
            {
                "claim": claim,
                "citations": citations_in(claim),
                "evidence": _evidence_for(citations_in(claim), citations),
            }
            for claim in claims
        ],
        "verdicts": sorted(VERDICTS),
    }

    try:
        raw = verify(payload)
        if isinstance(raw, dict):
            raw = raw.get("claims", raw.get("verdicts"))
        if raw is None:
            raise ValueError("verifier returned no claim verdicts")
        results = list(raw)
        if len(results) != len(claims):
            raise ValueError(
                f"verifier returned {len(results)} verdicts for {len(claims)} claims"
            )
        for claim, item in zip(claims, results):
            verdict, rationale, replacement = _coerce_verdict(item, claim)
            indices = citations_in(claim)
            report.claims.append(ClaimVerification(
                claim=claim,
                citations=indices,
                verdict=verdict,
                rationale=rationale,
                chunk_ids=[
                    cid for ev in _evidence_for(indices, citations)
                    for cid in ev["chunk_ids"]
                ],
                replacement=replacement,
            ))
    except ConfigurationError:
        raise  # caller error (bad verdict/shape) — surface it, don't swallow
    except Exception as exc:
        # A broken verifier must never destroy a valid answer.
        report.error = f"{type(exc).__name__}: {exc}"
        report.claims = []
        report.repaired_text = answer_text
        report.elapsed_ms = (time.monotonic() - started) * 1000
        return report

    report.repaired_text = _apply_mode(report.claims, mode, answer_text)
    report.elapsed_ms = (time.monotonic() - started) * 1000
    return report


def _apply_mode(
    claims: list[ClaimVerification], mode: str, original: str
) -> str:
    """Rebuild the answer according to the mode, recording each action."""
    if mode == "report":
        for claim in claims:
            claim.action = "kept"
        return original

    kept: list[str] = []
    for claim in claims:
        drop = claim.is_problem or (mode == "strict" and claim.verdict == UNCITED)

        if mode == "annotate":
            if claim.is_problem:
                tag = ("[unsupported]" if claim.verdict == UNSUPPORTED
                       else "[contradicted]")
                kept.append(f"{claim.claim} {tag}")
                claim.action = "annotated"
            else:
                kept.append(claim.claim)
                claim.action = "kept"
            continue

        # repair / strict
        if drop:
            if claim.replacement:
                kept.append(claim.replacement.strip())
                claim.action = "rewritten"
            else:
                claim.action = "removed"
            continue
        kept.append(claim.claim)
        claim.action = "kept"

    text = " ".join(kept).strip()
    if not text:
        # Everything was removed: say so instead of returning an empty answer.
        return (
            "No statement in the generated answer was supported by the "
            "retrieved context."
        )
    return text
