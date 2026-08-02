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
#: List markers count as a boundary too — without them a bulleted answer is a
#: single claim, so one bad bullet would condemn (or spare) the whole list.
_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\[]|[-*•+]\s|\d+[.)]\s)"
)


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
    #: Verdict of the second pass over the replacement itself, when one ran.
    replacement_verdict: Optional[str] = None
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
            "replacement_verdict": self.replacement_verdict,
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
    #: Set when the second pass over replacements raised. The repaired answer
    #: stands, but its replacements went unchecked — surfaced, not hidden.
    recheck_error: Optional[str] = None
    #: Per-facet coverage, when the caller decomposed the question and the
    #: verifier reported it. Fidelity (claims) and completeness (facets) are
    #: separate axes: an answer can be fully supported and still incomplete.
    facet_coverage: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """True when no claim was judged unsupported or contradicted."""
        return not any(c.is_problem for c in self.claims)

    @property
    def unsupported(self) -> list[ClaimVerification]:
        return [c for c in self.claims if c.is_problem]

    @property
    def uncovered_facets(self) -> list[dict]:
        """Facets the verifier judged unaddressed. Empty when the verifier
        did not report coverage — absence of a report is not evidence of
        coverage, so `complete` distinguishes the two."""
        return [f for f in self.facet_coverage if not f["covered"]]

    @property
    def complete(self) -> Optional[bool]:
        """True/False when facet coverage was reported, None when it was not."""
        if not self.facet_coverage:
            return None
        return not self.uncovered_facets

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "claims": [c.to_dict() for c in self.claims],
            "facet_coverage": list(self.facet_coverage),
            "complete": self.complete,
            "error": self.error,
            "recheck_error": self.recheck_error,
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
    return _split_with_separators(text)[0]


def _split_with_separators(text: str) -> tuple[list[str], list[str]]:
    """Split into claims *and* the whitespace that separated them.

    Returned so ``repair`` can rebuild the answer with its original layout:
    an answer written as a bulleted list or as paragraphs must not come back
    as one run-on line just because a claim was dropped. Invariant:
    ``claims[0] + seps[0] + claims[1] + ... == text.strip()``.
    """
    stripped = text.strip()
    if not stripped:
        return [], []
    claims: list[str] = []
    seps: list[str] = []
    pos = 0
    for match in _SENTENCE_RE.finditer(stripped):
        claims.append(stripped[pos:match.start()])
        seps.append(match.group(0))
        pos = match.end()
    claims.append(stripped[pos:])
    # Drop empties while keeping the separator alignment consistent.
    out_claims: list[str] = []
    out_seps: list[str] = []
    for i, claim in enumerate(claims):
        if not claim.strip():
            continue
        if out_claims:
            out_seps.append(seps[i - 1] if i - 1 < len(seps) else " ")
        out_claims.append(claim.strip())
    return out_claims, out_seps


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
            # Precedence-deciding fields travel with the evidence: without
            # them a verifier cannot tell a current rule from a revoked one.
            "metadata": dict(citation.metadata),
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
    facets: Optional[Sequence[str]] = None,
) -> VerificationReport:
    """Run the verification pass. See the module docstring for the contract.

    ``verify`` is called with a payload describing the question, the answer,
    the context, every claim with its resolved evidence, and (when the caller
    decomposed the question) the ``facets`` the answer was supposed to cover.
    It must return one verdict per claim: a list, or a dict with a ``claims``
    key and an optional ``facets`` list reporting per-facet coverage.
    """
    if mode not in VERIFICATION_MODES:
        raise ConfigurationError(
            f"unknown verification_mode {mode!r}; available: {list(VERIFICATION_MODES)}"
        )
    if not callable(verify):
        raise ConfigurationError("verify must be a callable")

    report = VerificationReport(mode=mode, repaired_text=answer_text)
    started = time.monotonic()

    claims, separators = _split_with_separators(answer_text)
    if not claims:
        report.elapsed_ms = (time.monotonic() - started) * 1000
        return report

    def build_payload(items: Sequence[str]) -> dict:
        return {
            "question": question,
            "answer": answer_text,
            "context": context,
            "claims": [
                {
                    "claim": claim,
                    "citations": citations_in(claim),
                    "evidence": _evidence_for(citations_in(claim), citations),
                }
                for claim in items
            ],
            "facets": list(facets or []),
            "verdicts": sorted(VERDICTS),
        }

    def run(items: Sequence[str]) -> tuple[list, Optional[list]]:
        """Call the verifier and normalize its shape."""
        raw = verify(build_payload(items))
        facet_report = None
        if isinstance(raw, dict):
            facet_report = raw.get("facets")
            raw = raw.get("claims", raw.get("verdicts"))
        if raw is None:
            raise ValueError("verifier returned no claim verdicts")
        results = list(raw)
        if len(results) != len(items):
            raise ValueError(
                f"verifier returned {len(results)} verdicts for {len(items)} claims"
            )
        return results, facet_report

    try:
        results, facet_report = run(claims)
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
        report.facet_coverage = _coerce_facets(facet_report, facets)
    except ConfigurationError:
        raise  # caller error (bad verdict/shape) — surface it, don't swallow
    except Exception as exc:
        # A broken verifier must never destroy a valid answer.
        report.error = f"{type(exc).__name__}: {exc}"
        report.claims = []
        report.repaired_text = answer_text
        report.elapsed_ms = (time.monotonic() - started) * 1000
        return report

    report.repaired_text = _apply_mode(report.claims, mode, answer_text, separators)

    # Second pass over rewritten claims only. The verifier's `replacement` is
    # generated text that entered the answer without ever being checked — the
    # repair could itself introduce an unsupported statement. Exactly one
    # extra pass (never a loop): a replacement that fails re-verification is
    # dropped rather than replaced again.
    rewritten = [c for c in report.claims if c.action == "rewritten" and c.replacement]
    if rewritten and mode in ("repair", "strict"):
        try:
            recheck, _ = run([c.replacement for c in rewritten])  # type: ignore[misc]
            for claim, item in zip(rewritten, recheck):
                verdict, rationale, _ = _coerce_verdict(item, claim.replacement or "")
                claim.replacement_verdict = verdict
                if verdict in _PROBLEM_VERDICTS:
                    claim.replacement = None
                    claim.action = "removed"
                    claim.rationale = (
                        f"{claim.rationale} | replacement also rejected "
                        f"({verdict}): {rationale}"
                    ).strip(" |")
            report.repaired_text = _apply_mode(
                report.claims, mode, answer_text, separators
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            # Re-verification is a safety net, not a gate: if it fails, keep
            # the repaired answer and say the replacements went unchecked.
            report.recheck_error = f"{type(exc).__name__}: {exc}"

    report.elapsed_ms = (time.monotonic() - started) * 1000
    return report


def _coerce_facets(
    raw: object, facets: Optional[Sequence[str]]
) -> list[dict]:
    """Normalize optional per-facet coverage reported by the verifier."""
    if not raw or not facets:
        return []
    out: list[dict] = []
    for item in list(raw)[: len(facets)]:
        if isinstance(item, dict):
            out.append({
                "facet": str(item.get("facet", "")),
                "covered": bool(item.get("covered", False)),
                "rationale": str(item.get("rationale", "") or ""),
            })
        else:
            out.append({"facet": str(item), "covered": True, "rationale": ""})
    return out


def _apply_mode(
    claims: list[ClaimVerification], mode: str, original: str,
    separators: Sequence[str],
) -> str:
    """Rebuild the answer according to the mode, recording each action.

    The original separators are reused so bullet lists and paragraph breaks
    survive a repair; a dropped claim takes its preceding separator with it.
    """
    if mode == "report":
        for claim in claims:
            claim.action = "kept"
        return original

    kept: list[tuple[str, str]] = []  # (separator before, text)
    for i, claim in enumerate(claims):
        before = separators[i - 1] if 0 < i <= len(separators) else ""
        drop = claim.is_problem or (mode == "strict" and claim.verdict == UNCITED)

        if mode == "annotate":
            if claim.is_problem:
                tag = ("[unsupported]" if claim.verdict == UNSUPPORTED
                       else "[contradicted]")
                kept.append((before, f"{claim.claim} {tag}"))
                claim.action = "annotated"
            else:
                kept.append((before, claim.claim))
                claim.action = "kept"
            continue

        # repair / strict
        if drop:
            if claim.replacement:
                kept.append((before, claim.replacement.strip()))
                claim.action = "rewritten"
            else:
                claim.action = "removed"
            continue
        kept.append((before, claim.claim))
        claim.action = "kept"

    parts: list[str] = []
    for i, (before, text) in enumerate(kept):
        if i:
            parts.append(before or " ")
        parts.append(text)
    text = "".join(parts).strip()
    if not text:
        # Everything was removed: say so instead of returning an empty answer.
        return (
            "No statement in the generated answer was supported by the "
            "retrieved context."
        )
    return text
