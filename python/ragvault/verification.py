"""Post-generation semantic validation for ``ask()`` and ``ask_multi()``.

Retrieval integrity (only real chunks can be cited) and citation-marker
sanity (no `[n]` beyond the context) are enforced upstream. What neither can
catch is the harder failure: a citation that *exists* but does not actually
support the claim attached to it, a claim that silently restates a premise
from the user's question as if a document said it, or a claim contradicted by
the source it cites.

This module adds an optional verification pass over the generated answer:

1. **Split** the answer into claims. The built-in segmentation is heuristic
   (sentence terminators, list markers, CJK/Arabic/Hebrew punctuation, an
   abbreviation guard). It cannot see two claims inside one sentence — for
   that a verifier may return its **own** segmentation, which costs no extra
   call since it is already an LLM reading the whole answer. Its claims must
   be verbatim substrings of the answer, so repair stays surgical.
2. **Attach** the `[n]` markers found in each claim to the citations they
   refer to, resolved back to the real chunks that produced them.
3. **Judge** each claim with a caller-supplied verifier — typically an LLM, but
   any callable works. The verifier returns a verdict per claim:
   ``supported`` / ``unsupported`` / ``contradicted`` / ``uncited`` /
   ``question_fact`` / ``inference``, with a rationale.
4. **Act** according to ``verification_mode``:
   - ``"report"`` — change nothing, attach the report (default when verifying);
   - ``"annotate"`` — mark problem claims inline with a visible tag;
   - ``"repair"`` — drop unsupported/contradicted claims;
   - ``"strict"`` — like repair, but also drops claims that cite nothing.

The verifier segments and classifies; it does not write. A ``replacement`` it
proposes is ignored unless the caller passes ``allow_replacements=True``,
because re-verifying one's own correction is self-endorsement — the second
opinion comes from the judge that produced the text, carrying whatever blind
spot produced it.

Two independent axes come out of this: ``ok`` (every claim holds) and
``complete`` (every facet the answer owed was fully covered). Both fail
closed — a crash, a partial return, an unevaluated facet, or structurally
unsound segmentation yields False, never a silent pass.

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

#: Abbreviations whose trailing period must not end a claim. Legal and
#: Portuguese text is dense with these ("Art. 5º", "Inc. II"), and splitting
#: there hands the verifier fragments instead of claims.
_ABBREVIATIONS = (
    "Art", "Arts", "Inc", "Ltda", "Ltd", "Cia", "Cf", "Ref", "Fig", "Eq",
    "Sr", "Sra", "Srta", "Dr", "Dra", "Prof", "Profa", "Mr", "Mrs", "Ms",
    "No", "Nº", "n", "p", "pp", "vs", "etc", "ed", "al", "par",
)
#: The split point sits *after* the period, so each guard must include it.
_ABBREV_GUARD = "".join(f"(?<!\\b{a}\\.)" for a in _ABBREVIATIONS)

#: What may start a new claim: a capital, a citation marker, a list bullet, or
#: any letter that has no lowercase form (Arabic, Hebrew, Devanagari, CJK…).
#: Requiring `[A-Z]` silently disabled per-claim verification for every script
#: without letter case.
_CLAIM_START = r"[A-Z\[]|[-*•+]\s|\d+[.)]\s|(?![a-zà-öø-ÿ])[^\W\d_]"

#: Citation markers sitting right after a terminator belong to the claim they
#: follow, not to the next one. Written as `... 30 days. [1]`, the marker used
#: to land on the *following* claim: the claim it actually sourced came back
#: `uncited` (and `strict` dropped it), the next claim was credited to a source
#: it never cited, and a trailing `[2]` became a "claim" that was only a
#: marker. `[ \t]*` and not `\s*` on purpose — a marker on the next line
#: stays with that line.
#:
#: Written as an atomic group (lookahead capture + backreference, since
#: possessive quantifiers need 3.11 and abi3 targets 3.9): allowed to
#: backtrack, the trail would give the markers back and let a terminal `[2]`
#: split off as a claim that is only a marker — the very failure this fixes.
_TRAILING_CITES = r"(?=(?P<trail>(?:[ \t]*\[\d+\])*))(?P=trail)"

#: Claim boundaries. Three alternatives, in order:
#:  1. Latin terminator + any trailing citation markers + whitespace +
#:     something that starts a new unit (capital, citation marker, or a list
#:     bullet — without bullets a whole list would be one claim, so one bad
#:     item condemned all of them);
#:  2. CJK/full-width terminators, where the space is optional and there is
#:     no case to look for — Chinese, Japanese and Korean answers were never
#:     split at all before, making per-claim verification a no-op for them;
#:  3. Arabic full stop / question mark, same reason.
_SENTENCE_RE = re.compile(
    rf"(?<=[.!?]){_ABBREV_GUARD}{_TRAILING_CITES}\s+(?={_CLAIM_START})"
    rf"|(?<=[。．！？]){_TRAILING_CITES.replace('trail', 'trail_cjk')}\s*"
    rf"|(?<=[؟۔]){_TRAILING_CITES.replace('trail', 'trail_rtl')}\s+"
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
    #: Span of the cited source the verifier says carries the support. Checked
    #: against the evidence text: a quote that is not in the source is the
    #: verifier attributing to it something it does not say, which no verdict
    #: of `supported` can survive.
    quote: Optional[str] = None
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
            "quote": self.quote,
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
    #: Which segmentation produced the claims: the built-in heuristic, or the
    #: verifier's own (which can split two claims inside one sentence).
    segmentation: str = "heuristic"
    #: Facets the answer was expected to cover (empty when the question was
    #: not decomposed). Kept so a *partial* coverage report can be detected:
    #: a facet the verifier never mentioned is not a covered facet.
    expected_facets: list[str] = field(default_factory=list)
    #: Structural defects in the verifier's own output — overlapping or
    #: out-of-order spans, unreported facets, answer text no claim covered.
    #: Separate from fidelity and coverage: the result can be structurally
    #: invalid regardless of what the verdicts say.
    structural_issues: list[str] = field(default_factory=list)
    #: Per-facet coverage, when the caller decomposed the question and the
    #: verifier reported it. Fidelity (claims) and completeness (facets) are
    #: separate axes: an answer can be fully supported and still incomplete.
    facet_coverage: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def valid(self) -> bool:
        """True when the verification ran to completion and its output was
        structurally sound. Fidelity and coverage are only meaningful when
        this holds."""
        return self.error is None and not self.structural_issues

    @property
    def ok(self) -> bool:
        """True only when verification completed, the result is structurally
        valid, and no claim was judged unsupported or contradicted.

        Deliberately fails closed: a crashed verifier used to leave `claims`
        empty, and `not any([])` is True — so a failure reported the answer as
        faithful. Anything that gates on `ok` would then ship unverified text.
        """
        if not self.valid:
            return False
        return not any(c.is_problem for c in self.claims)

    @property
    def unsupported(self) -> list[ClaimVerification]:
        return [c for c in self.claims if c.is_problem]

    @property
    def uncovered_facets(self) -> list[dict]:
        """Facets judged unaddressed, plus any expected facet the verifier
        never mentioned — an unreported facet is not a covered facet."""
        out = [f for f in self.facet_coverage if not f["covered"]]
        reported = {f["facet"] for f in self.facet_coverage}
        for facet in self.expected_facets:
            if facet not in reported:
                out.append({
                    "facet": facet, "covered": False,
                    "rationale": "the verifier did not report on this facet",
                })
        return out

    @property
    def complete(self) -> Optional[bool]:
        """True only when every expected facet was reported *and* covered.

        `None` means "unknown", and the only honest unknown is "no facets were
        expected". Once the caller declared facets, silence about them is not
        unknown — it is an unevaluated facet, which fails closed like a partial
        report: a verifier that ignored every facet has not shown the answer is
        complete, and neither has one that crashed.
        """
        if not self.expected_facets:
            return None
        if not self.valid:
            return False
        return not self.uncovered_facets

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "valid": self.valid,
            "structural_issues": list(self.structural_issues),
            "segmentation": self.segmentation,
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
            f"valid={self.valid}, complete={self.complete}, "
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
        # Citation markers trailing the terminator stay with the claim they
        # source; only what follows them is layout between claims.
        trail = (match.group("trail") or match.group("trail_cjk")
                 or match.group("trail_rtl") or "")
        claims.append(stripped[pos:match.start()] + trail)
        seps.append(stripped[match.start() + len(trail):match.end()])
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
            # The cited block's own text. A judge handed only the full context
            # has to find `[n]` in it by hand, and can just as easily justify
            # a claim from a block the claim never cited — the support has to
            # come from the source that was actually named.
            "text": citation.text,
        })
    return evidence


def _normalize_span(text: str) -> str:
    """Collapse whitespace and case for substring comparison.

    A quote is re-wrapped and re-cased on its way through a model; neither
    changes whose words they are. Nothing else is normalized — the words
    themselves must match.
    """
    return " ".join(text.split()).casefold()


#: What each verdict is allowed to rest on. A verdict is a claim *about* a
#: source: `supported` means the cited document says it, `question_fact` means
#: the question said it, `inference` and `contradicted` may rest on either.
#: `unsupported` and `uncited` assert an absence and need no ground.
#:
#: Without this table three verdicts were unfalsifiable: labelling a fabricated
#: sentence `supported`, `inference` or `question_fact` passed with `ok=True`
#: and no citation, no quote and nothing checked.
_GROUNDS = {
    SUPPORTED: ("sources",),
    QUESTION_FACT: ("question",),
    INFERENCE: ("sources", "question"),
    CONTRADICTED: ("sources", "question"),
    UNSUPPORTED: (),
    UNCITED: (),
}


def _quote_is_grounded(quote: str, evidence: Sequence[dict]) -> bool:
    """True when the quoted span really appears in one of the cited sources.

    Purely structural: substring, not meaning. A verifier that answers
    "supported, the source says X" where no cited source says X has attributed
    words to a document it did not contain them, and that is checkable without
    judging anything.
    """
    needle = _normalize_span(quote)
    if not needle:
        return False
    return any(needle in _normalize_span(ev.get("text") or "") for ev in evidence)


def _check_ground(
    verdict: str, rationale: str, quote: Optional[str], claim: str,
    evidence: Sequence[dict], question: str, *,
    require_evidence: bool, require_quotes: bool,
) -> tuple[str, str, Optional[str]]:
    """Hold a verdict to the ground it implicitly claims.

    Returns (verdict, rationale, structural issue or None). Every test here is
    structural — a substring, or the presence of a resolved citation — never a
    judgement about meaning.
    """
    grounds = _GROUNDS.get(verdict, ())
    if not grounds:
        return verdict, rationale, None

    def rejected(reason: str, downgrade: str) -> tuple[str, str, str]:
        return downgrade, f"{rationale} | {reason}".strip(" |"), (
            f"{reason} for claim {claim[:40]!r}"
        )

    if quote is not None:
        # The quote names where the support lives, so it is checked against
        # exactly that: a `question_fact` quoting the question was being
        # compared to the cited documents and rejected for not being in them.
        found = (
            ("sources" in grounds and _quote_is_grounded(quote, evidence))
            or ("question" in grounds
                and _normalize_span(quote) in _normalize_span(question))
        )
        if not found:
            where = " or ".join(grounds)
            return rejected(
                f"quoted span {quote.strip()[:60]!r} is not in the {where}",
                UNSUPPORTED,
            )
        return verdict, rationale, None

    if require_quotes:
        return rejected("no quoted span was given, and quotes are required",
                        UNSUPPORTED)
    if not require_evidence:
        return verdict, rationale, None

    if "sources" in grounds and evidence:
        return verdict, rationale, None
    if verdict == SUPPORTED:
        # Supported by what? With nothing cited, the verdict describes the
        # claim wrongly: it is uncited, which is a verdict of its own.
        return rejected("verdict is 'supported' but the claim cites nothing",
                        UNCITED)
    # `question_fact` and a citation-less `inference` can only be shown by
    # pointing at the question, and no span was offered.
    return rejected(
        f"verdict is {verdict!r} but names no ground: no citation and no "
        "quoted span from the question",
        UNSUPPORTED,
    )


def _coerce_verdict(
    raw: object, claim: str
) -> tuple[str, str, Optional[str], Optional[str]]:
    """Normalize one verifier result into (verdict, rationale, replacement,
    quote)."""
    if isinstance(raw, str):
        verdict, rationale, replacement, quote = raw, "", None, None
    elif isinstance(raw, dict):
        verdict = str(raw.get("verdict", "")).strip().lower()
        rationale = str(raw.get("rationale", "") or "")
        replacement = raw.get("replacement")
        replacement = None if replacement is None else str(replacement)
        quote = raw.get("quote")
        quote = None if quote is None else str(quote)
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
    return verdict, rationale, replacement, quote


def verify_answer(
    *,
    question: str,
    answer_text: str,
    context: str,
    citations: Sequence[Citation],
    verify: Callable,
    mode: str = "report",
    facets: Optional[Sequence[str]] = None,
    allow_replacements: bool = False,
    require_quotes: bool = False,
    require_evidence: bool = True,
) -> VerificationReport:
    """Run the verification pass. See the module docstring for the contract.

    ``verify`` is called with a payload describing the question, the answer,
    the context, every claim with its resolved evidence, and (when the caller
    decomposed the question) the ``facets`` the answer was supposed to cover.
    It must return one verdict per claim: a list, or a dict with a ``claims``
    key and an optional ``facets`` list reporting per-facet coverage.

    ``allow_replacements`` is off by default: the verifier segments and
    classifies, it does not write. A verifier that proposes a ``replacement``
    and then re-verifies it is grading its own text — self-endorsement, not
    verification — so by default a problem claim is removed rather than
    rewritten. Turn it on only if you accept that trade-off.

    Every verdict implies a ground, and the ground is checked (see
    ``_GROUNDS``). A ``quote`` is compared against exactly what the verdict
    rests on: the cited sources for ``supported``, the question for
    ``question_fact``, either for ``inference`` and ``contradicted``. With no
    quote, ``require_evidence`` (on by default) still demands that the claim
    cite something wherever a source is an admissible ground — otherwise
    ``supported``, ``inference`` and ``question_fact`` are unfalsifiable, and
    a fabricated sentence passes by being labelled one of them.

    Callers whose answers do not use citations at all should pass
    ``require_evidence=False``; ``ask()`` and ``ask_multi()`` derive it from
    their own ``citations`` argument. ``require_quotes`` goes further and
    demands a span for every verdict that has a ground. It is off by default
    because support is not always a contiguous span (a rule spread over two
    sentences, a table), and demanding one would reject claims that really are
    supported.
    """
    if mode not in VERIFICATION_MODES:
        raise ConfigurationError(
            f"unknown verification_mode {mode!r}; available: {list(VERIFICATION_MODES)}"
        )
    if not callable(verify):
        raise ConfigurationError("verify must be a callable")

    report = VerificationReport(mode=mode, repaired_text=answer_text)
    # Recorded before anything can fail: what the answer was *supposed* to
    # cover does not depend on the verifier surviving. A crash must leave the
    # completeness axis failing closed, not reverting to "no facets expected".
    report.expected_facets = list(facets or [])
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

    def run(
        items: Sequence[str], allow_resegmentation: bool = False
    ) -> tuple[list, Optional[list], Optional[list[str]]]:
        """Call the verifier and normalize its shape.

        The verifier may re-segment: heuristic boundaries cannot see two
        claims inside one sentence ("X takes 30 days [1] and Y takes 5 [2]"),
        and the verifier is already an LLM reading the whole answer, so its
        segmentation costs no extra call. Returned claim texts must appear
        verbatim in the answer — that keeps repair surgical (spans removed
        from the original) instead of rewriting the answer from model output.
        """
        raw = verify(build_payload(items))
        facet_report = None
        if isinstance(raw, dict):
            facet_report = raw.get("facets")
            raw = raw.get("claims", raw.get("verdicts"))
        if raw is None:
            raise ValueError("verifier returned no claim verdicts")
        results = list(raw)

        resegmented: Optional[list[str]] = None
        if allow_resegmentation:
            texts = [
                str(r["claim"]) for r in results
                if isinstance(r, dict) and str(r.get("claim", "")).strip()
            ]
            if len(texts) == len(results) and texts != list(items):
                missing = [t for t in texts if t not in answer_text]
                if missing:
                    raise ValueError(
                        "verifier re-segmented the answer but "
                        f"{len(missing)} claim(s) are not verbatim substrings "
                        f"of it, e.g. {missing[0][:60]!r}"
                    )
                resegmented = texts

        if resegmented is None and len(results) != len(items):
            raise ValueError(
                f"verifier returned {len(results)} verdicts for {len(items)} claims"
            )
        return results, facet_report, resegmented

    try:
        results, facet_report, resegmented = run(claims, allow_resegmentation=True)
        if resegmented is not None:
            claims, separators, issues = _locate_spans(resegmented, answer_text)
            report.segmentation = "verifier"
            report.structural_issues.extend(issues)
        for claim, item in zip(claims, results):
            verdict, rationale, replacement, quote = _coerce_verdict(item, claim)
            indices = citations_in(claim)
            evidence = _evidence_for(indices, citations)

            # The ground a verdict implies is the part the library can check
            # itself, and checking it costs nothing.
            verdict, rationale, issue = _check_ground(
                verdict, rationale, quote, claim, evidence, question,
                require_evidence=require_evidence,
                require_quotes=require_quotes,
            )
            if issue:
                report.structural_issues.append(issue)

            report.claims.append(ClaimVerification(
                claim=claim,
                citations=indices,
                verdict=verdict,
                rationale=rationale,
                chunk_ids=[cid for ev in evidence for cid in ev["chunk_ids"]],
                replacement=replacement,
                quote=quote,
            ))
        report.facet_coverage = _coerce_facets(facet_report, facets)
        if report.expected_facets and report.facet_coverage:
            reported = {f["facet"] for f in report.facet_coverage}
            missing = [f for f in report.expected_facets if f not in reported]
            if missing:
                report.structural_issues.append(
                    f"verifier reported coverage for {len(report.facet_coverage)} "
                    f"of {len(report.expected_facets)} facets; missing: "
                    + ", ".join(repr(m) for m in missing)
                )
    except ConfigurationError:
        raise  # caller error (bad verdict/shape) — surface it, don't swallow
    except Exception as exc:
        # A broken verifier must never destroy a valid answer.
        report.error = f"{type(exc).__name__}: {exc}"
        report.claims = []
        report.repaired_text = answer_text
        report.elapsed_ms = (time.monotonic() - started) * 1000
        return report

    report.repaired_text = _apply_mode(
        report.claims, mode, answer_text, separators, allow_replacements
    )

    # Second pass over rewritten claims only. The verifier's `replacement` is
    # generated text that entered the answer without ever being checked — the
    # repair could itself introduce an unsupported statement. Exactly one
    # extra pass (never a loop): a replacement that fails re-verification is
    # dropped rather than replaced again.
    rewritten = [c for c in report.claims if c.action == "rewritten" and c.replacement]
    if allow_replacements and rewritten and mode in ("repair", "strict"):
        try:
            recheck, _, _ = run([c.replacement for c in rewritten])  # type: ignore[misc]
            for claim, item in zip(rewritten, recheck):
                verdict, rationale, _, _ = _coerce_verdict(
                    item, claim.replacement or ""
                )
                claim.replacement_verdict = verdict
                # Fail closed: only an explicitly `supported` replacement may
                # enter the answer. `uncited`/`inference`/`question_fact` are
                # not endorsements, and this text was written by the verifier,
                # not by the model whose output the user chose to trust.
                if verdict != SUPPORTED:
                    claim.replacement = None
                    claim.action = "removed"
                    claim.rationale = (
                        f"{claim.rationale} | replacement not accepted "
                        f"({verdict}): {rationale}"
                    ).strip(" |")
            report.repaired_text = _apply_mode(
                report.claims, mode, answer_text, separators, allow_replacements
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            # Re-verification is a safety net, not a gate: if it fails, keep
            # the repaired answer and say the replacements went unchecked.
            report.recheck_error = f"{type(exc).__name__}: {exc}"

    report.elapsed_ms = (time.monotonic() - started) * 1000
    return report


def _locate_spans(
    texts: Sequence[str], answer: str
) -> tuple[list[str], list[str], list[str]]:
    """Map verifier-supplied claims back onto the answer.

    Returns (claims, separators, structural_issues). Working with spans of the
    original — rather than the verifier's copy — is what lets ``repair`` stay
    surgical: removing a claim removes that span and keeps every character
    around it.

    Order and non-overlap are *structural* requirements, checked here: spans
    that overlap would judge the same text twice and make repair produce
    garbage, and out-of-order spans make the separators between them
    meaningless. Both are rejected rather than silently worked around.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for text in texts:
        start = answer.find(text, cursor)
        if start < 0:
            start = answer.find(text)
        if start < 0:  # pragma: no cover - guarded by the caller
            raise ValueError(f"claim not found in the answer: {text[:60]!r}")
        spans.append((start, start + len(text)))
        cursor = start + len(text)

    for i in range(1, len(spans)):
        if spans[i][0] < spans[i - 1][1]:
            raise ValueError(
                "verifier segmentation overlaps or is out of order at claim "
                f"{i + 1}: {texts[i][:40]!r}"
            )

    issues: list[str] = []
    # Text no claim covers was never judged. Structural, not semantic: we do
    # not guess whether it was "material", only report that it went unseen.
    covered = sum(end - start for start, end in spans)
    gaps = [answer[:spans[0][0]]] if spans else [answer]
    gaps += [answer[spans[i][1]:spans[i + 1][0]] for i in range(len(spans) - 1)]
    if spans:
        gaps.append(answer[spans[-1][1]:])
    unseen = sum(len(g.strip()) for g in gaps)
    if unseen:
        issues.append(
            f"{unseen} character(s) of the answer were not covered by any "
            "claim and therefore went unverified"
        )

    seps = [answer[spans[i][1]:spans[i + 1][0]] for i in range(len(spans) - 1)]
    return list(texts), seps, issues


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
    separators: Sequence[str], allow_replacements: bool = False,
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
            if allow_replacements and claim.replacement:
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
