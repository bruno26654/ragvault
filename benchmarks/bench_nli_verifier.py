#!/usr/bin/env python3
"""Does the offline NLI verifier actually judge fidelity correctly?

The adapter in `ragvault.nli` maps NLI labels onto verdicts, and the mapping is
sound by construction. What is *not* sound by construction is whether a given
checkpoint gets the labels right on RAG-shaped input — long premises, numbers
that differ by one digit, distractor blocks that repeat the same figure in an
unrelated context. That is what this measures.

The number that decides whether the adapter may drive `repair`/`strict` is not
overall accuracy: it is the **false-contradicted rate**, the share of correct
claims labelled `contradicted`. Those are the ones `repair` deletes. A verifier
that misses a contradiction leaves a wrong sentence visible; one that invents a
contradiction silently removes a right one.

Two granularities are compared, because the default rests on the difference:
  sentence  premise split into sentences, max over them (SummaC-ZS)
  block     the whole retrieved chunk as one premise

Each pair is also run twice — once with the bare premise, once with the premise
padded by unrelated sentences from the same corpus, which is what a real
retrieved chunk looks like. Padding is where the two granularities are supposed
to diverge; without it the comparison would flatter the cheaper option.

Dataset: benchmarks/data/nli_verification_pairs.jsonl — claim/premise pairs
derived from the versioned-registry scenario in
tests/python/test_scenario_versioned_registry.py, which already encodes ground
truth (current rule vs. its own source, revoked figure, crossed entity,
distractor block). Committed, so the run is reproducible.

Every number written to RESULTS-VERIFICATION.md comes from an actual run. When
the environment blocks huggingface.co the file says so and prints the command
to run elsewhere, rather than reporting numbers nobody produced.

Usage:
    pip install "ragvault[nli]"
    python benchmarks/bench_nli_verifier.py
    python benchmarks/bench_nli_verifier.py --model <hf-id> --limit 20
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

DATA = Path(__file__).parent / "data" / "nli_verification_pairs.jsonl"
OUT = Path(__file__).parent / "RESULTS-VERIFICATION.md"

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

#: Padding that turns a one-sentence premise into a chunk-sized one. Drawn from
#: the same registry corpus, so it competes lexically instead of being obvious
#: filler — an easy padding would understate the long-premise effect.
PADDING = (
    "Secretariat circular 41. The procurement deadline is 30 calendar days "
    "from notification. This circular applies to all secretariat procedures "
    "and supersedes circular 1. "
    "Programs overview. ALPHA, BETA, GAMMA and DELTA are administered by the "
    "same secretariat and share a single appeals calendar. "
    "Program DELTA glossary. 'Continuous participation' means participation "
    "without a gap longer than 60 days."
)

LABELS = ("supported", "contradicted", "unsupported")


def load_pairs(limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def evidence_payload(claim: str, premise: str) -> dict:
    """The payload shape `verify=` receives from ask()/ask_multi()."""
    return {
        "question": "",
        "answer": claim,
        "context": premise,
        "claims": [{
            "claim": claim,
            "citations": [1],
            "evidence": [{
                "index": 1, "document_id": "d1", "document_version": 1,
                "chunk_ids": ["c1"], "title": "t", "metadata": {},
                "text": premise,
            }],
        }],
        "facets": [],
    }


def run_config(verifier, pairs: list[dict], pad: bool) -> dict:
    predicted, latencies = [], []
    for row in pairs:
        premise = f"{row['premise']} {PADDING}" if pad else row["premise"]
        started = time.perf_counter()
        out = verifier(evidence_payload(row["claim"], premise))
        latencies.append((time.perf_counter() - started) * 1000)
        predicted.append(out["claims"][0]["verdict"])

    expected = [row["label"] for row in pairs]
    per_label = {}
    for label in LABELS:
        tp = sum(1 for e, p in zip(expected, predicted) if e == p == label)
        fp = sum(1 for e, p in zip(expected, predicted) if p == label and e != label)
        fn = sum(1 for e, p in zip(expected, predicted) if e == label and p != label)
        per_label[label] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "support": tp + fn,
        }

    # The rate that gates repair/strict: correct claims called contradicted.
    not_contradicted = [
        (e, p) for e, p in zip(expected, predicted) if e != "contradicted"
    ]
    false_contra = sum(1 for _, p in not_contradicted if p == "contradicted")

    by_lang: dict[str, list[bool]] = {}
    for row, p in zip(pairs, predicted):
        by_lang.setdefault(row["lang"], []).append(p == row["label"])

    return {
        "accuracy": sum(1 for e, p in zip(expected, predicted) if e == p) / len(pairs),
        "per_label": per_label,
        "false_contradicted": (
            false_contra / len(not_contradicted) if not_contradicted else 0.0
        ),
        "false_contradicted_n": false_contra,
        "eligible_n": len(not_contradicted),
        "by_lang": {k: sum(v) / len(v) for k, v in sorted(by_lang.items())},
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
    }


def render(results: dict, pairs: list[dict], model: str) -> str:
    counts = {label: sum(1 for r in pairs if r["label"] == label) for label in LABELS}
    langs = sorted({r["lang"] for r in pairs})
    lines = [
        "# Offline NLI verification — measured, not assumed",
        "",
        f"Generated by `python benchmarks/bench_nli_verifier.py`. Every number "
        "is from an actual run on this machine; nothing here is estimated.",
        "",
        f"- machine: {platform.platform()} / python "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"- model: `{model}`",
        f"- pairs: {len(pairs)} "
        + ", ".join(f"{n} {label}" for label, n in counts.items())
        + f" — languages: {', '.join(langs)}",
        "- source: `benchmarks/data/nli_verification_pairs.jsonl`, derived from "
        "the versioned-registry scenario",
        "",
        "**Sample size is small and comes from one corpus.** These numbers say "
        "whether the adapter behaves sanely on registry-shaped text, not how it "
        "performs in general. Re-run on your own data before trusting it.",
        "",
        "| config | accuracy | false-contradicted | supported P/R | "
        "contradicted P/R | unsupported P/R | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        cells = []
        for label in LABELS:
            m = r["per_label"][label]
            cells.append(f"{m['precision']:.2f} / {m['recall']:.2f}")
        lines.append(
            f"| {name} | {r['accuracy']:.2f} | "
            f"**{r['false_contradicted']:.2f}** "
            f"({r['false_contradicted_n']}/{r['eligible_n']}) | "
            + " | ".join(cells)
            + f" | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} |"
        )

    lines += [
        "",
        "`false-contradicted` is the share of claims that are *not* "
        "contradictions which the verifier called `contradicted`. In `repair` "
        "mode each one deletes a correct sentence, so it gates whether the "
        "adapter may drive repair at all.",
        "",
        "## Accuracy by language",
        "",
        "| config | " + " | ".join(langs) + " |",
        "|---|" + "---|" * len(langs),
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | "
            + " | ".join(f"{r['by_lang'].get(lang, float('nan')):.2f}" for lang in langs)
            + " |"
        )
    lines += [
        "",
        "A multilingual checkpoint inherits ~100 pretraining languages from "
        "XLM-R but only the ~15 that XNLI fine-tunes on. Per-language rows are "
        "here because a single average would hide exactly that.",
        "",
        "## Reading",
        "",
    ]

    short = results.get("sentence / bare premise")
    long_sentence = results.get("sentence / padded premise")
    long_block = results.get("block / padded premise")
    if long_sentence and long_block:
        delta = long_sentence["accuracy"] - long_block["accuracy"]
        lines.append(
            f"- on chunk-sized premises, sentence granularity changes accuracy "
            f"by **{delta:+.2f}** against scoring the whole block — the "
            f"SummaC-ZS effect, and the reason it is the default."
        )
        lines.append(
            f"- it costs latency: **{long_sentence['p50_ms'] / max(long_block['p50_ms'], 1e-9):.1f}x** "
            "the block-granularity p50, since each premise sentence is its own pair."
        )
    if short and long_sentence:
        lines.append(
            f"- padding the premise changes accuracy by "
            f"**{long_sentence['accuracy'] - short['accuracy']:+.2f}** even at "
            "sentence granularity: retrieval noise is not free."
        )
    return "\n".join(lines) + "\n"


def render_blocked(model: str, reason: str) -> str:
    return f"""# Offline NLI verification — NOT YET MEASURED

`python benchmarks/bench_nli_verifier.py` could not run here: the model could
not be loaded from this environment.

    {reason}

**No numbers are published for this adapter yet.** The benchmark, its dataset
and the harness are committed and reproducible; what is missing is a machine
that can reach the model. Run it where huggingface.co is reachable:

    pip install "ragvault[nli]"
    python benchmarks/bench_nli_verifier.py --model {model}

Until that has been run, `ragvault.nli` should be treated as unmeasured:
usable in `report` and `annotate` mode, where a wrong verdict is visible and
costs nothing, and **not** in `repair`/`strict`, where a false `contradicted`
deletes a correct sentence. The gate is the false-contradicted rate, which
this benchmark reports as its headline column.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--granularity", default="sentence,block",
        help="comma-separated: sentence, block",
    )
    args = parser.parse_args()

    pairs = load_pairs(args.limit)
    print(f"{len(pairs)} labelled pairs from {DATA.name}")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from ragvault.nli import nli_verifier

    granularities = [g.strip() for g in args.granularity.split(",") if g.strip()]
    results = {}
    for granularity in granularities:
        try:
            verifier = nli_verifier(args.model, granularity=granularity)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {str(exc)[:300]}"
            print(f"BLOCKED — {reason}", file=sys.stderr)
            OUT.write_text(render_blocked(args.model, reason))
            print(f"wrote {OUT} (blocked)")
            return 1
        for pad in (False, True):
            name = f"{granularity} / {'padded' if pad else 'bare'} premise"
            print(f"  {name} …", flush=True)
            results[name] = run_config(verifier, pairs, pad)

    OUT.write_text(render(results, pairs, args.model))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
