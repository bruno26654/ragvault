"""RagVault CLI.

    ragvault init ./data
    ragvault sync ./data ./documents
    ragvault query ./data "minha pergunta"
    ragvault inspect ./data
    ragvault doctor ./data
    ragvault evaluate ./data evaluation.jsonl
    ragvault compact ./data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _open(path: str, **kwargs):
    from .kb import KnowledgeBase

    return KnowledgeBase(path, **kwargs)


def cmd_init(args: argparse.Namespace) -> int:
    kb = _open(args.path, preset=args.preset, create=True)
    print(f"initialized knowledge base at {args.path}")
    print(kb.config.explain())
    kb.close()
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    with _open(args.path) as kb:
        report = kb.sync(
            args.documents,
            include=args.include or None,
            exclude=args.exclude or None,
            delete_missing=not args.keep_missing,
        )
        print(report)
        for error in report.errors:
            print(f"  error: {error}", file=sys.stderr)
        return 1 if report.failed and args.strict else 0


def cmd_query(args: argparse.Namespace) -> int:
    with _open(args.path) as kb:
        result = kb.retrieve(
            args.query,
            k=args.k,
            token_budget=args.token_budget,
            explain=args.explain,
        )
        print(result.context)
        print()
        for citation in result.citations:
            line = f"[{citation.index}] {citation.title or citation.document_id}"
            if citation.uri:
                line += f" — {citation.uri}"
            print(line)
        if args.explain:
            print()
            print("plan:", json.dumps(result.plan, indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    with _open(args.path) as kb:
        print(json.dumps(kb.stats(), indent=2))
        print()
        print(kb.config.explain())
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    path = Path(args.path)
    problems: list[str] = []
    checks: list[str] = []

    if not path.exists():
        print(f"no knowledge base at {path}")
        return 1
    config_file = path / "ragvault.json"
    if config_file.exists():
        checks.append("config file present")
        try:
            json.loads(config_file.read_text())
            checks.append("config file parses")
        except json.JSONDecodeError as exc:
            problems.append(f"config file corrupt: {exc}")
    else:
        problems.append("missing ragvault.json")

    try:
        with _open(args.path) as kb:
            checks.append("vault opens (lock, manifest, WAL replay OK)")
            stats = kb.stats()
            checks.append(
                f"{stats['documents']} documents, {stats['live_chunks']} live chunks, "
                f"{stats['tombstones']} tombstones"
            )
            if stats["documents"] > 0:
                result = kb.retrieve("doctor self-test query", k=1)
                checks.append(f"test query OK ({len(result.chunks)} chunk(s) returned)")
            if stats["tombstones"] > max(100, stats["live_chunks"]):
                problems.append(
                    "many tombstones — run `ragvault compact` to reclaim space"
                )
    except Exception as exc:
        problems.append(f"vault failed to open or query: {exc}")

    for check in checks:
        print(f"  ok: {check}")
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    print("healthy" if not problems else f"{len(problems)} problem(s) found")
    return 0 if not problems else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    with _open(args.path) as kb:
        report = kb.evaluate(args.dataset, k=args.k)
        print(report.to_markdown())
        if args.output:
            report.to_json(args.output)
            print(f"\nfull report written to {args.output}")
    return 0


def cmd_studio(args: argparse.Namespace) -> int:
    from .studio import serve

    with _open(args.path) as kb:
        server = serve(kb, host=args.host, port=args.port,
                       open_browser=not args.no_browser)
        host, port = server.server_address[:2]
        print(f"RagVault Studio at http://{host}:{port}/ (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopping")
        finally:
            server.server_close()
    return 0


def cmd_compact(args: argparse.Namespace) -> int:
    with _open(args.path) as kb:
        before = kb.stats()
        kb.compact()
        after = kb.stats()
        print(
            f"compacted: tombstones {before['tombstones']} -> {after['tombstones']}, "
            f"rows {before['total_rows']} -> {after['total_rows']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ragvault",
        description="RagVault — documents to a high-quality RAG knowledge base.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a knowledge base")
    p.add_argument("path")
    p.add_argument("--preset", default="balanced")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("sync", help="sync a documents directory")
    p.add_argument("path")
    p.add_argument("documents")
    p.add_argument("--include", action="append")
    p.add_argument("--exclude", action="append")
    p.add_argument("--keep-missing", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.set_defaults(fn=cmd_sync)

    p = sub.add_parser("query", help="run a retrieval query")
    p.add_argument("path")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--token-budget", type=int, default=None)
    p.add_argument("--explain", action="store_true")
    p.set_defaults(fn=cmd_query)

    p = sub.add_parser("inspect", help="show stats and configuration")
    p.add_argument("path")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("doctor", help="check knowledge base health")
    p.add_argument("path")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("evaluate", help="run retrieval evaluation")
    p.add_argument("path")
    p.add_argument("dataset")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--output")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("compact", help="reclaim space from deletions")
    p.add_argument("path")
    p.set_defaults(fn=cmd_compact)

    p = sub.add_parser("studio", help="open the local inspection UI")
    p.add_argument("path")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7644)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_studio)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
