"""``recalllab`` CLI entry point.

Subcommands:

- ``recalllab init`` (v0.1) — scaffold the six example contract tests
  and a ``recalllab.toml`` config file.
- ``recalllab dashboard`` (v0.1) — serve the read-only Failure Gallery
  over the SQLite trace store.
- ``recalllab record`` (v0.2.1) — read a recorded ``ContractRun`` and
  emit a self-contained pytest regression that replays it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recalllab.cli.record import cmd_record
from recalllab.cli.scaffolds import RECALLLAB_TOML, SCAFFOLD_CONTRACTS


def cmd_init(target: Path) -> int:
    """Scaffold ``tests/memory/`` and ``recalllab.toml`` under *target*.

    Existing files are left alone (and reported as ``skip``). Re-running
    ``recalllab init`` is therefore safe.
    """
    target = target.resolve()
    tests_dir = target / "tests" / "memory"
    tests_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for name, content in SCAFFOLD_CONTRACTS.items():
        dest = tests_dir / name
        rel = dest.relative_to(target)
        if dest.exists():
            print(f"  skip   {rel} (exists)")
            skipped += 1
            continue
        dest.write_text(content)
        print(f"  write  {rel}")
        written += 1

    config_path = target / "recalllab.toml"
    if config_path.exists():
        print("  skip   recalllab.toml (exists)")
    else:
        config_path.write_text(RECALLLAB_TOML)
        print("  write  recalllab.toml")

    print()
    print(f"Wrote {written} contract files; skipped {skipped} existing.")
    print()
    print("Run them with:")
    print("    uv run pytest tests/memory")
    return 0


def cmd_dashboard(trace: Path, host: str, port: int) -> int:
    """Serve the Failure Gallery against an existing trace store.

    FastAPI/Jinja2/uvicorn are imported lazily so users without the
    ``[dashboard]`` extra get an actionable message rather than a stack
    trace at CLI start-up.
    """
    try:
        from recalllab.dashboard.app import run_server
    except ImportError as exc:
        print("Dashboard dependencies not installed.", file=sys.stderr)
        print(file=sys.stderr)
        print('    pip install "recalllab[dashboard]"', file=sys.stderr)
        print(file=sys.stderr)
        print(f"(import error: {exc})", file=sys.stderr)
        return 1

    if not trace.exists():
        print(f"Trace store not found at {trace}.", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "Run `recalllab init && uv run pytest tests/memory` first to "
            "populate it.",
            file=sys.stderr,
        )
        return 1

    run_server(trace, host=host, port=port)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recalllab",
        description=(
            "RecallLab — pytest for agent memory. Turn memory expectations "
            "into pytest contracts that run in CI."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    init_p = sub.add_parser(
        "init",
        help="Scaffold tests/memory/ and recalllab.toml in the target directory.",
    )
    init_p.add_argument(
        "--path",
        "-p",
        default=".",
        help="Target directory (default: current working directory).",
    )

    dash_p = sub.add_parser(
        "dashboard",
        help="Serve the read-only Failure Gallery for a trace store.",
    )
    dash_p.add_argument(
        "--trace",
        type=Path,
        default=Path(".recalllab/traces.sqlite"),
        help="Path to the SQLite trace store (default: .recalllab/traces.sqlite).",
    )
    dash_p.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind host (default: 127.0.0.1, localhost only). Pass 0.0.0.0 "
            "to expose the dashboard on your LAN — note that traces are "
            "served unauthenticated, so only do that on a trusted network."
        ),
    )
    dash_p.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bind port (default: 8080).",
    )

    rec_p = sub.add_parser(
        "record",
        help=(
            "Generate a pytest regression file from a recorded "
            "ContractRun (trace → test)."
        ),
    )
    rec_p.add_argument(
        "--trace",
        type=Path,
        default=Path(".recalllab/traces.sqlite"),
        help="Path to the SQLite trace store (default: .recalllab/traces.sqlite).",
    )
    rec_source = rec_p.add_mutually_exclusive_group(required=True)
    rec_source.add_argument(
        "--run-id",
        help="Replay a specific recorded run by its UUID.",
    )
    rec_source.add_argument(
        "--latest-failure",
        action="store_true",
        help=(
            "Pick the most recent run with status=failed. Useful for "
            "turning the freshest CI failure into a checked-in regression."
        ),
    )
    rec_p.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output path for the generated pytest file (e.g. "
            "tests/regressions/test_recorded_failure.py)."
        ),
    )
    rec_p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite the file at --out if it already exists. Default "
            "is to refuse and exit so a typo can't silently destroy a "
            "checked-in regression. The overwrite goes through an "
            "atomic rename so a crash mid-write can't leave a "
            "half-written file."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "init":
        return cmd_init(Path(args.path))
    if args.command == "dashboard":
        return cmd_dashboard(args.trace, args.host, args.port)
    if args.command == "record":
        return cmd_record(
            trace_path=args.trace,
            run_id=args.run_id,
            latest_failure=args.latest_failure,
            out_path=args.out,
            force=args.force,
        )
    parser.error(f"unknown command: {args.command!r}")
    return 2  # parser.error sys.exits, but keep mypy happy.


if __name__ == "__main__":
    sys.exit(main())
