# Contributing to RecallLab

Thanks for considering a contribution. This file covers the practical bits:
local setup, the gates CI runs, and conventions that keep the repo
shippable.

## Development setup

RecallLab uses [uv](https://github.com/astral-sh/uv) for both dependency
management and the virtual environment.

```bash
git clone https://github.com/iushv/recalllab.git
cd recalllab
uv sync --all-extras --dev
```

That installs every optional extra (`langgraph`, `mcp`, `dashboard`,
`judge`) plus the dev tools (`ruff`, `mypy`, `pytest-asyncio`,
`pytest-cov`, `pre-commit`). All subsequent `uv run` commands use this
environment.

## Gates (run these before opening a PR)

CI runs exactly three commands; mirror them locally:

```bash
uv run ruff check src/recalllab tests
uv run mypy src/recalllab tests
uv run pytest --no-header -q
```

Targets:

- **ruff**: no errors. The config in `pyproject.toml` enables E, F, I,
  N, W, B, UP, RUF, SIM, C4.
- **mypy**: no errors. `tool.mypy.strict = true`; both source and tests
  are type-checked.
- **pytest**: all green. The xdist warning test conditionally skips
  when `pytest-xdist` isn't installed; that's expected.

If any of the three fail, the PR's `CI` check fails. Don't squash a
red CI past the gate — fix it.

## Where to put things

| Adding a... | File / dir |
|---|---|
| New example contract | `examples/tests/test_*.py` (gets shipped by `recalllab init`) |
| New memory provider adapter | `src/recalllab/adapters/<your_adapter>/` mirroring `reference/` |
| New judge backend | `src/recalllab/core/judge/<your_judge>.py`, lazy-imported from `plugin._build_judge` |
| Unit test | `tests/unit/test_<feature>.py` |
| Integration test | `tests/integration/test_<feature>.py` |
| Docs section | `docs/concepts.md` or a focused page like `docs/judge-assertions.md` |

## Conventions

- **Commit messages** follow the existing pattern: short imperative
  subject, body explaining *why*. Look at `git log --oneline` for
  examples.
- **No emojis** in code, comments, commits, or docs unless explicitly
  requested by the user.
- **Tests are mandatory** for new behavior. If you can't reach the
  branch from a test, refactor until you can.
- **Type annotations are mandatory** under `mypy --strict`. Use
  `# type: ignore[<code>]` only for genuine type-system gaps with a
  comment explaining why.
- **Adversarial review.** RecallLab uses Codex adversarial reviews
  for substantial design changes (see `docs/judge-assertions.md` for
  the v0.2.2 history). For feature-sized PRs, consider running one
  before merge.

## Trace-store hygiene

`.recalllab/` is git-ignored on first use; never commit traces or the
SQLite file. They contain raw memory text (which may include test
fixtures or, in production traces, real conversation snippets).

## Release process

Maintainers only:

1. Bump version in `pyproject.toml`.
2. Update `CHANGELOG.md` with the new version section + compare-link.
3. Update `docs/concepts.md` with any new feature sections.
4. Merge to `main`.
5. Tag `vX.Y.Z` on `main` and push.
6. The `Release` workflow (`.github/workflows/release.yml`) builds and
   publishes to PyPI via trusted publishing. The one-time PyPI setup
   is documented at the top of that workflow file.

## Reporting issues

Open a GitHub issue with:

- What you tried.
- What you expected.
- What happened.
- Minimal repro if possible — a one-file contract that demonstrates
  the bug is ideal.

For security issues, please email the maintainer instead of opening a
public issue.
