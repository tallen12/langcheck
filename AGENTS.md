# Project instructions

These instructions are for every coding agent working in this repository (Claude Code,
Codex, Cursor, Copilot and others) and for humans. `AGENTS.md` is the single source of
truth: tool-specific files only point here. Keep this file short, imperative and current.

## Scope

- Design documents, when present, live in `design-docs/`. They do not authorize
  implementation; start work only when the user asks for it.
- When behavior and a design document disagree, stop and ask, or update the document in
  the same change with the reason.
- Keep the **Commands** section below accurate: tools and agents use it to find this
  repo's checks.

## Commands

All commands run through `just` and use `uv`. Never call `pip`, `python -m venv` or a
global interpreter.

- Setup: `just setup` (also installs the pre-commit hooks: gitleaks, ruff, ty)
- Format: `just format`
- Lint: `just lint`
- Type check: `just typecheck`
- Test layout and skills link: `just layout`
- Test: `just test`
- Test coverage: `just coverage`
- All required checks: `just check`
- Build the VS Code extension (`editors/vscode/`, TypeScript): `just vscode` (needs
  Node.js 22; type-checks, bundles and packages `langcheck.vsix`)
- Remove finished worktrees and merged branches: `just prune-worktrees` (`--fetch`,
  `--dry-run`)
- Render diagrams: `just diagrams` (needs Graphviz). Diagrams live in
  `design-docs/images/` as `.dot` sources; commit the rendered PNG next to each.

## Working across agents

- Put shared instructions here, never in `CLAUDE.md`, `.cursor/rules`,
  `.github/copilot-instructions.md` or similar. Those files, if present, only point to
  `AGENTS.md`. `CLAUDE.md` imports this file with `@AGENTS.md`.
- Skills live in `.agents/skills/<skill-name>/SKILL.md`. `.claude/skills` is a symlink to
  that folder so Claude Code loads the same skills. Do not create skills elsewhere.
- A skill folder holds `SKILL.md` (YAML front matter with `name` and `description`, then
  instructions) plus optional `scripts/`, `references/` and `assets/`. Skill scripts must
  follow the same Python rules as the codebase and run with `uv run`.
- Write instructions and skills in plain Markdown with relative paths. No tool-specific
  syntax, absolute paths or references to one agent's private features.
- Do not commit agent state or caches (`.claude/settings.local.json`, session files,
  scratch notes).

## Cross-platform rules

Development happens on macOS and Linux.

- Use `pathlib.Path` for all paths; never build paths with string concatenation or
  hard-code `/` separators, home directories or `/tmp`. Use `tempfile` for temporary
  files.
- Run subprocesses with argument lists, never `shell=True`. Keep process execution behind a
  protocol so it can be faked in tests.
- Do not rely on GNU-only or BSD-only command flags in code, recipes or tests. Prefer
  Python over shell for anything non-trivial.
- Text files use UTF-8 and LF line endings (`.gitattributes`, `.editorconfig`). Open files
  with an explicit `encoding="utf-8"`.
- Platform-specific behavior belongs behind a protocol in platform-specific adapters, with
  fakes for tests that run anywhere.

## Protocols and dependency injection

- Define behavioral interfaces with `typing.Protocol`. Pydantic schemas, dataclasses, enums
  and exceptions are value types and need no protocol.
- Keep protocols separate from implementations. Prefer small, capability-focused protocols.
- Type injected dependencies as protocols. Use constructor injection for classes and
  explicit parameters for functions.
- Construct concrete implementations only at composition roots (CLI startup, test
  fixtures). Domain code never instantiates its own infrastructure.
- No service locators or hidden globals. Framework injection stays in adapters.
- Rely on structural typing; add `@runtime_checkable` only when a runtime check is needed.
- Prefer composition over inheritance. Test doubles implement the same protocols.

## Types and data

- Type every parameter and return value, including `__init__`, fixtures and helpers.
- `ty` is the required type checker with every rule set to error. Do not weaken it.
- Avoid `Any`, bare containers, unchecked casts and blanket ignores. Treat unknown input as
  `object` and narrow it. Any targeted suppression needs a one-line reason.
- Model alternatives as sum types (unions of named variants, discriminated when
  serialized) and related fields as product types (frozen dataclasses internally, Pydantic
  models at boundaries). Prefer explicit success, failure and awaiting-decision variants
  over optional fields and boolean flags.
- Use enums or `Literal` for finite choices; no magic strings in domain logic.
- Prefer `match` for variants and finite states, exhaustive with `typing.assert_never`.
- Validate all external data (config, files, API responses, subprocess output) with
  explicit schemas.

## Code organization

- `src` layout: all application code lives under `src/langcheck/`. Tests live in
  `tests/`, design documents in `design-docs/`.
- Organize by domain or capability, not by horizontal layer. Entry-point adapters (such as
  `cli`) get their own packages. Each domain keeps its models, protocols and
  implementations together but in separate modules.
- Importing a module must have no side effects: no workers, clients, config reads, file,
  network or subprocess access at import time.
- Prefer small pure functions for policies and transformations; use classes for state,
  lifecycle or a cohesive capability.
- Favor immutable values (`@dataclass(frozen=True, slots=True)`). Use keyword-only
  parameters for options and same-typed arguments; no boolean positional arguments or
  mutable defaults.
- Represent expected outcomes with typed result variants. Use exceptions for exceptional
  failures, catch specific types, preserve causes and never swallow errors.
- Make resource ownership explicit with context managers. Never block the event loop.
- Inject clocks, ID generators, subprocess runners and external clients.
- Use absolute imports. Comments explain why, not what. Document public contracts.

## Tests

- Tests mirror `src/langcheck/`, split by type: the tests for
  `src/langcheck/<path>/<name>.py` are `tests/unit/<path>/test_<name>.py` and, when
  needed, `tests/integration/<path>/test_<name>.py`. Every module except `__init__.py`
  needs a unit test. Tests of shared fixtures in `tests/conftest.py` are
  `tests/<type>/test_conftest.py`. Every test directory has an `__init__.py`.
  `scripts/check_layout.py` (`just layout`) enforces this.
- Unit tests run in process with fakes and in-memory transports. Integration tests cross a
  real boundary: git repositories, sockets, or an adapter talking to another adapter.
- Test observable behavior and protocol contracts: state transitions, failures, recovery.
- Use realistic fakes over mocks tied to implementation details. Tests that touch the
  filesystem use `tmp_path`, never the network.
- The default suite needs no credentials, Docker or network access.

## Commits

- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
  Keep commits focused; one logical change each.
- Branch, land and push as the `git-workflow` skill describes, also in a session the host
  tool started in its own worktree or branch. Tasks land on the feature branch locally;
  the only pull request is a feature branch into `main`.
- Only commit or push when the user asks. The one standing exception: back up a work
  branch when a session may end mid-task, as the `git-workflow` skill describes.

## Completion

Before reporting work as done, run `just check` (format check, lint, type check and
tests). Report any check that could not run and why. Never claim a check passed without
running it. Documentation-only changes do not need the test suite. Do not add
dependencies or features beyond what was asked.
