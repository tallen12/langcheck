---
name: conventional-commits
description: Write git commit messages for this repository in the Conventional Commits format. Use whenever creating, amending or rewording a commit, or drafting a commit message for the user.
---

# Conventional commits

Every commit in this repository uses the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
format. Only commit or push when the user asks. For branches, worktrees and squash merges,
follow the `git-workflow` skill.

## Format

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

- **type** is one of:
  - `feat`: a new user-facing capability
  - `fix`: a bug fix
  - `docs`: documentation only (including `design-docs/`, `AGENTS.md`, skills)
  - `test`: adding or changing tests only
  - `refactor`: a code change that neither fixes a bug nor adds a feature
  - `chore`: tooling, dependencies, build, CI, repository housekeeping
- **scope** is optional and names the domain or adapter touched, for example
  `feat(stacks):`, `fix(cli):`, `test(health):`.
- **description** is imperative, lowercase, with no trailing period, and at most about 72
  characters including the prefix: `fix(branches): handle detached HEAD`.
- **body** explains what changed and why, not how. Wrap at 72 characters and separate it
  from the subject with a blank line. Omit it for trivial changes.
- **breaking changes** add `!` after the type or scope (`feat(mcp)!: rename scopes`) and a
  `BREAKING CHANGE: <what and how to migrate>` footer.
- Reference issues in a footer: `Refs: #123` or `Closes: #123`.

## Rules

- One logical change per commit. If the staged changes need two types, split them into two
  commits.
- Do not add `Co-Authored-By` trailers or any other AI or tool attribution (for example
  "Generated with ..." lines) to commit messages.
- Do not bypass commit signing or hooks (`--no-verify`, `-c commit.gpgsign=false`) unless
  the user explicitly asks. If signing fails, report the error and let the user fix it.
- Before committing code changes, run `just check` and report the result. Documentation-only
  commits do not need the test suite.
- Review `git status` before staging so agent state and caches (see `.gitignore`) are not
  committed.

## Examples

```text
feat(stacks): restack dependent branches after a merge

When a parent branch merges, rebase each child onto the new base in
order so the stack stays linear without manual intervention.
```

```text
docs: add conventional commits skill
```

```text
fix(cli)!: require --repo for the status command

BREAKING CHANGE: `langcheck status` no longer infers the repository
from the working directory. Pass `--repo <path>` explicitly.
```
