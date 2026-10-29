---
name: pull-requests
description: Write pull request titles and descriptions for this repository, with a summary, list of changes, impact, testing and risks. Use whenever opening or updating a pull request, or drafting a PR description for the user.
---

# Pull requests

Pull requests go from a feature branch into `main` (see the `git-workflow` skill). Open or
update one only when the user asks.

`main` only accepts squash merges, and GitHub uses the **PR title as the squash commit
subject and the PR description as its body**. Write both as the permanent record of the
feature, not as a chat message.

## Title

The title is a Conventional Commits subject (see the `conventional-commits` skill) that
describes the whole feature:

```text
feat(stacks): restack dependent branches after a merge
```

Use `!` for breaking changes, for example `feat(mcp)!: rename approval scopes`.

## Gather the facts first

Base the description on the actual changes, not on memory of the session:

```bash
git fetch origin
git log --format='%h %s%n%b' origin/main..HEAD
git diff --stat origin/main...HEAD
git diff origin/main...HEAD
```

Also check `design-docs/`, when present, for the milestone or decision the work
implements, and any issue it closes.

## Description template

Use these sections in this order. Leave out a section that has nothing real to say; never
write "N/A" or filler.

```markdown
## Summary

One or two short paragraphs: what this changes and why. Lead with the outcome for users or maintainers. Mention the design-doc milestone or decision it implements, if any.

## Changes

- Group by domain or adapter (for example `cli`, tooling, docs).
- One bullet per meaningful change, saying what changed and, if not obvious, why.
- Skip trivial edits such as renames or formatting.

## Impact

- **Users:** new or changed CLI commands, options or output.
- **Breaking changes:** what breaks and how to migrate. Match a `!` in the title.
- **Configuration and deployment:** new settings, environment variables, Docker or dependency changes.
- **Security:** any change to credentials, scopes, or permissions.

## Testing

- The checks that were run and their result, for example `just check` passed.
- New or changed tests and what behavior they cover.
- Manual verification steps a reviewer can repeat.
- Anything not verified, and why.

## Risks and follow-ups

- Known limitations, edge cases or risky areas worth reviewing closely.
- How to roll back if it is not just reverting the squash commit.
- Work deliberately left for later, with links to issues if they exist.

Closes #<issue>
```

## Rules

- Be accurate. Report only checks that actually ran; never claim a check passed without
  running it. Say plainly what was not tested.
- Be specific and brief. Name the files, commands, endpoints and behaviors; avoid vague
  phrases like "various improvements" or "cleanup".
- Write for a reviewer who has not seen the session. Explain decisions they would
  otherwise question.
- Do not add AI or tool attribution (for example "Generated with ..." lines or
  `Co-Authored-By` trailers). The description becomes the commit message on `main`.
- Do not paste work branch commit messages, full diffs or long logs. Summarize instead.
- Do not hard-wrap the description. Write each paragraph and each bullet as a single
  line; GitHub renders line breaks inside a paragraph literally, so wrapped text shows up
  as ragged short lines.

## Creating or updating the pull request

Write the description to a temporary file (use `tempfile` or the agent's scratch
directory, never a path inside the repository), then:

```bash
gh pr create --base main --head <type>/<feature> --title "<title>" --body-file <file>
gh pr edit <number> --title "<title>" --body-file <file>
```

After pushing more work to the feature branch, update the description so it still
matches the full diff against `main`.
