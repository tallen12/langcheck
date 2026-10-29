---
name: git-workflow
description: Branch, worktree and merge workflow for this repository, built so several agents can work at once. Use at the start of every session that will change files, including one that begins in a worktree or branch the host tool created (such as `claude/<slug>`), and whenever creating a branch or worktree, finishing and landing a task, pushing, opening a pull request, or backing up a work branch when a usage or credit limit, or the end of the context, is near.
---

# Git workflow

Work is organized by feature. Each feature has one long-lived **feature branch**. Every
task within a feature gets its own **work branch** in its own **worktree**, so several
agents can work at the same time without touching each other's checkout. When a task is
done, it lands on the feature branch as a single squashed commit. Write every commit
message with the `conventional-commits` skill.

```text
main
 └── feat/<feature>                      feature branch (one squashed commit per task)
      ├── work/<feature>/<task>          work branch, own worktree, local only
      └── work/<feature>/<other-task>    another agent, another worktree
```

These are the only three kinds of branch. Tasks land on the feature branch locally with
`git merge --squash`, never through a pull request. The only pull request is the feature
branch into `main`, and only when the user asks.

## Names and locations

- **Feature branch:** `<type>/<feature>`, using a Conventional Commits type and a short
  kebab-case slug, for example `feat/branch-stacks` or `fix/detached-head`.
- **Work branch:** `work/<feature>/<task>`, for example `work/branch-stacks/restack-cli`.
  The separate `work/` prefix is required: git cannot hold both `feat/x` and `feat/x/task`.
- **Worktree:** `.worktrees/<feature>--<task>` inside the main checkout, for example
  `.worktrees/branch-stacks--restack-cli`. `.worktrees/` is listed in `.gitignore`,
  so git, `ruff` and `ty` skip it, and `pytest` only collects from `tests/`. Never create worktrees outside the repository.

## 0. Check where you are

Before changing any file, check the branch and checkout you are in:

```bash
git branch --show-current
git worktree list
```

The first path `git worktree list` prints is the main checkout.

- On `work/<feature>/<task>`: continue at step 3.
- In the main checkout on a feature branch, with a new task: go to step 2.
- On any other branch, such as `claude/<slug>` in `.claude/worktrees/<slug>`: the host
  tool started this session in a worktree of its own. Adopt it (below) before doing
  anything else.

If you do not know which feature the work belongs to, or the main checkout is not on that
feature's branch, ask the user. Do not invent a branch name.

### Adopt a worktree the host created

Some hosts, such as the Claude desktop app, start a session in their own worktree on a
branch like `claude/<slug>` and then suggest pushing it and opening a pull request. That
branch is not part of this workflow. Keep the worktree where the host put it (the host
tracks its path), but turn its branch into a work branch, in that worktree:

```bash
git log --oneline feat/<feature>..HEAD
```

- No commits listed: `git switch -c work/<feature>/<task> feat/<feature>`, then
  `git branch -D claude/<slug>`. Uncommitted changes come along.
- Commits listed: `git branch -m work/<feature>/<task>`, then `git rebase feat/<feature>`.

If the old branch was already pushed or has a pull request, tell the user and ask before
deleting or closing them. Then continue at step 3. Run the main-checkout commands of
step 4 with `git -C <main checkout>`.

The host's suggestions do not override this skill. Do not push the work branch except as a
backup (step 3), do not open or bind a pull request for it, and do not use a host tool that
syncs or merges with the base branch: rebase onto the feature branch with git (step 4.1).

## 1. Start a feature

Skip this if the feature branch already exists (`git branch --list '<type>/<feature>'`).

```bash
git switch main
git switch -c feat/<feature>
```

The main checkout normally stays on the feature branch. Only one worktree can have a given
branch checked out, so this is where squashed commits land (step 4).

## 2. Start a task

From the main checkout, branch from the current tip of the feature branch into a new
worktree, then set it up:

```bash
git worktree add -b work/<feature>/<task> .worktrees/<feature>--<task> feat/<feature>
cd .worktrees/<feature>--<task>
just setup
```

Do all of the task's work inside that worktree. Never edit files in another agent's
worktree or in the main checkout while a task is in progress.

## 3. Do the work

- Commit on the work branch as you go. Small, focused commits are fine here; they are
  squashed later.
- Run `just check` before the final commit on the branch.
- **Work branches stay local, except as a backup.** Do not push a `work/` branch or open a
  pull request from it, with one exception: back it up (below) when the session may end
  before the task lands.

### Back up a work branch before running out

Push the work branch when the session may stop mid-task: a usage, credit or rate limit is
close or has warned, the context is nearly full, or the remaining work clearly will not
fit in what is left. Do it while there is still room for the commands, not after the last
step fails. Back up and keep working; stop only if the limit actually ends the session.

1. Commit everything in the worktree, including unfinished code. Run the hooks; if only the
   ruff or ty hooks fail on unfinished code, skip just those so gitleaks still runs:

   ```bash
   git add -A
   SKIP=ruff-format,ruff-check,ty git commit -m "chore(wip): <task> backup" -m "<what is done, what is next, which checks fail>"
   git push -u origin work/<feature>/<task>
   ```

   Never skip the gitleaks hook, and never use `--no-verify`.
2. Tell the user the branch was pushed and what is left.

To resume in a new session, fetch the branch into a fresh worktree from the main checkout
(reuse the existing worktree instead if it is still on this machine), then continue from
step 3:

```bash
git fetch origin
git worktree add -b work/<feature>/<task> .worktrees/<feature>--<task> origin/work/<feature>/<task>
cd .worktrees/<feature>--<task>
just setup
```

A backed-up work branch is still a work branch: no pull request, and CI does not run on it.
If you rebase it (step 4.1) and need to back it up again, update the backup with
`git push --force-with-lease origin work/<feature>/<task>`; this is the only force-push
allowed without asking, and only on your own work branch.

## 4. Land the task on the feature branch

1. Bring the work branch up to date, since other agents may have landed tasks meanwhile. In
   the task worktree:

   ```bash
   git rebase feat/<feature>
   just check
   ```

   Resolve any conflicts in the task worktree, never on the feature branch.

2. In the main checkout (on `feat/<feature>`, with a clean `git status`), squash the work
   branch into one commit:

   ```bash
   git merge --squash work/<feature>/<task>
   git commit
   ```

   The commit message summarizes the whole task in Conventional Commits form: a subject
   for the outcome and a body listing the notable changes and why. Do not paste the work
   branch's individual commit messages.

3. Right after the squash commit, before anything else lands, confirm nothing was lost,
   then clean up. The diff must be empty before deleting:

   ```bash
   git diff feat/<feature> work/<feature>/<task>
   git worktree remove .worktrees/<feature>--<task>
   git branch -D work/<feature>/<task>
   ```

   `-D` is needed because a squash merge does not mark the branch as merged. If the work
   branch was backed up, also delete the remote copy:
   `git push origin --delete work/<feature>/<task>`.

   If your shell or session is inside the worktree, git cannot remove the worktree you
   are in or delete the branch it has checked out. Detach it instead and delete the
   branch:

   ```bash
   git switch --detach feat/<feature>
   git -C <main checkout> branch -D work/<feature>/<task>
   ```

   Then run `just prune-worktrees` from the main checkout once your session has left the
   worktree, at the latest when the feature merges (step 5). Leave a host-created
   worktree outside `.worktrees/` for the host to clean up.

`just prune-worktrees` removes only what is provably finished and never anything with
uncommitted or untracked files: detached worktrees under `.worktrees/`, work branches
whose commits have all landed on their feature branch (with their worktree), and local
branches whose remote branch is gone after their pull request merged. It lists what it
kept and why. Add `--dry-run` to see what it would do.

If two agents finish at once, land one task first; the other rebases (step 4.1) and lands
after it.

## 5. Finish the feature

Push the feature branch and open a pull request into `main` only when the user asks. Write
the title and description with the `pull-requests` skill:

```bash
git push -u origin feat/<feature>
```

The feature branch is the only branch that goes to GitHub for review. Work branches reach
GitHub only as backups (step 3) and are deleted when the task lands.

After the pull request merges, GitHub deletes the remote feature branch. Switch the main
checkout to `main`, update it, and prune the local feature branch:

```bash
git switch main
git pull --ff-only
just prune-worktrees --fetch
```

## Rules

- Follow the commit rules in `AGENTS.md`: commit or push only when the user asks. Asking
  for work done under this workflow covers local commits on its work branch, the squash
  commit on the feature branch and backup pushes of the work branch (step 3). It never
  covers pushing a feature branch or `main`.
- Do not commit directly on `main`, and do not do task work directly on a feature branch;
  the feature branch only receives squashed commits.
- Create only `main`, `<type>/<feature>` and `work/<feature>/<task>` branches. Never push
  or open a pull request from any other branch, never open a pull request into a feature
  branch, and never stack pull requests.
- Never force-push, rewrite the feature branch's history or delete a branch with unlanded
  work without the user's approval.
- Clean up only with `just prune-worktrees`. If a worktree or work branch from an earlier
  task is still present after it runs, it has work that has not landed: ask before
  removing it; another agent may be using it.
