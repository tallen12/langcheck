"""Remove the worktrees and branches the git workflow no longer needs.

Only removes what is provably finished, and never anything with uncommitted or untracked
files:

- a worktree under ``.worktrees/`` with a detached HEAD (left after landing a task);
- a worktree on ``work/<feature>/<task>`` whose commits have all landed on the feature
  branch, together with that work branch;
- a local branch whose remote branch is gone (deleted after its pull request merged),
  whose changes are all on the default branch, and that no worktree has checked out.

"Landed" is checked by content, not ancestry, because tasks and pull requests land as
squash commits: a branch has landed when merging it into the target changes nothing.

Run with ``uv run python scripts/prune_worktrees.py [--fetch] [--dry-run]`` (``just
prune-worktrees``). ``--fetch`` runs ``git fetch --prune`` first so merged remote branches
show as gone. Works from the main checkout or any worktree.
"""

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

WORKTREES_DIR = ".worktrees"
WORK_PREFIX = "work/"
GONE = "[gone]"
FLAGS = frozenset({"--fetch", "--dry-run"})


@dataclass(frozen=True, slots=True)
class GitResult:
    """What a git command returned."""

    returncode: int
    stdout: str
    stderr: str


class Git(Protocol):
    """Runs git commands, so the policy can be exercised against a fake."""

    def run(self, args: Sequence[str], *, cwd: Path) -> GitResult:
        """Run ``git <args>`` in ``cwd``."""
        ...


class SubprocessGit:
    """Runs the real ``git`` executable."""

    def run(self, args: Sequence[str], *, cwd: Path) -> GitResult:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8"
        )
        return GitResult(done.returncode, done.stdout, done.stderr)


class GitError(Exception):
    """A git command that must succeed failed."""


@dataclass(frozen=True, slots=True)
class Worktree:
    """One entry of ``git worktree list``. ``branch`` is ``None`` when HEAD is detached."""

    path: Path
    branch: str | None
    locked: bool


@dataclass(frozen=True, slots=True)
class Remove:
    """Remove the worktree and then delete ``branch``, if any."""

    path: Path
    branch: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class Keep:
    """Leave the worktree alone."""

    path: Path
    reason: str


type Decision = Remove | Keep


def checked(git: Git, args: Sequence[str], *, cwd: Path) -> str:
    """Run a git command that must succeed and return its output."""
    result = git.run(args, cwd=cwd)
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def main_checkout(git: Git, cwd: Path) -> Path:
    """The main checkout, found from any worktree through the shared git directory."""
    common = checked(git, ["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=cwd)
    return Path(common.strip()).parent


def parse_worktrees(porcelain: str) -> list[Worktree]:
    """Parse ``git worktree list --porcelain``: blocks of lines separated by blank lines."""
    worktrees: list[Worktree] = []
    for block in porcelain.strip().split("\n\n"):
        fields = dict(line.partition(" ")[::2] for line in block.splitlines() if line)
        if "worktree" not in fields or "bare" in fields:
            continue
        ref = fields.get("branch")
        worktrees.append(
            Worktree(
                path=Path(fields["worktree"]),
                branch=ref.removeprefix("refs/heads/") if ref is not None else None,
                locked="locked" in fields,
            )
        )
    return worktrees


def feature_branch_for(work_branch: str, branches: Sequence[str]) -> str | None:
    """The feature branch ``<type>/<feature>`` of ``work/<feature>/<task>``, if exactly one."""
    parts = work_branch.removeprefix(WORK_PREFIX).split("/")
    if not work_branch.startswith(WORK_PREFIX) or len(parts) != 2:
        return None
    candidates = [
        branch
        for branch in branches
        if not branch.startswith(WORK_PREFIX) and branch.partition("/")[2] == parts[0]
    ]
    return candidates[0] if len(candidates) == 1 else None


def adds_changes(git: Git, root: Path, *, target: str, branch: str) -> bool:
    """Whether merging ``branch`` into ``target`` would change ``target``'s files.

    A conflicting merge counts as a change, so the branch is kept.
    """
    merged = git.run(["merge-tree", "--write-tree", target, branch], cwd=root)
    if merged.returncode != 0:
        return True
    target_tree = checked(git, ["rev-parse", f"{target}^{{tree}}"], cwd=root).strip()
    return merged.stdout.split("\n", 1)[0].strip() != target_tree


def has_own_commits(git: Git, root: Path, *, target: str, branch: str) -> bool:
    """Whether ``branch`` has commits that ``target`` does not."""
    count = checked(git, ["rev-list", "--count", f"{target}..{branch}"], cwd=root)
    return int(count.strip()) > 0


def is_clean(git: Git, path: Path) -> bool:
    """No modified, staged or untracked files (ignored files do not count)."""
    return checked(git, ["status", "--porcelain"], cwd=path).strip() == ""


def decide(
    git: Git, root: Path, worktree: Worktree, *, branches: Sequence[str], cwd: Path
) -> Decision:
    """Whether a worktree under ``.worktrees/`` is finished and can be removed."""
    path = worktree.path
    if cwd == path or path in cwd.parents:
        return Keep(path, "in use by the current shell")
    if worktree.locked:
        return Keep(path, "locked")
    if not path.is_dir():
        return Keep(path, "missing; cleared by git worktree prune")
    if not is_clean(git, path):
        return Keep(path, "has uncommitted or untracked files")
    if worktree.branch is None:
        return Remove(path, None, "detached HEAD")
    if not worktree.branch.startswith(WORK_PREFIX):
        return Keep(path, f"on {worktree.branch}, not a work branch")
    feature = feature_branch_for(worktree.branch, branches)
    if feature is None:
        return Keep(path, f"no single feature branch for {worktree.branch}")
    if not has_own_commits(git, root, target=feature, branch=worktree.branch):
        return Keep(path, f"{worktree.branch} has no commits yet")
    if adds_changes(git, root, target=feature, branch=worktree.branch):
        return Keep(path, f"{worktree.branch} has work not on {feature}")
    return Remove(path, worktree.branch, f"{worktree.branch} landed on {feature}")


def local_branches(git: Git, root: Path) -> dict[str, str]:
    """Local branch names mapped to their upstream tracking state (``[gone]``, ...)."""
    output = checked(
        git,
        ["for-each-ref", "--format=%(refname:short)%00%(upstream:track)", "refs/heads"],
        cwd=root,
    )
    return dict(line.split("\0", 1) for line in output.splitlines() if line)


def default_branch(git: Git, root: Path) -> str:
    """``origin``'s default branch when known, otherwise ``main``."""
    result = git.run(["rev-parse", "--abbrev-ref", "origin/HEAD"], cwd=root)
    return result.stdout.strip() if result.returncode == 0 else "main"


def prune(git: Git, cwd: Path, *, fetch: bool, dry_run: bool) -> list[str]:
    """Remove finished worktrees and branches; return one line per action or kept item."""
    root = main_checkout(git, cwd)
    report: list[str] = []
    if fetch:
        fetched = git.run(["fetch", "--prune", "--quiet"], cwd=root)
        if fetched.returncode != 0:
            report.append(f"fetch failed, using local state: {fetched.stderr.strip()}")
    verb = "would remove" if dry_run else "removed"

    branches = local_branches(git, root)
    worktrees = parse_worktrees(checked(git, ["worktree", "list", "--porcelain"], cwd=root))
    managed = root / WORKTREES_DIR
    checked_out = {worktree.branch for worktree in worktrees if worktree.branch is not None}
    for worktree in worktrees:
        if managed not in worktree.path.parents:
            continue
        decision = decide(git, root, worktree, branches=list(branches), cwd=cwd)
        match decision:
            case Remove(path=path, branch=branch, reason=reason):
                if not dry_run:
                    checked(git, ["worktree", "remove", str(path)], cwd=root)
                    if branch is not None:
                        checked(git, ["branch", "-D", branch], cwd=root)
                checked_out.discard(branch)
                report.append(f"{verb} {path.relative_to(root)} ({reason})")
            case Keep(path=path, reason=reason):
                report.append(f"kept {path.relative_to(root)} ({reason})")

    target = default_branch(git, root)
    for branch, track in branches.items():
        if track != GONE:
            continue
        if branch in checked_out:
            report.append(f"kept branch {branch} (remote gone, but checked out)")
        elif adds_changes(git, root, target=target, branch=branch):
            report.append(f"kept branch {branch} (remote gone, has work not on {target})")
        else:
            if not dry_run:
                checked(git, ["branch", "-D", branch], cwd=root)
            report.append(f"{verb} branch {branch} (merged into {target}, remote gone)")

    if not dry_run:
        checked(git, ["worktree", "prune"], cwd=root)
    return report


def main(argv: Sequence[str]) -> int:
    """Parse flags, prune, and print the report."""
    unknown = [arg for arg in argv if arg not in FLAGS]
    if unknown:
        print(f"unknown arguments: {' '.join(unknown)}; expected {sorted(FLAGS)}", file=sys.stderr)
        return 2
    try:
        report = prune(
            SubprocessGit(), Path.cwd(), fetch="--fetch" in argv, dry_run="--dry-run" in argv
        )
    except GitError as error:
        print(error, file=sys.stderr)
        return 1
    for line in report:
        print(line)
    if not report:
        print("nothing to prune")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
