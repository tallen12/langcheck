import dataclasses
import json
import re
from collections import Counter
from pathlib import Path

import httpx

from langcheck.cli.judging import render_summary
from langcheck.cli.main import build_app
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.judging.llm_judge import judge_name
from langcheck.judging.models import Source
from langcheck.judging.panel import Summary, Tally
from langcheck.judging.store import JsonlJudgmentStore, store_path
from tests.unit.cli.test_main import Cli

M2 = """S He go to school .
A 1 2|||R:VERB:TENSE|||went|||REQUIRED|||-NONE-|||0

S It is fine .
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0
"""
PANEL = ("judge-a:1b", "judge-b:1b", "other-c:1b")
URL = ("--llm-url", "http://llm.test/v1")


@dataclasses.dataclass
class JudgeServer:
    """Judges answer by model: ``judge-*`` find every edit valid, others find it wrong."""

    models: list[str] = dataclasses.field(default_factory=list)
    broken: bool = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.models.append(body["model"])
        if self.broken:
            return httpx.Response(200, json={"choices": [{"message": {"content": "no"}}]})
        edits = re.findall(r"^(\d+)\. ", body["messages"][1]["content"], re.M)
        verdict = "valid" if body["model"].startswith("judge-") else "wrong"
        reply = {
            "edits": [{"edit": int(n), "reason": "r", "verdict": verdict} for n in edits],
            "remaining_reason": "",
            "remaining_error": False,
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})


def run(tmp_path: Path, server: JudgeServer, *argv: str) -> tuple[int, Cli]:
    cli = Cli(tmp_path)
    env = dataclasses.replace(cli.environment(), transport=httpx.MockTransport(server))
    code = build_app(env)(list(argv))
    assert isinstance(code, int)
    return code, cli


def fetched(root: Path) -> None:
    [path] = dataset_files(Dataset.CWEB_G_DEV, root / "data" / "eval")
    path.parent.mkdir(parents=True)
    path.write_text(M2, encoding="utf-8")


def test_the_panel_judges_gold_and_run_edits_and_stores_each_verdict(tmp_path: Path) -> None:
    fetched(tmp_path)
    corrections = tmp_path / "run.jsonl"
    corrections.write_text(
        '{"text": "It is fine .", "corrected": "It is good ."}\n'
        '{"text": "Not in the set", "corrected": "Not in the set."}\n',
        encoding="utf-8",
    )
    server = JudgeServer()
    models = ("--models", *PANEL)

    code, cli = run(
        tmp_path,
        server,
        "judge",
        "run",
        "cweb-g-dev",
        *models,
        *URL,
        "--corrections",
        str(corrections),
    )

    assert code == 0, cli.err
    assert server.models == [m for m in PANEL for _ in range(2)]
    stored = JsonlJudgmentStore(store_path(tmp_path / "data" / "judgments", "cweb-g-dev")).load()
    assert len(stored) == 3 * (2 + 2)
    assert "1 of 2 lines match no sentence" in cli.output
    assert "gold edits: 1 (valid 1, neutral 0, wrong 0, disputed 0, pending 0)" in cli.output
    assert "run edits: 1 (valid 1, neutral 0, wrong 0, disputed 0, pending 0)" in cli.output

    again = JudgeServer()
    code, cli = run(tmp_path, again, "judge", "show", "cweb-g-dev", *models)

    assert code == 0
    assert again.models == []
    assert "gold edits: 1 (valid 1" in cli.output
    assert "run edits" not in cli.output


def test_a_judge_of_the_correctors_family_gives_way_to_the_standby(tmp_path: Path) -> None:
    fetched(tmp_path)
    server = JudgeServer()

    code, cli = run(
        tmp_path,
        server,
        "judge",
        "run",
        "cweb-g-dev",
        "--models",
        *PANEL,
        *URL,
        "--corrector",
        "other-c:7b",
        "--standby",
        "judge-d:1b",
        "--limit",
        "1",
    )

    assert code == 0, cli.err
    assert server.models == ["judge-a:1b", "judge-b:1b", "judge-d:1b"]
    assert "panel: judge-a:1b, judge-b:1b, judge-d:1b (quorum 2)" in cli.output


def test_unusable_replies_fail_the_run_and_store_nothing(tmp_path: Path) -> None:
    fetched(tmp_path)

    code, cli = run(
        tmp_path,
        JudgeServer(broken=True),
        "judge",
        "run",
        "cweb-g-dev",
        "--models",
        *PANEL,
        *URL,
        "--limit",
        "1",
    )

    assert code == 2
    assert f"{judge_name('judge-a:1b')}: judged 0, already had 0, failed 1" in cli.output
    assert not (tmp_path / "data" / "judgments").exists()


def test_an_unfetched_set_or_a_bad_quorum_is_refused(tmp_path: Path) -> None:
    code, cli = run(tmp_path, JudgeServer(), "judge", "run", "cweb-g-dev", *URL)

    assert code == 2
    assert "langcheck evaluate fetch cweb-g-dev" in cli.output + "\n".join(cli.err)

    code, cli = run(tmp_path, JudgeServer(), "judge", "show", "cweb-g-dev", "--quorum", "4")

    assert code == 2
    assert "--quorum takes 1 to 3" in "\n".join(cli.err)


def test_the_summary_names_each_judges_counts_and_agreement() -> None:
    summary = Summary(
        {source: Tally() for source in Source}, Tally(), {"a": Counter()}, {("a", "b"): 0.5}
    )

    text = render_summary(summary)

    assert "edits" not in text
    assert "error left in sentence: 0 (yes 0, no 0, disputed 0, pending 0)" in text
    assert "a: nothing judged" in text
    assert "agreement a / b: 50%" in text
