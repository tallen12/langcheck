import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import httpx

from langcheck.cli.suggesting import (
    CLEAN,
    FAILED,
    FOUND,
    render_outcome,
    render_rejected,
    suggest_app,
)
from langcheck.cli.tagging import Trained
from langcheck.correction.models import Corrected, Rejected, RejectionReason, Unavailable
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.m2 import AnnotatedSentence
from langcheck.settings.loading import ClientSettings, load_settings
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.tiers import FlagKind, TextFlag, Tier
from tests.conftest import WordTableTagger
from tests.unit.cli.test_main import Cli

FLAG = TextFlag(CharSpan(3, 5), 0.9, Tier.ERROR, FlagKind.GENERAL)


class WordTableRuntime:
    def train(
        self,
        sentences: Sequence[AnnotatedSentence],
        *,
        base_model: str,
        output: Path,
        epochs: int,
        batch_size: int,
        precision: float,
        context_sentences: int,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        raise NotImplementedError

    def load(self, path: Path, *, device: str) -> Trained:
        return Trained(WordTableTagger({"go": 0.9}), 0.5, "base", 0.5)

    def train_commas(
        self,
        path: Path,
        sentences: Sequence[AnnotatedSentence],
        *,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        raise NotImplementedError


def corrections_reply(request: httpx.Request) -> httpx.Response:
    content = json.dumps(
        {
            "flags": [
                {
                    "flag": "F1",
                    "corrections": [
                        {"replacement": "goes", "kind": "VerbForm", "explanation": "Agreement."},
                        {"replacement": "go", "kind": "VerbForm", "explanation": "Same."},
                    ],
                }
            ]
        }
    )
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def run(cli: Cli, *argv: str) -> int:
    environment = replace(cli.environment(), transport=httpx.MockTransport(corrections_reply))

    def settings(_: Path | None, __: object) -> ClientSettings:
        return load_settings(ClientSettings, overrides={}, yaml_data={})

    app = suggest_app(
        environment, runtime=WordTableRuntime(), settings=settings, guarded=lambda body: body()
    )
    code = app(list(argv))
    assert isinstance(code, int)
    return code


def test_suggest_prints_the_flag_and_its_ranked_corrections(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, "tagger", "He go home.", "--device", "cpu") == FOUND

    assert "3-5\t0.900\terror\tgo" in cli.output
    assert "1.\t-0.900\thigh\t'go' -> 'goes'\tVerbForm\tAgreement." in cli.output
    assert "no_change" not in cli.output


def test_suggest_can_show_what_it_rejected(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, "tagger", "He go home.", "--show-rejected") == FOUND

    assert "no_change" in cli.output


def test_clean_text_exits_clean(tmp_path: Path) -> None:
    assert run(Cli(tmp_path), "tagger", "All is well.") == CLEAN


def test_measure_scores_a_fetched_set_and_saves_the_report(tmp_path: Path) -> None:
    [path] = dataset_files(Dataset.BEA19_DEV, tmp_path / "data" / "eval")
    path.parent.mkdir(parents=True)
    path.write_text(
        "S He go home .\nA 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0\n", encoding="utf-8"
    )
    cli = Cli(tmp_path)

    code = run(
        cli, "measure", "tagger", "bea19-dev", "--output", str(tmp_path / "out" / "report.json")
    )

    assert code == CLEAN
    assert "tagger-ranked\t1.000\t1.000\t1.000" in cli.output
    assert "bea19-dev" in (tmp_path / "out" / "report.json").read_text(encoding="utf-8")


def test_measure_asks_for_a_set_not_yet_fetched(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, "measure", "tagger", "bea19-dev") == FAILED
    assert "fetch bea19-dev" in "\n".join(cli.err)


def test_measure_file_scores_text_and_corrected_lines(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        '{"text": "He go home .", "corrected": "He goes home .", "model": "x"}\n', encoding="utf-8"
    )
    cli = Cli(tmp_path)

    assert run(cli, "measure-file", "tagger", str(pairs)) == CLEAN

    assert "pairs: 1 sentences, 1 flags" in cli.output
    assert "tagger-ranked\t1.000\t1.000\t1.000" in cli.output


def test_an_unavailable_flag_says_why() -> None:
    lines = render_outcome("He go home.", Unavailable(FLAG, "LLM error: down"), show_rejected=False)

    assert lines[1] == "  unavailable: LLM error: down"


def test_a_flag_with_no_corrections_is_only_its_line() -> None:
    lines = render_outcome("He go home.", Corrected(FLAG, (), ()), show_rejected=True)

    assert len(lines) == 1


def test_a_rejection_after_judging_shows_its_reduction() -> None:
    line = render_rejected(Rejected("go", "gos", RejectionReason.NO_IMPROVEMENT, 0.05))

    assert line == "  x\t-0.050\t'go' -> 'gos'\tno_improvement"
