import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langcheck.cli.noising import (
    FAILED,
    Base,
    NoiseStart,
    Saved,
    corrupted,
    mark_edits,
    marks_in,
    noise_app,
    plausible,
)
from langcheck.evaluation.context_pairs import read_pairs, write_pairs
from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from langcheck.noising.requests import SEPARATOR, NoiseRequest
from langcheck.synthesis.clean_text import CleanSentence
from tests.unit.cli.test_main import Cli

PAIR = AnnotatedSentence(("He", "go", "home", "."), ((GoldEdit(1, 2, ("went",), "R:VERB"),),))
PROSE = "The council met on Monday. It approved the new budget for the city. Work starts soon."


@dataclass
class FakeRuntime:
    trained: list[tuple[int, NoiseStart]] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    def train(
        self,
        pairs: Sequence[AnnotatedSentence],
        *,
        start: NoiseStart,
        output: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        self.trained.append((len(pairs), start))

    def write(
        self, model: Path, prompts: Sequence[str], *, device: str, seed: int, top_p: float
    ) -> list[str]:
        self.prompts += prompts
        # Drop the sentence's first article, or its final full stop.
        answers = []
        for found in prompts:
            clean = found.partition(SEPARATOR)[2]
            answers.append(clean.replace("the ", "", 1) if "the " in clean else clean[:-1])
        return answers


def run(cli: Cli, runtime: FakeRuntime, *argv: str) -> int:
    app = noise_app(cli.environment(), runtime=runtime, guarded=lambda body: body())
    code = app(list(argv))
    assert isinstance(code, int)
    return code


def test_train_reads_the_pairs_and_can_continue_a_saved_model(tmp_path: Path) -> None:
    (tmp_path / "p.jsonl").write_text(write_pairs([(PAIR, "")] * 3), encoding="utf-8")
    runtime = FakeRuntime()
    pairs = ("--pairs", str(tmp_path / "p.jsonl"), "--device", "cpu")

    assert run(Cli(tmp_path), runtime, "train", "out", *pairs) == 0
    assert run(Cli(tmp_path), runtime, "train", "out", *pairs, "--start-from", "first") == 0

    assert runtime.trained == [(3, Base("facebook/bart-base")), (3, Saved(Path("first")))]


def test_corrupt_writes_each_sentence_with_its_context(tmp_path: Path) -> None:
    (tmp_path / "prose.jsonl").write_text(json.dumps({"text": PROSE}) + "\n", encoding="utf-8")
    (tmp_path / "kinds.jsonl").write_text(write_pairs([(PAIR, "")]), encoding="utf-8")
    cli, runtime = Cli(tmp_path), FakeRuntime()

    code = run(
        cli,
        runtime,
        "corrupt",
        "model",
        "--clean",
        str(tmp_path / "prose.jsonl"),
        "--kinds-from",
        str(tmp_path / "kinds.jsonl"),
        "--errors",
        "1",
        "--output",
        str(tmp_path / "out.jsonl"),
    )

    assert code == 0
    made = read_pairs((tmp_path / "out.jsonl").read_text(encoding="utf-8"))
    assert [len(s.annotators[0]) for s, _ in made] == [1, 1]
    assert made[1][1] == "The council met on Monday."
    assert all(p.startswith("1 VERB | ") for p in runtime.prompts)
    assert "2 as asked" in cli.out


def test_corrupt_refuses_a_negative_error_count(tmp_path: Path) -> None:
    cli = Cli(tmp_path)
    argv = ("corrupt", "m", "--clean", "c", "--kinds-from", "k", "--output", "o")

    assert run(cli, FakeRuntime(), *argv, "--errors=-1") == FAILED
    assert "--errors" in cli.err[0]


def test_answers_that_rewrite_the_sentence_are_not_plausible() -> None:
    request = NoiseRequest(("DET",))
    small = AnnotatedSentence(("a", "b", "c", "d"), ((GoldEdit(0, 0, ("x",), "M:NOISE"),),))
    rewrite = AnnotatedSentence(
        ("a", "b", "c", "d"), ((GoldEdit(0, 3, ("x", "y", "z"), "R:NOISE"),),)
    )
    too_many = AnnotatedSentence(
        ("a", "b", "c", "d", "e", "f", "g", "h"),
        (tuple(GoldEdit(i, i, ("x",), "M:NOISE") for i in (0, 2, 4)),),
    )

    assert plausible(small, request)
    assert not plausible(rewrite, request)
    assert not plausible(too_many, request)
    assert not plausible(AnnotatedSentence(("a",), ((),)), request)


def test_corrupted_keeps_clean_requests_and_drops_failed_answers() -> None:
    found = [
        CleanSentence(("The", "cat", "sat", "."), "Before."),
        CleanSentence(("A", "dog", "ran", "."), ""),
    ]
    requests = [NoiseRequest(()), NoiseRequest(("DET",))]

    made, outcomes = corrupted(found, requests, ["A dog ran."])

    assert made == [(AnnotatedSentence(("The", "cat", "sat", "."), ((),)), "Before.")]
    assert outcomes == {"clean": 1, "dropped": 1}


def test_mark_edits_count_quotes_and_parentheses_an_edit_touches() -> None:
    made = AnnotatedSentence(
        ("The", '"', "cat", '"', "sat", "."),
        ((GoldEdit(1, 2, (), "U:NOISE"), GoldEdit(4, 4, ("(",), "M:NOISE")),),
    )

    assert marks_in(made.tokens) == 2
    assert mark_edits(made) == 2


def test_corrupted_caps_edits_on_quote_marks() -> None:
    quoted = CleanSentence(
        tuple("The cat named ( Tom ) sat on the mat , the dog said .".split()), ""
    )
    plain = CleanSentence(tuple("The dog sat on the mat today .".split()), "")
    found = [quoted, plain, plain]
    requests = [NoiseRequest(()), NoiseRequest(("PUNCT",)), NoiseRequest(("DET",))]

    made, outcomes = corrupted(
        found, requests, ['The " dog sat on the mat today .', "A dog sat on the mat today ."]
    )

    assert [m.tokens for m, _ in made] == [
        quoted.words,
        tuple("A dog sat on the mat today .".split()),
    ]
    assert outcomes == {"clean": 1, "too many mark edits": 1, "as asked": 1}
