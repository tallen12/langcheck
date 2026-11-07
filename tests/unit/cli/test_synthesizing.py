from pathlib import Path

from langcheck.cli.synthesizing import FAILED, synth_app
from langcheck.evaluation.context_pairs import Origin, pairs_origin, read_pairs, write_pairs
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from tests.unit.cli.test_main import Cli

M2 = """S we go in home now .
A 2 3|||R:PREP|||to|||REQUIRED|||-NONE-|||0

S we go in home now .
A 2 3|||R:PREP|||to|||REQUIRED|||-NONE-|||0
"""


def run(cli: Cli, *argv: str) -> int:
    code = synth_app(cli.environment(), guarded=lambda body: body())(list(argv))
    assert isinstance(code, int)
    return code


def prose(root: Path) -> str:
    path = root / "clean.jsonl"
    path.write_text('{"text": "First we go to bed. Then we go to town."}\n', encoding="utf-8")
    return str(path)


def written(path: Path) -> list[tuple[AnnotatedSentence, str]]:
    return read_pairs(path.read_text(encoding="utf-8"))


def test_errors_follow_the_tables_learned_from_labeled_pairs(tmp_path: Path) -> None:
    (tmp_path / "labeled.m2").write_text(M2, encoding="utf-8")
    output = tmp_path / "out" / "made.jsonl"
    cli = Cli(tmp_path)
    tables = ("--tables", str(tmp_path / "labeled.m2"), "--errors", "1")

    assert run(cli, "corrupt", "--clean", prose(tmp_path), "--output", str(output), *tables) == 0

    made = written(output)
    assert [sentence.tokens for sentence, _ in made] == [
        ("First", "we", "go", "in", "bed", "."),
        ("Then", "we", "go", "in", "town", "."),
    ]
    assert made[1][1] == "First we go in bed."
    assert {edit.code for sentence, _ in made for edit in sentence.annotators[0]} == {"R:PREP"}
    assert pairs_origin(output.read_text(encoding="utf-8")) is Origin.OPEN
    assert "2 sentences, 2 errors (R:PREP 2); open data" in cli.output


def test_without_tables_the_built_in_ones_are_used_and_copies_repeat(tmp_path: Path) -> None:
    output = tmp_path / "made.jsonl"
    args = ("--output", str(output), "--copies", "3", "--context", "0")

    assert run(Cli(tmp_path), "corrupt", "--clean", prose(tmp_path), *args) == 0

    assert len(written(output)) == 6


def test_errors_from_an_evaluation_set_are_marked_as_evaluation_data(tmp_path: Path) -> None:
    [path] = dataset_files(Dataset.CONLL14_TEST, tmp_path / "data" / "eval")
    path.parent.mkdir(parents=True)
    path.write_text(M2, encoding="utf-8")
    output = tmp_path / "eval.jsonl"
    cli = Cli(tmp_path)

    code = run(
        cli,
        "corrupt",
        "--clean",
        prose(tmp_path),
        "--output",
        str(output),
        "--from-set",
        "conll14-test",
    )

    assert code == 0
    assert pairs_origin(output.read_text(encoding="utf-8")) is Origin.EVALUATION
    assert "evaluation data" in cli.output


def test_the_mark_carries_over_from_marked_pairs(tmp_path: Path) -> None:
    pair = AnnotatedSentence(("we", "go", "in", "home"), ((GoldEdit(2, 3, ("to",), "R:PREP"),),))
    marked = tmp_path / "marked.jsonl"
    marked.write_text(write_pairs([(pair, "")] * 2, origin=Origin.EVALUATION), encoding="utf-8")
    output = tmp_path / "made.jsonl"

    code = run(
        Cli(tmp_path),
        "corrupt",
        "--clean",
        prose(tmp_path),
        "--output",
        str(output),
        "--tables",
        str(marked),
    )

    assert code == 0
    assert pairs_origin(output.read_text(encoding="utf-8")) is Origin.EVALUATION


def test_a_set_without_typed_edits_or_not_fetched_is_refused(tmp_path: Path) -> None:
    cli = Cli(tmp_path)
    base = ("corrupt", "--clean", prose(tmp_path), "--output", str(tmp_path / "o.jsonl"))

    assert run(cli, *base, "--from-set", "jfleg-dev") == FAILED
    assert run(cli, *base, "--from-set", "bea19-dev") == FAILED
    assert run(cli, *base, "--copies", "0") == FAILED

    assert "no typed edits" in cli.err[0]
    assert "evaluate fetch bea19-dev" in cli.err[1]
    assert "--copies" in cli.err[2]
