"""``langcheck noise``: train a tagged noise model and corrupt prose with it."""

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Protocol, assert_never

from cyclopts import App, Parameter

from langcheck.cli.environment import CliEnvironment
from langcheck.cli.tagging import read_pair_file
from langcheck.evaluation.context_pairs import write_pairs
from langcheck.evaluation.detokenize import detokenize
from langcheck.evaluation.m2 import AnnotatedSentence
from langcheck.noising.requests import NoiseRequest, kind_mix, noisy_sentence, prompt
from langcheck.synthesis.clean_text import CleanSentence, clean_sentences, read_texts

CLEAN: Final = 0
FAILED: Final = 2
DEFAULT_MODEL: Final = "facebook/bart-base"
MAX_LENGTH: Final = 128
MAX_CHANGED_SHARE: Final = 0.5
"""An answer that changes more of the sentence than this rewrote it rather than erred."""
MAX_EXTRA_EDITS: Final = 1
"""An answer may make one edit more than asked; more is a model running off."""
MARKS: Final = frozenset({'"', "\u201c", "\u201d", "'", "\u2018", "\u2019", "(", ")"})
MAX_MARK_EDIT_SHARE: Final = 0.05
"""Edits on quote marks and parentheses, against the marks in the kept text, are capped near
OmniGEC's rate. The noise model makes about twice as many, mostly quotes around names, and
they taught the tagger that correct quotes and parentheses are mistakes."""

type Guard = Callable[[Callable[[], int]], int]


@dataclass(frozen=True, slots=True)
class Base:
    """Start from a pretrained sequence-to-sequence model."""

    name: str


@dataclass(frozen=True, slots=True)
class Saved:
    """Continue training a saved noise model."""

    path: Path


type NoiseStart = Base | Saved


class NoiseRuntime(Protocol):
    """Training and sampling, kept behind a protocol so the commands test without them."""

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
        """Train a noise model on ``pairs``, written as their writers wrote them; save it."""
        ...

    def write(
        self, model: Path, prompts: Sequence[str], *, device: str, seed: int, top_p: float
    ) -> list[str]:
        """The saved noise model's sampled answer to each prompt."""
        ...


class TransformerNoiseRuntime:
    """The real runtime, importing PyTorch only when a command needs it."""

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
        """Train a noise model on ``pairs``, written as their writers wrote them; save it."""
        import torch

        from langcheck.noising.model import (
            NoiseInfo,
            NoiseOptions,
            load_seq2seq,
            read_noise_info,
            save_noise_model,
            train_noise_model,
        )
        from langcheck.noising.requests import training_example
        from langcheck.tagging.encoding import load_tokenizer
        from langcheck.tagging.loaded import torch_limit
        from langcheck.tagging.training import split_holdout

        target = torch.device(device)
        torch_limit(target)
        match start:
            case Base(name=name):
                base_model = name
                source: str | Path = name
            case Saved(path=path):
                info = read_noise_info(path)
                base_model = info.base_model
                source = path
                log(f"continuing from {path} (epoch {info.epoch})")
            case _:
                assert_never(start)
        model = load_seq2seq(source, device=target)
        tokenizer = load_tokenizer(source)
        examples = [found for s in pairs if (found := training_example(s)) is not None]
        training, validation = split_holdout(examples)
        log(f"{len(training)} training pairs, {len(validation)} validation pairs")
        options = NoiseOptions(
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, max_length=MAX_LENGTH
        )
        result = train_noise_model(
            model, tokenizer, training, validation, device=target, options=options, log=log
        )
        save_noise_model(
            output,
            model,
            tokenizer,
            NoiseInfo(
                base_model=base_model,
                train_pairs=len(training),
                validation_pairs=len(validation),
                epoch=result.epoch,
                validation_loss=result.validation_loss,
                count_match=result.count_match,
            ),
        )

    def write(
        self, model: Path, prompts: Sequence[str], *, device: str, seed: int, top_p: float
    ) -> list[str]:
        """The saved noise model's sampled answer to each prompt."""
        import torch

        from langcheck.noising.model import Sampling, load_noise_model
        from langcheck.tagging.loaded import torch_limit

        target = torch.device(device)
        torch_limit(target)
        noise, _ = load_noise_model(model, device=target, max_length=MAX_LENGTH)
        return noise.write(prompts, Sampling(top_p=top_p, seed=seed))


def plausible(made: AnnotatedSentence, request: NoiseRequest) -> bool:
    """Whether an answer made about the errors asked for, rather than a rewrite."""
    edits = made.annotators[0]
    if not edits or len(edits) > len(request.kinds) + MAX_EXTRA_EDITS:
        return False
    changed = sum(max(e.end - e.start, len(e.correction)) for e in edits)
    return changed <= MAX_CHANGED_SHARE * max(1, len(made.tokens))


def marks_in(words: Sequence[str]) -> int:
    """How many of ``words`` are quote marks or parentheses."""
    return sum(word in MARKS for word in words)


def mark_edits(made: AnnotatedSentence) -> int:
    """How many quote marks and parentheses ``made``'s edits remove, change or insert."""
    return sum(
        marks_in((*made.tokens[edit.start : edit.end], *edit.correction))
        for edit in made.annotators[0]
    )


def corrupted(
    found: Sequence[CleanSentence],
    requests: Sequence[NoiseRequest],
    answers: Sequence[str],
) -> tuple[list[tuple[AnnotatedSentence, str]], Counter[str]]:
    """Each sentence as the model wrote it, or clean when asked for no errors; answers
    that are not plausible are dropped, as are answers whose edits on quote marks and
    parentheses would pass ``MAX_MARK_EDIT_SHARE`` of the marks kept so far. ``answers``
    follow the requests with errors.
    """
    made: list[tuple[AnnotatedSentence, str]] = []
    outcomes: Counter[str] = Counter()
    pending = iter(answers)
    marks = edited_marks = 0
    for sentence, request in zip(found, requests, strict=True):
        if not request.kinds:
            made.append((AnnotatedSentence(sentence.words, ((),)), sentence.context))
            marks += marks_in(sentence.words)
            outcomes["clean"] += 1
            continue
        written = noisy_sentence(sentence.words, next(pending))
        if not plausible(written, request):
            outcomes["dropped"] += 1
            continue
        on_marks = mark_edits(written)
        written_marks = marks_in(written.tokens)
        if on_marks and edited_marks + on_marks > MAX_MARK_EDIT_SHARE * (marks + written_marks):
            outcomes["too many mark edits"] += 1
            continue
        marks += written_marks
        edited_marks += on_marks
        edits = len(written.annotators[0])
        outcomes["as asked" if edits == len(request.kinds) else "one more"] += 1
        made.append((written, sentence.context))
    return made, outcomes


def noise_app(env: CliEnvironment, *, runtime: NoiseRuntime, guarded: Guard) -> App:
    """The ``noise`` commands."""
    app = App(
        name="noise",
        help="Train a tagged noise model and write synthetic errors with it.",
        result_action="return_value",
    )

    @app.command(name="train")
    def train_command(
        output: Path,
        /,
        *,
        pairs: Annotated[
            list[Path],
            Parameter(help="Sentences with their edits (M2 or JSONL) to learn errors from."),
        ],
        model: Annotated[str, Parameter(help="Pretrained model to fine-tune.")] = DEFAULT_MODEL,
        start_from: Annotated[
            Path | None, Parameter(help="Continue training this saved noise model instead.")
        ] = None,
        epochs: int = 2,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
        limit: Annotated[int | None, Parameter(help="Only the first N sentences.")] = None,
        device: str = "cuda",
    ) -> int:
        """Train a noise model to write corrected sentences back with their errors."""

        def run() -> int:
            found = [s for path in pairs for s, _ in read_pair_file(path)]
            runtime.train(
                found[:limit],
                start=Base(model) if start_from is None else Saved(start_from),
                output=output,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                log=env.out,
            )
            env.out(f"saved: {output}")
            return CLEAN

        return guarded(run)

    @app.command(name="corrupt")
    def corrupt_command(
        noise_model: Path,
        /,
        *,
        clean: Annotated[Path, Parameter(help="JSONL of edited prose, one text per line.")],
        output: Annotated[Path, Parameter(help="JSONL of the corrupted sentences.")],
        kinds_from: Annotated[
            list[Path],
            Parameter(help="Pairs (M2 or JSONL) whose kinds of error to ask for as often."),
        ],
        errors: Annotated[
            list[int],
            Parameter(
                help="Error counts to draw from, such as 0 1 2 3 for a quarter each.",
                consume_multiple=True,
            ),
        ],
        context: Annotated[int, Parameter(help="Preceding sentences to keep.")] = 2,
        limit: Annotated[int | None, Parameter(help="Only the first N sentences.")] = None,
        top_p: float = 0.9,
        seed: int = 0,
        device: str = "cuda",
    ) -> int:
        """Write each sentence of CLEAN with errors drawn as asked, keeping its context."""

        def run() -> int:
            if any(count < 0 for count in errors):
                env.err("langcheck: --errors takes counts of 0 or more")
                return FAILED
            found = list(clean_sentences(read_texts(clean), context_sentences=context))[:limit]
            mix = kind_mix(
                [s for path in kinds_from for s, _ in read_pair_file(path)], counts=errors
            )
            rng = random.Random(seed)
            requests = [mix.draw(rng) for _ in found]
            prompts = [
                prompt(detokenize(s.words), r)
                for s, r in zip(found, requests, strict=True)
                if r.kinds
            ]
            env.out(f"asking for errors in {len(prompts)} of {len(found)} sentences")
            answers = runtime.write(noise_model, prompts, device=device, seed=seed, top_p=top_p)
            made, outcomes = corrupted(found, requests, answers)
            output.write_text(write_pairs(made), encoding="utf-8")
            env.out(", ".join(f"{n} {outcome}" for outcome, n in sorted(outcomes.items())))
            env.out(f"saved: {output}")
            return CLEAN

        return guarded(run)

    return app
