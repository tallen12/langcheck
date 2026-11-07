"""Trace the tagger and proposer on a clean set, sentence by sentence, as JSON.

For every sentence: the gold edits with the judges' verdicts, the clean references, each
tagger flag, the proposer's whole beam before and after reranking, what the filters did
with the top proposals, the output, its score, and where each reference edit was lost.

Run from the repository root:

    uv run python .agents/skills/model-trace/scripts/trace.py \
        --tagger runs/tagger-deberta-marks --proposer runs/raw/proposer-deberta-marks \
        --dataset cweb-g-dev --limit 300 --out trace.json

Sentences run one at a time so each flag's rerank can be tied to its outcome. The
settings are those of ``langcheck suggest measure-file``, so the totals match its report.
"""

import argparse
import asyncio
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import assert_never

from pydantic import BaseModel, ConfigDict

from langcheck.cli.environment import load_edit_annotator
from langcheck.cli.judging import pooled
from langcheck.cli.tagging import DEFAULT_POLICY
from langcheck.correction.measure import apply_all
from langcheck.correction.models import (
    Candidate,
    Corrected,
    CorrectionPolicy,
    Unavailable,
)
from langcheck.correction.pipeline import CorrectionPipeline
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.detokenize import detokenize
from langcheck.evaluation.examples import Example, load_corrected_pairs, load_examples
from langcheck.evaluation.models import ScoredEdit
from langcheck.evaluation.protocols import EditAnnotator
from langcheck.evaluation.scoring import Counts, best_annotator
from langcheck.judging.consensus import DEFAULT_QUORUM, Agreed, Consensus, Disputed, Pending
from langcheck.judging.llm_judge import judge_name
from langcheck.judging.panel import ItemConsensus, consensus_by_item
from langcheck.judging.render import written_span
from langcheck.judging.roster import DEFAULT_PANEL
from langcheck.judging.store import JsonlJudgmentStore, store_path
from langcheck.proposal.corrector import JUDGED, ProposerCorrector, proposer_policy
from langcheck.proposal.loaded import load_proposing
from langcheck.proposal.rerank import RerankPolicy, Scored
from langcheck.tagging.tiers import TextFlag, TierPolicy


class Stage(StrEnum):
    """Where the model lost a reference edit, or that it made it."""

    OFFERED = "offered"
    FILTERED = "filtered"
    """The right fix was among the proposals the filters saw, and none offered it."""
    BEAM_NOT_TOP8 = "beam_not_top8"
    """The right fix was in the beam but ranked below the proposals the filters see."""
    NOT_PROPOSED = "not_proposed"
    NOT_FLAGGED = "not_flagged"


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BeamRow(Record):
    text: str
    logp: float
    drop: float
    rerank: int
    rerank_score: float


class Offer(Record):
    text: str
    rank: int
    confidence: str
    reduction: float


class Refusal(Record):
    text: str
    rank: int | None
    """``None`` when it was refused before it was placed in the text."""
    reason: str
    reduction: float | None


class FlagTrace(Record):
    start: int
    end: int
    text: str
    score: float
    tier: str
    kind: str
    beam: tuple[BeamRow, ...] = ()
    offered: tuple[Offer, ...] = ()
    rejected: tuple[Refusal, ...] = ()
    unavailable: str | None = None


class GoldEdit(Record):
    original: str
    correction: str
    verdict: str
    votes: int
    linked: bool


class GoldVersion(Record):
    label: str
    edits: tuple[GoldEdit, ...]


class ReferenceEdit(Record):
    original: str
    correction: str
    at: int
    stage: Stage
    why: tuple[str, ...]


class SentenceTrace(Record):
    text: str
    references: tuple[str, ...]
    scored_against: int
    gold: tuple[GoldVersion, ...]
    preference: str | None
    flags: tuple[FlagTrace, ...]
    reference_edits: tuple[tuple[ReferenceEdit, ...], ...]
    output: str
    tp: int
    fp: int
    fn: int


class Summary(Record):
    dataset: str
    tagger: str
    proposer: str
    tp: int
    fp: int
    fn: int
    main_reference_edits: int
    scored_reference_edits: int
    stages: dict[Stage, int]


class TraceFile(Record):
    summary: Summary
    sentences: tuple[SentenceTrace, ...]


class CleanLine(BaseModel):
    """The fields of a ``langcheck judge clean`` line this script reads."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    sentence: str
    corrected: str
    alternatives: tuple[str, ...] = ()


@dataclass
class Recorder:
    """A ``RerankObserver`` keeping the last sentence's reranks by flag span."""

    seen: dict[tuple[int, int], tuple[Sequence[Scored], Sequence[Scored]]] = field(
        default_factory=dict
    )

    def reranked(
        self, flag: TextFlag, proposals: Sequence[Scored], ordered: Sequence[Scored]
    ) -> None:
        self.seen[flag.span.start, flag.span.end] = (proposals, ordered)


def beam_rows(
    proposals: Sequence[Scored], ordered: Sequence[Scored], policy: RerankPolicy
) -> tuple[BeamRow, ...]:
    rank = {id(scored): index for index, scored in enumerate(ordered)}
    return tuple(
        BeamRow(
            text=p.text,
            logp=round(p.log_probability, 2),
            drop=round(p.drop, 2),
            rerank=rank[id(p)] + 1,
            rerank_score=round(
                p.log_probability + policy.weight * p.drop - policy.penalty * p.removed, 2
            ),
        )
        for p in proposals
    )


def verdict[T: Hashable](outcome: Consensus[T]) -> tuple[str, int]:
    match outcome:
        case Agreed(value=value, votes=votes):
            return str(value), votes
        case Disputed():
            return "disputed", 0
        case Pending():
            return "pending", 0
        case unreachable:
            assert_never(unreachable)


def gold_versions(decided: ItemConsensus) -> tuple[GoldVersion, ...]:
    item = decided.item
    return tuple(
        GoldVersion(
            label=version.label,
            edits=tuple(
                GoldEdit(
                    original=written_span(item.tokens, start, end, item.written)
                    if end > start
                    else "",
                    correction=detokenize(list(correction)),
                    verdict=verdict(decided.edits[start, end, correction])[0],
                    votes=verdict(decided.edits[start, end, correction])[1],
                    linked=(start, end, correction) in decided.linked,
                )
                for start, end, correction in version.edits
            ),
        )
        for version in item.versions
    )


def token_offsets(text: str, tokens: Sequence[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = 0
    for token in tokens:
        start = text.find(token, position)
        spans.append((start, start + len(token)))
        position = start + len(token)
    return spans


def fixes(
    annotator: EditAnnotator, tokens: Sequence[str], text: str, flag: FlagTrace, replacement: str
) -> set[tuple[int, int, tuple[str, ...]]]:
    fixed = text[: flag.start] + replacement + text[flag.end :]
    return {edit.key for edit in annotator.system_edits(tokens, fixed)}


def reference_edits(
    annotator: EditAnnotator,
    example: Example,
    flags: Sequence[FlagTrace],
    output: str,
) -> tuple[tuple[ReferenceEdit, ...], ...]:
    text, tokens = example.text, example.source
    spans = token_offsets(text, tokens)
    made = {edit.key for edit in annotator.system_edits(tokens, output)}
    result: list[tuple[ReferenceEdit, ...]] = []
    for reference in example.annotators:
        edits: list[ReferenceEdit] = []
        for edit in reference:
            low = spans[edit.start][0] if edit.start < len(spans) else len(text)
            high = spans[edit.end - 1][1] if edit.end > edit.start else low
            near = [f for f in flags if f.start <= high and f.end >= low]
            judged = [
                _fate(choice)
                for f in near
                for choice in (*f.offered, *f.rejected)
                if edit.key in fixes(annotator, tokens, text, f, choice.text)
            ]
            in_beam = any(
                edit.key in fixes(annotator, tokens, text, f, row.text)
                for f in near
                for row in f.beam
            )
            if edit.key in made:
                stage = Stage.OFFERED
            elif not near:
                stage = Stage.NOT_FLAGGED
            elif judged:
                stage = Stage.FILTERED
            elif in_beam:
                stage = Stage.BEAM_NOT_TOP8
            else:
                stage = Stage.NOT_PROPOSED
            edits.append(
                ReferenceEdit(
                    original=text[low:high],
                    correction=detokenize(list(edit.correction)),
                    at=low,
                    stage=stage,
                    why=tuple(dict.fromkeys(judged)),
                )
            )
        result.append(tuple(edits))
    return tuple(result)


def _fate(choice: Offer | Refusal) -> str:
    match choice:
        case Offer():
            return Stage.OFFERED.value
        case Refusal(reason=reason):
            return reason
        case unreachable:
            assert_never(unreachable)


def flag_trace(
    text: str, outcome: Corrected | Unavailable, recorder: Recorder, policy: RerankPolicy
) -> tuple[FlagTrace, Candidate | None]:
    flag = outcome.flag
    reranks = recorder.seen.get((flag.span.start, flag.span.end))
    beam = beam_rows(*reranks, policy) if reranks is not None else ()
    base = FlagTrace(
        start=flag.span.start,
        end=flag.span.end,
        text=text[flag.span.start : flag.span.end],
        score=round(flag.score, 3),
        tier=str(flag.tier),
        kind=str(flag.kind),
        beam=beam,
    )
    match outcome:
        case Corrected(ranked=ranked, rejected=rejected):
            offered = tuple(
                Offer(
                    text=r.candidate.replacement,
                    rank=r.candidate.position + 1,
                    confidence=str(r.confidence),
                    reduction=round(r.reduction, 3),
                )
                for r in ranked
            )
            refused = tuple(
                Refusal(
                    text=r.replacement,
                    rank=None if r.candidate is None else r.candidate.position + 1,
                    reason=str(r.reason),
                    reduction=None if r.reduction is None else round(r.reduction, 3),
                )
                for r in rejected
            )
            first = ranked[0].candidate if ranked else None
            return base.model_copy(update={"offered": offered, "rejected": refused}), first
        case Unavailable(reason=reason):
            return base.model_copy(update={"unavailable": reason}), None
        case unreachable:
            assert_never(unreachable)


class Options(Record):
    """The command line, validated."""

    tagger: Path
    proposer: Path
    dataset: Dataset
    limit: int | None
    clean: Path | None
    data: Path
    store: Path
    device: str
    out: Path

    @property
    def clean_set(self) -> Path:
        """The clean set to trace, by default the local one beside the set's judgments."""
        if self.clean is not None:
            return self.clean
        return store_path(self.store, self.dataset.value).parent / "clean-local.jsonl"


async def trace(options: Options) -> TraceFile:
    annotator = load_edit_annotator()
    lines = [
        CleanLine.model_validate_json(line)
        for line in options.clean_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples = load_corrected_pairs(options.clean_set, annotator, limit=len(lines))
    files = dataset_files(options.dataset, options.data)
    items = pooled(
        load_examples(options.dataset, files, annotator, limit=options.limit),
        (),
        annotator,
        print,
    )
    judges = [judge_name(model, think=False) for model in DEFAULT_PANEL]
    stored = JsonlJudgmentStore(store_path(options.store, options.dataset.value)).load()
    decided = {
        d.item.key: d for d in consensus_by_item(items, stored, judges, quorum=DEFAULT_QUORUM)
    }

    loaded = load_proposing(options.tagger, options.proposer, device=options.device)
    policy = proposer_policy(CorrectionPolicy(), loaded.gate)
    recorder = Recorder()
    corrector = ProposerCorrector(loaded.source, loaded.trained.reader, policy, observer=recorder)
    pipeline = CorrectionPipeline(
        corrector=corrector,
        reader=loaded.trained.reader,
        tiers=TierPolicy(error=DEFAULT_POLICY.error, suggestion=DEFAULT_POLICY.suggestion),
        policy=policy,
        context_sentences=loaded.trained.context_sentences,
    )

    running = Counts()
    stages: Counter[Stage] = Counter()
    main = scored = 0
    sentences: list[SentenceTrace] = []
    for index, (line, example) in enumerate(zip(lines, examples, strict=True)):
        recorder.seen.clear()
        traced: list[FlagTrace] = []
        chosen: list[Candidate] = []
        for outcome in await pipeline.suggest(example.text):
            flag, first = flag_trace(example.text, outcome, recorder, corrector.rerank)
            traced.append(flag)
            if first is not None:
                chosen.append(first)
        output = apply_all(example.text, chosen)
        system: list[ScoredEdit] = annotator.system_edits(example.source, output)
        against, score = best_annotator(system, example.annotators, running=running)
        running = running + score.overall
        main += len(example.annotators[0])
        scored += len(example.annotators[against])
        edits = reference_edits(annotator, example, traced, output)
        stages.update(edit.stage for edit in edits[0])
        sentence = decided[line.sentence]
        preference = None if sentence.preference is None else verdict(sentence.preference)[0]
        sentences.append(
            SentenceTrace(
                text=example.text,
                references=(line.corrected, *line.alternatives),
                scored_against=against,
                gold=gold_versions(sentence),
                preference=preference,
                flags=tuple(traced),
                reference_edits=edits,
                output=output,
                tp=score.overall.tp,
                fp=score.overall.fp,
                fn=score.overall.fn,
            )
        )
        if (index + 1) % 25 == 0:
            print(f"{index + 1}/{len(lines)} sentences", flush=True)
    return TraceFile(
        summary=Summary(
            dataset=options.dataset.value,
            tagger=options.tagger.name,
            proposer=options.proposer.name,
            tp=running.tp,
            fp=running.fp,
            fn=running.fn,
            main_reference_edits=main,
            scored_reference_edits=scored,
            stages=dict(stages),
        ),
        sentences=tuple(sentences),
    )


def parse(argv: Sequence[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tagger", type=Path, required=True)
    parser.add_argument("--proposer", type=Path, required=True)
    parser.add_argument("--dataset", type=Dataset, required=True)
    parser.add_argument(
        "--limit", type=int, default=None, help="The --limit the clean set was made with."
    )
    parser.add_argument("--clean", type=Path, help="Default <store>/<set>/clean-local.jsonl.")
    parser.add_argument("--data", type=Path, default=Path("data") / "eval")
    parser.add_argument("--store", type=Path, default=Path("data") / "judgments")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    return Options.model_validate(vars(parser.parse_args(argv)))


def main() -> None:
    options = parse()
    result = asyncio.run(trace(options))
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(result.model_dump_json(), encoding="utf-8")
    s = result.summary
    print(
        f"tp {s.tp} fp {s.fp} fn {s.fn}; reference edits {s.main_reference_edits} main, "
        f"{s.scored_reference_edits} scored; stages "
        + ", ".join(f"{stage} {s.stages.get(stage, 0)}" for stage in Stage)
    )
    print(f"proposals per flag through the filters: {JUDGED}; trace: {options.out}")


if __name__ == "__main__":
    main()
