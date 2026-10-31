from typing import override

import torch
from transformers import PreTrainedTokenizerFast

from langcheck.proposal.model import ProposerShape, TokenProposer
from langcheck.proposal.protocols import ProposalRequest
from langcheck.proposal.search import BeamProposer, beam_search, target_ids, vocabulary
from langcheck.proposal.targets import WordSpan
from tests.unit.tagging.test_model import tiny_tagger

CPU = torch.device("cpu")


class Scripted(TokenProposer):
    """Prefers writing token 7 once and then ending."""

    def __init__(self, tokens: int) -> None:
        super().__init__(
            ProposerShape(
                encoder_size=4, embedding_size=4, vocab_size=tokens, size=4, layers=1, heads=1
            )
        )

    @override
    def next_logits(
        self,
        states: torch.Tensor,
        states_mask: torch.Tensor,
        segments: torch.Tensor,
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.zeros(tokens.size(0), embeddings.size(0))
        logits[:, 7] = 5.0 if tokens.size(1) == 1 else -5.0
        logits[:, 2] = 5.0 if tokens.size(1) > 1 else 1.0
        logits[:, 4] = 2.0 if tokens.size(1) == 1 else -5.0
        return logits


def real_proposer(vocab: int) -> TokenProposer:
    torch.manual_seed(0)
    shape = ProposerShape(
        encoder_size=16, embedding_size=16, vocab_size=vocab, size=8, layers=1, heads=2, dropout=0.0
    )
    return TokenProposer(shape).eval()


def test_targets_run_from_start_to_end(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    tokens = vocabulary(tiny_tokenizer)

    assert target_ids(None, tiny_tokenizer, tokens) == (tokens.start, tokens.keep, tokens.end)
    assert target_ids((), tiny_tokenizer, tokens) == (tokens.start, tokens.end)
    written = target_ids(("dog",), tiny_tokenizer, tokens)
    assert written[0] == tokens.start
    assert written[-1] == tokens.end
    assert len(written) > 2


def test_beam_search_finishes_deletions_keeps_and_text_best_first(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    tokens = vocabulary(tiny_tokenizer)
    [found] = beam_search(
        Scripted(20),
        torch.zeros(1, 3, 4),
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.long),
        torch.zeros(20, 4),
        tokens,
        width=3,
    )
    written = [ids for ids, _ in found]

    assert written[0] == (7,)
    assert (tokens.keep,) in written
    assert () in written
    assert [score for _, score in found] == sorted((score for _, score in found), reverse=True)


def test_searching_together_gives_what_searching_alone_does(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    tokens = vocabulary(tiny_tokenizer)
    proposer = real_proposer(len(tiny_tokenizer))
    embeddings = torch.rand(len(tiny_tokenizer), 16)
    states = torch.rand(3, 5, 16)
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2, [True] * 4 + [False]])
    segments = torch.tensor([[0, 1, 1, 0, 0], [0, 2, 2, 0, 0], [1, 0, 0, 0, 0]])

    def search(rows: list[int], chunk: int) -> list[list[tuple[tuple[int, ...], float]]]:
        with torch.inference_mode():
            return beam_search(
                proposer,
                states[rows],
                mask[rows],
                segments[rows],
                embeddings,
                tokens,
                width=3,
                chunk=chunk,
            )

    together = search([0, 1, 2], 2)
    for row, found in enumerate(together):
        [alone] = search([row], 512)
        assert [ids for ids, _ in found] == [ids for ids, _ in alone]
        for (_, first), (_, second) in zip(found, alone, strict=True):
            assert abs(first - second) < 1e-3


def test_proposals_are_distinct_limited_and_skip_spans_past_the_limit(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    tagger = tiny_tagger(tiny_tokenizer).eval()
    source = BeamProposer(
        tagger, Scripted(len(tiny_tokenizer)), tiny_tokenizer, device=CPU, max_length=8, width=2
    )
    request = ProposalRequest(("The", "dog", ".") * 10, "", (WordSpan(0, 1), WordSpan(25, 26)))
    short = ProposalRequest(("The", "dog"), "", (WordSpan(1, 1),))

    [[near, far], [gap]] = source.propose([request, short])

    assert 0 < len(near) <= 2
    assert len({(p.text, p.keep) for p in near}) == len(near)
    assert far == ()
    assert gap
