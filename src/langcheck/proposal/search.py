"""Writing proposals: beam search over the proposer, one encoder pass per sentence."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from langcheck.evaluation.detokenize import detokenize
from langcheck.proposal.model import (
    MAX_TARGET,
    OUTSIDE,
    TokenProposer,
    pad_rows,
    span_fits,
    span_segments,
    word_embeddings,
)
from langcheck.proposal.protocols import Proposal, ProposalRequest
from langcheck.tagging.encoding import collate, encode, to_device
from langcheck.tagging.model import SlotTagger

SENTENCES_PER_BATCH = 32
HYPOTHESES_PER_PASS = 512
"""Limits one decoder pass, whose memory grows with hypotheses times sentence length."""


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """The special tokens a proposal is written with. The mask token means "keep"."""

    start: int
    end: int
    keep: int
    banned: tuple[int, ...]


def vocabulary(tokenizer: PreTrainedTokenizerBase) -> Vocabulary:
    """The special tokens of ``tokenizer``."""
    start = int(tokenizer.bos_token_id or 0)
    end = int(tokenizer.eos_token_id or 0)
    keep = int(tokenizer.mask_token_id or 0)
    banned = (start, int(tokenizer.pad_token_id or 0), int(tokenizer.unk_token_id or 0))
    return Vocabulary(start, end, keep, banned)


def target_ids(
    words: Sequence[str] | None, tokenizer: PreTrainedTokenizerBase, tokens: Vocabulary
) -> tuple[int, ...]:
    """A target from its start to its end token; ``None`` is the keep target. Every word
    starts with a space, whatever comes before it, so one word has one spelling.
    """
    if words is None:
        return (tokens.start, tokens.keep, tokens.end)
    if not words:
        return (tokens.start, tokens.end)
    encoded = tokenizer(f" {detokenize(words)}", add_special_tokens=False)["input_ids"]
    return (tokens.start, *(int(token) for token in encoded), tokens.end)


def beam_search(
    proposer: TokenProposer,
    states: torch.Tensor,
    mask: torch.Tensor,
    segments: torch.Tensor,
    embeddings: torch.Tensor,
    tokens: Vocabulary,
    *,
    width: int,
    chunk: int = HYPOTHESES_PER_PASS,
) -> list[list[tuple[tuple[int, ...], float]]]:
    """For each case (a row of ``states``, with ``mask`` true where a state is real), the
    best finished subwords, without start and end, with log probabilities, best first.

    Every live hypothesis has the same length at each step, so all cases advance together.
    """
    device = states.device
    finished: list[list[tuple[tuple[int, ...], float]]] = [[] for _ in range(states.size(0))]
    beams: list[list[tuple[tuple[int, ...], float]]] = [
        [((tokens.start,), 0.0)] for _ in range(states.size(0))
    ]
    for step in range(1, MAX_TARGET):
        if step == 2:
            # A keep is whole: it ends here, and only here.
            for case, beam in enumerate(beams):
                finished[case] += [((tokens.keep,), s) for w, s in beam if w[-1] == tokens.keep]
                beams[case] = [(w, s) for w, s in beam if w[-1] != tokens.keep]
        live = [(case, *item) for case, beam in enumerate(beams) for item in beam]
        if not live:
            break
        tops: list[torch.Tensor] = []
        picks: list[torch.Tensor] = []
        for begin in range(0, len(live), chunk):
            part = live[begin : begin + chunk]
            cases = torch.tensor([case for case, _, _ in part], dtype=torch.long, device=device)
            logits = proposer.next_logits(
                states[cases],
                mask[cases],
                segments[cases],
                torch.tensor([written for _, written, _ in part], dtype=torch.long, device=device),
                embeddings,
            )
            logits[:, list(tokens.banned)] = -torch.inf
            if step > 1:
                logits[:, tokens.keep] = -torch.inf
            top, pick = torch.log_softmax(logits, dim=-1).topk(min(width, logits.size(-1)), dim=-1)
            tops.append(top.cpu())
            picks.append(pick.cpu())
        values = torch.cat(tops).tolist()
        indices = torch.cat(picks).tolist()
        pools: list[list[tuple[tuple[int, ...], float]]] = [[] for _ in beams]
        for (case, written, total), row_values, row_indices in zip(
            live, values, indices, strict=True
        ):
            for value, index in zip(row_values, row_indices, strict=True):
                if index == tokens.end:
                    finished[case].append((written[1:], total + value))
                else:
                    pools[case].append(((*written, index), total + value))
        beams = [sorted(pool, key=lambda i: i[1], reverse=True)[:width] for pool in pools]
    return [sorted(found, key=lambda i: i[1], reverse=True) for found in finished]


class BeamProposer:
    """A ``ProposalSource`` reading with the tagger's encoder and writing with beam search,
    in batches of sentences of similar length.
    """

    def __init__(
        self,
        tagger: SlotTagger,
        proposer: TokenProposer,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: torch.device,
        max_length: int,
        width: int,
        batch_size: int = SENTENCES_PER_BATCH,
    ) -> None:
        self._tagger = tagger
        self._proposer = proposer
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        self._width = width
        self._batch_size = batch_size
        self._tokens = vocabulary(tokenizer)

    def propose(self, requests: Sequence[ProposalRequest]) -> list[list[tuple[Proposal, ...]]]:
        """Up to the beam width of distinct proposals per span, best first. A span past the
        length limit gets none. Blocking.
        """
        encoded = [
            encode(r.words, self._tokenizer, max_length=self._max_length, context=r.context)
            for r in requests
        ]
        results: list[list[tuple[Proposal, ...]]] = [[()] * len(r.spans) for r in requests]
        order = sorted(range(len(encoded)), key=lambda index: len(encoded[index].input_ids))
        self._proposer.eval()
        pad = int(self._tokenizer.pad_token_id or 0)
        for begin in range(0, len(order), self._batch_size):
            indices = order[begin : begin + self._batch_size]
            where = [
                (row, index, position, requests[index].spans[position])
                for row, index in enumerate(indices)
                for position in range(len(requests[index].spans))
                if span_fits(encoded[index], requests[index].spans[position])
            ]
            if not where:
                continue
            batch = to_device(
                collate([encoded[i] for i in indices], pad_token_id=pad), self._device
            )
            with torch.inference_mode():
                states = self._tagger.encoder(
                    input_ids=batch.input_ids, attention_mask=batch.attention_mask
                ).last_hidden_state
                rows = torch.tensor([w[0] for w in where], dtype=torch.long, device=self._device)
                segments = pad_rows(
                    [span_segments(encoded[index], span) for _, index, _, span in where],
                    OUTSIDE,
                    self._device,
                )
                width = batch.input_ids.size(1)
                segments = torch.nn.functional.pad(segments, (0, width - segments.size(1)))
                found = beam_search(
                    self._proposer,
                    states[rows],
                    batch.attention_mask[rows].bool(),
                    segments,
                    word_embeddings(self._tagger),
                    self._tokens,
                    width=self._width,
                )
            for (_, index, position, _), written in zip(where, found, strict=True):
                results[index][position] = self._distinct(written)
        return results

    def _distinct(self, found: Sequence[tuple[tuple[int, ...], float]]) -> tuple[Proposal, ...]:
        seen: set[str] = set()
        proposals: list[Proposal] = []
        for ids, log_probability in found:
            keep = ids == (self._tokens.keep,)
            decoded = "" if keep else self._tokenizer.decode(list(ids))
            text = decoded.strip() if isinstance(decoded, str) else ""
            key = "keep" if keep else f"text:{text}"
            if key not in seen:
                seen.add(key)
                proposals.append(Proposal(text, log_probability, keep=keep))
            if len(proposals) == self._width:
                break
        return tuple(proposals)
