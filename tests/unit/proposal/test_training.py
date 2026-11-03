import pytest
import torch
from transformers import PreTrainedTokenizerFast

from langcheck.proposal.model import ProposerShape, TokenProposer
from langcheck.proposal.search import vocabulary
from langcheck.proposal.stores import MemoryStore
from langcheck.proposal.targets import Keep, Rewrite, WordSpan
from langcheck.proposal.training import (
    ProposerItem,
    ProposerOptions,
    _prepare,
    encode_once,
    train_proposer,
)
from langcheck.tagging.encoding import collate
from tests.unit.tagging.test_model import tiny_tagger

CPU = torch.device("cpu")
ITEM = ProposerItem(
    ("The", "dog", "."),
    "",
    (Rewrite(WordSpan(1, 2), ("dog",)), Keep(WordSpan(2, 3)), Rewrite(WordSpan(0, 1), ())),
)


def proposer_for(vocab: int) -> TokenProposer:
    torch.manual_seed(0)
    return TokenProposer(
        ProposerShape(
            encoder_size=16, embedding_size=16, vocab_size=vocab, size=16, layers=1, heads=2
        )
    )


def test_training_lowers_the_loss_and_leaves_the_tagger_alone(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    tagger = tiny_tagger(tiny_tokenizer)
    before = {name: value.clone() for name, value in tagger.state_dict().items()}
    lines: list[str] = []

    result = train_proposer(
        tagger,
        proposer_for(len(tiny_tokenizer)),
        tiny_tokenizer,
        [ITEM] * 16,
        [ITEM],
        device=CPU,
        options=ProposerOptions(epochs=8, batch_size=4, learning_rate=1e-3, max_length=32),
        log=lines.append,
        store=MemoryStore(),
    )
    first = float(next(line for line in lines if "validation loss" in line).split()[-1])

    assert result.loss < first
    for name, value in tagger.state_dict().items():
        assert torch.equal(value, before[name])


def test_nothing_to_train_on_is_refused(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    empty = ProposerItem(("a",), "", ())

    with pytest.raises(ValueError, match="no targets"):
        train_proposer(
            tiny_tagger(tiny_tokenizer),
            proposer_for(len(tiny_tokenizer)),
            tiny_tokenizer,
            [empty],
            [empty],
            device=CPU,
            options=ProposerOptions(max_length=32),
            log=lambda _: None,
            store=MemoryStore(),
        )


def test_no_epochs_is_refused(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    with pytest.raises(ValueError, match="epochs must be positive"):
        train_proposer(
            tiny_tagger(tiny_tokenizer),
            proposer_for(len(tiny_tokenizer)),
            tiny_tokenizer,
            [ITEM],
            [ITEM],
            device=CPU,
            options=ProposerOptions(epochs=0, max_length=32),
            log=lambda _: None,
            store=MemoryStore(),
        )


def test_after_epoch_runs_once_per_epoch_after_the_loss(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    proposer = proposer_for(len(tiny_tokenizer))
    lines: list[str] = []
    seen: list[tuple[int, bool, str]] = []

    train_proposer(
        tiny_tagger(tiny_tokenizer),
        proposer,
        tiny_tokenizer,
        [ITEM] * 4,
        [ITEM],
        device=CPU,
        options=ProposerOptions(epochs=2, batch_size=4, max_length=32),
        log=lines.append,
        store=MemoryStore(),
        after_epoch=lambda epoch: seen.append((epoch, proposer.training, lines[-1])),
    )

    assert [(epoch, training) for epoch, training, _ in seen] == [(1, False), (2, False)]
    assert all(line.startswith(f"epoch {epoch}: validation loss") for epoch, _, line in seen)


def test_each_sentence_is_encoded_once_however_many_epochs(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    tagger = tiny_tagger(tiny_tokenizer)
    rows: list[int] = []
    tagger.encoder.register_forward_hook(
        lambda _module, _args, kwargs, _output: rows.append(kwargs["input_ids"].size(0)),
        with_kwargs=True,
    )
    lines: list[str] = []

    train_proposer(
        tagger,
        proposer_for(len(tiny_tokenizer)),
        tiny_tokenizer,
        [ITEM] * 8,
        [ITEM] * 2,
        device=CPU,
        options=ProposerOptions(epochs=3, batch_size=4, max_length=32),
        log=lines.append,
        store=MemoryStore(),
    )

    assert sum(rows) == 10
    assert lines[0] == "encoding 10 sentences once"


def test_cached_states_match_the_encoder_on_each_sentence_alone(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    tagger = tiny_tagger(tiny_tokenizer).eval()
    tokens = vocabulary(tiny_tokenizer)
    short = ProposerItem(("The", "dog", "."), "", (Keep(WordSpan(2, 3)),))
    long = ProposerItem(
        ("The", "dog", "saw", "the", "big", "cat", "."), "", (Keep(WordSpan(1, 2)),)
    )
    prepared = [_prepare(item, tiny_tokenizer, tokens, max_length=32) for item in (long, short)]
    pad = int(tiny_tokenizer.pad_token_id or 0)

    cached = encode_once(tagger, prepared, pad=pad, device=CPU, store=MemoryStore())

    for item, found in zip(prepared, cached, strict=True):
        batch = collate([item.encoded], pad_token_id=pad)
        with torch.no_grad():
            alone = tagger.encoder(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            ).last_hidden_state[0]
        assert found.prepared is item
        assert found.states.dtype == torch.float16
        assert torch.allclose(found.states.float(), alone, atol=1e-2)
