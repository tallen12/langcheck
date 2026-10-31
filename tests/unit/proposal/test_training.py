import pytest
import torch
from transformers import PreTrainedTokenizerFast

from langcheck.proposal.model import ProposerShape, TokenProposer
from langcheck.proposal.targets import Keep, Rewrite, WordSpan
from langcheck.proposal.training import ProposerItem, ProposerOptions, train_proposer
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
    )
    first = float(lines[0].split()[-1])

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
        )
