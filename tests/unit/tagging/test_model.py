import pytest
import torch
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

from langcheck.tagging.encoding import Batch, collate, encode
from langcheck.tagging.labels import SlotLabels
from langcheck.tagging.model import SlotTagger, slot_loss


def tiny_tagger(tokenizer: PreTrainedTokenizerFast) -> SlotTagger:
    """A random two-layer RoBERTa, seeded so results repeat."""
    torch.manual_seed(0)
    config = RobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
    )
    return SlotTagger(RobertaModel(config))


def test_logits_match_the_slots_and_the_loss_ignores_padding(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    pad = int(tiny_tokenizer.pad_token_id or 0)
    first = encode(("dog",), tiny_tokenizer, max_length=32)
    second = encode(("The", "dog", "."), tiny_tokenizer, max_length=32)
    labels = [SlotLabels((True,), (False, False)), SlotLabels((False,) * 3, (False,) * 4)]
    tagger = tiny_tagger(tiny_tokenizer).eval()

    batch = collate([first, second], pad_token_id=pad, labels=labels)
    words, gaps = tagger(batch)

    assert words.shape == batch.word_mask.shape
    assert gaps.shape == batch.gap_mask.shape
    changed = words.detach().clone()
    changed[0, 2] = 100.0
    assert slot_loss(batch, changed, gaps).item() == slot_loss(batch, words, gaps).item()


def test_error_weight_scales_only_the_loss_on_error_slots(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    pad = int(tiny_tokenizer.pad_token_id or 0)
    encoded = [encode(("The", "dog", "."), tiny_tokenizer, max_length=32)]
    flagged = collate(
        encoded, pad_token_id=pad, labels=[SlotLabels((False, True, False), (False,) * 4)]
    )
    clean = collate(encoded, pad_token_id=pad, labels=[SlotLabels((False,) * 3, (False,) * 4)])
    words, gaps = tiny_tagger(tiny_tokenizer).eval()(flagged)

    def loss(batch: Batch, weight: float) -> float:
        return slot_loss(batch, words, gaps, error_weight=weight).item()

    assert loss(clean, 3.0) == pytest.approx(loss(clean, 1.0))
    assert loss(flagged, 3.0) > loss(flagged, 1.0)
