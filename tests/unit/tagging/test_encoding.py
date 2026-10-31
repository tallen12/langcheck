import pytest
import torch
from transformers import PreTrainedTokenizerFast

from langcheck.tagging.encoding import collate, encode, load_tokenizer, to_device
from langcheck.tagging.labels import SlotLabels

WORDS = ("The", "dog", "sleeps", ".")


def test_every_slot_reads_the_subwords_around_it(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    encoded = encode(WORDS, tiny_tokenizer, max_length=64)

    assert len(encoded.word_first) == len(encoded.word_last) == 4
    assert all(a <= b for a, b in zip(encoded.word_first, encoded.word_last, strict=True))
    assert encoded.gap_left[0] == 0
    assert encoded.gap_right[-1] == len(encoded.input_ids) - 1
    assert encoded.gap_left[1:] == encoded.word_last
    assert encoded.gap_right[:-1] == encoded.word_first


def test_the_model_sees_normal_spacing(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    encoded = encode(("dog", "."), tiny_tokenizer, max_length=64)

    assert tiny_tokenizer.decode(list(encoded.input_ids), skip_special_tokens=True) == "dog."


def test_words_past_the_limit_are_left_out(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    encoded = encode(("dog",) * 40, tiny_tokenizer, max_length=8)

    assert 0 < len(encoded.word_first) < 40
    assert len(encoded.gap_left) == len(encoded.gap_right) == len(encoded.word_first)
    assert max(encoded.word_last) < len(encoded.input_ids) - 1


def test_collate_pads_and_cuts_labels_to_the_scored_slots(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    short = encode(("dog",), tiny_tokenizer, max_length=64)
    long = encode(WORDS, tiny_tokenizer, max_length=64)
    labels = [
        SlotLabels((True,), (False, True)),
        SlotLabels((False, False, True, False), (True, False, False, False, False)),
    ]

    batch = collate([short, long], pad_token_id=1, labels=labels)

    assert batch.word_mask.tolist() == [[True, False, False, False], [True] * 4]
    assert batch.gap_mask.sum().item() == 7
    assert batch.word_labels.tolist() == [[1, 0, 0, 0], [0, 0, 1, 0]]
    assert batch.input_ids[0, -1].item() == 1
    assert to_device(batch, torch.device("cpu")).gap_labels.tolist()[0][:2] == [0, 1]


def test_a_missing_tokenizer_is_refused(tmp_path: object) -> None:
    with pytest.raises((ValueError, OSError)):
        load_tokenizer(f"{tmp_path}/none")


def test_context_is_read_but_every_slot_stays_on_the_sentence(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    plain = encode(WORDS, tiny_tokenizer, max_length=64)

    with_context = encode(WORDS, tiny_tokenizer, max_length=64, context="The dog.")

    shift = len(with_context.input_ids) - len(plain.input_ids)
    assert shift > 2
    assert with_context.word_first == tuple(p + shift for p in plain.word_first)
    assert with_context.gap_right[-1] == len(with_context.input_ids) - 1
    assert with_context.input_ids[-len(plain.input_ids) + 1 :] == plain.input_ids[1:]
    # The separator right before the sentence stands where its start token did.
    assert with_context.gap_left[0] == shift
    assert with_context.input_ids[shift] == tiny_tokenizer.sep_token_id


def test_context_is_cut_from_its_start_to_fit(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    plain = encode(("dog",), tiny_tokenizer, max_length=16)

    context = "The cat. " * 30 + "dog"

    encoded = encode(("dog",), tiny_tokenizer, max_length=16, context=context)

    assert len(encoded.input_ids) == 16
    nearest = tiny_tokenizer(context, add_special_tokens=False)["input_ids"][-1]
    assert encoded.input_ids[-len(plain.input_ids) - 2] == nearest


def test_a_sentence_filling_the_limit_gets_no_context(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    plain = encode(("dog",) * 40, tiny_tokenizer, max_length=8)

    assert encode(("dog",) * 40, tiny_tokenizer, max_length=8, context="The cat.") == plain
