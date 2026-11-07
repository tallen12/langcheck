from pathlib import Path

import pytest
import torch
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartModel,
    PreTrainedTokenizerFast,
)

from langcheck.noising.model import (
    NoiseInfo,
    NoiseModel,
    NoiseOptions,
    Sampling,
    count_match,
    load_noise_model,
    read_noise_info,
    save_noise_model,
    train_noise_model,
)

CPU = torch.device("cpu")
PAIRS = [("1 VERB | He went home.", "He go home."), ("0 | The cat sat.", "The cat sat.")] * 4


def tiny_bart(tokenizer: PreTrainedTokenizerFast) -> BartForConditionalGeneration:
    config = BartConfig(
        vocab_size=len(tokenizer),
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    return BartForConditionalGeneration(config)


def test_training_logs_each_epoch_and_keeps_the_lowest_loss(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    lines: list[str] = []
    model = tiny_bart(tiny_tokenizer)

    result = train_noise_model(
        model,
        tiny_tokenizer,
        PAIRS,
        PAIRS[:2],
        device=CPU,
        options=NoiseOptions(epochs=3, batch_size=4, learning_rate=1e-3, max_length=32, checked=2),
        log=lines.append,
    )

    epochs = [line for line in lines if "validation loss" in line]
    assert [line.split(":")[0] for line in epochs] == ["epoch 1", "epoch 2", "epoch 3"]
    losses = [float(line.split("loss ")[1].split(",")[0]) for line in epochs]
    assert result.validation_loss == pytest.approx(min(losses), abs=1e-4)
    assert 0.0 <= result.count_match <= 1.0
    assert lines[-1] == f"keeping epoch {result.epoch}"


def test_training_needs_validation_and_an_epoch(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    model = tiny_bart(tiny_tokenizer)
    with pytest.raises(ValueError, match="validation"):
        train_noise_model(
            model, tiny_tokenizer, PAIRS, [], device=CPU, options=NoiseOptions(), log=print
        )
    with pytest.raises(ValueError, match="epochs"):
        train_noise_model(
            model,
            tiny_tokenizer,
            PAIRS,
            PAIRS,
            device=CPU,
            options=NoiseOptions(epochs=0),
            log=print,
        )


def test_writing_answers_each_prompt_the_same_way_for_a_seed(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    noise = NoiseModel(tiny_bart(tiny_tokenizer), tiny_tokenizer, device=CPU, max_length=16)
    prompts = ["1 VERB | He went home.", "0 | The cat sat.", "2 DET PUNCT | A dog ran."]

    first = noise.write(prompts, Sampling(batch_size=2, seed=1))

    assert len(first) == 3
    assert noise.write(prompts, Sampling(batch_size=2, seed=1)) == first


def test_writing_samples_even_when_the_model_was_saved_for_beam_search(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    model = tiny_bart(tiny_tokenizer)
    model.generation_config.num_beams = 4
    model.generation_config.no_repeat_ngram_size = 3
    noise = NoiseModel(model, tiny_tokenizer, device=CPU, max_length=16)
    prompts = [f"1 DET | The cat sat {n}." for n in range(12)]

    first = noise.write(prompts, Sampling(top_p=1.0, seed=1))
    second = noise.write(prompts, Sampling(top_p=1.0, seed=2))

    assert first != second


def test_a_model_that_cannot_generate_is_refused(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    encoder_only = BartModel(tiny_bart(tiny_tokenizer).config)
    with pytest.raises(TypeError, match="cannot generate"):
        NoiseModel(encoder_only, tiny_tokenizer, device=CPU, max_length=16)


def test_count_match_compares_edits_made_with_edits_asked() -> None:
    prompts = ["1 VERB | He went home.", "2 DET VERB | He went home."]
    assert count_match(prompts, ["He go home.", "He go home."]) == 0.5
    assert count_match([], []) == 1.0


def test_a_saved_noise_model_loads_with_its_info(
    tmp_path: Path, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    info = NoiseInfo(
        base_model="tiny",
        train_pairs=8,
        validation_pairs=2,
        epoch=1,
        validation_loss=1.5,
        count_match=0.5,
    )
    save_noise_model(tmp_path / "noise", tiny_bart(tiny_tokenizer), tiny_tokenizer, info)

    noise, loaded = load_noise_model(tmp_path / "noise", device=CPU, max_length=16)

    assert loaded == info == read_noise_info(tmp_path / "noise")
    assert len(noise.write(["0 | The cat sat."], Sampling())) == 1
