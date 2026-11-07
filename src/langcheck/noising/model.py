"""Training a sequence-to-sequence noise model and sampling errors from it."""

import copy
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForSeq2SeqLM, GenerationMixin, PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from langcheck.noising.requests import SEPARATOR, noisy_sentence
from langcheck.tagging.alignment import split_words
from langcheck.tagging.encoding import load_tokenizer

LOG_EVERY: Final = 200
INFO: Final = "noise.json"
IGNORED: Final = -100
"""Label of padding, which the loss leaves out."""


class NoiseInfo(BaseModel):
    """What a saved noise model was trained from, and how it did."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_model: str
    train_pairs: int
    validation_pairs: int
    epoch: int
    validation_loss: float
    count_match: float


@dataclass(frozen=True, slots=True)
class NoiseOptions:
    """How to train. ``checked`` validation prompts are sampled after each epoch to see
    how often the model makes as many errors as it was asked for.
    """

    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_share: float = 0.05
    max_length: int = 128
    checked: int = 500
    seed: int = 0


@dataclass(frozen=True, slots=True)
class NoiseResult:
    """The kept epoch and how it did on the validation pairs."""

    epoch: int
    validation_loss: float
    count_match: float


@dataclass(frozen=True, slots=True)
class Sampling:
    """How answers are sampled: nucleus sampling keeps errors varied but plausible."""

    top_p: float = 0.9
    batch_size: int = 64
    seed: int = 0


class NoiseModel:
    """A trained noise model on a device."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: torch.device,
        max_length: int,
    ) -> None:
        if not isinstance(model, GenerationMixin):
            msg = f"{type(model).__name__} cannot generate text"
            raise TypeError(msg)
        self._model = model
        self._generator = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length

    @property
    def model(self) -> PreTrainedModel:
        """The underlying sequence-to-sequence model."""
        return self._model

    def write(self, prompts: Sequence[str], sampling: Sampling) -> list[str]:
        """A sampled answer to each prompt. Blocking."""
        torch.manual_seed(sampling.seed)
        self._model.eval()
        answers: list[str] = []
        with torch.no_grad():
            for begin in range(0, len(prompts), sampling.batch_size):
                batch = self._tokenizer(
                    list(prompts[begin : begin + sampling.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                # Plain sampling: a pretrained model's saved settings (BART's beam search
                # and ban on repeated trigrams) would make every sentence's errors the
                # likeliest ones and block answers that repeat words.
                made = self._generator.generate(
                    **batch,
                    do_sample=True,
                    top_p=sampling.top_p,
                    num_beams=1,
                    no_repeat_ngram_size=0,
                    early_stopping=False,
                    max_length=self._max_length,
                )
                if not isinstance(made, torch.Tensor):
                    msg = "generation returned no token ids"
                    raise TypeError(msg)
                answers += self._tokenizer.batch_decode(made, skip_special_tokens=True)
        return [answer.strip() for answer in answers]


def _encode(
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[tuple[str, str]],
    *,
    max_length: int,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    batch = tokenizer(
        [source for source, _ in pairs],
        text_target=[target for _, target in pairs],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            msg = f"the tokenizer returned no tensor for {key}"
            raise TypeError(msg)
        encoded[key] = value.to(device)
    encoded["labels"][encoded["labels"] == tokenizer.pad_token_id] = IGNORED
    return encoded


def _loss(model: PreTrainedModel, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    loss = model(**batch).loss
    if not isinstance(loss, torch.Tensor):
        msg = "the model returned no loss"
        raise TypeError(msg)
    return loss


def count_match(prompts: Sequence[str], answers: Sequence[str]) -> float:
    """The share of answers with as many edits as their prompt asked for."""
    if not prompts:
        return 1.0
    matched = 0
    for prompt, answer in zip(prompts, answers, strict=True):
        head, _, clean = prompt.partition(SEPARATOR)
        wanted = int(head.split()[0])
        made = noisy_sentence(split_words(clean), answer).annotators[0]
        matched += len(made) == wanted
    return matched / len(prompts)


def train_noise_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[tuple[str, str]],
    validation: Sequence[tuple[str, str]],
    *,
    device: torch.device,
    options: NoiseOptions,
    log: Callable[[str], None],
) -> NoiseResult:
    """Fine-tune ``model`` to answer each prompt in ``pairs``; it ends with the weights
    of the epoch with the lowest validation loss.
    """
    if not validation:
        msg = "no validation pairs to choose an epoch by"
        raise ValueError(msg)
    if options.epochs < 1:
        msg = f"epochs must be positive, got {options.epochs}"
        raise ValueError(msg)
    torch.manual_seed(options.seed)
    shuffler = random.Random(options.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    per_epoch = max(1, -(-len(pairs) // options.batch_size))
    total = per_epoch * options.epochs
    warmup = max(1, int(total * options.warmup_share))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup, max(0.0, (total - step) / max(1, total - warmup))),
    )
    checked = list(validation[: options.checked])
    sampler = NoiseModel(model, tokenizer, device=device, max_length=options.max_length)
    best: NoiseResult | None = None
    best_state: Mapping[str, object] = {}
    for epoch in range(1, options.epochs + 1):
        model.train()
        order = list(range(len(pairs)))
        shuffler.shuffle(order)
        for step, begin in enumerate(range(0, len(order), options.batch_size), start=1):
            batch = _encode(
                tokenizer,
                [pairs[i] for i in order[begin : begin + options.batch_size]],
                max_length=options.max_length,
                device=device,
            )
            loss = _loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            if step % LOG_EVERY == 0:
                log(f"epoch {epoch} step {step}/{per_epoch} loss {loss.item():.4f}")
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for begin in range(0, len(validation), options.batch_size):
                part = validation[begin : begin + options.batch_size]
                batch = _encode(tokenizer, part, max_length=options.max_length, device=device)
                total_loss += _loss(model, batch).item() * len(part)
        validation_loss = total_loss / len(validation)
        answers = sampler.write([source for source, _ in checked], Sampling(seed=options.seed))
        matched = count_match([source for source, _ in checked], answers)
        log(
            f"epoch {epoch}: validation loss {validation_loss:.4f}, "
            f"{matched:.3f} of {len(checked)} answers make the errors asked for"
        )
        if best is None or validation_loss < best.validation_loss:
            best = NoiseResult(epoch, validation_loss, matched)
            best_state = copy.deepcopy(model.state_dict())
    if best is None:
        msg = "no epoch was trained"
        raise ValueError(msg)
    model.load_state_dict(best_state)
    log(f"keeping epoch {best.epoch}")
    return best


def save_noise_model(
    directory: Path, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, info: NoiseInfo
) -> None:
    """Save ``model``, its tokenizer and ``info`` to ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    (directory / INFO).write_text(info.model_dump_json(indent=2), encoding="utf-8")


def load_seq2seq(source: str | Path, *, device: torch.device) -> PreTrainedModel:
    """The sequence-to-sequence model ``source`` names, on ``device``."""
    model = AutoModelForSeq2SeqLM.from_pretrained(source)
    if not isinstance(model, PreTrainedModel):
        msg = f"{source} is not a sequence-to-sequence model"
        raise TypeError(msg)
    # nn.Module.to moves the parameters in place and returns the same module.
    model.to(device)
    return model


def read_noise_info(directory: Path) -> NoiseInfo:
    """What the noise model saved in ``directory`` was trained from."""
    return NoiseInfo.model_validate_json((directory / INFO).read_text(encoding="utf-8"))


def load_noise_model(
    directory: Path, *, device: torch.device, max_length: int
) -> tuple[NoiseModel, NoiseInfo]:
    """The noise model saved in ``directory``, on ``device``."""
    info = read_noise_info(directory)
    model = load_seq2seq(directory, device=device)
    tokenizer = load_tokenizer(directory)
    return NoiseModel(model, tokenizer, device=device, max_length=max_length), info
