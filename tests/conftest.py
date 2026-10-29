"""Fixtures shared across test types."""

import sys
import types
from collections.abc import Callable, Iterator, Sequence

import pytest
from transformers import PreTrainedTokenizerFast

from langcheck.auth.models import User, UserExistsError
from langcheck.llm.models import ChatMessage, ChatOptions, LLMError
from langcheck.scoring.models import ScoredText

type Responder = Callable[[Sequence[ChatMessage]], str | LLMError]


class ScriptedLLM:
    """An ``AsyncLLMClient`` fake that answers from a script and records every chat.

    Each entry is a reply, an ``LLMError`` to raise, or a function of the chat returning
    either, for replies that depend on the request (such as which sentences it holds).
    The last entry repeats once the script runs out.
    """

    def __init__(self, script: Sequence[str | LLMError | Responder]) -> None:
        if not script:
            msg = "the script needs at least one entry"
            raise ValueError(msg)
        self._script = list(script)
        self.chats: list[list[ChatMessage]] = []
        self.options: list[ChatOptions] = []

    async def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        """Return the next scripted reply for ``messages``."""
        self.chats.append(list(messages))
        self.options.append(options)
        entry = self._script[min(len(self.chats), len(self._script)) - 1]
        reply = entry if isinstance(entry, (str, LLMError)) else entry(messages)
        match reply:
            case LLMError():
                raise reply
            case str():
                return reply


@pytest.fixture
def scripted_llm() -> Callable[[Sequence[str | LLMError | Responder]], ScriptedLLM]:
    """Build a :class:`ScriptedLLM` from a script."""
    return ScriptedLLM


class MemoryApiKeyStore:
    """An ``ApiKeyStore`` fake that keeps key hashes in a dictionary."""

    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}

    def user_for(self, token_hash: str) -> User | None:
        """The user whose key has ``token_hash``."""
        return next((User(n) for n, h in self.hashes.items() if h == token_hash), None)

    def add(self, name: str, token_hash: str) -> None:
        """Store the hash; refuse a second key for one user."""
        if name in self.hashes:
            raise UserExistsError(name)
        self.hashes[name] = token_hash

    def revoke(self, name: str) -> bool:
        """Remove ``name``'s key."""
        return self.hashes.pop(name, None) is not None

    def users(self) -> list[str]:
        """Every user with a key, sorted."""
        return sorted(self.hashes)


class TableScorer:
    """A ``SentenceScorer`` and ``AsyncSentenceScorer`` fake that looks scores up by text.

    Unknown texts score ``default``. Every batch it is asked to score is recorded.
    """

    def __init__(self, table: dict[str, float], *, default: float = 0.0) -> None:
        self._table = table
        self._default = default
        self.batches: list[list[ScoredText]] = []

    def score_now(self, texts: Sequence[ScoredText]) -> list[float]:
        """Score ``texts`` and record the batch."""
        self.batches.append(list(texts))
        return [self._table.get(t.text, self._default) for t in texts]

    async def score(self, texts: Sequence[ScoredText]) -> list[float]:
        """The async form, for rankers."""
        return self.score_now(texts)


class SyncTableScorer:
    """The blocking form of :class:`TableScorer`, for code that runs scorers in threads."""

    def __init__(self, table: dict[str, float], *, default: float = 0.0) -> None:
        self.table = TableScorer(table, default=default)

    def score(self, texts: Sequence[ScoredText]) -> list[float]:
        """Score ``texts``."""
        return self.table.score_now(texts)

    @property
    def batches(self) -> list[list[ScoredText]]:
        """Every batch scored so far."""
        return self.table.batches


TINY_CORPUS = [
    "The dog sleeps in the garden every morning.",
    "A cat walked to the park and the river.",
    "They are going home. Their house is old. I think they're right.",
] * 20


@pytest.fixture(scope="session")
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A RoBERTa-style byte-level BPE tokenizer trained on a few sentences, so scorer
    tests run without downloading a model. Like RoBERTa it wraps text in ``<s> ... </s>``.
    """
    # Only scorer tests need these, so they load with the fixture.
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    special = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    bpe = Tokenizer(models.BPE(unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600,
        min_frequency=1,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    bpe.train_from_iterator(TINY_CORPUS, trainer)
    # The trainer gives special tokens the first ids, in order.
    bpe.post_processor = processors.RobertaProcessing(
        ("</s>", special.index("</s>")), ("<s>", special.index("<s>"))
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        bos_token="<s>",
        cls_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
    )


class _UnoStruct:
    """A UNO struct: attributes set after construction, keyword arguments allowed."""

    def __init__(self, *args: object, **fields: object) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


class _Locale(_UnoStruct):
    def __init__(self, Language: str = "", Country: str = "", Variant: str = "") -> None:
        super().__init__(Language=Language, Country=Country, Variant=Variant)


class _ImplementationHelper:
    def __init__(self) -> None:
        self.implementations: list[tuple[object, str, tuple[str, ...]]] = []

    def addImplementation(self, ctor: object, name: str, services: tuple[str, ...]) -> None:
        self.implementations.append((ctor, name, services))


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def fake_uno(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stand-ins for the UNO modules LibreOffice provides, so the LibreOffice extension
    imports and runs under our Python. Modules that import them are reloaded per test.
    """
    interfaces = {
        name: type(name, (), {})
        for name in [
            "XServiceInfo",
            "XServiceName",
            "XServiceDisplayName",
            "XProofreader",
            "XSupportedLocales",
            "XLinguServiceEventBroadcaster",
        ]
    }
    modules = {
        "unohelper": _module(
            "unohelper", Base=type("Base", (), {}), ImplementationHelper=_ImplementationHelper
        ),
        "com": _module("com"),
        "com.sun": _module("com.sun"),
        "com.sun.star": _module("com.sun.star"),
        "com.sun.star.lang": _module(
            "com.sun.star.lang",
            Locale=_Locale,
            **{k: v for k, v in interfaces.items() if k.startswith("XService")},
        ),
        "com.sun.star.beans": _module(
            "com.sun.star.beans", PropertyValue=type("PropertyValue", (_UnoStruct,), {})
        ),
        "com.sun.star.linguistic2": _module(
            "com.sun.star.linguistic2",
            ProofreadingResult=type("ProofreadingResult", (_UnoStruct,), {}),
            SingleProofreadingError=type("SingleProofreadingError", (_UnoStruct,), {}),
            LinguServiceEvent=type("LinguServiceEvent", (_UnoStruct,), {}),
            **{k: v for k, v in interfaces.items() if not k.startswith("XService")},
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "langcheck.libreoffice.component", raising=False)
    yield
    sys.modules.pop("langcheck.libreoffice.component", None)
