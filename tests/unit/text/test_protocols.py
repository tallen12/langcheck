from langcheck.text.protocols import SentenceSplitter
from langcheck.text.regex_splitter import RegexSentenceSplitter


def test_regex_splitter_satisfies_the_protocol() -> None:
    splitter: SentenceSplitter = RegexSentenceSplitter()

    assert splitter.split("One. Two.") != []
