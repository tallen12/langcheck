from langcheck.taxonomy.kinds import ErrorKind, Formality, Operation


def test_there_are_nineteen_kinds_with_unique_values() -> None:
    assert len(ErrorKind) == 19
    assert len({kind.value for kind in ErrorKind}) == 19


def test_kind_values_have_no_label_separator() -> None:
    assert all("-" not in kind.value for kind in ErrorKind)


def test_operations_use_errant_prefixes() -> None:
    assert [op.value for op in Operation] == ["M", "U", "R"]


def test_formality_values() -> None:
    assert [f.value for f in Formality] == ["error", "informal"]
