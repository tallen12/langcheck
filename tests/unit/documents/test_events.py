from langcheck.documents.events import Snapshot
from langcheck.documents.models import CheckState, Document, DocumentKey


def test_snapshot_copies_the_current_version() -> None:
    document = Document(
        key=DocumentKey("ana", "file:///a.md"), text="Hi.", version=4, state=CheckState.DONE
    )

    assert Snapshot.of(document) == Snapshot(4, CheckState.DONE, (), ())
