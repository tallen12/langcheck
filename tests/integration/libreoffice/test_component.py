"""The built ``langcheck.oxt`` in a real LibreOffice, against a local stand-in service.

Needs ``soffice``, ``unopkg`` and a ``python3`` that can ``import uno`` (LibreOffice's
Python); run with ``just test-libreoffice``.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request

pytestmark = pytest.mark.libreoffice

ROOT = Path(__file__).resolve().parents[3]
TEXT = "Read teh book today.\nSecond paragraph is fine."

DRIVER = r"""
import json, os, subprocess, sys, time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue
from com.sun.star.lang import Locale

profile, port, text = sys.argv[1], sys.argv[2], sys.argv[3]
office = subprocess.Popen([
    "soffice", f"-env:UserInstallation={profile}", "--headless", "--invisible", "--norestore",
    f"--accept=socket,host=127.0.0.1,port={port};urp;",
])
try:
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(120):
        try:
            ctx = resolver.resolve(
                f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext")
            break
        except Exception:
            time.sleep(0.5)
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    # A view (not Hidden): the spike showed re-checks run in Writer's layout idle loop.
    document = desktop.loadComponentFromURL("private:factory/swriter", "_blank", 0, ())
    body = document.getText()
    cursor = body.createTextCursor()
    cursor.CharLocale = Locale("en", "US", "")
    first, _, second = text.partition("\n")
    body.insertString(cursor, first, False)
    body.insertControlCharacter(cursor, 0, False)
    body.insertString(cursor, second, False)
    iterator = smgr.createInstanceWithContext(
        "com.sun.star.linguistic2.ProofreadingIterator", ctx)
    iterator.startProofreading(document, document)
    errors = []
    for _ in range(60):
        time.sleep(0.5)
        paragraph = document.getFlatParagraphIterator(4, False).getFirstPara()
        result = iterator.checkSentenceAtPosition(
            document, paragraph, paragraph.getText(), Locale("en", "US", ""), 0, 0, -1)
        errors = [
            {"start": e.nErrorStart, "length": e.nErrorLength, "options": list(e.aSuggestions)}
            for e in result.aErrors
        ]
        if errors:
            break
    print(json.dumps(errors))
    document.close(True)
finally:
    office.terminate()
    office.wait(timeout=30)
"""


@dataclass
class StandIn:
    """A documents service that flags "teh" and records what it was sent."""

    puts: list[tuple[str, str]] = field(default_factory=list)
    polls: int = 0


def outside_venv(directory: str) -> bool:
    return not Path(directory).resolve().is_relative_to(Path(sys.prefix).resolve())


def office_environment() -> dict[str, str]:
    """This environment without the virtualenv: LibreOffice's Python loader crashes
    registering the extension when it sees the virtualenv's Python first.
    """
    path = os.pathsep.join(d for d in os.get_exec_path() if outside_venv(d))
    hidden = {"VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"}
    return {
        **{k: v for k, v in os.environ.items() if k not in hidden and not k.startswith("UV_")},
        "PATH": path,
    }


def system_python() -> str:
    """A Python that can ``import uno``: ``LANGCHECK_TEST_UNO_PYTHON``, else the first
    ``python3`` on ``PATH`` outside this virtualenv (which never has uno).
    """
    override = os.environ.get("LANGCHECK_TEST_UNO_PYTHON", "")
    directories = [d for d in os.get_exec_path() if outside_venv(d)]
    search = os.pathsep.join(directories)
    found = [shutil.which(name, path=search) for name in ("python3", "python")]
    candidates = [override] if override != "" else [p for p in found if p is not None]
    for path in candidates:
        probe = subprocess.run([path, "-c", "import uno"], capture_output=True, check=False)
        if probe.returncode == 0:
            return path
    pytest.skip("no Python with LibreOffice's uno module")


@pytest.fixture
def service() -> Iterator[tuple[str, StandIn]]:
    stand_in = StandIn()
    app = FastAPI()

    @app.put("/documents/{uri:path}")
    async def put(uri: str, request: Request) -> dict[str, int]:
        body: object = json.loads(await request.body())
        assert isinstance(body, dict)
        stand_in.puts.append((uri, str(body["text"])))
        return {"version": len(stand_in.puts)}

    @app.get("/documents/{uri:path}/suggestions")
    async def suggestions(uri: str) -> dict[str, object]:
        stand_in.polls += 1
        text = stand_in.puts[-1][1]
        start = text.find("teh")
        found = (
            []
            if start == -1
            else [
                {
                    "id": "sg_1",
                    "start": start,
                    "end": start + 3,
                    "original": "teh",
                    "kind": "Spelling",
                    "operation": "R",
                    "explanation": "Misspelt.",
                    "options": [{"replacement": "the", "score": None, "delta": None}],
                }
            ]
        )
        return {
            "version": len(stand_in.puts),
            "state": "done",
            "suggestions": found,
            "unchecked": [],
        }

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        threading.Event().wait(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", stand_in
    server.should_exit = True
    thread.join(timeout=5)


def test_writer_gets_langcheck_suggestions(tmp_path: Path, service: tuple[str, StandIn]) -> None:
    if shutil.which("soffice") is None or shutil.which("unopkg") is None:
        pytest.skip("LibreOffice is not installed")
    python = system_python()
    url, stand_in = service
    oxt = tmp_path / "langcheck.oxt"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_oxt.py"), str(oxt)], check=True)
    profile = (tmp_path / "profile").as_uri()
    env = {
        **office_environment(),
        "LANGCHECK_SERVER__URL": url,
        "LANGCHECK_SERVER__API_KEY": "lc_test",
    }
    subprocess.run(
        ["unopkg", "add", "-f", f"-env:UserInstallation={profile}", str(oxt)], check=True, env=env
    )
    driver = tmp_path / "drive.py"
    driver.write_text(DRIVER, encoding="utf-8")

    run = subprocess.run(
        [python, str(driver), profile, "2093", TEXT],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=True,
    )

    errors = json.loads(run.stdout.strip().splitlines()[-1])
    assert errors == [{"start": 5, "length": 3, "options": ["the"]}]
    # Writer may send the empty starting document and intermediate versions while the
    # text is inserted; all of them must be the same document, ending with the full text.
    sent = [(uri, text) for uri, text in stand_in.puts if text]
    uris = {uri for uri, _ in sent}
    assert len(uris) == 1, sent
    assert next(iter(uris)).startswith("untitled:")
    assert TEXT in {text for _, text in sent}, sent
    assert stand_in.polls >= 1
