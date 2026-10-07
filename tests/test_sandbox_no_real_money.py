import ast
import subprocess
import sys
from pathlib import Path

import pytest

CODE = """
import importlib, pkgutil, sys
import sandbox
for m in pkgutil.iter_modules(sandbox.__path__):
    importlib.import_module("sandbox." + m.name)
print("highcast.kalshi_client" in sys.modules)
"""

SANDBOX_SRC = Path(__file__).resolve().parents[1] / "src" / "sandbox"
CLIENT = "kalshi_client"


def imports_kalshi_client(source: str) -> bool:
    """True if the code imports highcast's authenticated client in any spelling."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(CLIENT in a.name.split(".") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if CLIENT in (node.module or "").split(".") or any(a.name == CLIENT for a in node.names):
                return True
        elif isinstance(node, ast.Call):  # importlib.import_module("highcast.kalshi_client")
            fn = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if fn in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and CLIENT in arg.value:
                    return True
    return False


def test_sandbox_never_imports_the_real_money_client():
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True,
                         check=True, timeout=120)
    assert out.stdout.strip() == "False", out.stderr


def test_no_sandbox_source_file_imports_the_real_money_client():
    files = sorted(SANDBOX_SRC.rglob("*.py"))
    assert files, "found no sandbox sources to scan"
    offenders = [str(f.relative_to(SANDBOX_SRC)) for f in files
                 if imports_kalshi_client(f.read_text(encoding="utf-8"))]
    assert offenders == []


@pytest.mark.parametrize("line", [
    "import highcast.kalshi_client",
    "import highcast.kalshi_client as kc",
    "from highcast import kalshi_client",
    "from highcast.kalshi_client import KalshiClient",
    "from . import kalshi_client",
    "importlib.import_module('highcast.kalshi_client')",
])
def test_the_scanner_catches_every_import_spelling(line):
    assert imports_kalshi_client(line)


def test_the_scanner_ignores_unrelated_code():
    assert not imports_kalshi_client("from highcast.fees import taker_fee\n"
                                     "# highcast.kalshi_client is never imported here\n")
