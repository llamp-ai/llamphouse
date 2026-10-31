"""The 2.0 runtime stands on its own: nothing of the 1.x core is imported."""

import subprocess
import sys


def test_the_runtime_and_its_tracing_never_import_the_1x_core():
    code = ("import sys, llamphouse.runtime, llamphouse.runtime.api, llamphouse.runtime.server, "
            "llamphouse.runtime.tracing.stores; print(any(m.startswith('llamphouse.core') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_an_engine_without_tracing_never_imports_opentelemetry():
    code = "import sys, llamphouse.runtime; print(any(m.startswith('opentelemetry') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
