"""llamphouse.sdk imports pydantic and the standard library only."""

import subprocess
import sys

ALLOWED = {
    "llamphouse",
    "pydantic", "pydantic_core", "annotated_types", "typing_extensions", "typing_inspection",
    # the interpreter's own start-up (virtualenv, site customisation)
    "_virtualenv", "sitecustomize", "usercustomize",
}


def test_the_sdk_imports_nothing_beyond_pydantic():
    code = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import llamphouse.sdk, llamphouse.sdk.testing, llamphouse.sdk.wire\n"
        "std = set(sys.stdlib_module_names)\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before} - std\n"
        "print(sorted(m for m in new if not m.startswith('_sysconfigdata')))\n"  # the stdlib's per-platform config
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    loaded = set(eval(out))
    assert loaded <= ALLOWED, f"llamphouse.sdk pulls in {sorted(loaded - ALLOWED)}"


def test_importing_the_sdk_does_not_load_the_1x_runtime():
    code = "import sys, llamphouse.sdk; print('llamphouse.core' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
