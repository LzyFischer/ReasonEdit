from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, ATTR_SCORES_DIR
#!/usr/bin/env python3
"""
Patch hard-coded paths to use `config.paths`.

Backs up each modified .py as .py.bak — totally safe to run.
"""

import re
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
PATTERNS = {
    r'Path\(str(DATA_DIR / "([^")]+)"\)'        : r'DATA_DIR / "\1"',
    r'str(DATA_DIR / "([^")]+)"'                : r'str(DATA_DIR / "\1")',
    r'Path\(str(OUTPUTS_DIR / "([^")]+)"\)'      : r'OUTPUTS_DIR / "\1"',
    r'str(OUTPUTS_DIR / "([^")]+)"'              : r'str(OUTPUTS_DIR / "\1")',
    r'"/scratch[^"]*?/output/([^"]+)"': r'str(OUTPUTS_DIR / "\1")',
    r'str(OUTPUTS_DIR / "([^")]+)"'      : r'str(OUTPUTS_DIR / "\1")',
}

for py in ROOT.rglob("*.py"):
    original = text = py.read_text()
    for pat, rep in PATTERNS.items():
        text = re.sub(pat, rep, text)
    if text != original:
        py.with_suffix(".py.bak").write_text(original)
        # ensure import once at file top
        if "config.paths" not in text.splitlines()[0]:
            text = (
                "from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, "
                "ATTR_SCORES_DIR\n" + text
            )
        py.write_text(text)
        print(f"patched {py.relative_to(ROOT)}")

print("✓ All done – originals saved as *.py.bak")