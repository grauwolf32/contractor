"""Offline fixture check; only a passing check produces a report."""

import json
from pathlib import Path

from calculator import add

cases = [(2, 3, 5), (-2, 3, 1), (0, 0, 0)]
for a, b, expected in cases:
    actual = add(a, b)
    if actual != expected:
        raise SystemExit(f"add({a}, {b}) = {actual}, expected {expected}")

Path("report.json").write_text(
    json.dumps({"passed": len(cases), "status": "ok"}, sort_keys=True) + "\n",
    encoding="utf-8",
)
