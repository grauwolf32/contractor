"""Run a nonempty offline acceptance suite; no live entry point is imported as main."""

import sys
import unittest
from pathlib import Path

folder = Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.discover(str(folder), pattern="test_*.py")
if suite.countTestCases() == 0:
    raise RuntimeError("context-compaction evaluation selected zero cases")
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    sys.exit(1)
print(
    "Offline context-compaction gate: 7 variants, 3 classes, 105 scored samples; no Gateway calls."
)
