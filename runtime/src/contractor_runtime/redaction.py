"""The one policy for matching Runtime private values in recorded text.

This module imports nothing from the toolsets, allocation or Worker packages, so
each of them can share it without an import cycle.
"""

from __future__ import annotations

# A private value this long is specific enough to be matched anywhere in
# model- or Worker-authored text; a shorter one (a proxy username, a short
# password) only as a complete string, so ordinary words cannot fail closed.
MIN_PRIVATE_SUBSTRING_BYTES = 16
# Retained in place of a redacted private value.
REDACTED = "[REDACTED]"
