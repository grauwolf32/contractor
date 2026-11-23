"""Placement location grammar shared with the Server through golden cases."""

from __future__ import annotations

import re

_LABEL = re.compile(r"(?:cloud|local)(?:/[a-z][a-z0-9_-]*)*")


def validate_placement_labels(labels: list[str]) -> None:
    """Require sorted, bounded cloud/local paths forming one ancestor chain."""

    if len(labels) > 32:
        raise ValueError("placement labels must contain at most 32 values")
    previous = ""
    for label in labels:
        if not 1 <= len(label) <= 63 or _LABEL.fullmatch(label) is None:
            raise ValueError("placement labels must be cloud/local paths of 1-63 ASCII bytes")
        if previous and (label <= previous or not label.startswith(previous + "/")):
            raise ValueError("placement labels must be sorted, unique and form one ancestor chain")
        previous = label
