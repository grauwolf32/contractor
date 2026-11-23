"""Canonical private-wire encoder used only to assert test fixtures."""

import jcs

from contractor_runtime.contracts import WireModel


def encode_private(value: WireModel) -> bytes:
    dumped = value.model_dump(mode="json", by_alias=True, exclude_none=True)
    return jcs.canonicalize(dumped)
