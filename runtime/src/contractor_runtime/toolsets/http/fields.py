"""The cookie rule built on the HTTP/1.1 field grammar of the Runtime contracts.

``contractor_runtime.contracts.fields`` owns the field-value grammar that a
model-supplied header, bearer token or cookie must satisfy (HTP-R33).
"""

from __future__ import annotations

from contractor_runtime.contracts.fields import is_field_value


def is_cookie_pair(name: str, value: str | None) -> bool:
    """Whether a cookie can be sent back inside a Cookie header.

    httpx joins ``name=value`` pairs with ``"; "``. A non-empty pair that is
    itself a field value starts and ends with a field-vchar, so any join of
    such pairs is a field value too.
    """

    pair = name if value is None else f"{name}={value}"
    return bool(name) and is_field_value(pair)
