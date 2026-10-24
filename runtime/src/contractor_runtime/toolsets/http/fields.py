"""The HTTP/1.1 field grammar that a request must satisfy before it is sent.

httpx encodes a header value as ASCII, and h11 refuses to serialize a field
value that is not field-content as it implements RFC 9110 section 5.5: no NUL,
no CR, LF, VT or FF, and no leading or trailing space or tab. Checking the same
grammar where the model supplies a header, bearer token or cookie makes such
input ``http_request_invalid`` up front, instead of a transport failure at send
time or a stored session value that breaks every later request.
"""

from __future__ import annotations

import re

# h11's field-value: field-vchar (any byte but NUL and whitespace), with runs of
# space or tab only between them. On bytes, \s is exactly [ \t\n\r\f\v].
_FIELD_VALUE = re.compile(rb"(?:[^\x00\s]+(?:[ \t]+[^\x00\s]+)*)?")


def is_field_value(value: str) -> bool:
    """Whether httpx and h11 can send ``value`` as one header field value."""

    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return _FIELD_VALUE.fullmatch(encoded) is not None


def is_cookie_pair(name: str, value: str | None) -> bool:
    """Whether a cookie can be sent back inside a Cookie header.

    httpx joins ``name=value`` pairs with ``"; "``. A non-empty pair that is
    itself a field value starts and ends with a field-vchar, so any join of
    such pairs is a field value too.
    """

    pair = name if value is None else f"{name}={value}"
    return bool(name) and is_field_value(pair)
