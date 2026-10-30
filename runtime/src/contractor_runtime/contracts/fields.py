"""The HTTP/1.1 field grammars that a header value must satisfy before it is sent.

httpx encodes a header value as ASCII, and h11 refuses to serialize a field
value that is not field-content as it implements RFC 9110 section 5.5: no NUL,
no CR, LF, VT or FF, and no leading or trailing space or tab. Checking the same
grammar where the model supplies a header, bearer token or cookie makes such
input ``http_request_invalid`` up front, instead of a transport failure at send
time or a stored session value that breaks every later request (HTP-R33).

A credential token or header value that the Server delivers in RuntimeSettings
is also sent by the Server's own Go clients, which refuse every other control
character and trim surrounding whitespace, so it follows the stricter
intersection of both (RCF-R113, RCF-R114). The Go Server's
``contracts.ValidCredentialFieldValue`` is the same rule, and
api/testdata/v1alpha1/credential-field-value-cases.json holds both sides and
the private allocation schema to it.
"""

from __future__ import annotations

import re

# h11's field-value: field-vchar (any byte but NUL and whitespace), with runs of
# space or tab only between them. On bytes, \s is exactly [ \t\n\r\f\v].
_FIELD_VALUE = re.compile(rb"(?:[^\x00\s]+(?:[ \t]+[^\x00\s]+)*)?")

# Visible ASCII characters, with runs of space or tab only between them.
_CREDENTIAL_FIELD_VALUE = re.compile(r"[!-~]+(?:[ \t]+[!-~]+)*")
_COOKIE_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,256}")


def valid_cookie_name(value: str) -> bool:
    return _COOKIE_NAME.fullmatch(value) is not None


def valid_cookie_value(value: str) -> bool:
    return all(0x21 <= ord(c) <= 0x7E and c not in '",;\\' for c in value)


def is_field_value(value: str) -> bool:
    """Whether httpx and h11 can send ``value`` as one header field value."""

    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return _FIELD_VALUE.fullmatch(encoded) is not None


def is_credential_field_value(value: str) -> bool:
    """Whether every client sends the credential ``value`` unchanged as one field value.

    It is a non-empty ``is_field_value`` without control characters.
    """

    return _CREDENTIAL_FIELD_VALUE.fullmatch(value) is not None
