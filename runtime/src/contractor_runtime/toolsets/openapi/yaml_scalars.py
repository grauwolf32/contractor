"""The YAML scalar and mapping-key rule of OpenAPI source documents (SCN-R124).

Server scan preparation (internal/scanplan) and the openapi-operations@1
inventory (internal/auditdomain) read YAML with go-yaml v4 under the rule of
internal/helpers/documentyaml. The Toolset ``openapi@1`` reads and writes the same
documents, so this module ports that rule exactly instead of using PyYAML's
YAML 1.1 resolvers, and api/testdata/openapi-yaml-scalar-cases.json holds both
languages to the same verdicts:

- A plain scalar resolves as go-yaml v4 resolves it: ``yes``, ``on``, ``y``
  and ``1:30`` are strings; ``0o17``, ``017``, ``0x1F``, ``0b101``, ``1_000``
  and ``1e3`` are numbers; a plain date stays its source text.
- A number must be finite and within the exact integer range of IEEE 754
  doubles (LIM-SCN-67). A plain scalar spelled like a number that does not
  resolve to one, such as ``0x10000000000000000``, is rejected rather than
  read as a string.
- An explicit tag must be a core tag. ``!!int``, ``!!float``, ``!!bool`` and
  ``!!null`` accept only text that means a value of that type.

The mapping-key rule (a plain key is the string of its source text) and the
rejection of merge keys, anchors and aliases live in the loader that uses this
module. The writer quotes every string that this rule or YAML 1.1 would not
read back as the same string, so the bytes ``openapi@1`` writes mean one value
to every reader.
"""

from __future__ import annotations

import math
import re

import yaml

from contractor_runtime.contracts.base import MAX_EXACT_INTEGER
from contractor_runtime.core.input_errors import ToolInputError

STR_TAG = "tag:yaml.org,2002:str"
INT_TAG = "tag:yaml.org,2002:int"
FLOAT_TAG = "tag:yaml.org,2002:float"
BOOL_TAG = "tag:yaml.org,2002:bool"
NULL_TAG = "tag:yaml.org,2002:null"
MERGE_TAG = "tag:yaml.org,2002:merge"
MAP_TAG = "tag:yaml.org,2002:map"
SEQ_TAG = "tag:yaml.org,2002:seq"
# A plain scalar spelled like a number that go-yaml leaves a string only
# because the number overflowed; every reader rejects it.
NUMERIC_STRING_TAG = "tag:contractor.invalid,2026:numeric-string"

# The explicit tags a scalar may carry.
SCALAR_TAGS = frozenset({STR_TAG, INT_TAG, FLOAT_TAG, BOOL_TAG, NULL_TAG})

MAX_SAFE_INTEGER = MAX_EXACT_INTEGER
_MAX_SAFE_INTEGER_TEXT = str(MAX_SAFE_INTEGER)
_MAX_SAFE_INTEGER_DIGITS = len(_MAX_SAFE_INTEGER_TEXT)
NUMBER_MESSAGE = (
    f"OpenAPI document numbers must be finite and within ±{MAX_SAFE_INTEGER}; "
    "quote a string that is spelled like a number"
)
TAGGED_SCALAR_MESSAGE = "OpenAPI YAML scalar text does not match its explicit tag"

TRUE_SPELLINGS = frozenset({"true", "True", "TRUE"})
FALSE_SPELLINGS = frozenset({"false", "False", "FALSE"})
NULL_SPELLINGS = frozenset({"", "~", "null", "Null", "NULL"})

# go-yaml v4 resolveMap: exact plain spellings. NaN and the infinities resolve
# to floats that the readers then reject as non-finite.
_RESOLVE_MAP: dict[str, str] = {
    **dict.fromkeys(TRUE_SPELLINGS | FALSE_SPELLINGS, BOOL_TAG),
    **dict.fromkeys(NULL_SPELLINGS, NULL_TAG),
    **dict.fromkeys(
        (
            ".nan", ".NaN", ".NAN",
            ".inf", ".Inf", ".INF",
            "+.inf", "+.Inf", "+.INF",
            "-.inf", "-.Inf", "-.INF",
            "-0", "-0.0",
        ),
        FLOAT_TAG,
    ),
    "<<": MERGE_TAG,
}  # fmt: skip

# go-yaml v4 yamlStyleFloat, matched after removing underscores.
_YAML_STYLE_FLOAT = re.compile(r"[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?")
# internal/helpers/documentnumber numericYAMLScalar.
_NUMERIC = re.compile(
    r"[+-]?(?:0[xX][0-9a-fA-F_]+|0[oO][0-7_]+|0[bB][01_]+"
    r"|(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9][0-9_]*)(?:[eE][+-]?[0-9]+)?)"
)
# The decimal syntax of strconv.ParseFloat. The mantissa needs a digit, and
# underscores must separate digits (_underscores_ok).
_GO_DECIMAL_FLOAT = re.compile(r"[+-]?([0-9_]*(?:\.[0-9_]*)?)(?:[eE][+-]?[0-9][0-9_]*)?")
_BASE_PREFIXES = {"b": 2, "B": 2, "o": 8, "O": 8, "x": 16, "X": 16}
_DIGITS = "0123456789abcdef"
# YAML 1.1 booleans that PyYAML's resolver leaves out.
_YAML_11_BOOL_LETTERS = frozenset({"y", "Y", "n", "N"})


def looks_numeric(text: str) -> bool:
    """documentnumber.LooksNumeric: a plain scalar spelled like a YAML number."""

    return _NUMERIC.fullmatch(text) is not None


def _parse_uint(text: str) -> int | None:
    """strconv.ParseUint(text, 0, 64) for a text without underscores."""

    if not text:
        return None
    base = 10
    if text[0] == "0":
        if len(text) >= 3 and text[1] in _BASE_PREFIXES:
            base = _BASE_PREFIXES[text[1]]
            text = text[2:]
        else:
            base = 8
            text = text[1:]
    value = 0
    for character in text:
        digit = _DIGITS.find(character.lower()) if character.isascii() else -1
        if digit < 0 or digit >= base:
            return None
        value = value * base + digit
        if value >= 1 << 64:
            # strconv stops at the first overflow; either way the text is not
            # a 64-bit number.
            return None
    return value


def _parse_int(text: str) -> int | None:
    """strconv.ParseInt(text, 0, 64) for a text without underscores."""

    negative = text[:1] == "-"
    unsigned = _parse_uint(text[1:] if text[:1] in ("+", "-") else text)
    if unsigned is None or unsigned > (1 << 63) - (0 if negative else 1):
        return None
    return -unsigned if negative else unsigned


def _underscores_ok(text: str) -> bool:
    """strconv.underscoreOK for a decimal text: underscores separate digits."""

    saw = "^"
    for character in text[1:] if text[:1] in ("+", "-") else text:
        if "0" <= character <= "9":
            saw = "0"
        elif character == "_":
            if saw != "0":
                return False
            saw = "_"
        elif saw == "_":
            return False
        else:
            saw = "!"
    return saw != "_"


def _parse_decimal_float(text: str) -> float | None:
    """strconv.ParseFloat(text, 64) for a decimal text, or None on any error.

    A result that underflows to zero is returned, as strconv returns it; one
    that overflows is an error.
    """

    match = _GO_DECIMAL_FLOAT.fullmatch(text)
    if match is None or not any("0" <= character <= "9" for character in match.group(1)):
        return None
    if "_" in text and not _underscores_ok(text):
        return None
    value = float(text.replace("_", ""))
    return value if math.isfinite(value) else None


def resolve_plain(text: str) -> str:
    """The tag go-yaml v4 resolves for an untagged plain scalar.

    A plain date resolves to a string here, where go-yaml says timestamp; the
    readers return its source text either way. A string spelled like a number
    resolves to NUMERIC_STRING_TAG, which every reader rejects.
    """

    tag = _RESOLVE_MAP.get(text)
    if tag is not None:
        return tag
    first = text[:1]
    if first == ".":
        if _parse_decimal_float(text) is not None:
            return FLOAT_TAG
    elif first in ("+", "-") or "0" <= first <= "9":
        plain = text.replace("_", "")
        if _parse_int(plain) is not None or _parse_uint(plain) is not None:
            return INT_TAG
        if _YAML_STYLE_FLOAT.fullmatch(plain) is not None and math.isfinite(float(plain)):
            return FLOAT_TAG
    return NUMERIC_STRING_TAG if looks_numeric(text) else STR_TAG


def document_integer(text: str) -> int | None:
    """documentnumber.Integer: an integer within ±(2**53 - 1), or None."""

    value = _parse_int(text.replace("_", ""))
    if value is None or abs(value) > MAX_SAFE_INTEGER:
        return None
    return value


def document_float(text: str) -> float | None:
    """documentnumber.Float of an underscore-free text, or None.

    The exact decimal magnitude is checked before the float is accepted, so
    9007199254740991.1 and an underflow to zero are rejected.
    """

    if not looks_numeric(text):
        return None
    absolute = text.removeprefix("+").removeprefix("-")
    if absolute[:2].lower() in ("0x", "0o", "0b"):
        # strconv.ParseFloat has no octal or binary floats, and a hexadecimal
        # float needs a p exponent that the numeric spelling cannot contain.
        return None
    value = _parse_decimal_float(text)
    if value is None or abs(value) > MAX_SAFE_INTEGER:
        return None
    mantissa, _, exponent_text = absolute.lower().partition("e")
    integer, _, fraction = mantissa.partition(".")
    digits = (integer + fraction).lstrip("0")
    if not digits:
        return value
    if value == 0:
        # Underflow would silently turn an explicit nonzero example into zero.
        return None
    exponent = 0
    if exponent_text:
        # strconv.ParseInt(exponent, 10, 32); leading zeros do not count.
        magnitude = exponent_text.lstrip("+-").lstrip("0")
        if len(magnitude) > 10:
            return None
        exponent = int(magnitude or "0") * (-1 if exponent_text[0] == "-" else 1)
        if not -(1 << 31) <= exponent < 1 << 31:
            return None
    integer_digits = len(digits) - len(fraction) + exponent
    if integer_digits > _MAX_SAFE_INTEGER_DIGITS:
        return None
    if integer_digits == _MAX_SAFE_INTEGER_DIGITS:
        whole = (digits + "0" * _MAX_SAFE_INTEGER_DIGITS)[:_MAX_SAFE_INTEGER_DIGITS]
        if whole > _MAX_SAFE_INTEGER_TEXT or (
            whole == _MAX_SAFE_INTEGER_TEXT
            and len(digits) > _MAX_SAFE_INTEGER_DIGITS
            and digits[_MAX_SAFE_INTEGER_DIGITS:].strip("0")
        ):
            return None
    return value


def scalar_value(tag: str, text: str) -> str | int | float | bool | None:
    """The JSON value of a scalar with a resolved or explicit tag.

    Raises ToolInputError when the rule rejects the scalar.
    """

    if tag == STR_TAG:
        return text
    if tag in (NULL_TAG, BOOL_TAG):
        if tag == NULL_TAG and text in NULL_SPELLINGS:
            return None
        if tag == BOOL_TAG and text in TRUE_SPELLINGS | FALSE_SPELLINGS:
            return text in TRUE_SPELLINGS
        raise ToolInputError(TAGGED_SCALAR_MESSAGE)
    if tag in (INT_TAG, FLOAT_TAG, NUMERIC_STRING_TAG):
        number: int | float | None = None
        if tag == INT_TAG:
            number = document_integer(text)
        elif tag == FLOAT_TAG:
            number = document_float(text.replace("_", ""))
        if number is None:
            raise ToolInputError(NUMBER_MESSAGE)
        return number
    if tag == MERGE_TAG:
        raise ToolInputError("OpenAPI YAML merge keys (<<) are not supported")
    raise ToolInputError("OpenAPI YAML tags other than the core tags are not supported")


def writer_tag(text: str) -> str:
    """The tag a writer must assume a plain scalar has, to decide on quoting.

    A string may be written plain only when this rule and YAML 1.1 (PyYAML's
    resolvers, plus the y and n booleans of the 1.1 type repository) both
    read it as a string.
    """

    tag = resolve_plain(text)
    if tag != STR_TAG:
        return tag
    if text in _YAML_11_BOOL_LETTERS:
        return BOOL_TAG
    resolvers = yaml.resolver.Resolver.yaml_implicit_resolvers
    for candidate, pattern in resolvers.get(text[:1], []) + resolvers.get(None, []):
        if pattern.match(text):
            return str(candidate)
    return STR_TAG
