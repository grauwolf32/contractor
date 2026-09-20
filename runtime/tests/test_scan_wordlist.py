from __future__ import annotations

import pytest

from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.scan.wordlist import (
    MAX_WORDLIST_ARTIFACT_BYTES,
    MAX_WORDLIST_LINES,
    MAX_WORDLIST_PAYLOAD_BYTES,
    parse_wordlist,
)


@pytest.mark.parametrize(
    "data,expected,count",
    [
        (b"one", b"one", 1),
        (b"one\n", b"one\n", 1),
        (b"one\r\n", b"one\n", 1),
        (b"one\r\ntwo\nthree", b"one\ntwo\nthree", 3),
        (b"\n", b"\n", 1),
        (b"\r\n", b"\n", 1),
        (b"\n\n", b"\n\n", 2),
        (b"one\n\n", b"one\n\n", 2),
        (b"\r\none\r\n\r\n", b"\none\n\n", 3),
        (b"\none\n\ntwo\n", b"\none\n\ntwo\n", 4),
    ],
)
def test_line_endings_and_explicit_empty_payloads(data, expected, count):
    prepared = parse_wordlist(data)
    assert prepared.raw == expected
    assert prepared.line_count == count


def test_spaces_comments_duplicates_and_utf8_are_preserved():
    raw = "  spaced  \n#comment\ninline # comment\nrepeat\nrepeat\nключ/🔑\n  ".encode()
    prepared = parse_wordlist(raw)
    assert prepared.raw == raw
    assert prepared.line_count == 7


def test_unicode_separators_and_noninitial_bom_are_payload_data():
    raw = "first\u0085second\u2028third\u2029fourth\ufefflast\nnext".encode()
    prepared = parse_wordlist(raw)
    assert prepared.raw == raw
    assert prepared.line_count == 2


def test_prepared_representation_does_not_disclose_payloads():
    prepared = parse_wordlist(b"secret-canary\n")
    assert prepared.raw == b"secret-canary\n"
    assert "canary" not in repr(prepared)


@pytest.mark.parametrize(
    "data",
    [b"FFUFHASH", b"prefix-FFUFHASH-secret-canary\n", b"first\r\nFFUFHASH\r\nlast"],
)
def test_builtin_ffuf_hash_substitution_is_rejected_without_payload(data):
    with pytest.raises(ToolInputError, match="substitution marker") as caught:
        parse_wordlist(data)
    assert "canary" not in str(caught.value)


def test_nonreserved_marker_text_is_preserved():
    raw = b"FUZZ\nffufhash\nFFUfHASH\nFFUFHAS\n"
    prepared = parse_wordlist(raw)
    assert prepared.raw == raw
    assert prepared.line_count == 4


@pytest.mark.parametrize("ending", [b"", b"\n", b"\r\n"])
@pytest.mark.parametrize(
    "payload", [b"a" * MAX_WORDLIST_PAYLOAD_BYTES, "я".encode() * (MAX_WORDLIST_PAYLOAD_BYTES // 2)]
)
def test_payload_byte_limit_excludes_line_delimiters(payload, ending):
    prepared = parse_wordlist(payload + ending)
    assert prepared.raw == payload + ending.replace(b"\r\n", b"\n")
    assert prepared.line_count == 1
    with pytest.raises(ToolInputError, match="byte limit"):
        parse_wordlist(payload + b"x" + ending)


def test_payload_limit_uses_utf8_bytes_instead_of_character_count():
    with pytest.raises(ToolInputError, match="byte limit"):
        parse_wordlist(("я" * MAX_WORDLIST_PAYLOAD_BYTES).encode())


@pytest.mark.parametrize("final_delimiter", [False, True])
def test_maximum_payload_count_includes_duplicates(final_delimiter):
    raw = b"\n".join([b"repeat"] * MAX_WORDLIST_LINES)
    if final_delimiter:
        raw += b"\n"
    prepared = parse_wordlist(raw)
    assert prepared.raw == raw
    assert prepared.line_count == MAX_WORDLIST_LINES
    with pytest.raises(ToolInputError, match="count limit"):
        parse_wordlist(raw + (b"repeat\n" if final_delimiter else b"\nrepeat"))


def test_maximum_payload_count_includes_empty_payloads():
    prepared = parse_wordlist(b"\n" * MAX_WORDLIST_LINES)
    assert prepared.line_count == MAX_WORDLIST_LINES
    with pytest.raises(ToolInputError, match="count limit"):
        parse_wordlist(b"\n" * (MAX_WORDLIST_LINES + 1))


def test_exact_artifact_byte_limit_is_accepted():
    block = b"a" * MAX_WORDLIST_PAYLOAD_BYTES + b"\n"
    count, remainder = divmod(MAX_WORDLIST_ARTIFACT_BYTES, len(block))
    raw = block * count + b"a" * remainder
    assert len(raw) == MAX_WORDLIST_ARTIFACT_BYTES
    assert parse_wordlist(raw).raw == raw
    with pytest.raises(ToolInputError, match="size limit"):
        parse_wordlist(raw + b"\n")


def test_artifact_byte_limit_is_checked_before_crlf_normalization():
    raw = (b"a" * 1022 + b"\r\n") * 1024 + b"\n"
    assert len(raw) == MAX_WORDLIST_ARTIFACT_BYTES + 1
    assert len(raw.replace(b"\r\n", b"\n")) < MAX_WORDLIST_ARTIFACT_BYTES
    with pytest.raises(ToolInputError, match="size limit"):
        parse_wordlist(raw)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        None,
        "secret-canary",
        bytearray(b"secret-canary"),
        b"\xef\xbb\xbfsecret-canary",
        b"secret-canary\xff",
        b"secret-canary\xc3",
        b"secret-canary\xc0\xaf",
        b"secret-canary\xed\xa0\x80",
        b"secret-canary\xf4\x90\x80\x80",
        b"secret-canary\r",
        b"secret-canary\rsecond",
        b"secret-canary\r\r\n",
    ],
)
def test_invalid_input_uses_only_fixed_diagnostics(data):
    with pytest.raises(ToolInputError) as caught:
        parse_wordlist(data)
    assert caught.value.code == "tool_input_invalid"
    assert not caught.value.retryable
    assert "canary" not in str(caught.value)
    assert "canary" not in caught.value.diagnostic_message


@pytest.mark.parametrize("control", [*range(10), *range(11, 13), *range(14, 32), 127])
def test_unsupported_ascii_controls_are_rejected_without_payload(control):
    with pytest.raises(ToolInputError, match="control characters") as caught:
        parse_wordlist(b"secret-canary" + bytes([control]))
    assert "canary" not in str(caught.value)
