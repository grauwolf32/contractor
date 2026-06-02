"""Unit tests for opaque request tagging shared by http + caido tools.

Tags are injected as ``X-Request-Id`` so a vuln's proof chain can later be
collected deterministically from Caido history. They must be opaque (no vuln
name in live traffic) and zero-padded so no id is a substring of another under
Caido's ``req.raw.cont`` filter.
"""
from __future__ import annotations

from contractor.tools.caido import (
    CaidoTools,
    _find_placeholder_offsets,
    _inject_raw_header,
)
from contractor.tools.http import HTTPClient


class TestHttpRequestTag:
    def test_disabled_without_prefix(self):
        cli = HTTPClient(name="t")
        assert cli._make_request_tag(1) == ""

    def test_zero_padded_and_prefixed(self):
        cli = HTTPClient(name="t", request_tag_prefix="rABC")
        assert cli._make_request_tag(1) == "rABC-h000001"
        assert cli._make_request_tag(12) == "rABC-h000012"

    def test_no_id_is_substring_of_another(self):
        cli = HTTPClient(name="t", request_tag_prefix="rABC")
        t1 = cli._make_request_tag(1)
        t10 = cli._make_request_tag(10)
        assert t1 not in t10 and t10 not in t1


class TestCaidoReplayTag:
    def test_disabled_without_prefix(self):
        backend = CaidoTools(cli=None)  # type: ignore[arg-type]
        assert backend._make_request_tag() == ""

    def test_c_infix_and_increments(self):
        backend = CaidoTools(cli=None, request_tag_prefix="rABC")  # type: ignore[arg-type]
        assert backend._make_request_tag() == "rABC-c000001"
        assert backend._make_request_tag() == "rABC-c000002"


class TestInjectRawHeader:
    def test_inserts_after_request_line_crlf(self):
        raw = "GET / HTTP/1.1\r\nHost: x\r\n\r\n"
        out = _inject_raw_header(raw, "X-Request-Id", "rABC-c000001")
        assert out == (
            "GET / HTTP/1.1\r\n"
            "X-Request-Id: rABC-c000001\r\n"
            "Host: x\r\n\r\n"
        )

    def test_handles_bare_lf(self):
        raw = "GET / HTTP/1.1\nHost: x\n\n"
        out = _inject_raw_header(raw, "X-Request-Id", "t")
        assert out == "GET / HTTP/1.1\nX-Request-Id: t\nHost: x\n\n"

    def test_no_line_break_left_untouched(self):
        assert _inject_raw_header("garbage", "X", "y") == "garbage"


class TestFindPlaceholderOffsets:
    """Regression for M4: offsets must be byte positions into the raw request,
    not character indices into a decoded str."""

    def test_ascii_offsets(self):
        raw = b"GET /?id=42 HTTP/1.1\r\n\r\n"
        assert _find_placeholder_offsets(raw, "42") == [(9, 11)]

    def test_multibyte_prefix_shifts_byte_offset(self):
        # 'é' is two UTF-8 bytes, so the target's byte offset is one greater
        # than its character index. Pre-fix (char indices) this was wrong.
        raw = "GET /?q=é&id=42 HTTP/1.1\r\n\r\n".encode()
        offsets = _find_placeholder_offsets(raw, "42")
        start, end = offsets[0]
        # The bytes at the reported offsets really are the target.
        assert raw[start:end] == b"42"
        # And it is shifted by the extra byte from 'é'.
        char_index = "GET /?q=é&id=42 HTTP/1.1\r\n\r\n".index("42")
        assert start == char_index + 1

    def test_multiple_occurrences(self):
        raw = b"a=1&a=1"
        assert _find_placeholder_offsets(raw, "a=1") == [(0, 3), (4, 7)]

    def test_target_absent(self):
        assert _find_placeholder_offsets(b"GET / HTTP/1.1", "nope") == []
