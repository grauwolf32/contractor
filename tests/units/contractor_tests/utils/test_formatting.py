import unicodedata

import pytest

from contractor.utils.formatting import (
    norm_unicode,
    norm_unicode_strict,
    normalize_slashes,
    xml_escape,
)


def test_norm_unicode_returns_none_for_none():
    assert norm_unicode(None) is None


def test_norm_unicode_normalizes_to_nfc():
    # "é" can be encoded as a single codepoint (NFC) or e + combining (NFD).
    nfd = unicodedata.normalize("NFD", "café")
    nfc = unicodedata.normalize("NFC", "café")

    assert nfd != nfc  # sanity: input forms differ
    assert norm_unicode(nfd) == nfc


def test_norm_unicode_passes_through_ascii():
    assert norm_unicode("hello") == "hello"


def test_norm_unicode_strict_raises_on_none():
    with pytest.raises(ValueError, match="Cannot normalize"):
        norm_unicode_strict(None)  # type: ignore[arg-type]


def test_norm_unicode_strict_returns_normalized():
    nfd = unicodedata.normalize("NFD", "naïve")
    assert norm_unicode_strict(nfd) == unicodedata.normalize("NFC", "naïve")


def test_normalize_slashes_converts_backslashes():
    assert normalize_slashes(r"a\b\c") == "a/b/c"


def test_normalize_slashes_idempotent_on_forward_slashes():
    assert normalize_slashes("a/b/c") == "a/b/c"


def test_normalize_slashes_handles_empty():
    assert normalize_slashes("") == ""


def test_xml_escape_handles_all_entities():
    raw = "<tag attr=\"v\" attr2='v'>a & b</tag>"
    out = xml_escape(raw)

    assert "<" not in out
    assert ">" not in out
    assert "&amp;" in out
    assert "&lt;" in out
    assert "&gt;" in out
    assert "&quot;" in out
    assert "&apos;" in out


def test_xml_escape_amp_first_then_others():
    # Order matters: '&' must be escaped first so we don't double-escape entities.
    assert xml_escape("&lt;") == "&amp;lt;"
