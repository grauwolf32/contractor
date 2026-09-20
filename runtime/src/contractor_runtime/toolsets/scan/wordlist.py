"""Validate bounded UTF-8 payload lists before handing a private file to ffuf."""

from __future__ import annotations

from dataclasses import dataclass, field

from contractor_runtime.toolsets.common.input_errors import ToolInputError

MAX_WORDLIST_ARTIFACT_BYTES = 1024 * 1024
MAX_WORDLIST_LINES = 10_000
MAX_WORDLIST_PAYLOAD_BYTES = 4096


class WordlistInputError(ToolInputError):
    """Fixed diagnostics never include supplied payloads or decoder errors."""


@dataclass(frozen=True, slots=True)
class PreparedWordlist:
    raw: bytes = field(repr=False)
    line_count: int


def parse_wordlist(data: bytes) -> PreparedWordlist:
    """Normalize CRLF only, preserving ffuf's line and empty-payload semantics.

    A final LF terminates its preceding payload without creating another one:
    ``b"a\\n"`` has one payload, ``b"a\\n\\n"`` has two, and ``b"\\n"`` has
    one empty payload. Empty files are invalid. Missing final LF is preserved.
    Spaces, comments, duplicates and Unicode separators remain payload data.
    """
    if not isinstance(data, bytes) or len(data) > MAX_WORDLIST_ARTIFACT_BYTES:
        raise WordlistInputError("wordlist artifact exceeds its size limit or is not bytes")
    if not data:
        raise WordlistInputError("wordlist must contain at least one payload")
    if data.startswith(b"\xef\xbb\xbf"):
        raise WordlistInputError("wordlist must not begin with a UTF-8 byte order mark")
    try:
        data.decode("utf-8")
    except UnicodeError:
        raise WordlistInputError("wordlist must contain valid UTF-8 text") from None

    raw = data.replace(b"\r\n", b"\n")
    if b"\r" in raw:
        raise WordlistInputError("wordlist must use LF or CRLF line endings")
    if any((byte < 32 and byte != 10) or byte == 127 for byte in raw):
        raise WordlistInputError("wordlist contains unsupported control characters")
    # ffuf adds this built-in keyword to its input map. Depending on Go's map
    # iteration order, it can substitute a hash inside an already-inserted payload.
    if b"FFUFHASH" in raw:
        raise WordlistInputError("wordlist contains an unsupported ffuf substitution marker")

    # ffuf uses Go's bufio.ScanLines. Count only ASCII LF, and exclude the
    # delimiter's empty tail without dropping intentional empty payloads.
    line_count = raw.count(b"\n") + (not raw.endswith(b"\n"))
    if line_count > MAX_WORDLIST_LINES:
        raise WordlistInputError("wordlist exceeds its payload count limit")
    if any(len(payload) > MAX_WORDLIST_PAYLOAD_BYTES for payload in raw.split(b"\n")):
        raise WordlistInputError("wordlist payload exceeds its UTF-8 byte limit")
    return PreparedWordlist(raw=raw, line_count=line_count)
