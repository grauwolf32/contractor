"""Bounded reads of streamed httpx response bodies, including content decoding.

httpx decodes a whole received chunk at once, so a single 64 KiB gzip chunk can
inflate to tens of MiB before a caller sees it and checks its limit. This reader
limits the bytes received and decodes incrementally, never producing more than
one byte beyond the limit before it stops.
"""

from __future__ import annotations

import zlib

import httpx

# The content codings this reader decodes. They match httpx's decoders for the
# locked dependency set; any other coding is kept exactly as received.
DECODED_CONTENT_CODINGS = frozenset({"identity", "gzip", "deflate"})


class BodyTooLarge(Exception):
    """The body exceeded its limit as received or after decoding."""


class _ZlibStage:
    """One gzip/deflate layer with httpx's raw-deflate fallback."""

    def __init__(self, coding: str) -> None:
        self._raw_fallback = coding == "deflate"
        self._first = True
        self._decompressor = zlib.decompressobj(
            zlib.MAX_WBITS | 16 if coding == "gzip" else zlib.MAX_WBITS
        )
        self._produced = 0

    def decode(self, data: bytes, limit: int) -> bytes:
        first = self._first
        self._first = False
        try:
            # One byte beyond the limit is enough to know the body is too large.
            output = self._decompressor.decompress(data, limit - self._produced + 1)
        except zlib.error:
            if first and self._raw_fallback:
                # Some servers send raw deflate without the zlib wrapper.
                self._decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
                return self.decode(data, limit)
            raise httpx.DecodingError("response content could not be decoded") from None
        return self._count(output, limit)

    def flush(self, limit: int) -> bytes:
        try:
            output = self._decompressor.flush()
        except zlib.error:
            raise httpx.DecodingError("response content could not be decoded") from None
        return self._count(output, limit)

    def _count(self, output: bytes, limit: int) -> bytes:
        self._produced += len(output)
        if self._produced > limit:
            raise BodyTooLarge
        return output


def _stages(headers: httpx.Headers) -> list[_ZlibStage]:
    codings = [
        value.strip().lower() for value in headers.get_list("content-encoding", split_commas=True)
    ]
    # Codings are listed in the order they were applied; decode in reverse.
    return [_ZlibStage(coding) for coding in reversed(codings) if coding in {"gzip", "deflate"}]


async def read_limited_body(response: httpx.Response, limit: int) -> bytes:
    """Return the decoded body of a streamed response, at most ``limit`` bytes.

    ``limit`` applies to the bytes received and, separately, to the decoded
    body. Raises BodyTooLarge when either is exceeded and httpx.DecodingError
    for malformed compressed content; transport errors propagate from httpx.
    """

    if response.is_stream_consumed:
        # httpx reads and decodes content handed to Response() up front, as
        # mock transports do; only its size remains to be checked.
        content = response.content
        if len(content) > limit:
            raise BodyTooLarge
        return content
    stages = _stages(response.headers)
    body = bytearray()
    received = 0

    def append(data: bytes) -> None:
        if len(body) + len(data) > limit:
            raise BodyTooLarge
        body.extend(data)

    async for chunk in response.aiter_raw():
        received += len(chunk)
        if received > limit:
            raise BodyTooLarge
        for stage in stages:
            chunk = stage.decode(chunk, limit)
        append(chunk)
    remainder = b""
    for stage in stages:
        remainder = stage.decode(remainder, limit) + stage.flush(limit)
    append(remainder)
    return bytes(body)
