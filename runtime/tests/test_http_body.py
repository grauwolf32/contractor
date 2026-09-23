"""Bounded response reads decode incrementally and limit received bytes."""

from __future__ import annotations

import asyncio
import gzip
import tracemalloc
import zlib
from pathlib import Path

import httpx
import pytest
from httpx._decoders import SUPPORTED_DECODERS
from test_caido_read_tools import FakeArtifactClient as CaidoArtifactClient
from test_caido_read_tools import create_tools as create_caido_tools
from test_http_toolset import FakeArtifactClient, close_tools, create_tools

import contractor_runtime.adapters.caido_graphql as caido_graphql
import contractor_runtime.toolsets.http.tools as http_tools
from contractor_runtime.http_body import (
    DECODED_CONTENT_CODINGS,
    BodyTooLarge,
    read_limited_body,
)
from contractor_runtime.toolsets.caido.tools import CaidoToolError
from contractor_runtime.toolsets.http.tools import HTTPToolError

BOMB = gzip.compress(b"\0" * (32 * 1024 * 1024))


async def chunks(content: bytes, size: int):
    for index in range(0, len(content), size):
        yield content[index : index + size]


def streamed(content: bytes, encoding: str | None, *, chunk: int = 64 * 1024) -> httpx.Response:
    headers = {} if encoding is None else {"content-encoding": encoding}
    return httpx.Response(200, headers=headers, content=chunks(content, chunk))


def read(response: httpx.Response, limit: int) -> bytes:
    return asyncio.run(read_limited_body(response, limit))


def test_every_httpx_content_decoder_is_decoded_incrementally() -> None:
    # A new httpx decoder (for example brotli) would bypass the bound below.
    assert set(SUPPORTED_DECODERS) == DECODED_CONTENT_CODINGS


def test_a_compression_bomb_never_inflates_past_the_limit() -> None:
    assert len(BOMB) < 64 * 1024
    tracemalloc.start()
    try:
        with pytest.raises(BodyTooLarge):
            read(streamed(BOMB, "gzip"), 1024 * 1024)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # httpx alone decodes the single received chunk to 32 MiB at once.
    assert peak < 8 * 1024 * 1024


@pytest.mark.parametrize(
    ("encoding", "encode"),
    [
        (None, lambda data: data),
        ("identity", lambda data: data),
        ("gzip", gzip.compress),
        ("deflate", zlib.compress),
        # Raw deflate without the zlib wrapper, as httpx also accepts.
        ("deflate", lambda data: zlib.compress(data, wbits=-zlib.MAX_WBITS)),
        ("deflate, gzip", lambda data: gzip.compress(zlib.compress(data))),
        ("GZip", gzip.compress),
        # Unsupported codings are kept exactly as received.
        ("br", lambda data: data),
    ],
)
def test_decoding_matches_httpx_for_supported_codings(encoding, encode) -> None:
    data = bytes(range(256)) * 400
    assert read(streamed(encode(data), encoding, chunk=1000), len(data)) == data
    with pytest.raises(BodyTooLarge):
        read(streamed(encode(data), encoding, chunk=1000), len(data) - 1)


def test_received_bytes_are_limited_even_when_decoding_ignores_them() -> None:
    # Trailing data after a gzip member decodes to nothing but still arrives.
    padded = gzip.compress(b"ok") + b"\0" * 4096
    with pytest.raises(BodyTooLarge):
        read(streamed(padded, "gzip", chunk=512), 1024)


def test_malformed_compressed_content_is_a_decoding_error() -> None:
    with pytest.raises(httpx.DecodingError):
        read(streamed(b"not gzip at all", "gzip"), 1024)


def test_content_already_read_by_httpx_is_only_size_checked() -> None:
    assert read(httpx.Response(200, content=b"x" * 10), 10) == b"x" * 10
    with pytest.raises(BodyTooLarge):
        read(httpx.Response(200, content=b"x" * 11), 10)


def test_http_request_rejects_a_compression_bomb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http_tools, "MAX_RESPONSE_BODY_BYTES", 1024 * 1024)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=chunks(BOMB, 64 * 1024),
            headers={"content-encoding": "gzip", "content-type": "text/plain"},
            request=request,
        )

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, _state = await create_tools(tmp_path, handler, artifacts=artifacts)
        with pytest.raises(HTTPToolError) as failure:
            await tools["http_request"]("https://target.example/bomb")
        assert failure.value.code == "http_response_too_large"
        assert artifacts.writes == 0
        await close_tools(tools)

    asyncio.run(scenario())


def test_caido_rejects_a_compressed_response_beyond_its_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(caido_graphql, "MAX_CAIDO_RESPONSE_BYTES", 1024 * 1024)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=chunks(BOMB, 64 * 1024),
            headers={"content-encoding": "gzip", "content-type": "application/json"},
            request=request,
        )

    async def scenario() -> None:
        tools, _state, handle = await create_caido_tools(
            tmp_path, handler, CaidoArtifactClient(), selected={"caido_scope"}
        )
        with pytest.raises(CaidoToolError) as failure:
            await tools["caido_scope"]()
        assert failure.value.code == "caido_response_too_large"
        await handle.close()

    asyncio.run(scenario())
