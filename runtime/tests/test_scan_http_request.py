from __future__ import annotations

import json

import pytest

from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.scan.http_request import (
    MAX_BODY_BYTES,
    MAX_HEADER_VALUE_BYTES,
    MAX_HEADERS,
    MAX_REQUEST_ARTIFACT_BYTES,
    MAX_TEST_PARAMETERS,
    parse_http_request,
)


def request_document(**changes):
    document = {
        "schemaVersion": 1,
        "method": "POST",
        "url": "https://target.invalid:8443/items%2Fsearch?id=7&id=8",
        "headers": [
            {"name": "Authorization", "value": "Bearer header-canary"},
            {"name": "Cookie", "value": "session=cookie-canary"},
            {"name": "Content-Type", "value": "application/json; charset=utf-8"},
        ],
        "body": '{"id":7,"token":"body-canary-ключ"}',
        "testParameters": ["id"],
    }
    document.update(changes)
    return document


def parse_document(**changes):
    return parse_http_request(json.dumps(request_document(**changes)).encode())


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
def test_request_preserves_target_headers_utf8_body_and_selected_parameters(method):
    body = '{\n  "id":7,"token":"body-canary-ключ"\n}'
    request = parse_document(method=method, body=body, testParameters=["id", "item.name"])
    head, separator, raw_body = request.raw.partition(b"\r\n\r\n")
    assert request.method == method
    assert request.test_parameters == ("id", "item.name")
    assert head.startswith(
        f"{method} https://target.invalid:8443/items%2Fsearch?id=7&id=8 HTTP/1.1\r\n".encode()
    )
    assert b"Host: target.invalid:8443\r\n" in head
    assert b"Authorization: Bearer header-canary\r\n" in head
    assert b"Cookie: session=cookie-canary\r\n" in head
    assert head.endswith(f"Content-Length: {len(body.encode())}".encode())
    assert separator == b"\r\n\r\n"
    assert raw_body == body.encode()


@pytest.mark.parametrize(
    "url,host",
    [
        ("http://target.invalid:80/?id=7", "target.invalid:80"),
        ("https://target.invalid:443/?id=7", "TARGET.invalid:443"),
        ("https://[2001:DB8::1]:8443/?id=7", "[2001:db8::1]:8443"),
    ],
)
def test_explicit_host_and_port_are_preserved(url, host):
    request = parse_document(url=url, headers=[{"name": "hOsT", "value": host}], body="")
    assert f"POST {url} HTTP/1.1\r\n".encode() in request.raw
    assert request.raw.count(b"hOsT:") == 1
    assert b"Host:" not in request.raw
    assert f"hOsT: {host}\r\n".encode() in request.raw


@pytest.mark.parametrize(
    "changes",
    [
        {"schemaVersion": True},
        {"schemaVersion": "1"},
        {"schemaVersion": 1.0},
        {"unexpected": "secret-canary"},
        {"method": "POST\r\nsecret-canary"},
        {"method": "post"},
        {"method": "CONNECT"},
        {"method": []},
        {"url": "https://user:secret-canary@target.invalid/path"},
        {"url": "https://target.invalid/path#secret-canary"},
        {"url": "https://target.invalid/path#"},
        {"url": "https://target.invalid:0/path"},
        {"url": "https://target.invalid:65536/path"},
        {"url": "https://target.invalid:/path"},
        {"url": "https://target.invalid\r\nsecret-canary/path"},
        {"url": "https://target.invalid\\secret-canary/path"},
        {"url": "https://target.invalid/path%zz"},
        {"url": "https://target.invalid/ключ"},
        {"url": "file:///secret-canary"},
        {"url": "https://[2001:db8::1]secret-canary/path"},
        {"body": None},
        {"body": "secret-canary\ud800"},
        {"body": "secret-canary\x00"},
        {"body": "secret-canary\r\nvalue"},
        {"body": "secret-canary\n"},
        {"body": "secret-canary\n \t"},
        {"body": '{"id":"*"}'},
        {"body": "secret-canary %INJECT_HERE%"},
        {"body": "secret-canary %inject here%"},
        {"body": "secret-canary ==========\nGET http://target.invalid/"},
        {"body": "secret-canary ### Conversation"},
        {"body": 'secret-canary <request base64="true">'},
        {"body": 'secret-canary <REQUEST BASE64="TRUE">'},
        {"body": 'secret-canary <ReQuEsT BaSe64="true">'},
        {"url": "https://target.invalid/?id=*"},
        {"testParameters": []},
        {"testParameters": "secret-canary"},
        {"testParameters": ["id", "id"]},
        {"testParameters": ["id,secret-canary"]},
        {"testParameters": ["id\nsecret-canary"]},
        {"testParameters": ["*"]},
        {"testParameters": [1]},
        {"headers": {"secret-canary": "value"}},
        {"headers": [{"name": "secret-canary"}]},
        {"headers": [{"name": "X-Header", "value": "value", "secret-canary": True}]},
    ],
)
def test_invalid_request_fields_use_safe_diagnostics(changes):
    with pytest.raises(ToolInputError) as caught:
        parse_document(**changes)
    assert "canary" not in str(caught.value)
    assert caught.value.code == "tool_input_invalid"
    assert not caught.value.retryable


@pytest.mark.parametrize(
    "headers",
    [
        [{"name": "Bad Name", "value": "secret-canary"}],
        [{"name": "X-Header\r\n", "value": "secret-canary"}],
        [{"name": "X-Header", "value": "secret-canary\r\nInjected: yes"}],
        [{"name": "X-Header", "value": "secret-canary\tvalue"}],
        [{"name": "X-Header", "value": "secret-canary\x7f"}],
        [{"name": "X-Header", "value": "ключ"}],
        [{"name": "X-Header", "value": " secret-canary"}],
        [{"name": "X-Header", "value": "secret-canary "}],
        [{"name": "X-*", "value": "secret-canary"}],
        [{"name": "X-Header", "value": "secret-canary*"}],
        [{"name": "Host", "value": "target.invalid"}],
        [{"name": "Host", "value": "target.invalid:443"}],
        [{"name": "Content-Length", "value": "1"}],
        [{"name": "Content-Type", "value": "text/plain; charset=iso-8859-1"}],
        [{"name": "Content-Type", "value": "text/plain; charset=utf-8; charset=utf-8"}],
        [{"name": "Cookie", "value": "a=1"}, {"name": "cookie", "value": "b=2"}],
    ],
)
def test_invalid_headers_are_rejected_without_values(headers):
    with pytest.raises(ToolInputError) as caught:
        parse_document(headers=headers)
    assert "canary" not in str(caught.value)


@pytest.mark.parametrize(
    "name",
    [
        "Transfer-Encoding",
        "Content-Encoding",
        "Connection",
        "Proxy-Connection",
        "Keep-Alive",
        "TE",
        "Trailer",
        "Upgrade",
        "Expect",
        "If-Modified-Since",
        "If-None-Match",
    ],
)
def test_unsupported_framing_and_sqlmap_removed_headers_are_rejected(name):
    with pytest.raises(ToolInputError):
        parse_document(headers=[{"name": name.swapcase(), "value": "secret-canary"}])


def test_content_length_uses_utf8_bytes_and_is_not_duplicated():
    body = "ключ"
    request = parse_document(
        body=body, headers=[{"name": "content-length", "value": str(len(body.encode()))}]
    )
    assert request.raw.count(b"content-length: 8\r\n") == 1
    assert b"Content-Length:" not in request.raw
    with pytest.raises(ToolInputError):
        parse_document(body=body, headers=[{"name": "Content-Length", "value": str(len(body))}])


def test_empty_get_has_framing_for_header_only_parameter_selection():
    request = parse_document(
        method="GET",
        url="https://target.invalid/",
        body="",
        headers=[{"name": "User-Agent", "value": "example"}, {"name": "Accept", "value": "*/*"}],
        testParameters=["User-Agent"],
    )
    assert request.raw.endswith(b"Content-Length: 0\r\n\r\n")
    assert b"Accept: */*\r\n" in request.raw


@pytest.mark.parametrize(
    "data",
    [
        b"\xffsecret-canary",
        b"{",
        b"[]",
        b"null",
        b'{"schemaVersion":1,"schemaVersion":1}',
        b'{"field":{"name":"secret-canary","name":"secret-canary"}}',
        b'{"secret-canary":NaN}',
        b'{"secret-canary":Infinity}',
        b"[" * 2000 + b"]" * 2000,
        b" " * (MAX_REQUEST_ARTIFACT_BYTES + 1),
    ],
)
def test_invalid_json_and_oversized_artifact_are_rejected_safely(data):
    with pytest.raises(ToolInputError) as caught:
        parse_http_request(data)
    assert "canary" not in str(caught.value)


def test_all_request_dimensions_are_bounded():
    assert parse_document(body="a" * MAX_BODY_BYTES).raw.endswith(b"a" * MAX_BODY_BYTES)
    oversized = [
        {"body": "a" * (MAX_BODY_BYTES + 1)},
        {"body": "я" * MAX_BODY_BYTES},
        {"url": "https://target.invalid/" + "a" * 8192},
        {"headers": [{"name": "X-" + str(i), "value": "x"} for i in range(MAX_HEADERS + 1)]},
        {"headers": [{"name": "X-Header", "value": "x" * (MAX_HEADER_VALUE_BYTES + 1)}]},
        {"headers": [{"name": "X-" + str(i), "value": "x" * 8192} for i in range(4)]},
        {"testParameters": ["id" + str(i) for i in range(MAX_TEST_PARAMETERS + 1)]},
        {"testParameters": ["a" * 129]},
    ]
    for changes in oversized:
        with pytest.raises(ToolInputError):
            parse_document(**changes)
