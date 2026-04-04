import json

import fsspec
import pytest
from google.genai import types

from contractor.tools.openapi import (
    PathItem,
    SecurityScheme,
    openapi_tools,
    validate_files,
    validate_model,
)


class MockToolContext:
    def __init__(self):
        self.artifacts = {}

    async def save_artifact(self, name, artifact, meta):
        self.artifacts[name] = artifact.text
        return len(self.artifacts)

    async def load_artifact(self, filename, version=None):
        if filename not in self.artifacts:
            return None
        return types.Part.from_text(text=self.artifacts[filename])


@pytest.fixture
def tool_context():
    return MockToolContext()


@pytest.fixture
def fs():
    fs = fsspec.filesystem("memory")
    fs.pipe("main.py", b"print('ok')")
    fs.pipe("service.go", b"package main")
    fs.pipe("pets.py", b"def pets(): pass")
    fs.pipe("responses.py", b"RESPONSES = {}")
    fs.pipe("README.md", b"# readme")
    return fs


def test_validate_files_banned_extensions(fs):
    err = validate_files(["README.md", "main.py"], fs)
    assert "PROHIBITED" in err


def test_validate_files_no_files(fs):
    err = validate_files([], fs)
    assert "No files provided" in err


def test_validate_files_ok(fs):
    err = validate_files(["main.py", "service.go"], fs)
    assert err is None


def test_validate_model_valid_security_scheme():
    ok, err = validate_model(
        SecurityScheme,
        {"type": "http", "scheme": "basic"},
    )
    assert ok is True
    assert err is None


def test_validate_model_invalid_security_scheme():
    ok, err = validate_model(SecurityScheme, {"scheme": "basic"})
    assert ok is False

    payload = json.loads(err)
    assert payload["component"] == "SecurityScheme"
    assert payload["input"] == {"scheme": "basic"}
    assert any(e["field"] == "type" for e in payload["errors"])


@pytest.mark.asyncio
async def test_upsert_and_get_path(tool_context, fs):
    tools = openapi_tools("openapi", fs)
    upsert_path = next(t for t in tools if t.__name__ == "upsert_path")
    get_path = next(t for t in tools if t.__name__ == "get_path")

    path_def = PathItem(
        get={
            "operationId": "listPets",
            "responses": {"200": {"description": "ok"}},
        }
    ).model_dump()

    res = await upsert_path(
        "/pets",
        path_def,
        path_files=["pets.py"],
        tool_context=tool_context,
    )
    assert "result" in res

    res = await get_path("/pets", tool_context)
    assert res["result"]["get"]["operationId"] == "listPets"


@pytest.mark.asyncio
async def test_remove_path(tool_context, fs):
    tools = openapi_tools("openapi", fs)
    upsert_path = next(t for t in tools if t.__name__ == "upsert_path")
    remove_path = next(t for t in tools if t.__name__ == "remove_path")

    await upsert_path(
        "/pets",
        PathItem(
            get={"operationId": "x", "responses": {"200": {"description": "ok"}}}
        ).model_dump(),
        ["pets.py"],
        tool_context,
    )

    res = await remove_path("/pets", tool_context)
    assert "result" in res


@pytest.mark.asyncio
async def test_upsert_and_list_components(tool_context, fs):
    tools = openapi_tools("openapi", fs)
    upsert_component = next(t for t in tools if t.__name__ == "upsert_component")
    list_components = next(t for t in tools if t.__name__ == "list_components")

    comp = {"description": "OK"}

    res = await upsert_component(
        "responses",
        "MyResponse",
        comp,
        component_files=["responses.py"],
        tool_context=tool_context,
    )
    assert "result" in res

    res = await list_components("responses", tool_context)
    assert "MyResponse" in res["result"]


@pytest.mark.asyncio
async def test_set_and_get_info(tool_context, fs):
    tools = openapi_tools("openapi", fs)
    set_info = next(t for t in tools if t.__name__ == "set_info")
    get_info = next(t for t in tools if t.__name__ == "get_info")

    await set_info(
        title="My API",
        framework="FastAPI",
        code_language="Python",
        tool_context=tool_context,
    )

    res = await get_info(tool_context)
    assert res["result"]["title"] == "My API"
    assert res["result"]["x-framework"] == "FastAPI"


@pytest.mark.asyncio
async def test_add_and_remove_server(tool_context, fs):
    tools = openapi_tools("openapi", fs)
    add_server = next(t for t in tools if t.__name__ == "add_server")
    remove_server = next(t for t in tools if t.__name__ == "remove_server")
    list_servers = next(t for t in tools if t.__name__ == "list_servers")

    await add_server("https://api.example.com", "prod", tool_context)

    res = await list_servers(tool_context)
    assert res["result"][0]["url"] == "https://api.example.com"

    res = await remove_server("https://api.example.com", tool_context)
    assert "result" in res