"""Tests for openapi_ref_resolver"""

import copy

import pytest

from contractor.tools.openapi import resolve_local_refs, resolve_refs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def petstore_schema():
    return {
        "openapi": "3.0.3",
        "components": {
            "schemas": {
                "Owner": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                    },
                },
                "Tag": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "label": {"type": "string"},
                    },
                },
                "Pet": {
                    "type": "object",
                    "required": ["id", "name"],
                    "properties": {
                        "id": {"type": "integer", "format": "int64"},
                        "name": {"type": "string"},
                        "owner": {"$ref": "#/components/schemas/Owner"},
                        "tags": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/Tag"},
                        },
                    },
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "integer"},
                        "message": {"type": "string"},
                    },
                },
            },
        },
        "paths": {
            "/pets": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Pet"},
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
    }


@pytest.fixture
def circular_schema():
    return {
        "components": {
            "schemas": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "child": {"$ref": "#/components/schemas/Node"},
                    },
                }
            }
        }
    }


@pytest.fixture
def mutual_circular_schema():
    return {
        "components": {
            "schemas": {
                "A": {
                    "type": "object",
                    "properties": {
                        "b": {"$ref": "#/components/schemas/B"},
                    },
                },
                "B": {
                    "type": "object",
                    "properties": {
                        "a": {"$ref": "#/components/schemas/A"},
                    },
                },
            }
        }
    }


# ---------------------------------------------------------------------------
# resolve_refs: basic
# ---------------------------------------------------------------------------


class TestResolveRefsBasic:
    def test_simple_ref(self, petstore_schema):
        component = {"$ref": "#/components/schemas/Owner"}
        result = resolve_refs(component, petstore_schema)
        assert result == {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
            },
        }

    def test_no_refs_passthrough(self, petstore_schema):
        component = {"type": "string", "minLength": 1}
        result = resolve_refs(component, petstore_schema)
        assert result == {"type": "string", "minLength": 1}

    def test_empty_dict(self, petstore_schema):
        assert resolve_refs({}, petstore_schema) == {}


# ---------------------------------------------------------------------------
# resolve_refs: nested
# ---------------------------------------------------------------------------


class TestResolveRefsNested:
    def test_nested_ref(self, petstore_schema):
        component = {"$ref": "#/components/schemas/Pet"}
        result = resolve_refs(component, petstore_schema)
        owner = result["properties"]["owner"]
        assert "$ref" not in owner
        assert owner["type"] == "object"
        assert "name" in owner["properties"]

    def test_ref_inside_array_items(self, petstore_schema):
        component = {
            "type": "array",
            "items": {"$ref": "#/components/schemas/Tag"},
        }
        result = resolve_refs(component, petstore_schema)
        assert result["items"]["type"] == "object"
        assert "label" in result["items"]["properties"]

    def test_deeply_nested_ref(self, petstore_schema):
        component = {"$ref": "#/components/schemas/Pet"}
        result = resolve_refs(component, petstore_schema)
        tag = result["properties"]["tags"]["items"]
        assert "$ref" not in tag
        assert tag["properties"]["label"] == {"type": "string"}


# ---------------------------------------------------------------------------
# resolve_refs: lists
# ---------------------------------------------------------------------------


class TestResolveRefsList:
    def test_refs_in_list(self, petstore_schema):
        component = {
            "oneOf": [
                {"$ref": "#/components/schemas/Pet"},
                {"$ref": "#/components/schemas/Error"},
            ]
        }
        result = resolve_refs(component, petstore_schema)
        assert len(result["oneOf"]) == 2
        assert all("$ref" not in item for item in result["oneOf"])
        assert result["oneOf"][0]["required"] == ["id", "name"]
        assert "code" in result["oneOf"][1]["properties"]


# ---------------------------------------------------------------------------
# resolve_refs: sibling merge
# ---------------------------------------------------------------------------


class TestResolveRefsSiblings:
    def test_sibling_keys_override(self, petstore_schema):
        component = {
            "$ref": "#/components/schemas/Owner",
            "description": "The pet owner",
        }
        result = resolve_refs(component, petstore_schema)
        assert result["description"] == "The pet owner"
        assert result["type"] == "object"


# ---------------------------------------------------------------------------
# resolve_refs: circular
# ---------------------------------------------------------------------------


class TestResolveRefsCircular:
    def test_self_circular(self, circular_schema):
        component = {"$ref": "#/components/schemas/Node"}
        result = resolve_refs(component, circular_schema)
        assert result["type"] == "object"
        child = result["properties"]["child"]
        assert child == {"$circular_ref": "#/components/schemas/Node"}

    def test_mutual_circular(self, mutual_circular_schema):
        component = {"$ref": "#/components/schemas/A"}
        result = resolve_refs(component, mutual_circular_schema)
        b = result["properties"]["b"]
        assert b["type"] == "object"
        a_again = b["properties"]["a"]
        assert a_again == {"$circular_ref": "#/components/schemas/A"}


# ---------------------------------------------------------------------------
# resolve_refs: immutability
# ---------------------------------------------------------------------------


class TestResolveRefsImmutability:
    def test_original_component_unchanged(self, petstore_schema):
        component = {"$ref": "#/components/schemas/Owner"}
        original = component.copy()
        resolve_refs(component, petstore_schema)
        assert component == original

    def test_original_schema_unchanged(self, petstore_schema):
        original = copy.deepcopy(petstore_schema)
        resolve_refs({"$ref": "#/components/schemas/Pet"}, petstore_schema)
        assert petstore_schema == original


# ---------------------------------------------------------------------------
# resolve_refs: errors
# ---------------------------------------------------------------------------


class TestResolveRefsErrors:
    def test_missing_ref_raises_key_error(self, petstore_schema):
        component = {"$ref": "#/components/schemas/NonExistent"}
        with pytest.raises(KeyError, match="NonExistent"):
            resolve_refs(component, petstore_schema)

    def test_non_local_ref_left_untouched(self, petstore_schema):
        component = {"$ref": "https://example.com/schemas/External"}
        result = resolve_refs(component, petstore_schema)
        assert result == {"$ref": "https://example.com/schemas/External"}

    def test_max_depth_exceeded(self, petstore_schema):
        component = {"$ref": "#/components/schemas/Pet"}
        with pytest.raises(RecursionError, match="Max resolution depth"):
            resolve_refs(component, petstore_schema, max_depth=1)


# ---------------------------------------------------------------------------
# resolve_refs: pointer escaping
# ---------------------------------------------------------------------------


class TestResolveRefsPointerEscaping:
    def test_tilde_and_slash_escaping(self):
        schema = {
            "components": {
                "schemas": {
                    "a/b": {
                        "type": "object",
                        "properties": {"x~y": {"type": "string"}},
                    }
                }
            }
        }
        component = {"$ref": "#/components/schemas/a~1b"}
        result = resolve_refs(component, schema)
        assert result["type"] == "object"
        assert "x~y" in result["properties"]


# ---------------------------------------------------------------------------
# resolve_refs: edge cases
# ---------------------------------------------------------------------------


class TestResolveRefsEdgeCases:
    def test_ref_resolves_to_scalar(self):
        schema = {"definitions": {"name": "hello"}}
        component = {"value": {"$ref": "#/definitions/name"}}
        result = resolve_refs(component, schema)
        assert result["value"] == "hello"

    def test_multiple_refs_same_target_independent(self, petstore_schema):
        component = {
            "a": {"$ref": "#/components/schemas/Owner"},
            "b": {"$ref": "#/components/schemas/Owner"},
        }
        result = resolve_refs(component, petstore_schema)
        assert result["a"] == result["b"]
        result["a"]["extra"] = True
        assert "extra" not in result["b"]


# ---------------------------------------------------------------------------
# resolve_local_refs (whole-schema convenience)
# ---------------------------------------------------------------------------


class TestResolveLocalRefs:
    def test_resolves_entire_schema(self, petstore_schema):
        result = resolve_local_refs(petstore_schema)
        # Check that the path-level ref got resolved
        items = result["paths"]["/pets"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["items"]
        assert "$ref" not in items
        assert items["type"] == "object"
        assert items["required"] == ["id", "name"]

    def test_components_also_resolved(self, petstore_schema):
        result = resolve_local_refs(petstore_schema)
        pet = result["components"]["schemas"]["Pet"]
        owner = pet["properties"]["owner"]
        assert "$ref" not in owner
        assert owner["type"] == "object"

    def test_no_refs_remain(self, petstore_schema):
        import json

        result = resolve_local_refs(petstore_schema)
        assert '"$ref"' not in json.dumps(result)

    def test_original_schema_unchanged(self, petstore_schema):
        original = copy.deepcopy(petstore_schema)
        resolve_local_refs(petstore_schema)
        assert petstore_schema == original

    def test_circular_whole_schema(self, circular_schema):
        result = resolve_local_refs(circular_schema)
        node = result["components"]["schemas"]["Node"]
        assert node == {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "child": {"$circular_ref": "#/components/schemas/Node"},
            },
        }
