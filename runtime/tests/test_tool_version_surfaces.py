"""Inspect the actual ADK declarations without reading shipped configuration."""

import ast
import inspect
import sys

from contractor_runtime.factories import built_in_factories
from contractor_runtime.toolsets.code_execution.tools import (
    CodeExecutionToolsetFactory,
    ExecCommandTool,
)
from contractor_runtime.toolsets.security_findings.trace import TraceFindingTool
from contractor_runtime.worker.function_tools import ContractorFunctionTool


def _description(cls):
    description = getattr(cls, "description", None) or cls.__dict__.get("__doc__")
    if isinstance(description, str):
        return description
    for node in ast.walk(ast.parse(inspect.getsource(inspect.getmodule(cls)))):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Assign)
                    and any(
                        isinstance(target, ast.Attribute) and target.attr == "__doc__"
                        for target in child.targets
                    )
                    and isinstance(child.value, ast.Constant)
                    and isinstance(child.value.value, str)
                ):
                    return child.value.value
    raise AssertionError(f"Missing description: {cls}")


def _check(schema):
    if isinstance(schema, dict):
        assert not {"revision", "expected_revision", "expected_source"} & set(
            schema.get("properties", {})
        )
        for value in schema.values():
            _check(value)
    elif isinstance(schema, list):
        for value in schema:
            _check(value)


def test_all_registered_model_tool_schemas_hide_technical_versions(tmp_path):
    factories = dict(built_in_factories(tmp_path).toolsets)
    factories["code-execution@1"] = CodeExecutionToolsetFactory()
    classes = {}
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("contractor_runtime.toolsets."):
            continue
        for cls in vars(module).values():
            if (
                inspect.isclass(cls)
                and cls.__module__ == module_name
                and any("__call__" in base.__dict__ for base in cls.__mro__)
            ):
                name = getattr(cls, "name", None)
                if isinstance(name, str):
                    classes.setdefault(name, set()).add(cls)
    seen = set()
    for ref, factory in factories.items():
        for name in factory.exported_tools:
            candidates = classes[name]
            cls = factory.tool_class if name == "finding" else next(iter(candidates))
            if ref == "security-findings-trace@1" and name == "finding":
                cls = TraceFindingTool
            assert name == "finding" or len(candidates) == 1
            owner = ExecCommandTool(None, None) if cls is ExecCommandTool else object.__new__(cls)
            if cls is not ExecCommandTool:
                owner.__name__, owner.__doc__ = name, _description(cls)
            schema = (
                ContractorFunctionTool(owner, logical_invocation_id=lambda _: "test")
                ._get_declaration()
                .parameters_json_schema
            )
            _check(schema)
            seen.add((ref, name))
    assert len(seen) == 117
