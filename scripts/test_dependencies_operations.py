"""The documentation graph shares the ordinary operation execution boundary."""

import importlib.util
import unittest
from pathlib import Path


class OperationGraphTests(unittest.TestCase):
    def test_operation_stage_has_no_synthetic_planner_or_worker(self):
        path = Path(__file__).parent / "dependencies" / "build_graph.py"
        spec = importlib.util.spec_from_file_location("dependency_graph_operations", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        graph = module.Graph()
        source = "docs/spec/execution/server-operations.md"
        graph.node("Workflow", "prepare@1", source)
        module.workflow_edges(
            graph,
            "prepare@1",
            {
                "entryStage": "prepare",
                "stages": {
                    "prepare": {
                        "objective": "Prepare selected operation work",
                        "operation": {"ref": "openapi-prepare-work@1", "arguments": {}},
                    }
                },
            },
            source,
        )
        self.assertEqual(
            {node["kind"] for node in graph.nodes.values()},
            {"Workflow", "Stage", "ServerOperation"},
        )
        self.assertEqual(len([edge for edge in graph.edges if edge["label"] == "operation"]), 2)
