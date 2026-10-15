#!/usr/bin/env python3
"""Build an offline documentation graph from the shipped configuration catalog."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs/spec/audit-workflow-dependencies.html"
CONFIGS = ROOT / "configs"
CATALOG = {
    "audit-profiles": "AuditProfile",
    "workflows": "Workflow",
    "agent-templates": "AgentTemplate",
    "model-policies": "ModelPolicy",
    "llm-gateways": "LLMGatewayConfig",
    "execution-configs": "ExecutionConfig",
}
BUILTINS = {
    "Planner": "internal/config/resolved/descriptors.go",
    "WorkerRuntime": "internal/config/resolved/descriptors.go",
    "Toolset": "internal/config/resolved/descriptors.go",
    "SandboxProfile": "internal/config/resolved/descriptors.go",
    "Inventory": "internal/config/auditprofile/audit_profile.go",
    "CompletionContract": "docs/spec/audits/worker-completion.md",
    "Credential": "configs/README.md",
}


class ConfigLoader(yaml.SafeLoader):
    """Keep Workflow's `on` key intact instead of using YAML 1.1 booleans."""

    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise ValueError(f"duplicate YAML key: {key!r}")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


ConfigLoader.yaml_implicit_resolvers = {
    char: [(tag, pattern) for tag, pattern in resolvers if tag != "tag:yaml.org,2002:bool"]
    for char, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false|True|False|TRUE|FALSE)$"), list("tTfF")
)


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.inputs = set()

    def read(self, path):
        self.inputs.add(path)
        return path.read_text(encoding="utf-8")

    def node(self, kind, label, source, description="", details=None, **extra):
        key = f"{kind}:{label}"
        if key not in self.nodes:
            if not (ROOT / source).is_file():
                raise ValueError(f"missing source: {source}")
            self.nodes[key] = {
                "id": key,
                "kind": kind,
                "label": label,
                "source": source,
                "description": description,
                "details": details or {},
                **extra,
            }
        return key

    def edge(
        self,
        origin,
        target,
        label,
        source,
        field="",
        relation="dependency",
        view="both",
        context="",
    ):
        if origin not in self.nodes or target not in self.nodes:
            raise ValueError(f"unresolved dependency: {origin} -> {target} ({source}: {field})")
        self.edges.append(
            {
                "from": origin,
                "to": target,
                "label": label,
                "source": source,
                "field": field,
                "relation": relation,
                "view": view,
                "context": context,
            }
        )

    def reference(self, origin, kind, selector, source, field, label=None, **extra):
        target = f"{kind}:{selector}"
        if kind in BUILTINS:
            description = "Встроенная реализация из реестра возможностей."
            if kind == "Credential":
                description = "Только ссылка на ID. Содержимое учётных данных в граф не входит."
            self.node(kind, selector, BUILTINS[kind], description)
        self.edge(origin, target, label or field.rsplit(".", 1)[-1], source, field, **extra)
        return target

    def instructions(self, origin, resource, source, field, view="both"):
        if not resource:
            return
        path = CONFIGS / resource["ref"]
        self.read(path)
        target = self.node(
            "Instructions",
            resource["ref"],
            str(path.relative_to(ROOT)),
            "Точный текст инструкций закрепляется при разрешении конфигурации.",
        )
        self.edge(origin, target, "instructions", source, field, view=view)

    def execution(self, origin, config, source, field, view="both"):
        """Visit reference leaves at every workflow, stage and escalation layer."""
        if not isinstance(config, dict):
            return
        for name, value in config.items():
            location = f"{field}.{name}"
            kinds = {
                "modelPolicy": "ModelPolicy",
                "llmGateway": "LLMGatewayConfig",
                "credential": "Credential",
                "ref": "ExecutionConfig",
            }
            if name in kinds and isinstance(value, str):
                self.reference(origin, kinds[name], value, source, location, view=view)
            elif isinstance(value, dict):
                self.execution(origin, value, source, location, view)


def load_catalog(graph):
    documents = []
    for directory, kind in CATALOG.items():
        for path in sorted((CONFIGS / directory).rglob("*")):
            if path.suffix not in {".yaml", ".yml"}:
                continue
            document = yaml.load(graph.read(path), Loader=ConfigLoader)
            if document["kind"] != kind:
                raise ValueError(f"unexpected configuration kind: {path}")
            label = f"{document['metadata']['name']}@{document['metadata']['version']}"
            if f"{kind}:{label}" in graph.nodes:
                raise ValueError(f"duplicate configuration identity: {kind}:{label}")
            spec = document["spec"]
            source = str(path.relative_to(ROOT))
            description = spec.get("description") or spec.get("presentation", {}).get(
                "description", ""
            )
            graph.node(kind, label, source, description, spec)
            documents.append((kind, label, spec, source))
    for path in sorted((CONFIGS / "audit-standards").glob("*/standard.json")):
        package = json.loads(graph.read(path))
        standard = package["standard"]
        graph.node(
            "Standard",
            f"{standard['scheme']}@{standard['version']}",
            str(path.relative_to(ROOT)),
            standard["description"],
            {
                "title": standard["title"],
                "entries": len(package["entries"]),
                "mappings": len(package["mappings"]),
                "evidenceContracts": len(package["evidenceContracts"]),
            },
        )
    for path in sorted((CONFIGS / "skills").glob("*/SKILL.md")):
        graph.read(path)
        graph.node(
            "Skill",
            f"skills/{path.parent.name}",
            str(path.relative_to(ROOT)),
            "Логическая ссылка на Skill. При старте фиксируется точная ревизия артефакта.",
        )
    return documents


def template_edges(graph, origin, spec, source):
    for field, kind in [
        ("runtime", "WorkerRuntime"),
        ("modelPolicy", "ModelPolicy"),
        ("sandboxProfile", "SandboxProfile"),
    ]:
        if field in spec:
            graph.reference(origin, kind, spec[field], source, f"spec.{field}")
    graph.instructions(origin, spec.get("instructions"), source, "spec.instructions")
    for index, selection in enumerate(spec.get("toolsets", [])):
        graph.reference(
            origin,
            "Toolset",
            selection["ref"],
            source,
            f"spec.toolsets[{index}]",
            f"tools · {len(selection['tools'])}",
        )
    for index, skill in enumerate(spec.get("skills", [])):
        graph.reference(
            origin,
            "Skill",
            f"{skill['namespace']}/{skill['name']}",
            source,
            f"spec.skills[{index}]",
            "skill",
        )
    summarizer = spec.get("summarizer")
    if summarizer:
        graph.reference(
            origin,
            "ModelPolicy",
            summarizer["modelPolicy"],
            source,
            "spec.summarizer.modelPolicy",
            "summarizer",
        )
        graph.instructions(
            origin, summarizer.get("instructions"), source, "spec.summarizer.instructions"
        )


def stage_actions(graph, workflow, stage, action, source, field):
    if not isinstance(action, dict):
        return
    for name, value in action.items():
        location = f"{field}.{name}"
        if name == "next":
            graph.edge(
                stage,
                f"Stage:{workflow}/{value}",
                "next",
                source,
                location,
                relation="transition",
                view="stages",
            )
        elif name == "planner":
            for origin, view in [(stage, "stages"), (f"Workflow:{workflow}", "catalog")]:
                graph.reference(
                    origin,
                    "Planner",
                    value,
                    source,
                    location,
                    "escalate",
                    relation="escalation",
                    view=view,
                )
        elif name == "executionConfig":
            graph.execution(stage, value, source, location, "stages")
            graph.execution(f"Workflow:{workflow}", value, source, location, "catalog")
        elif isinstance(value, dict):
            stage_actions(graph, workflow, stage, value, source, location)


def workflow_edges(graph, label, spec, source):
    origin = f"Workflow:{label}"
    graph.execution(origin, spec.get("executionConfig"), source, "spec.executionConfig")
    for index, (name, stage) in enumerate(spec["stages"].items()):
        stage_id = graph.node(
            "Stage",
            f"{label}/{name}",
            source,
            stage["objective"],
            stage,
            workflow=origin,
            ordinal=index,
            entry=name == spec["entryStage"],
        )
        graph.edge(
            origin,
            stage_id,
            "entryStage" if name == spec["entryStage"] else "stage",
            source,
            f"spec.stages.{name}",
            relation="structure",
            view="stages",
        )
    for name, stage in spec["stages"].items():
        stage_id = f"Stage:{label}/{name}"
        field = f"spec.stages.{name}"
        for consumer, view in [(origin, "catalog"), (stage_id, "stages")]:
            graph.reference(
                consumer,
                "Planner",
                stage["planner"],
                source,
                f"{field}.planner",
                "planner",
                view=view,
            )
            graph.instructions(
                consumer, stage.get("instructions"), source, f"{field}.instructions", view
            )
            for agent, binding in stage["agents"].items():
                graph.reference(
                    consumer,
                    "AgentTemplate",
                    binding["template"],
                    source,
                    f"{field}.agents.{agent}.template",
                    f"{name} · {agent}",
                    view=view,
                )
        stage_actions(graph, label, stage_id, stage.get("on", {}), source, f"{field}.on")


def profile_edges(graph, label, spec, source):
    origin = f"AuditProfile:{label}"
    for standard in spec.get("standards", []):
        graph.reference(
            origin,
            "Standard",
            f"{standard['scheme']}@{standard['version']}",
            source,
            "spec.standards",
            "standard",
            view="catalog",
        )
    graph.reference(
        origin,
        "Inventory",
        spec["inventory"]["implementation"],
        source,
        "spec.inventory.implementation",
        "inventory",
        view="catalog",
    )
    for role, binding in spec["workflows"].items():
        field = f"spec.workflows.{role}"
        workflow = graph.reference(
            origin,
            "Workflow",
            binding["ref"],
            source,
            f"{field}.ref",
            f"{binding['kind']} · {role}",
            view="catalog",
        )
        if completion := binding.get("workerCompletion"):
            graph.reference(
                origin,
                "CompletionContract",
                completion["kind"],
                source,
                f"{field}.workerCompletion",
                f"completion · {role}",
                view="catalog",
            )
        for name, mapping in binding.get("inputs", {}).items():
            if mapping["source"] not in {"prepare-output", "retained-output"}:
                continue
            producer = spec["workflows"][mapping["role"]]
            output = producer["outputs"][mapping["name"]]
            graph.edge(
                workflow,
                f"Workflow:{producer['ref']}",
                f"{output} → {name}",
                source,
                f"{field}.inputs.{name}",
                relation="data",
                view="catalog",
                context=origin,
            )


def build():
    graph = Graph()
    documents = load_catalog(graph)
    for kind, label, spec, source in documents:
        if kind == "AgentTemplate":
            template_edges(graph, f"{kind}:{label}", spec, source)
        elif kind == "Workflow":
            workflow_edges(graph, label, spec, source)
        elif kind == "AuditProfile":
            profile_edges(graph, label, spec, source)
        elif kind == "ExecutionConfig":
            graph.execution(f"{kind}:{label}", spec, source, "spec")
    architecture = json.loads(graph.read(HERE / "architecture.json"))
    for node in architecture["nodes"]:
        for source in node["sources"]:
            graph.read(ROOT / source)
    arch_ids = {node["id"] for node in architecture["nodes"]}
    for edge in architecture["edges"]:
        if edge["from"] not in arch_ids or edge["to"] not in arch_ids:
            raise ValueError(f"unresolved architecture edge: {edge}")
        graph.read(ROOT / edge["source"])
    # Include built-in implementation sources in the fingerprint too.
    for node in graph.nodes.values():
        graph.read(ROOT / node["source"])
    fingerprint = hashlib.sha256()
    for path in sorted(graph.inputs):
        fingerprint.update(str(path.relative_to(ROOT)).encode())
        fingerprint.update(b"\0" + path.read_bytes() + b"\0")
    data = {
        "fingerprint": fingerprint.hexdigest()[:12],
        "counts": dict(sorted(Counter(kind for kind, _, _, _ in documents).items())),
        "architecture": architecture,
        "catalog": {"nodes": list(graph.nodes.values()), "edges": graph.edges},
    }
    # No catalog text may terminate or inject markup into the JSON script element.
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return (HERE / "template.html").read_text(encoding="utf-8").replace('"__GRAPH_DATA__"', encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed HTML is stale")
    arguments = parser.parse_args()
    page = build()
    if arguments.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != page:
            raise SystemExit("dependency graph is stale; run make docs-dependencies")
        print("Audit/Workflow dependency graph is current")
    else:
        OUTPUT.write_text(page, encoding="utf-8")
        print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
