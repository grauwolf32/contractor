"""Author V40 release artifacts from retained snapshots; never publish a catalog.

Requires PyYAML (available in the sibling playground evals environment).
The output is an overlay for the frozen catalog, NOT for today's default configs.
"""

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUT = ROOT / "evaluation-configs"


class Loader(yaml.SafeLoader):
    pass


# Workflow `on` is a string, not the YAML 1.1 boolean understood by SafeLoader.
Loader.yaml_implicit_resolvers = {
    key: [(tag, rx) for tag, rx in values if tag != "tag:yaml.org,2002:bool"]
    for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$"), list("tf")
)


def main():
    frozen = json.loads((ROOT / "catalog-baseline.json").read_text())
    variants = json.loads((ROOT / "variants.json").read_text())
    files = frozen["files"]
    workflows = {}
    for pair in variants["workflows"]:
        workflows[pair["baseline_selector"]] = yaml.load(
            files[pair["baseline"]], Loader
        )
        workflows[pair["candidate_selector"]] = yaml.load(
            (REPO / pair["candidate"]).read_text(), Loader
        )
    profile = next(
        yaml.load(raw, Loader)
        for name, raw in files.items()
        if name.startswith("configs/audit-profiles/")
        and name.endswith(".yaml")
        and yaml.load(raw, Loader)["metadata"]["name"] == "source-checklist"
    )
    pairs = {
        "d1": ("openapi-from-workspace@5", "openapi-from-workspace@6"),
        "o1": ("openapi-from-workspace@5", "openapi-from-workspace@6"),
        "l1": ("likec4-from-workspace@5", "likec4-from-workspace@6"),
        "t1": ("taint-trace-from-workspace@2", "taint-trace-from-workspace@3"),
        "t2": ("taint-trace-from-workspace@2", "taint-trace-from-workspace@3"),
        "a1": ("audit-source-check@1", "audit-source-check@2"),
    }
    manifest = {
        "status": "unpublished-preparation-not-live-ready",
        "baseline_catalog_sha256": hashlib.sha256(
            (ROOT / "catalog-baseline.json").read_bytes()
        ).hexdigest(),
        "variants_sha256": hashlib.sha256(
            (ROOT / "variants.json").read_bytes()
        ).hexdigest(),
        "cases": {},
        "files": {},
    }

    def write(kind, name, document):
        path = OUT / "configs" / kind / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = yaml.safe_dump(document, sort_keys=False).encode()
        path.write_bytes(raw)
        manifest["files"][path.relative_to(OUT).as_posix()] = hashlib.sha256(
            raw
        ).hexdigest()

    for case, selectors in pairs.items():
        manifest["cases"][case] = {}
        for arm, source in zip(("baseline", "candidate"), selectors, strict=True):
            workflow = copy.deepcopy(workflows[source])
            name = f"eval-v40-{case}-{arm}"
            workflow["metadata"] = {"name": name, "version": "1"}
            if case == "d1":
                # Same extraction in both arms: retain the discovery worker contract,
                # hydration, model policy and retries; export its result and stop.
                spec = workflow["spec"]
                stage = spec["stages"]["dependency_discovery"]
                stage["workflowOutputs"] = {
                    role: role for role in stage["result"]["artifacts"]
                }
                stage["on"]["succeeded"] = {"succeed": {}}
                spec["stages"] = {"dependency_discovery": stage}
                spec["inputs"].pop("existing_openapi", None)
                spec["outputs"] = {
                    role: {key: value for key, value in slot.items() if key != "from"}
                    for role, slot in stage["result"]["artifacts"].items()
                }
                spec["outputs"]["dependency_report"]["primary"] = True
            write("workflows", name, workflow)
            entry = {"source_workflow": source, "workflow": name + "@1"}
            if case == "a1":
                wrapper = copy.deepcopy(profile)
                wrapper["metadata"] = {"name": name, "version": "1"}
                wrapper["spec"]["workflows"]["check"]["ref"] = name + "@1"
                write("audit-profiles", name, wrapper)
                entry["audit_profile"] = name + "@1"
            manifest["cases"][case][arm] = entry
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
