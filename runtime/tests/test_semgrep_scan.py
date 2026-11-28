from __future__ import annotations

import asyncio
import io
import json
import shutil
import stat
import sys
import time
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes.worker_state import WorkerState

from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.scan import semgrep
from contractor_runtime.toolsets.scan.nuclei_package import PackageError
from contractor_runtime.toolsets.scan.process import ProcessResult
from contractor_runtime.toolsets.scan.tools import ScanToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

RULE = b"""rules:
- id: python-eval
  languages: [python]
  severity: ERROR
  message: Dynamic evaluation of code
  pattern: eval(...)
"""
SOURCE = ArtifactRef(namespace="inputs", name="source", revision="rev-source")
PACK = ArtifactRef(namespace="inputs", name="rulepack", revision="rev-rules")


def archive(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as z:
        for name, data in files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                z.writestr(name, data)
    return output.getvalue()


def snapshot(tmp_path, data=RULE, media="application/yaml"):
    return semgrep.snapshot_rules(
        tmp_path / "absent",
        tmp_path / "rules.json",
        time.monotonic() + 60,
        data=data,
        media_type=media,
    )


def test_rulepack_exact_digest_and_qualified_ids(tmp_path):
    first = snapshot(tmp_path)
    assert first["ruleCount"] == 1
    assert first["ruleFiles"] == 1
    assert len(first["rulesDigest"]) == 71
    assert snapshot(tmp_path) == first
    changed = snapshot(tmp_path, RULE.replace(b"eval", b"exec"))
    assert changed["rulesDigest"] != first["rulesDigest"]
    packed = snapshot(
        tmp_path, archive([("a.yaml", RULE), ("b.yaml", RULE)]), semgrep.RULEPACK_MEDIA_TYPE
    )
    assert packed["ruleCount"] == 2 and len(packed["ruleIds"]) == 2


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"rules: []",
        b"rules: []\nrules: []",
        b"rules: &x [*x]",
        b"rules: !!python/object:os.system {}",
        b"config: https://semgrep.dev/p/default",
        RULE + b"  message: duplicate\n",
        RULE.replace(b"python-eval", b"../unsafe"),
        b"x" * (semgrep.MAX_FILE_BYTES + 1),
        RULE.replace(b"Dynamic evaluation of code", b'"\\0"'),
        RULE.replace(b"Dynamic evaluation of code", b'"   "'),
    ],
)
def test_invalid_yaml_never_falls_back(tmp_path, data):
    with pytest.raises(PackageError, match="semgrep_rulepack_invalid"):
        snapshot(tmp_path, data)
    assert not (tmp_path / "rules.json").exists()


@pytest.mark.parametrize(
    "entries",
    [
        [("../escape.yaml", RULE)],
        [("a.yaml", RULE), ("a.yaml", RULE)],
        [("a.txt", RULE)],
        [("a.yaml", RULE), ("a.yaml/b.yaml", RULE)],
    ],
)
def test_unsafe_custom_zip(tmp_path, entries):
    with pytest.raises(PackageError, match="semgrep_rulepack_invalid"):
        snapshot(tmp_path, archive(entries), semgrep.RULEPACK_MEDIA_TYPE)


def test_zip_symlink(tmp_path):
    info = zipfile.ZipInfo("link.yaml")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(PackageError, match="semgrep_rulepack_invalid"):
        snapshot(tmp_path, archive([(info, b"/etc/passwd")]), semgrep.RULEPACK_MEDIA_TYPE)


def test_installed_package_snapshot_and_bounds(tmp_path):
    installed = tmp_path / "installed"
    installed.mkdir()
    # Realistic rule count, beyond small fixture coverage.
    for i in range(300):
        (installed / f"{i}.yaml").write_bytes(RULE)
    (installed / "fixture.test.yaml").write_text("not: [valid")
    (installed / "template.yaml").write_text("placeholder")
    d = semgrep.snapshot_rules(installed, tmp_path / "rules.json", time.monotonic() + 60)
    assert d["ruleFiles"] == 300 and len(d["ruleIds"]) == 300
    original = (tmp_path / "rules.json").read_bytes()
    (installed / "0.yaml").write_bytes(RULE.replace(b"eval", b"exec"))
    assert (tmp_path / "rules.json").read_bytes() == original
    (installed / "linked.yaml").symlink_to(installed / "0.yaml")
    with pytest.raises(PackageError):
        semgrep.snapshot_rules(installed, tmp_path / "another.json", time.monotonic() + 60)


def test_native_observation_is_bounded_and_diagnostics_are_private(tmp_path):
    d = snapshot(tmp_path)
    rule = next(iter(d["ruleIds"]))
    prepared = {**d, "sourceFiles": {"app.py"}}
    row = {
        "check_id": rule,
        "path": "source/app.py",
        "start": {"line": 1},
        "end": {"line": 1},
        "extra": {
            "message": "eval use",
            "severity": "ERROR",
            "lines": "SECRET-SOURCE",
            "metavars": {"secret": "TOKEN"},
        },
    }

    def observe(rows, errors=None, **changes):
        doc = {
            "version": semgrep.ENGINE_VERSION,
            "results": rows,
            "errors": errors or [],
            "paths": {"scanned": ["source/app.py"]},
            **changes,
        }
        return semgrep.observation(ProcessResult(0, json.dumps(doc).encode(), b"SECRET"), prepared)

    complete = observe([row])
    assert complete["scanComplete"] and len(complete["results"]) == 1
    assert not any(word in json.dumps(complete) for word in ["SECRET", "TOKEN", "metavars"])
    assert observe([row] * 101)["resultsTruncated"]
    assert not observe([row] * 101)["scanComplete"]
    partial = observe([row], [{"message": "SECRET"}])
    assert partial["status"] == "failed" and len(partial["results"]) == 1
    assert not partial["scanComplete"]
    bad = observe([{**row, "path": "../../etc/passwd"}])
    assert bad["invalidResultRows"] == 1 and bad["status"] == "failed"
    assert not observe([], paths={"scanned": []})["scanComplete"]
    assert observe([], paths={"scanned": [42]})["errorCode"] == "invalid_scanner_output"
    assert observe([], paths={"scanned": ["source/foreign.py"]})["status"] == "failed"
    assert semgrep.observation(ProcessResult(0, b"bad"), prepared)["status"] == "failed"


class Store:
    def __init__(self):
        self.calls = []
        self.values = {
            SOURCE.name: (
                SOURCE,
                "application/zip",
                archive([("app.py", b"eval(input())\n"), ("safe.py", b'print("hello")\n')]),
            ),
            PACK.name: (PACK, "application/yaml", RULE),
        }

    async def read_artifact(self, ref, **_kwargs):
        self.calls.append(ref)
        exact, media, data = self.values[ref.name]
        return SimpleNamespace(artifact=exact, media_type=media, data=data)


async def tools_for(tmp_path, store, rules=None):
    factory = ScanToolsetFactory(lambda *_: store, semgrep_rules_directory=rules)
    tools = await factory.create_selected(
        selected=["scan_semgrep"],
        allocation_id="allocation",
        run_id="run",
        namespace="scanner",
        runtime_settings=RuntimeSettings(
            artifactApiUrl="https://artifacts.invalid", requestTimeoutSeconds=60
        ),
        workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
        resources=WorkerState().resources,
    )
    return tools["scan_semgrep"]


def fake_engine(tmp_path, monkeypatch):
    path = tmp_path / "bin"
    path.mkdir()
    engine = path / "semgrep"
    engine.write_text(f'''#!{sys.executable}
import json,sys
from pathlib import Path
if "--version" in sys.argv:
 print("{semgrep.ENGINE_VERSION}")
else:
 rules=json.loads(Path("rules.json").read_text())["rules"]
 print(json.dumps({{"version":"{semgrep.ENGINE_VERSION}","results":[{{"check_id":rules[0]["id"],
 "path":"source/app.py","start":{{"line":1}},"end":{{"line":1}},
 "extra":{{"message":"eval use","severity":"ERROR"}}}}],"errors":[],
 "paths":{{"scanned":["source/app.py","source/safe.py"]}}}}))
''')
    engine.chmod(0o700)
    monkeypatch.setenv("PATH", str(path))


def test_real_process_custom_source_and_cleanup(tmp_path, monkeypatch):
    fake_engine(tmp_path, monkeypatch)

    async def scenario():
        store = Store()
        tool = await tools_for(tmp_path, store)
        result = await tool.contractor_trusted_call(
            source_ref=SOURCE.model_dump(), rulepack_ref=PACK.model_dump()
        )
        await tool.close()
        assert result["status"] == "completed", result
        assert result["scanComplete"] and len(result["results"]) == 1
        assert result["sourceArtifact"] == SOURCE.model_dump(by_alias=True)
        assert result["rulepackArtifact"] == PACK.model_dump(by_alias=True)
        assert store.calls == [SOURCE, PACK]
        assert not list(tmp_path.glob("scan_semgrep-*"))

    asyncio.run(scenario())


def test_source_and_rulepack_failure_do_not_launch(tmp_path, monkeypatch):
    fake_engine(tmp_path, monkeypatch)

    async def scenario():
        store = Store()
        store.values["rulepack"] = (PACK, "application/yaml", b"rules: []")
        tool = await tools_for(tmp_path, store)
        result = await tool.contractor_trusted_call(
            source_ref=SOURCE.model_dump(), rulepack_ref=PACK.model_dump()
        )
        assert result["errorCode"] == "semgrep_rulepack_invalid"
        assert "engineVersion" not in result
        store.values["source"] = (SOURCE, "application/zip", archive([("../app.py", b"bad")]))
        result = await tool.contractor_trusted_call(source_ref=SOURCE.model_dump())
        assert result["errorCode"] == "semgrep_source_invalid"
        await tool.close()
        assert not list(tmp_path.glob("scan_semgrep-*"))

    asyncio.run(scenario())


@pytest.mark.skipif(not shutil.which("semgrep"), reason="real Semgrep executable not installed")
def test_real_semgrep_engine(tmp_path):
    async def scenario():
        tool = await tools_for(tmp_path, Store())
        result = await tool.contractor_trusted_call(
            source_ref=SOURCE.model_dump(), rulepack_ref=PACK.model_dump()
        )
        await tool.close()
        assert result["status"] == "completed", result
        assert result["engineVersion"] == semgrep.ENGINE_VERSION
        assert result["scannedFiles"] == 2
        assert len(result["results"]) == 1 and result["results"][0]["file"] == "app.py"
        assert result["source"] == "rulepack"
        assert not list(tmp_path.glob("scan_semgrep-*"))

    asyncio.run(scenario())


@pytest.mark.skipif(not shutil.which("semgrep"), reason="real Semgrep executable not installed")
@pytest.mark.parametrize("publish", [False, True])
def test_real_tool_worker_retains_semgrep_report(tmp_path, publish):
    from test_tool_worker import ArtifactStore, request, worker

    from contractor_runtime.contracts.tool_execution import ToolArgumentBinding

    async def scenario():
        from contractor_runtime.contracts import AllocationSpec

        store = ArtifactStore()
        spec = AllocationSpec.model_validate_json(
            (Path(__file__).parent / "testdata/semgrep-allocation.json").read_bytes()
        )
        spec.agent_template.execution.tool = "scan_semgrep"
        spec.agent_template.execution.arguments = {
            "source_ref": ToolArgumentBinding(source="artifact", name="source"),
            "rulepack_ref": ToolArgumentBinding(source="artifact", name="rulepack"),
        }
        from contractor_runtime.artifacts import ArtifactClient

        for name, (ref, media, data) in Store().values.items():
            store.values[(ref.namespace, name)] = (data, media, ref.revision)
        tool = await tools_for(tmp_path, ArtifactClient(spec.allocation_id, store))
        runtime = await worker(tmp_path, store, tool, spec=spec)
        selected = request(spec)
        selected.artifacts = {"source": SOURCE, "rulepack": PACK}
        if publish:
            selected.result_artifacts.update(
                {
                    name: ArtifactRef(namespace=spec.namespace, name=name)
                    for name in ("candidates", "evidence")
                }
            )
        result = await runtime.invoke(selected)
        assert result.failure is None, result
        report = json.loads(store.values[(spec.namespace, "report")][0])
        if publish:
            assert report["observation"]["findingSummary"]["candidateCount"] == 1
            batch = json.loads(store.values[(spec.namespace, "candidates")][0])
            assert batch["candidates"][0]["producer"] == "semgrep"
            assert batch["candidates"][0]["location"]["file"] == "app.py"
            proof = json.loads(store.values[(spec.namespace, "evidence")][0])
            assert proof["records"][0]["text"] == "eval(input())\n"
            assert proof["records"][0]["sourceArtifact"] == SOURCE.model_dump(by_alias=True)
        else:
            assert report["observation"]["results"][0]["file"] == "app.py"
            assert (spec.namespace, "candidates") not in store.values
        assert report["inputArtifacts"] == {
            "source": SOURCE.model_dump(by_alias=True),
            "rulepack": PACK.model_dump(by_alias=True),
        }
        replayed = await runtime.invoke(selected)
        assert replayed == result
        await tool.close()

    asyncio.run(scenario())


def test_selected_code_evidence_is_bounded_and_info_stays_out_of_findings(tmp_path):
    from contractor_runtime.toolsets.scan.findings import encode

    source = archive([(f"src/{i}.py", ("x" * 3500 + "\n").encode()) for i in range(100)])
    import hashlib

    observed = {
        "status": "completed",
        "scanComplete": True,
        "sourceArtifact": SOURCE.model_dump(by_alias=True),
        "sourceDigest": "sha256:" + hashlib.sha256(source).hexdigest(),
        "rulesDigest": "sha256:" + "a" * 64,
        "results": [
            {
                "file": f"src/{i}.py",
                "startLine": 1,
                "endLine": 1,
                "rule": "fixture",
                "severity": "ERROR",
                "message": "Possible issue",
            }
            for i in range(100)
        ],
    }
    batch = semgrep.finding_candidates(observed, source)
    assert 0 < len(batch["candidates"]) < 100
    assert not batch["complete"] and len(encode(batch)) <= 256 << 10
    assert len(batch["evidence"]["records"]) == len(batch["candidates"])
    observed["results"] = [{**observed["results"][0], "severity": "INFO"}]
    batch = semgrep.finding_candidates(observed, source)
    assert batch["candidates"] == [] and batch["skipped"] == {"informational": 1}
    with pytest.raises(ValueError, match="source evidence changed"):
        semgrep.finding_candidates(observed, source + b"changed")


def test_semgrep_cancellation_joins_materialization_before_cleanup(tmp_path, monkeypatch):
    import threading

    fake_engine(tmp_path, monkeypatch)
    original = semgrep.snapshot_rules
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        started.set()
        if not release.wait(30):
            raise RuntimeError("test did not release rules materialization")
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(semgrep, "snapshot_rules", blocked)

    async def scenario():
        tool = await tools_for(tmp_path, Store())
        task = asyncio.create_task(
            tool.contractor_trusted_call(
                source_ref=SOURCE.model_dump(), rulepack_ref=PACK.model_dump()
            )
        )
        try:
            await asyncio.wait_for(asyncio.to_thread(started.wait, 30), 35)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
        finally:
            release.set()
            await tool.close()
        assert not list(tmp_path.glob("scan_semgrep-*"))

    asyncio.run(scenario())


def test_zip_entry_bound_precedes_member_allocation(tmp_path, monkeypatch):
    import struct

    data = bytearray(archive([("rule.yaml", RULE)]))
    end = data.rfind(b"PK\x05\x06")
    struct.pack_into("<HH", data, end + 8, semgrep.MAX_RULES + 1, semgrep.MAX_RULES + 1)

    def unexpected(*_args, **_kwargs):
        pytest.fail("oversized directory reached ZipFile allocation")

    monkeypatch.setattr(semgrep.zipfile, "ZipFile", unexpected)
    with pytest.raises(PackageError, match="semgrep_rulepack_invalid"):
        snapshot(tmp_path, bytes(data), semgrep.RULEPACK_MEDIA_TYPE)
