"""Offline fixtures and a bounded external Managed Eval consumer. No model selection."""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
MAX_DOCUMENT = 1 << 20
TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
VALIDATOR_ENV = "CONTRACTOR_CONTEXT_EVAL_OPENAPI_VALIDATOR"


class Rejected(ValueError):
    """Safe fixture/preflight failure; never includes input values."""


def require(condition, reason):
    if not condition:
        raise Rejected(reason)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def scorer_digest():
    """Pin the scorer and the exact local validator source/dependency closure."""
    return digest(
        b"\0".join(
            path.read_bytes()
            for path in (
                ROOT / "harness.py",
                ROOT / "openapivalidator/main.go",
                REPO / "go.mod",
                REPO / "go.sum",
            )
        )
    )


@contextlib.contextmanager
def openapi_validator():
    """Build once without dependency downloads; remove the owned binary on exit."""
    with tempfile.TemporaryDirectory(prefix="context-eval-validator-") as folder:
        binary = str(Path(folder) / "validate-openapi")
        result = subprocess.run(
            ["go", "build", "-o", binary, "./tests/eval/context_compaction/openapivalidator"],
            cwd=REPO,
            capture_output=True,
            timeout=60,
            env={**os.environ, "GOPROXY": "off", "GOSUMDB": "off", "GOTOOLCHAIN": "local"},
        )
        require(result.returncode == 0, "local OpenAPI validator build failed")
        previous = os.environ.get(VALIDATOR_ENV)
        os.environ[VALIDATOR_ENV] = binary
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(VALIDATOR_ENV, None)
            else:
                os.environ[VALIDATOR_ENV] = previous


def valid_openapi(document):
    binary = os.environ.get(VALIDATOR_ENV)
    if binary is None:
        raise RuntimeError("local OpenAPI validator is not prepared")
    raw = encoded(document)
    if len(raw) > MAX_DOCUMENT:
        return False
    result = subprocess.run([binary], input=raw, capture_output=True, timeout=5)
    if result.returncode not in (0, 1):
        raise RuntimeError("local OpenAPI validator failed")
    return result.returncode == 0


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def decode(raw):
    require(len(raw) <= MAX_DOCUMENT, "document exceeds 1 MiB")
    value = json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(Rejected("nonfinite JSON")),
    )

    def depth(v, n=0):
        require(n <= 32, "document depth exceeds 32")
        for child in v.values() if isinstance(v, dict) else v if isinstance(v, list) else ():
            depth(child, n + 1)

    depth(value)
    return value


def load(path):
    with Path(path).open("rb") as stream:
        return decode(stream.read(MAX_DOCUMENT + 1))


def projection(case):
    """Allowlisted recorded-provider payload; public Eval recipes wrap it as an artifact."""
    return copy.deepcopy(
        {key: case[key] for key in ("id", "task", "inputs", "requires", "outputs")}
    )


def fixtures():
    variants = load(ROOT / "variants.json")
    require(
        len(variants["variants"]) == 7 and len(variants["cases"]) == 3,
        "empty/incomplete pilot matrix",
    )
    require(
        digest((ROOT / "private/truth.json").read_bytes()) == variants["truthSha256"],
        "truth drift",
    )
    require(
        scorer_digest() == variants["scorerSha256"],
        "scorer drift",
    )
    require(
        set(variants["catalogFiles"])
        == {str(p.relative_to(ROOT)) for p in (ROOT / "catalog").rglob("*") if p.is_file()}
        and all(
            digest((ROOT / path).read_bytes()) == pin
            for path, pin in variants["catalogFiles"].items()
        ),
        "catalog drift",
    )
    cases = {}
    for name, expected in variants["cases"].items():
        raw = (ROOT / "visible" / (name + ".json")).read_bytes()
        require(digest(raw) == expected, "visible case drift")
        cases[name] = projection(decode(raw))
    return variants, cases, load(ROOT / "private/truth.json")


def score(case, answer, oracle, truth, *, continuation_required=True, future_required=False):
    """Judge artifacts against source/route facts and evaluator-owned observations."""
    try:
        checks = {
            "constraints": set(answer["constraints"]) == set(truth["constraints"]),
            "read_only": oracle["mutations"] == 0,
            "no_repeated_effects": len(oracle["effects"]) == len(set(oracle["effects"])),
        }
        if continuation_required:
            checks["continuation"] = oracle["usefulAfterCheckpoint"] > 0
        if future_required:
            checks["future_answer"] = answer.get("futureAnswer") == truth["futureAnswer"]
        expected = truth["expected"]
        if case["id"] == "source-review":
            # Independently derive concatenation sinks from the actual source.
            actual = []
            for path, source in case["inputs"]["source"].items():
                tree = ast.parse("\n".join(source))
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "execute"
                        and node.args
                        and isinstance(node.args[0], ast.BinOp)
                    ):
                        actual.append(
                            {
                                "path": path,
                                "line": node.lineno,
                                "kind": "sql_concatenation",
                            }
                        )
            checks["finding"] = answer["findings"] == actual == [expected["finding"]]
            checks["factual_support"] = set(answer["observed"]) == set(
                expected["observed"]
            ) and set(answer["uncertain"]) == set(expected["uncertain"])
        elif case["id"] == "api-dependency":
            document = answer["openapi"]
            checks["openapi_valid"] = valid_openapi(document)
            routes = case["inputs"]["routes"]
            operations = [document["paths"][r["path"]][r["method"]] for r in routes]
            checks["api_contract"] = (
                document["openapi"] == "3.1.0"
                and [o["operationId"] for o in operations] == expected["operations"]
                and all(o["security"] == [{"tenant": []}] for o in operations)
            )
            optional = [
                p["name"]
                for o in operations
                for p in o.get("parameters", [])
                if p["required"] is False
            ]
            checks["side_constraint"] = optional == expected["optional"]
            checks["document_structure"] = (
                bool(document["info"]["title"])
                and bool(document["info"]["version"])
                and document["components"]["securitySchemes"]["tenant"]["type"] == "apiKey"
                and all(
                    bool(o["responses"]["200"]["description"])
                    and any(
                        p["name"] == "tenant" and p["in"] == "path" and p["required"] is True
                        for p in o["parameters"]
                    )
                    for o in operations
                )
            )
            edges = [line.split(" -> ") for line in answer["likec4"].splitlines()]
            checks["dependency_model"] = sorted(edges) == sorted(expected["edges"])
        else:
            checks["negative_memory"] = all(
                set(answer[k]) == set(expected[k]) for k in ("negative", "observed", "uncertain")
            )
            checks["retrieval"] = set(expected["retrieved"]).issubset(oracle["retrieved"])
            checks["no_repeated_probe"] = not set(oracle["newProbes"]).intersection(
                expected["negative"]
            )
        return {"pass": all(checks.values()), "checks": checks}
    except (KeyError, TypeError, AttributeError, IndexError, SyntaxError, ValueError):
        return {"pass": False, "checks": {"well_formed_evidence": False}}


def preflight(manifest):
    variants, cases, _ = fixtures()
    require(
        set(manifest)
        == {
            "schemaVersion",
            "id",
            "repetitions",
            "seed",
            "thresholds",
            "runtime",
            "budgets",
            "variants",
            "experiments",
            "resources",
            "fixtureSha256",
            "observerSha256",
        },
        "incomplete live manifest",
    )
    require(
        manifest["schemaVersion"] == "context-compaction-live/v1" and ID.fullmatch(manifest["id"]),
        "unknown manifest",
    )
    require(manifest["thresholds"] == variants["thresholds"], "unfrozen thresholds")
    require(
        type(manifest["repetitions"]) is int and 5 <= manifest["repetitions"] <= 100,
        "invalid repeat count",
    )
    require(
        type(manifest["seed"]) is int and manifest["seed"] == variants["order"]["seed"],
        "unfrozen seed",
    )
    require(
        manifest["fixtureSha256"] == digest((ROOT / "variants.json").read_bytes())
        and DIGEST.fullmatch(manifest["observerSha256"]),
        "unfrozen fixture/observer",
    )
    runtime = manifest["runtime"]
    require(
        set(runtime) == {"adk", "lockSha256", "build"}
        and runtime["adk"] == "2.13.0"
        and DIGEST.fullmatch(runtime["build"]),
        "invalid Runtime provenance",
    )
    require(
        runtime["lockSha256"] == digest((REPO / "runtime/uv.lock").read_bytes()),
        "Runtime lock mismatch",
    )
    budgets = manifest["budgets"]
    require(
        set(budgets)
        == {
            "wallMs",
            "memberMs",
            "maxCalls",
            "maxTokens",
            "maxToolCalls",
            "memberCalls",
            "memberTokens",
            "memberToolCalls",
        }
        and all(type(v) is int and 0 < v <= 2**53 - 1 for v in budgets.values()),
        "finite aggregate/member budgets required",
    )
    require(budgets["memberMs"] <= budgets["wallMs"], "member deadline exceeds experiment")
    definitions = {v["id"]: v for v in variants["variants"] if v["provider"] == "contractor"}
    require(
        set(manifest["variants"]) == set(definitions),
        "all production variants required",
    )
    equal = None
    for name, pin in manifest["variants"].items():
        require(
            set(pin)
            == {
                "workflow",
                "closureSha256",
                "models",
                "capabilitiesSha256",
                "instructionsSha256",
                "skillsSha256",
                "reasoningAvailable",
                "runtime",
            },
            "incomplete closure pins",
        )
        require(
            pin["workflow"] == definitions[name]["workflow"] and pin["runtime"] == runtime,
            "variant Runtime/Workflow mismatch",
        )
        require(
            all(
                DIGEST.fullmatch(pin[k])
                for k in (
                    "closureSha256",
                    "capabilitiesSha256",
                    "instructionsSha256",
                    "skillsSha256",
                )
            ),
            "invalid closure digest",
        )
        require(
            type(pin["reasoningAvailable"]) is bool,
            "reasoning availability must be explicit",
        )
        require(
            pin["models"]
            and all(
                set(model)
                == {
                    "policyRef",
                    "alias",
                    "gatewaySha256",
                    "policySha256",
                    "weightsRevision",
                }
                and isinstance(model["policyRef"], str)
                and bool(model["policyRef"])
                and isinstance(model["alias"], str)
                and bool(model["alias"])
                and DIGEST.fullmatch(model["gatewaySha256"])
                and DIGEST.fullmatch(model["policySha256"])
                and (
                    model["weightsRevision"] is None
                    or (isinstance(model["weightsRevision"], str) and model["weightsRevision"])
                )
                for model in pin["models"]
            ),
            "configured model provenance required",
        )
        dimensions = [
            pin[k]
            for k in (
                "models",
                "capabilitiesSha256",
                "instructionsSha256",
                "skillsSha256",
                "runtime",
                "reasoningAvailable",
            )
        ]
        require(equal is None or dimensions == equal, "unequal required dimensions")
        equal = dimensions
    require(
        len(manifest["experiments"]) == len(variants["comparisons"])
        and len(manifest["resources"]) <= 2048,
        "incomplete/big experiment set",
    )
    seen, expected = set(), []
    for experiment in manifest["experiments"]:
        require(
            set(experiment)
            == {
                "id",
                "portableId",
                "planSha256",
                "variants",
                "setupSha256",
                "memberOrder",
                "casePins",
            }
            and ID.fullmatch(experiment["id"])
            and ID.fullmatch(experiment["portableId"]),
            "invalid existing experiment identity",
        )
        pair = tuple(experiment["variants"])
        require(
            list(pair) in variants["comparisons"] and pair not in seen,
            "invalid/duplicate comparison",
        )
        seen.add(pair)
        require(
            DIGEST.fullmatch(experiment["planSha256"])
            and DIGEST.fullmatch(experiment["setupSha256"]),
            "exact frozen plan/setup required",
        )
        require(
            set(experiment["casePins"]) == set(cases)
            and all(DIGEST.fullmatch(pin) for pin in experiment["casePins"].values()),
            "registered case pins required",
        )
        order = experiment["memberOrder"]
        require(
            isinstance(order, list)
            and 1 <= len(order) <= 10000
            and len(set(order)) == len(order)
            and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in order),
            "invalid frozen member order",
        )
        require(
            len(order) == len(cases) * 2 * manifest["repetitions"],
            "frozen membership count mismatch",
        )
        expected.extend(order)

    require(len(set(expected)) == len(expected), "member reused across comparisons")
    paths = set()
    for resource in manifest["resources"]:
        require(
            set(resource) == {"path", "sha256"}
            and re.fullmatch(
                r"/v1/(?:workflows|configurations)/[A-Za-z0-9_./@:-]+", resource["path"]
            )
            and ".." not in resource["path"]
            and DIGEST.fullmatch(resource["sha256"]),
            "invalid read-only resource pin",
        )
        require(resource["path"] not in paths, "duplicate resource pin")
        paths.add(resource["path"])
    require(
        all(
            "/v1/workflows/" + pin["workflow"].replace("@", "/versions/") in paths
            for pin in manifest["variants"].values()
        ),
        "Workflow preflight reads required",
    )
    return expected


class Client:
    def __init__(self, base, token, timeout=10):
        parsed = urllib.parse.urlsplit(base)
        require(
            parsed.scheme in {"http", "https"}
            and parsed.netloc
            and not parsed.username
            and not parsed.query
            and not parsed.fragment,
            "invalid API base",
        )
        self.base, self.token, self.timeout = base.rstrip("/"), token, timeout
        self.deadline, self.clock = None, time.monotonic
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, method, path, body=None, *, key=None, revision=None, raw=False):
        require(
            path.startswith("/v1/") and not path.startswith("//") and ".." not in path,
            "invalid API path",
        )
        headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
        }
        if key:
            headers["Idempotency-Key"] = key
        if revision is not None:
            headers["If-Match"] = '"' + str(revision) + '"'
        timeout = self.timeout
        if self.deadline is not None:
            require(self.clock() < self.deadline, "experiment transport deadline")
            timeout = min(timeout, self.deadline - self.clock())
        request = urllib.request.Request(
            self.base + path,
            None if body is None else encoded(body),
            headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                data = response.read(MAX_DOCUMENT + 1)
                require(len(data) <= MAX_DOCUMENT, "API response exceeds bound")
                return data if raw else decode(data)
        except urllib.error.HTTPError as error:
            raise Rejected("API rejected request: " + str(error.code)) from None

    def pages(self, path, *, limit=100):
        result, cursor, snapshot = [], None, None
        seen = set()
        for _ in range(10001):
            query = {"limit": limit}
            if cursor:
                query["cursor"] = cursor
            page = self.request("GET", path + "?" + urllib.parse.urlencode(query))
            current = page.get("viewSnapshot")
            require(
                snapshot is None or snapshot == current,
                "Eval view changed; resume the frozen experiment",
            )
            snapshot = current
            result.extend(page["items"])
            require(len(result) <= 10000, "paged inventory exceeds limit")
            if not page["page"]["hasMore"]:
                return result
            cursor = page["page"]["nextCursor"]
            require(
                isinstance(cursor, str) and bool(cursor) and cursor not in seen,
                "nonadvancing cursor",
            )
            seen.add(cursor)
        raise Rejected("page limit exhausted")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise Rejected("API redirects forbidden")


def safe_value(usage, name):
    measure = usage.get(name) if usage else None
    return measure["value"] if measure and measure["completeness"] == "complete" else None


def run_observer(observer, run_id, case, *, timeout):
    """A pinned evaluator-private observer; never a model-visible tool or artifact."""
    import resource
    import signal
    import subprocess

    def output_limit():
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_DOCUMENT + 1, MAX_DOCUMENT + 1))

    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", observer],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            preexec_fn=output_limit,
        )
        try:
            process.communicate(
                encoded({"runId": run_id, "case": projection(case)}), timeout=timeout
            )
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        require(process.returncode == 0, "private observer failed")
        output.seek(0)
        result = decode(output.read(MAX_DOCUMENT + 1))
    require(
        set(result) == {"mutations", "effects", "usefulAfterCheckpoint", "retrieved", "newProbes"},
        "invalid private observations",
    )
    require(
        type(result["mutations"]) is int
        and result["mutations"] >= 0
        and type(result["usefulAfterCheckpoint"]) is int
        and result["usefulAfterCheckpoint"] >= 0,
        "invalid observer counts",
    )
    require(
        all(
            isinstance(result[k], list)
            and len(result[k]) <= 1024
            and all(isinstance(v, str) and len(v) <= 128 for v in result[k])
            for k in ("effects", "retrieved", "newProbes")
        ),
        "invalid observer identities",
    )
    return result


def verify_remote(client, manifest):
    """Validate all read-only pins before admitting the frozen Eval members."""
    resources = {}
    for resource in manifest["resources"]:
        current = client.request("GET", resource["path"])
        require(digest(encoded(current)) == resource["sha256"], "published configuration drift")
        resources[resource["path"]] = current
    configurations = {
        value["ref"]["digest"]: value
        for path, value in resources.items()
        if path.startswith("/v1/configurations/")
    }
    for pin in manifest["variants"].values():
        for model in pin["models"]:
            policy = configurations.get(model["policySha256"])
            gateway = configurations.get(model["gatewaySha256"])
            require(
                policy is not None
                and policy["ref"]["kind"] == "model-policies"
                and policy["ref"]["name"] + "@" + policy["ref"]["version"] == model["policyRef"]
                and policy["body"]["model"] == model["alias"],
                "configured model policy mismatch",
            )
            require(
                gateway is not None and gateway["ref"]["kind"] == "llm-gateways",
                "configured Gateway pin missing",
            )
        workflow = resources["/v1/workflows/" + pin["workflow"].replace("@", "/versions/")]
        verify_workflow_models(workflow, pin["models"])
    _, cases, _ = fixtures()
    expected = []
    for experiment in manifest["experiments"]:
        path = "/v1/eval-experiments/" + experiment["id"]
        current = client.request("GET", path)
        require(
            current["controlMode"] == "external"
            and current["portableExperimentId"] == experiment["portableId"]
            and current["planSha256"] == experiment["planSha256"]
            and digest(encoded(current["setup"])) == experiment["setupSha256"],
            "frozen remote experiment changed",
        )
        checks = current["setup"]["checks"]
        require(
            len(checks) == 1
            and checks[0]["required"]
            and checks[0]["implementationSha256"] == scorer_digest(),
            "remote scorer closure mismatch",
        )
        require(
            {v["id"]: v["selector"] for v in current["setup"]["variants"]}
            == {name: manifest["variants"][name]["workflow"] for name in experiment["variants"]},
            "registered Workflow selection mismatch",
        )
        gates = current["setup"]["comparison"]["gates"]
        require(
            gates
            == {
                "minCandidateEndToEndPass": manifest["thresholds"]["minEndToEndPass"],
                "maxQualityDrop": manifest["thresholds"]["maxQualityDrop"],
                "maxTotalTokensRatio": manifest["thresholds"]["maxTokenRatio"],
            },
            "registered thresholds mismatch",
        )
        rows = client.pages(path + "/members")
        by_id = {r["member"]["memberId"]: r["member"] for r in rows}
        require(
            len(by_id) == len(rows) and set(by_id) == set(experiment["memberOrder"]),
            "remote expected matrix mismatch",
        )
        tuples = set()
        for identity in experiment["memberOrder"]:
            member = by_id[identity]
            key = (member["caseId"], member["sample"], member["variantId"])
            require(
                key not in tuples
                and member["caseId"] in cases
                and 1 <= member["sample"] <= manifest["repetitions"]
                and member["variantId"] in experiment["variants"],
                "invalid frozen member dimensions",
            )
            tuples.add(key)
            require(
                member["bindingSha256"]
                == manifest["variants"][member["variantId"]]["closureSha256"],
                "registered execution closure changed",
            )
            require(
                member["caseSha256"] == experiment["casePins"][member["caseId"]],
                "registered case changed",
            )
            expected.append(
                {
                    "memberId": identity,
                    "caseId": member["caseId"],
                    "sample": member["sample"],
                    "variantId": member["variantId"],
                    "experimentId": experiment["id"],
                    "planSha256": experiment["planSha256"],
                    "eligibility": member["eligibility"],
                }
            )
    return expected


def verify_workflow_models(workflow, models):
    """Match every effective consumer, including continuation/retry variants.

    All arms declare the same policy set; an arm can use a subset (for example,
    a terminal-only arm has no compactor). Every actual route must be in it.
    An omitted Worker Gateway depends on deployment defaults and cannot prove
    the frozen route, so this experiment requires an explicit Workflow route.
    """
    declared = {
        (model["policyRef"], model["policySha256"], model["alias"], model["gatewaySha256"])
        for model in models
    }
    for stage in workflow["stages"].values():
        configurations = [stage["executionConfig"]]
        configurations.extend(
            action["executionConfig"]["effective"]
            for action in stage["on"].values()
            if action["kind"] == "escalate"
        )
        for configuration in configurations:
            consumers = list(configuration["agents"].values())
            if configuration.get("planner") is not None:
                consumers.append(configuration["planner"])
            for consumer in consumers:
                if "modelPolicy" not in consumer:
                    require(not consumer, "unverifiable model-free consumer")
                    continue
                gateway = consumer.get("llmGateway")
                require(gateway is not None, "explicit effective Gateway required")
                for policy_field, alias_field in (
                    ("modelPolicy", "model"),
                    ("summarizerModelPolicy", "summarizerModel"),
                    ("compactorModelPolicy", "compactorModel"),
                ):
                    policy = consumer.get(policy_field)
                    if policy is None:
                        require(alias_field not in consumer, "model alias has no policy")
                        continue
                    require(
                        (
                            policy["policyId"] + "@" + policy["version"],
                            policy["digest"],
                            consumer.get(alias_field),
                            gateway["digest"],
                        )
                        in declared,
                        "effective Workflow model route mismatch",
                    )


def observe_member(client, member, deadline, clock, pause):
    path = "/v1/eval-experiments/" + member["experimentId"] + "/members"
    while clock() < deadline:
        row = next(r for r in client.pages(path) if r["member"]["memberId"] == member["memberId"])
        if row.get("execution") and row["execution"]["state"] in TERMINAL:
            # Join allocation cleanup through the ordinary Run read model.
            ref = row["execution"]["ref"]
            require(ref["kind"] == "run", "pilot requires Workflow members")
            run = client.request("GET", "/v1/runs/" + ref["id"])
            if run["deletable"]:
                return row, run
        pause(min(1, max(0, deadline - clock())))
    raise TimeoutError("member observation deadline")


def submit_member(client, member):
    path = (
        "/v1/eval-experiments/"
        + member["experimentId"]
        + "/members/"
        + member["memberId"]
        + "/submissions"
    )
    return keyed_post(
        client, path, {"planSha256": member["planSha256"]}, key="eval-" + member["memberId"]
    )


def keyed_post(client, path, body, *, key, revision=None):
    # A lost response replays the same durable request, including its CAS.
    for attempt in range(3):
        try:
            return client.request("POST", path, body, key=key, revision=revision)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise


def collect_member(client, member, row, run, case, truth, observer, observer_timeout):
    run_id = row["execution"]["ref"]["id"]
    passed, checks = False, {"execution": False}
    artifact, answer = None, None
    if row["execution"]["state"] == "succeeded":
        ref = run["outputs"].get("result")
        if ref:
            path = (
                "/v1/runs/"
                + urllib.parse.quote(run_id, safe="")
                + "/artifacts/"
                + "/".join(urllib.parse.quote(ref[k], safe="") for k in ("namespace", "name"))
                + "?"
                + urllib.parse.urlencode({"revision": ref["revision"]})
            )
            raw = client.request("GET", path, raw=True)
            artifact = {
                "scope": "run",
                "scopeId": run_id,
                **ref,
                "mediaType": "application/json",
                "sizeBytes": len(raw),
                "sha256": digest(raw),
            }
            try:
                answer = decode(raw)
            except (ValueError, Rejected):
                checks = {"well_formed_evidence": False}
            else:
                verdict = score(
                    case,
                    answer,
                    run_observer(observer, run_id, case, timeout=observer_timeout),
                    truth,
                    continuation_required=member["variantId"]
                    in {
                        "compact-off",
                        "compact-on",
                        "terminal-handoff",
                        "controlled-compact",
                        "masking",
                    },
                    future_required=member["variantId"] in {"controlled-compact", "masking"},
                )
                passed, checks = verdict["pass"], verdict["checks"]
    prefix = "/v1/eval-experiments/" + member["experimentId"] + "/members/" + member["memberId"]
    if row["usage"] is not None:
        current = client.request("GET", "/v1/eval-experiments/" + member["experimentId"])
        result = {
            "schemaVersion": "contractor.eval-result-input/v1",
            "planSha256": member["planSha256"],
            "memberId": member["memberId"],
            "source": current["setup"]["source"],
            "execution": row["execution"],
            "collection": {
                "status": "complete" if artifact else "unavailable",
                "gaps": [] if artifact else ["required_output_missing"],
            },
            "outputs": {"report": artifact} if artifact else {},
            "evidence": [{"id": "report", "artifact": artifact}] if artifact else [],
            "usage": row["usage"],
            "previousResultSha256": None,
        }
        receipt = keyed_post(
            client, prefix + "/results", result, key="cc-result-" + member["memberId"]
        )
        check_pin = current["setup"]["checks"][0]
        require(
            check_pin["implementationSha256"] == scorer_digest(),
            "remote scorer pin drift",
        )
        assessment = {
            "schemaVersion": "contractor.eval-assessment-input/v1",
            "resultSha256": receipt["recordSha256"],
            "source": {
                "kind": "external",
                "producerId": "context-compaction",
                "recordSha256": digest(encoded({"member": member["memberId"], "checks": checks})),
            },
            "checks": [
                {
                    "id": check_pin["id"],
                    "evaluator": check_pin["evaluator"],
                    "implementationSha256": check_pin["implementationSha256"],
                    "status": "pass" if passed else "fail",
                    "reason": "Independent artifact and environment checks",
                    "evidenceRefs": ["report"] if artifact else [],
                }
            ],
            "previousAssessmentSha256": None,
        }
        assessment_receipt = keyed_post(
            client,
            prefix + "/assessments",
            assessment,
            key="cc-assess-" + member["memberId"],
        )
        select_records(
            client,
            member,
            receipt["recordSha256"],
            assessment_receipt["recordSha256"],
            selected=row,
        )
    return {
        **usage_record(client, member, row, run),
        "passed": passed,
        "checks": checks,
    }


def select_records(client, member, result_sha256, assessment_sha256, *, selected=None):
    """Select exact receipts under CAS; ambiguous retries keep the same request."""
    if selected is not None and (
        selected.get("resultSha256") == result_sha256
        and selected.get("assessmentSha256") == assessment_sha256
    ):
        return
    path = "/v1/eval-experiments/" + member["experimentId"]
    revision = client.request("GET", path)["revision"]
    selection = {
        "planSha256": member["planSha256"],
        "selections": [
            {
                "memberId": member["memberId"],
                "resultSha256": result_sha256,
                "assessmentSha256": assessment_sha256,
            }
        ],
    }
    key = "cc-select-" + digest(encoded([selection, revision]))[7:]
    return keyed_post(client, path + "/selections", selection, key=key, revision=revision)


def usage_record(client, member, row, run):
    """Project cumulative usage; breakdowns are never added again."""
    run_id = row["execution"]["ref"]["id"]
    try:
        costs = client.request("GET", "/v1/runs/" + run_id + "/llm-cost")
    except (Rejected, urllib.error.URLError, TimeoutError, ConnectionError):
        costs = {"coverage": "unavailable", "activity": []}
    progress = []
    progress_available = True
    try:
        for attempt in run["attempts"]:
            items = client.pages(
                "/v1/runs/"
                + run_id
                + "/attempts/"
                + attempt["stageExecutionId"]
                + "/runtime-diagnostics",
                limit=10,
            )
            for item in items:
                if "compaction" in item or "modelPhases" in item:
                    progress.append(
                        {
                            "stageExecutionId": attempt["stageExecutionId"],
                            "logicalAgent": item["logicalAgent"],
                            **{
                                key: item[key]
                                for key in ("compaction", "modelPhases")
                                if key in item
                            },
                        }
                    )
    except (Rejected, urllib.error.URLError, TimeoutError, ConnectionError):
        progress_available = False
    # The Eval cumulative member usage includes retries and Planner handoffs.
    # Role costs and progress are breakdowns, never added a second time.
    return {
        **member,
        "state": row["execution"]["state"],
        "calls": safe_value(row["usage"], "modelCalls"),
        "tokens": safe_value(row["usage"], "totalTokens"),
        "toolCalls": safe_value(row["usage"], "toolCalls"),
        "elapsedMs": safe_value(row["usage"], "wallMs"),
        "usd": costs["amount"] if costs["coverage"] == "complete" else None,
        "roleCosts": costs["activity"],
        "progress": progress,
        "progressAvailable": progress_available,
    }


def cancel_and_join(client, member, deadline, clock, pause):
    path = "/v1/eval-experiments/" + member["experimentId"]
    current = client.request("GET", path)
    if current["state"] not in {"cancelled", "finished"}:
        client.request(
            "POST",
            path + "/commands",
            {"kind": "cancel", "planSha256": member["planSha256"]},
            key="cc-cancel-" + member["experimentId"],
            revision=current["revision"],
        )
    return observe_member(client, member, deadline, clock, pause)


def live(
    manifest,
    client,
    *,
    observer,
    clock=time.monotonic,
    pause=time.sleep,
    emit=lambda _record: None,
):
    until = clock() + manifest["budgets"]["wallMs"] / 1000
    client.deadline, client.clock = until, clock
    preflight(manifest)
    expected = verify_remote(client, manifest)
    _, cases, truth = fixtures()
    observed, active = {}, None
    budget = manifest["budgets"]
    spent = {"calls": 0, "tokens": 0, "toolCalls": 0}
    coverage_complete = True
    try:
        for member in expected:
            if member["eligibility"] != "eligible":
                observed[member["memberId"]] = {
                    **member,
                    "state": member["eligibility"],
                }
                emit(observed[member["memberId"]])
                observed[member["memberId"]] = member["eligibility"]
                continue
            # Reserve a whole bounded task before dispatch; no auxiliary role
            # gains unaccounted dispatch credit. Missing usage closes dispatch.
            if not coverage_complete:
                break
            if clock() + budget["memberMs"] / 1000 > until or any(
                spent[k] + budget[member_key] > budget[max_key]
                for k, member_key, max_key in [
                    ("calls", "memberCalls", "maxCalls"),
                    ("tokens", "memberTokens", "maxTokens"),
                    ("toolCalls", "memberToolCalls", "maxToolCalls"),
                ]
            ):
                break
            active = member
            member_until = min(until, clock() + budget["memberMs"] / 1000)
            client.deadline = member_until
            receipt = submit_member(client, member)
            if receipt["state"] == "rejected":
                record = {**member, "state": "blocked", "passed": False}
                emit(record)
                observed[member["memberId"]] = "blocked"
                coverage_complete = False
                active = None
                continue
            row, run = observe_member(
                client,
                member,
                member_until,
                clock,
                pause,
            )
            if clock() >= member_until:
                raise TimeoutError("member collection deadline")
            observed[member["memberId"]] = collect_member(
                client,
                member,
                row,
                run,
                cases[member["caseId"]],
                truth[member["caseId"]],
                observer,
                min(30, member_until - clock()),
            )
            record = observed[member["memberId"]]
            coverage_complete = all(record.get(key) is not None for key in spent)
            if coverage_complete:
                for key in spent:
                    spent[key] += record[key]
            emit(record)
            observed[member["memberId"]] = record["state"]
            active = None
            client.deadline = until
    except (Exception, KeyboardInterrupt):
        if active is not None:
            # Cleanup owns a separate finite deadline; failure remains visible.
            try:
                client.deadline = clock() + 60
                row, run = cancel_and_join(client, active, client.deadline, clock, pause)
                observed[active["memberId"]] = {
                    **usage_record(client, active, row, run),
                    "passed": False,
                    "cleanupJoined": True,
                }
            except (Exception, KeyboardInterrupt):
                observed[active["memberId"]] = {
                    **active,
                    "state": "timed_out",
                    "passed": False,
                    "cleanupJoined": False,
                }
    if active is not None:
        record = observed[active["memberId"]]
        emit(record)
        observed[active["memberId"]] = record["state"]
    # Use the existing server comparison, without reconstructing denominators.
    client.deadline = clock() + 10
    comparisons = []
    for experiment in manifest["experiments"]:
        current = comparison_snapshot(client, experiment["id"])
        while current.get("freshness") != "current" and clock() < client.deadline:
            pause(min(1, max(0, client.deadline - clock())))
            if clock() < client.deadline:
                current = comparison_snapshot(client, experiment["id"])
        comparisons.append(
            {
                key: current.get(key)
                for key in (
                    "experimentId",
                    "state",
                    "viewSnapshot",
                    "summary",
                    "freshness",
                )
            }
        )
    result = {
        "expected": len(expected),
        "members": [
            {
                "memberId": row["memberId"],
                "state": observed.get(row["memberId"], "not_submitted"),
            }
            for row in expected
        ],
        "comparisons": comparisons,
    }
    result["manifestSha256"] = digest(encoded(manifest))
    result["pairedReplay"] = "not_applicable: live Projects are independent end-to-end members"
    result["masking"] = "Test-only SDK request projection in a separate controlled comparison."
    return result


def comparison_snapshot(client, experiment_id):
    try:
        return client.request("GET", "/v1/eval-experiments/" + experiment_id)
    except (Rejected, urllib.error.URLError, TimeoutError, ConnectionError):
        return {"experimentId": experiment_id, "freshness": "pending", "summary": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--api", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load(args.manifest)
    preflight(manifest)  # Invalid requests perform no network reads/submissions.
    require(
        args.observer.is_file()
        and digest(args.observer.read_bytes()) == manifest["observerSha256"],
        "exact evaluator observer required",
    )
    require(
        not args.observer.resolve().is_relative_to(ROOT / "visible")
        and not args.observer.resolve().is_relative_to(ROOT / "catalog"),
        "observer is model-visible",
    )
    require(not args.output.exists(), "safe report already exists")
    token = os.environ.get("CONTRACTOR_CONTEXT_EVAL_TOKEN")
    require(bool(token), "API token required")
    records_path = args.output.with_suffix(".members.jsonl")
    require(not records_path.exists(), "member records already exist")
    with openapi_validator(), records_path.open("xb") as records:

        def emit(record):
            raw = encoded(record)
            require(len(raw) <= MAX_DOCUMENT, "member record exceeds document bound")
            records.write(raw + b"\n")
            records.flush()

        result = live(
            manifest,
            Client(args.api, token),
            observer=str(args.observer.resolve()),
            emit=emit,
        )
    with args.output.open("xb") as stream:
        stream.write(encoded(result))
    return 0 if measurement_complete(result) else 1


def measurement_complete(result):
    """Consume authoritative completion counts without interpreting quality."""
    expected, members = result["expected"], result["members"]
    if (
        type(expected) is not int
        or expected < 1
        or len(members) != expected
        or len({row["memberId"] for row in members}) != expected
        or any(row["state"] not in TERMINAL for row in members)
    ):
        return False
    accounted = 0
    for comparison in result["comparisons"]:
        summary = comparison["summary"]
        if comparison["freshness"] != "current" or summary is None:
            return False
        counts = summary.get("counts", {})
        if len(counts) != 2:
            return False
        for arm in counts.values():
            if any(
                type(arm.get(key)) is not int
                for key in (
                    "expected",
                    "submitted",
                    "terminal",
                    "scored",
                    "missing",
                    "conflicting",
                )
            ):
                return False
            count = arm["expected"]
            if (
                count < 1
                or any(arm[key] != count for key in ("submitted", "terminal", "scored"))
                or arm["missing"]
                or arm["conflicting"]
            ):
                return False
            accounted += count
    return accounted == expected


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Exception, KeyboardInterrupt):
        print(
            "Context-compaction evaluation failed; private diagnostics omitted.",
            file=sys.stderr,
        )
        sys.exit(2)
