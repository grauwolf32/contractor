"""Deterministic acceptance: no Gateway, network, publication or external side effects."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import harness as h


def deny_network(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("deterministic eval attempted network I/O")


sys.addaudithook(deny_network)


def oracle():
    return {
        "mutations": 0,
        "effects": ["publish-once"],
        "usefulAfterCheckpoint": 1,
        "retrieved": ["obs-P17"],
        "newProbes": [],
    }


def complete_manifest():
    variants, _, _ = h.fixtures()
    runtime = {
        "adk": "2.13.0",
        "lockSha256": h.digest((h.REPO / "runtime/uv.lock").read_bytes()),
        "build": "sha256:" + "b" * 64,
    }
    model = {
        "policyRef": "worker@1",
        "alias": "configured-model",
        "gatewaySha256": "sha256:" + "a" * 64,
        "policySha256": "sha256:" + "c" * 64,
        "weightsRevision": None,
    }
    pins = {
        v["id"]: {
            "workflow": v["workflow"],
            "closureSha256": "sha256:" + "2" * 64,
            "models": [copy.deepcopy(model)],
            "capabilitiesSha256": "sha256:" + "3" * 64,
            "instructionsSha256": "sha256:" + "4" * 64,
            "skillsSha256": h.digest(b"[]"),
            "reasoningAvailable": True,
            "runtime": copy.deepcopy(runtime),
        }
        for v in variants["variants"]
        if v["provider"] == "contractor"
    }
    resources = [
        {
            "path": "/v1/workflows/" + pin["workflow"].replace("@", "/versions/"),
            "sha256": h.digest(h.encoded(published_workflow(pin["workflow"].split("@")[0]))),
        }
        for pin in pins.values()
    ]
    resources.extend(
        {"path": path, "sha256": h.digest(h.encoded(body))}
        for path, body in published_configurations().items()
    )
    return {
        "schemaVersion": "context-compaction-live/v1",
        "id": "offline",
        "fixtureSha256": h.digest((h.ROOT / "variants.json").read_bytes()),
        "observerSha256": "sha256:" + "e" * 64,
        "repetitions": 5,
        "seed": 417,
        "thresholds": variants["thresholds"],
        "runtime": runtime,
        "budgets": {
            "wallMs": 1000000,
            "memberMs": 1000,
            "maxCalls": 4,
            "maxTokens": 400,
            "maxToolCalls": 4,
            "memberCalls": 4,
            "memberTokens": 400,
            "memberToolCalls": 4,
        },
        "variants": pins,
        "experiments": [
            {
                "id": "experiment-" + str(i),
                "portableId": "portable-" + str(i),
                "planSha256": "sha256:" + str(i + 1) * 64,
                "setupSha256": h.digest(h.encoded(setup(pair))),
                "casePins": {
                    name: h.digest(h.encoded(case)) for name, case in h.fixtures()[1].items()
                },
                "memberOrder": [
                    row["memberId"] for row in registered_members("portable-" + str(i), pair)
                ],
                "variants": pair,
            }
            for i, pair in enumerate(variants["comparisons"][:3])
        ],
        "resources": resources,
    }


def registered_members(portable_id, variants):
    # Test server fixture only. Production consumes registered IDs and order.
    import hashlib

    return [
        {
            "memberId": hashlib.sha256(
                f"{portable_id}:{case_id}:{sample}:{variant}".encode()
            ).hexdigest(),
            "caseId": case_id,
            "sample": sample,
            "variantId": variant,
        }
        for case_id in sorted(h.fixtures()[1])
        for sample in range(1, 6)
        for variant in variants
    ]


def setup(pair):
    return {
        "comparison": {
            "gates": {
                "minCandidateEndToEndPass": 1,
                "maxQualityDrop": 0,
                "maxTotalTokensRatio": 1.0,
            }
        },
        "source": {"system": "fixture", "id": "context", "revision": None, "sourceSha256": None},
        "checks": [
            {
                "id": "artifact-quality",
                "evaluator": "context@1",
                "implementationSha256": h.scorer_digest(),
                "required": True,
            }
        ],
        "variants": [
            {"id": name, "kind": "workflow", "selector": "context-eval-" + name + "@1"}
            for name in pair
        ],
    }


def published_configurations():
    return {
        "/v1/configurations/model-policies/worker/versions/1": {
            "ref": {
                "kind": "model-policies",
                "name": "worker",
                "version": "1",
                "digest": "sha256:" + "c" * 64,
            },
            "body": {"model": "configured-model"},
            "source": "operator",
        },
        "/v1/configurations/llm-gateways/local/versions/1": {
            "ref": {
                "kind": "llm-gateways",
                "name": "local",
                "version": "1",
                "digest": "sha256:" + "a" * 64,
            },
            "body": {"protocol": "openai-compatible@1", "url": "http://gateway.invalid/v1"},
            "source": "operator",
        },
    }


def published_workflow(name):
    consumer = {
        "modelPolicy": {"policyId": "worker", "version": "1", "digest": "sha256:" + "c" * 64},
        "model": "configured-model",
        "llmGateway": {"gatewayId": "local", "version": "1", "digest": "sha256:" + "a" * 64},
    }
    if "compact" in name:
        consumer["compactorModelPolicy"] = copy.deepcopy(consumer["modelPolicy"])
        consumer["compactorModel"] = consumer["model"]
    else:
        consumer["summarizerModelPolicy"] = copy.deepcopy(consumer["modelPolicy"])
        consumer["summarizerModel"] = consumer["model"]
    return {
        "ref": {"name": name, "version": "1"},
        "stages": {
            "work": {
                "executionConfig": {"agents": {"builder": consumer}},
                "on": {
                    "failed": {
                        "kind": "escalate",
                        "executionConfig": {
                            "effective": {
                                "agents": {"builder": copy.deepcopy(consumer)},
                                "planner": {
                                    k: v
                                    for k, v in consumer.items()
                                    if k in {"modelPolicy", "model", "llmGateway"}
                                },
                            }
                        },
                    }
                },
            }
        },
    }


class FakeAPI:
    def __init__(self, manifest):
        self.manifest = manifest
        self.calls, self.submissions = [], {}
        self.lost = False
        self.cancelled = False
        self.timeout = False
        self.rows = {}
        self.workflows = {
            r["path"]: published_workflow(r["path"].split("/")[3])
            for r in manifest["resources"]
            if r["path"].startswith("/v1/workflows/")
        }
        _, cases, _ = h.fixtures()
        for experiment in manifest["experiments"]:
            rows = registered_members(experiment["portableId"], experiment["variants"])
            self.rows[experiment["id"]] = [
                {
                    "member": {
                        **r,
                        "caseSha256": h.digest(h.encoded(cases[r["caseId"]])),
                        "eligibility": "eligible",
                        "bindingSha256": "sha256:" + "2" * 64,
                    },
                    "execution": None,
                    "usage": None,
                }
                for r in rows
            ]

    def request(self, method, path, body=None, **kwargs):
        self.calls.append((method, path, copy.deepcopy(body), kwargs.get("key")))
        if method == "GET" and path.startswith("/v1/workflows/"):
            return copy.deepcopy(self.workflows[path])
        if method == "GET" and path in published_configurations():
            return published_configurations()[path]
        if method == "GET" and path.endswith("/llm-cost"):
            return {"coverage": "unavailable", "activity": []}
        if method == "GET" and path.startswith("/v1/runs/"):
            return {"deletable": True, "outputs": {}, "attempts": []}
        if path.endswith("/commands"):
            self.cancelled = True
            for rows in self.rows.values():
                for row in rows:
                    if row["execution"]:
                        row["execution"]["state"] = "cancelled"
            return {}
        if path.endswith("/submissions"):
            member = path.split("/")[-2]
            self.submissions.setdefault(
                member, {"state": "accepted", "execution": {"kind": "run", "id": "run-1"}}
            )
            for rows in self.rows.values():
                for row in rows:
                    if row["member"]["memberId"] == member:
                        row["execution"] = {
                            "ref": {"kind": "run", "id": "run-1"},
                            "state": "running" if self.timeout else "succeeded",
                        }
            if self.lost:
                self.lost = False
                raise TimeoutError("response lost after commit")
            return self.submissions[member]
        if path.startswith("/v1/eval-experiments/"):
            experiment = next(
                e for e in self.manifest["experiments"] if e["id"] == path.split("/")[3]
            )
            return {
                "controlMode": "external",
                "portableExperimentId": experiment["portableId"],
                "planSha256": experiment["planSha256"],
                "setup": setup(experiment["variants"]),
                "experimentId": experiment["id"],
                "summary": {"conclusion": "inconclusive", "counts": {"fixture": {"expected": 30}}},
                "freshness": "current",
                "viewSnapshot": "view-1",
                "state": "running",
                "revision": 1,
            }
        raise AssertionError("unexpected fake API operation")

    def pages(self, path, *, limit=100):
        if path.endswith("/runtime-diagnostics"):
            return []
        self.calls.append(("GET", path, None, None))
        return copy.deepcopy(self.rows[path.split("/")[3]])


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = h.openapi_validator()
        cls.validator.__enter__()
        cls.addClassCleanup(cls.validator.__exit__, None, None, None)
        cls.variants, cls.cases, cls.truth = h.fixtures()
        cls.answers = h.load(h.ROOT / "private/recorded-results.json")

    def test_every_declared_variant_scores_five_samples_of_every_class(self):
        n = 0
        for variant in self.variants["variants"]:
            for case_id, case in self.cases.items():
                for sample in range(5):
                    with self.subTest(variant=variant["id"], case=case_id, sample=sample):
                        result = h.score(
                            case,
                            copy.deepcopy(self.answers[case_id]),
                            oracle(),
                            self.truth[case_id],
                            continuation_required=variant["strategy"] == "compact@1"
                            or variant["id"] == "terminal-handoff",
                        )
                        self.assertTrue(result["pass"], result["checks"])
                        n += 1
        self.assertEqual(n, 90)

    def test_scorers_reject_deliberately_plausible_corruption(self):
        mutations = [
            ("source-review", lambda a: a["findings"][0].update(line=2)),
            (
                "source-review",
                lambda a: a.update(observed=["lookup-exploitability"], uncertain=[]),
            ),
            ("source-review", lambda a: a.update(constraints=["read-only"])),
            (
                "api-dependency",
                lambda a: a["openapi"]["paths"]["/tenants/{tenant}/orders"]["post"].update(
                    operationId="createOrder"
                ),
            ),
            ("api-dependency", lambda a: a.update(likec4="orders -> inventory")),
            (
                "api-dependency",
                lambda a: a["openapi"]["paths"]["/tenants/{tenant}/orders"]["post"]["parameters"][
                    0
                ].update(required=True),
            ),
            (
                "negative-memory",
                lambda a: a.update(uncertain=[], observed=["all-other-inputs"]),
            ),
        ]
        for case_id, mutate in mutations:
            with self.subTest(case=case_id, mutation=mutate):
                answer = copy.deepcopy(self.answers[case_id])
                mutate(answer)
                self.assertFalse(
                    h.score(self.cases[case_id], answer, oracle(), self.truth[case_id])["pass"]
                )

    def test_openapi_scoring_rejects_invalid_structure_and_references(self):
        mutations = [
            lambda d: d["components"]["securitySchemes"]["tenant"].pop("name"),
            lambda d: d["components"]["securitySchemes"]["tenant"].pop("in"),
            lambda d: d["components"]["securitySchemes"]["tenant"].update(type="unknown"),
            lambda d: d["paths"]["/tenants/{tenant}/items"]["get"]["responses"].update(
                {"404": {"description": 123}}
            ),
            lambda d: d["components"].update(schemas={"Broken": {"type": "invalid"}}),
            lambda d: d["paths"]["/tenants/{tenant}/items"]["get"]["responses"]["200"].update(
                content={"application/json": {"schema": {"$ref": "#/components/schemas/Absent"}}}
            ),
            lambda d: d["paths"]["/tenants/{tenant}/items"]["get"]["responses"]["200"].update(
                content={"application/json": {"schema": {"$ref": "http://127.0.0.1:1/schema"}}}
            ),
        ]
        case_id = "api-dependency"
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                answer = copy.deepcopy(self.answers[case_id])
                mutate(answer["openapi"])
                verdict = h.score(self.cases[case_id], answer, oracle(), self.truth[case_id])
                self.assertFalse(verdict["pass"])
                self.assertFalse(verdict["checks"]["openapi_valid"])
        document = copy.deepcopy(self.answers[case_id]["openapi"])
        document["components"]["schemas"] = {"Item": {"type": "string"}}
        document["paths"]["/tenants/{tenant}/items"]["get"]["responses"]["200"]["content"] = {
            "application/json": {"schema": {"$ref": "#/components/schemas/Item"}}
        }
        self.assertTrue(h.valid_openapi(document))

    def test_scorer_pin_covers_validator_source_and_dependencies(self):
        original = h.scorer_digest()
        for changed in (h.ROOT / "openapivalidator/main.go", h.REPO / "go.mod", h.REPO / "go.sum"):
            read = Path.read_bytes
            with patch.object(
                Path,
                "read_bytes",
                lambda path, read=read, changed=changed: (
                    read(path) + (b"changed" if path == changed else b"")
                ),
            ):
                self.assertNotEqual(original, h.scorer_digest())

    def test_terminal_answer_does_not_claim_continuing_work(self):
        observed = {**oracle(), "usefulAfterCheckpoint": 0}
        case = self.cases["source-review"]
        answer = self.answers["source-review"]
        self.assertTrue(
            h.score(
                case, answer, observed, self.truth["source-review"], continuation_required=False
            )["pass"]
        )
        self.assertFalse(h.score(case, answer, observed, self.truth["source-review"])["pass"])

    def test_environment_receipts_and_useful_continuation_are_independent(self):
        for update in [
            {"mutations": 1},
            {"effects": ["publish", "publish"]},
            {"usefulAfterCheckpoint": 0},
            {"retrieved": []},
            {"newProbes": ["P17"]},
        ]:
            self.assertFalse(
                h.score(
                    self.cases["negative-memory"],
                    self.answers["negative-memory"],
                    {**oracle(), **update},
                    self.truth["negative-memory"],
                )["pass"]
            )

    def test_private_truth_and_future_questions_never_enter_projection(self):
        for name, case in self.cases.items():
            private = {
                **case,
                "evaluation": self.truth[name],
                "futureQuestion": "PRIVATE_FUTURE",
                "privateHook": "/private/oracle.py",
            }
            visible = h.projection(private)
            self.assertEqual(set(visible), {"id", "task", "inputs", "requires", "outputs"})
            self.assertNotIn(b"EVALUATOR_PRIVATE", h.encoded(visible))
            self.assertNotIn(b"PRIVATE_FUTURE", h.encoded(visible))
            self.assertNotIn(b"privateHook", h.encoded(visible))
            authored = h.encoded(visible) + b"".join(
                p.read_bytes() for p in (h.ROOT / "catalog").rglob("*") if p.is_file()
            )
            for question in self.truth[name]["futureQuestions"]:
                self.assertNotIn(question.encode(), authored)

    def test_controlled_replay_is_independent_and_masking_has_no_external_effect(self):
        events = [{"tool": "read", "receipt": "R1", "body": "large observation"}]
        self.assertEqual(h.mask_observations(events), [{"tool": "read", "receipt": "R1"}])
        folders = []

        def change(folder, state):
            folders.append(folder)
            state["receipts"].append("new")
            return len(state["receipts"])

        def inspect(folder, state):
            folders.append(folder)
            return len(state["receipts"])

        starts, results = h.controlled_pair(
            {"files": {"state.json": "{}"}, "receipts": ["R1"]}, [change, inspect]
        )
        self.assertEqual(starts[0], starts[1])
        self.assertEqual(results, [2, 1])
        self.assertNotEqual(folders[0], folders[1])
        self.assertTrue(all(not p.exists() for p in folders))
        with self.assertRaises(h.Rejected):
            h.controlled_pair({"url": "https://external.example"}, [change, inspect])

    def test_authoritative_usage_is_consumed_without_repricing_or_phase_addition(self):
        usage = {
            "modelCalls": {"value": 7, "completeness": "complete"},
            "totalTokens": {"value": 145, "completeness": "complete"},
        }
        self.assertEqual(h.safe_value(usage, "modelCalls"), 7)
        self.assertEqual(h.safe_value(usage, "totalTokens"), 145)
        usage["totalTokens"]["completeness"] = "partial"
        self.assertIsNone(h.safe_value(usage, "totalTokens"))
        self.assertIsNone(h.safe_value(None, "modelCalls"))

    def test_preflight_pins_reject_confounded_or_incomplete_experiments(self):
        manifest = complete_manifest()
        self.assertEqual(len(h.preflight(manifest)), 90)
        namespaced = copy.deepcopy(manifest)
        for pin in namespaced["variants"].values():
            pin["models"][0].update(alias="provider/model:revision", policyRef="worker@2026.11")
        self.assertEqual(len(h.preflight(namespaced)), 90)
        # The published-resource owner, rather than an invented alias grammar,
        # settles whether those identities match the selected configuration.
        mutations = [
            lambda m: m.pop("budgets"),
            lambda m: m.update(repetitions=4),
            lambda m: m["runtime"].update(adk="2.8.0"),
            lambda m: m["runtime"].update(lockSha256="sha256:" + "0" * 64),
            lambda m: m["variants"]["compact-on"]["runtime"].update(build="sha256:" + "c" * 64),
            lambda m: m["variants"]["compact-on"]["models"][0].update(alias="substituted-model"),
            lambda m: m["experiments"][0].update(variants=["a", "b", "c"]),
            lambda m: m["budgets"].update(maxTokens=None),
            lambda m: m.update(resources=[]),
        ]
        for mutate in mutations:
            value = copy.deepcopy(manifest)
            mutate(value)
            with self.assertRaises((h.Rejected, KeyError)):
                h.preflight(value)
        for count in (0, 4, 101, 10000):
            value = copy.deepcopy(manifest)
            value["repetitions"] = count
            with self.assertRaises(h.Rejected):
                h.preflight(value)
        with self.assertRaises(h.Rejected):
            h.decode(b"{" + b" " * (h.MAX_DOCUMENT + 1) + b"}")
        with self.assertRaises(h.Rejected):
            h.decode(b'{"id":1,"id":2}')

    def test_all_remote_configuration_and_matrix_checks_precede_dispatch(self):
        manifest = complete_manifest()
        api = FakeAPI(manifest)
        rows = h.verify_remote(api, manifest)
        self.assertEqual(len(rows), 90)
        self.assertTrue(all(call[0] == "GET" for call in api.calls))
        manifest["resources"][-1]["sha256"] = "sha256:" + "f" * 64
        with self.assertRaises(h.Rejected):
            h.verify_remote(api, manifest)
        self.assertFalse(api.submissions)

    def test_effective_workflow_model_routes_are_verified_before_dispatch(self):
        for target in (
            "main",
            "summarizer",
            "compactor",
            "planner",
            "failed",
            "interrupted",
            "summarized",
        ):
            for field in ("policy", "alias", "gateway", "missing_gateway"):
                with self.subTest(target=target, field=field):
                    manifest = complete_manifest()
                    api = FakeAPI(manifest)
                    resource = next(
                        r
                        for r in manifest["resources"]
                        if r["path"].startswith("/v1/workflows/")
                        and ("compact" in r["path"]) == (target == "compactor")
                    )
                    stage = api.workflows[resource["path"]]["stages"]["work"]
                    consumer = stage["executionConfig"]["agents"]["builder"]
                    policy, alias = "modelPolicy", "model"
                    if target in {"summarizer", "compactor"}:
                        policy, alias = target + "ModelPolicy", target + "Model"
                    elif target == "planner":
                        stage["executionConfig"]["planner"] = copy.deepcopy(consumer)
                        consumer = stage["executionConfig"]["planner"]
                    elif target in {"failed", "interrupted", "summarized"}:
                        stage["on"][target] = copy.deepcopy(stage["on"]["failed"])
                        consumer = stage["on"][target]["executionConfig"]["effective"]["planner"]
                    if field == "policy":
                        consumer[policy]["digest"] = "sha256:" + "d" * 64
                    elif field == "alias":
                        consumer[alias] = "another-model"
                    elif field == "gateway":
                        consumer["llmGateway"]["digest"] = "sha256:" + "d" * 64
                    else:
                        consumer.pop("llmGateway")
                    # Exact published bytes are pinned: this is a selection mismatch,
                    # not the existing resource-drift check.
                    resource["sha256"] = h.digest(h.encoded(api.workflows[resource["path"]]))
                    with self.assertRaises(h.Rejected):
                        h.live(manifest, api, observer="unused")
                    self.assertFalse(api.submissions)

    def test_selection_replays_exact_cas_and_preserves_conflicts(self):
        member = {"memberId": "a" * 64, "experimentId": "e", "planSha256": "sha256:" + "b" * 64}
        api = h.Client("http://127.0.0.1", "unused")
        for failure in (TimeoutError(), ConnectionError()):
            with patch.object(
                api, "request", side_effect=[{"revision": 7}, failure, {"revision": 8}]
            ) as call:
                h.select_records(api, member, "sha256:" + "c" * 64, "sha256:" + "d" * 64)
                self.assertEqual(call.call_args_list[1], call.call_args_list[2])
                self.assertEqual(call.call_args.kwargs["revision"], 7)
                self.assertLessEqual(len(call.call_args.kwargs["key"]), 128)
        with patch.object(api, "request", side_effect=[{"revision": 7}, h.Rejected("412")]) as call:
            with self.assertRaises(h.Rejected):
                h.select_records(api, member, "r", "a")
            self.assertEqual(call.call_count, 2)
        with patch.object(api, "request") as call:
            h.select_records(
                api, member, "r", "a", selected={"resultSha256": "r", "assessmentSha256": "a"}
            )
            call.assert_not_called()

    def test_comparisons_wait_for_publication_and_stop_at_deadline(self):
        for settle in (True, False):
            manifest = complete_manifest()
            api = FakeAPI(manifest)
            tick, reads = [0], []

            def pause(seconds, tick=tick):
                tick[0] += seconds

            def comparison(_client, experiment, reads=reads, settle=settle, tick=tick):
                reads.append(experiment)
                return {
                    "experimentId": experiment,
                    "freshness": "current" if settle and tick[0] >= 2 else "stale",
                    "summary": {"conclusion": "pass"},
                }

            with (
                patch.object(h, "comparison_snapshot", comparison),
                patch.object(
                    h,
                    "collect_member",
                    lambda _client, member, *_args: {
                        **member,
                        "state": "succeeded",
                        "calls": 4,
                        "tokens": 400,
                        "toolCalls": 4,
                    },
                ),
            ):
                result = h.live(
                    manifest, api, observer="unused", clock=lambda tick=tick: tick[0], pause=pause
                )
            self.assertEqual(tick[0], 2 if settle else 10)
            self.assertEqual(len(result["comparisons"]), 3)
            self.assertGreater(len(reads), 3)
            self.assertEqual(
                result["comparisons"][0]["freshness"], "current" if settle else "stale"
            )

    def test_lost_submission_response_replays_exact_existing_receipt(self):
        manifest = complete_manifest()
        api = FakeAPI(manifest)
        row = h.verify_remote(api, manifest)[0]
        api.lost = True
        h.submit_member(api, row)
        writes = [c for c in api.calls if c[0] == "POST"]
        self.assertEqual(len(writes), 2)
        self.assertEqual(writes[0], writes[1])
        self.assertEqual(len(api.submissions), 1)

    def test_aggregate_budget_stops_dispatch_and_retains_expected_matrix(self):
        manifest = complete_manifest()
        api = FakeAPI(manifest)

        def collected(_client, member, *_args):
            return {
                **member,
                "state": "succeeded",
                "passed": True,
                "calls": 4,
                "tokens": 400,
                "toolCalls": 4,
            }

        with patch.object(h, "collect_member", collected):
            result = h.live(manifest, api, observer="unused", clock=lambda: 0, pause=lambda _: None)
        self.assertEqual((result["expected"], len(api.submissions)), (90, 1))
        self.assertEqual(sum(row["state"] == "not_submitted" for row in result["members"]), 89)
        self.assertFalse(h.successful(result))
        self.assertEqual(result["comparisons"][0]["summary"]["counts"]["fixture"]["expected"], 30)

    def test_member_timeout_cancels_and_joins_without_resetting_fixture(self):
        manifest = complete_manifest()
        api = FakeAPI(manifest)
        api.timeout = True
        tick = [0]

        def pause(delta):
            tick[0] += delta

        records = []
        result = h.live(
            manifest,
            api,
            observer="unused",
            clock=lambda: tick[0],
            pause=pause,
            emit=records.append,
        )
        self.assertTrue(api.cancelled)
        self.assertEqual(len(api.submissions), 1)
        self.assertTrue(next(r for r in records if r["state"] == "cancelled")["cleanupJoined"])
        self.assertEqual(result["expected"], 90)

    def test_real_collection_posts_valid_public_records_and_keeps_authoritative_usage(self):
        import yaml
        from jsonschema import Draft202012Validator

        components = yaml.safe_load((h.REPO / "api/openapi/contractor-public-v1.yaml").read_text())[
            "components"
        ]
        manifest = complete_manifest()
        api = FakeAPI(manifest)
        member = h.verify_remote(api, manifest)[0]
        answer = h.encoded(self.answers[member["caseId"]])
        execution = {
            "ref": {"kind": "run", "id": "run-1"},
            "state": "succeeded",
            "startedAt": None,
            "finishedAt": None,
            "reason": None,
        }
        usage = {}
        for name, value in {
            "inputTokens": 120,
            "outputTokens": 25,
            "totalTokens": 145,
            "cachedInputTokens": 0,
            "modelCalls": 7,
            "toolCalls": 4,
            "toolFailures": 1,
            "wallMs": 200,
        }.items():
            usage[name] = {
                "value": value,
                "unit": "tokens"
                if "Tokens" in name
                else "milliseconds"
                if name == "wallMs"
                else "calls",
                "completeness": "complete",
                "sourceRefs": [],
                "scope": {
                    "memberId": member["memberId"],
                    "kind": "workflow",
                    "executions": [execution["ref"]],
                    "missing": [],
                    "interval": None,
                },
            }
        row = {"execution": execution, "usage": usage}
        run = {
            "outputs": {"result": {"namespace": "builder", "name": "result", "revision": "r1"}},
            "attempts": [{"stageExecutionId": "stage-1"}],
        }
        posted = []
        reads = []

        def request(method, path, body=None, **kwargs):
            if kwargs.get("raw"):
                self.assertIn("revision=r1", path)
                return answer
            if method == "POST":
                schema = (
                    "EvalSelectionInput"
                    if path.endswith("/selections")
                    else "EvalResultInput"
                    if path.endswith("/results")
                    else "EvalAssessmentInput"
                )
                Draft202012Validator(
                    {"$ref": "#/components/schemas/" + schema, "components": components}
                ).validate(body)
                posted.append(copy.deepcopy(body))
                if path.endswith("/selections"):
                    self.assertEqual(kwargs["revision"], 7)
                    return {"revision": 8, "viewSnapshot": None}
                return {"recordSha256": h.digest(h.encoded(body))}
            if path.endswith("/llm-cost"):
                return {
                    "coverage": "complete",
                    "amount": "0.70",
                    "activity": [{"phase": "summarization", "calls": "1"}],
                }
            return {"setup": setup(manifest["experiments"][0]["variants"]), "revision": 7}

        def pages(path, *, limit=100):
            reads.append((path, limit))
            return [
                {
                    "logicalAgent": "builder",
                    "modelPhases": {
                        "main": {"modelCalls": 4},
                        "compaction": {"modelCalls": 1},
                        "finalization": {"modelCalls": 1},
                        "summarization": {"modelCalls": 1},
                    },
                }
            ]

        with (
            patch.object(api, "request", request),
            patch.object(api, "pages", pages),
            patch.object(h, "run_observer", return_value=oracle()),
        ):
            record = h.collect_member(
                api,
                member,
                row,
                run,
                self.cases[member["caseId"]],
                self.truth[member["caseId"]],
                "private",
                1,
            )
        self.assertTrue(record["passed"])
        self.assertEqual((record["calls"], record["tokens"], record["usd"]), (7, 145, "0.70"))
        self.assertEqual(posted[0]["usage"], usage)
        self.assertEqual(len(posted), 3)
        self.assertEqual(
            posted[2]["selections"],
            [
                {
                    "memberId": member["memberId"],
                    "resultSha256": h.digest(h.encoded(posted[0])),
                    "assessmentSha256": h.digest(h.encoded(posted[1])),
                }
            ],
        )
        self.assertEqual(reads, [("/v1/runs/run-1/attempts/stage-1/runtime-diagnostics", 10)])
        self.assertNotIn(b"EVALUATOR_PRIVATE", h.encoded(record))
        self.assertNotIn(b"credential-canary", h.encoded(record))
        # Malformed output is a failed quality check, with a normal receipt.
        answer = b"{broken"
        with patch.object(api, "request", request), patch.object(api, "pages", pages):
            malformed = h.collect_member(
                api,
                member,
                row,
                run,
                self.cases[member["caseId"]],
                self.truth[member["caseId"]],
                "unused",
                1,
            )
        self.assertFalse(malformed["passed"])
        self.assertEqual(malformed["checks"], {"well_formed_evidence": False})
        self.assertEqual(posted[-2]["checks"][0]["status"], "fail")
        answer = h.encoded(self.answers[member["caseId"]])
        # Missing usage and prices stay unknown, including in the retained record.
        row["usage"]["totalTokens"]["completeness"] = "partial"
        original = request

        def missing(method, path, body=None, **kwargs):
            if path.endswith("/llm-cost"):
                return {"coverage": "partial", "amount": "0.10", "activity": []}
            return original(method, path, body, **kwargs)

        with (
            patch.object(api, "request", missing),
            patch.object(api, "pages", pages),
            patch.object(h, "run_observer", return_value=oracle()),
        ):
            record = h.collect_member(
                api,
                member,
                row,
                run,
                self.cases[member["caseId"]],
                self.truth[member["caseId"]],
                "private",
                1,
            )
        self.assertIsNone(record["tokens"])
        self.assertIsNone(record["usd"])

    def test_missing_usage_stops_dispatch_without_dropping_expected_failures(self):
        manifest = complete_manifest()
        api = FakeAPI(manifest)
        # These are registered, server-owned eligibility outcomes.
        api.rows["experiment-0"][0]["member"]["eligibility"] = "unsupported"
        api.rows["experiment-0"][1]["member"]["eligibility"] = "blocked"
        for state in ("failed", "cancelled", "timed_out"):
            records = []

            def collected(_client, member, *_args, state=state):
                return {
                    **member,
                    "state": state,
                    "passed": False,
                    "calls": 2,
                    "tokens": None,
                    "toolCalls": 1,
                }

            with patch.object(h, "collect_member", collected):
                result = h.live(
                    manifest,
                    api,
                    observer="private",
                    clock=lambda: 0,
                    pause=lambda _: None,
                    emit=records.append,
                )
            self.assertEqual(result["expected"], 90)
            states = [row["state"] for row in result["members"]]
            self.assertEqual(states[:3], ["unsupported", "blocked", state])
            self.assertEqual(states.count("not_submitted"), 87)
            self.assertEqual(len(records), 3)
            self.assertFalse(h.successful(result))

    def test_optional_diagnostic_failure_keeps_scoring_and_costs(self):
        # Exercised through the real collector above; auxiliary reads cannot
        # change the execution or its independently scored assessment.
        api = FakeAPI(complete_manifest())
        member = h.verify_remote(api, api.manifest)[0]
        row = {
            "execution": {"ref": {"kind": "run", "id": "run-1"}, "state": "failed"},
            "usage": None,
        }
        with (
            patch.object(api, "request", side_effect=h.Rejected("diagnostics unavailable")),
            patch.object(api, "pages", side_effect=h.Rejected("diagnostics unavailable")),
        ):
            result = h.collect_member(
                api,
                member,
                row,
                {"attempts": [{"stageExecutionId": "s"}]},
                self.cases[member["caseId"]],
                self.truth[member["caseId"]],
                "unused",
                1,
            )
        self.assertEqual(result["state"], "failed")
        self.assertIsNone(result["usd"])
        self.assertFalse(result["progressAvailable"])

    def test_private_observer_is_bounded_and_joined(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            observer = Path(folder) / "observer.py"
            observer.write_text(
                "import json,sys; data=json.load(sys.stdin); print("
                + repr(json.dumps(oracle()))
                + ")"
            )
            self.assertEqual(
                h.run_observer(str(observer), "run-1", self.cases["negative-memory"], timeout=2),
                oracle(),
            )
            observer.write_text("import sys; sys.stdout.write('x' * (2 << 20))")
            with self.assertRaises(h.Rejected):
                h.run_observer(str(observer), "run-1", self.cases["negative-memory"], timeout=2)
            observer.write_text("import time; time.sleep(60)")
            with self.assertRaises(subprocess.TimeoutExpired):
                h.run_observer(str(observer), "run-1", self.cases["negative-memory"], timeout=0.05)

    def test_only_fresh_server_comparison_can_report_success(self):
        result = {
            "members": [{"state": "succeeded"}],
            "comparisons": [{"freshness": "current", "summary": {"conclusion": "pass"}}],
        }
        self.assertTrue(h.successful(result))
        for key, value in [
            ("freshness", "stale"),
            ("summary", None),
            ("summary", {"conclusion": "inconclusive"}),
        ]:
            changed = copy.deepcopy(result)
            changed["comparisons"][0][key] = value
            self.assertFalse(h.successful(changed))

    def test_invalid_cli_manifests_fail_before_network_or_output(self):
        import os
        import subprocess
        import tempfile

        guard = (
            "import sys,runpy; "
            "sys.addaudithook(lambda event,args: sys.exit(99) "
            "if event in {'socket.connect','socket.getaddrinfo'} else None); "
            "sys.argv=['harness.py']+sys.argv[1:]; runpy.run_path("
            + repr(str(h.ROOT / "harness.py"))
            + ",run_name='__main__')"
        )
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            invalid = folder / "manifest.json"
            for raw in (b"{}", b"{broken", b" " * (h.MAX_DOCUMENT + 1)):
                invalid.write_bytes(raw)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-c",
                        guard,
                        "--manifest",
                        str(invalid),
                        "--observer",
                        str(folder / "private.py"),
                        "--api",
                        "http://127.0.0.1:9",
                        "--output",
                        str(folder / "safe.json"),
                    ],
                    capture_output=True,
                    timeout=10,
                    env={**os.environ, "CONTRACTOR_CONTEXT_EVAL_TOKEN": "credential-canary"},
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn(b"credential-canary", result.stdout + result.stderr)
                self.assertFalse((folder / "safe.json").exists())
                self.assertFalse((folder / "safe.members.jsonl").exists())

    def test_full_page_continuation_is_bounded_and_snapshot_consistent(self):
        api = h.Client("http://127.0.0.1", "credential-canary")
        responses = [
            {
                "items": list(range(100)),
                "viewSnapshot": "v1",
                "page": {"hasMore": True, "nextCursor": "next"},
            },
            {"items": [100], "viewSnapshot": "v1", "page": {"hasMore": False}},
        ]
        with patch.object(api, "request", side_effect=responses) as read:
            self.assertEqual(len(api.pages("/v1/eval-experiments/e/members")), 101)
            self.assertIn("cursor=next", read.call_args.args[1])
        responses[1]["viewSnapshot"] = "v2"
        with (
            patch.object(api, "request", side_effect=responses),
            self.assertRaises(h.Rejected),
        ):
            api.pages("/v1/eval-experiments/e/members")


if __name__ == "__main__":
    unittest.main(verbosity=2)
