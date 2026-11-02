"""Actual hybrid client driven by TestPostgresHybridBootstrapMutationIdentity."""

import copy
import sys
import unittest
from http.client import RemoteDisconnected
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "deploy/hybrid"))
import bootstrap


class RecordingAPI(bootstrap.API):
    def __init__(self, url, token):
        super().__init__(url, token)
        self.writes = []
        self.lose_next = False

    def request(self, method, path, body=None, headers=None, missing=False):
        if method not in {"GET", "HEAD", "OPTIONS"}:
            self.writes.append(copy.deepcopy((method, path, body, headers)))
            if self.lose_next:
                self.lose_next = False
                headers = {**(headers or {}), "X-Test-Lose-Response": "yes"}
        return super().request(method, path, body, headers, missing)


class BootstrapPostgres(unittest.TestCase):
    def setUp(self):
        self.state = Path(state)
        self.api = RecordingAPI(url, self.state / "api-token")
        self.args = SimpleNamespace(
            credential_id="development-worker",
            config_version="1",
            model_policies=["worker@1"],
            budget=20,
            rebind_label=True,
            pki_root=self.state,
        )
        self.binding_path = "/v1/operations/runtime-labels/cloud-llm"
        self.principal_path = "/v1/operations/runtime-agent-principals/" + principal_id

    def configure(self, version):
        self.args.config_version = version
        bootstrap.configure(self.api, self.state, self.args)

    def current(self, path):
        return self.api.request("GET", path)[0]

    def check_replay(self, request, path, revision):
        first, _ = self.api.request(*request)
        second, _ = self.api.request(*request)
        self.assertEqual(first, second)
        self.assertEqual(self.current(path)["revision"], revision)

    def binding(self):
        for version, revision in (("1", "1"), ("2", "2"), ("1", "3")):
            self.configure(version)
            self.check_replay(self.api.writes[-1], self.binding_path, revision)
        count = len(self.api.writes)
        self.configure("1")
        self.assertEqual(len(self.api.writes), count, "unchanged configure must only read")
        # The binding commits but the HTTP response is lost. Reconcile the exact
        # request through its receipt, then rerun configure using resource reads.
        self.api.lose_next = True
        with self.assertRaises(RemoteDisconnected):
            self.configure("2")
        committed = self.api.writes[-1]
        self.check_replay(committed, self.binding_path, "4")
        count = len(self.api.writes)
        self.configure("2")
        self.assertEqual(len(self.api.writes), count)
        body = {"config": self.current(self.binding_path)["config"]}
        self.reject("PUT", self.binding_path, body, {"If-Match": '"2"'}, 412)
        self.reject(
            "PUT",
            self.binding_path,
            {"config": {"name": "absent", "version": "1", "digest": "sha256:" + "a" * 64}},
            {"If-Match": '"4"'},
            400,
        )
        # An explicit reused key with a different precondition stays a permanent
        # idempotency conflict, even though using a fresh key could avoid it.
        headers = {"If-Match": '"4"', "Idempotency-Key": "explicit-bootstrap-key"}
        self.api.request("PUT", self.binding_path, body, headers)
        self.reject("PUT", self.binding_path, body, {**headers, "If-Match": '"1"'}, 409)
        self.assertEqual(self.current(self.binding_path)["revision"], "4")

    def principal(self):
        self.configure("1")
        doc = {"hosts": [{"ssh": "fixture", "slots": 1}]}

        def assign():
            bootstrap.assign(self.api, self.state, doc, self.args)

        assign()
        original = self.api.writes[-1]
        self.check_replay(original, self.principal_path, "3")
        self.api.request(
            "PUT",
            self.principal_path + "/labels",
            {"labels": ["unrelated"]},
            {"If-Match": '"3"', "Idempotency-Key": "external-remove"},
        )
        self.api.lose_next = True
        with self.assertRaises(RemoteDisconnected):
            assign()
        restored = self.api.writes[-1]
        self.check_replay(restored, self.principal_path, "5")
        self.assertEqual(self.current(self.principal_path)["labels"], ["cloud-llm", "unrelated"])
        count = len(self.api.writes)
        assign()
        self.assertEqual(len(self.api.writes), count, "unchanged assign must only read")
        self.reject(
            "PUT",
            self.principal_path + "/labels",
            {"labels": ["unrelated"]},
            {"If-Match": '"1"'},
            412,
        )

    def reject(self, method, path, body, headers, status):
        before = len(self.api.writes)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "HTTP " + str(status)):
                self.api.request(method, path, body, headers)
        self.assertEqual(len(self.api.writes), before + 2, "no automatic permanent-error retries")


url, state, scenario, principal_id = sys.argv[1:]
result = unittest.TextTestRunner().run(unittest.TestSuite([BootstrapPostgres(scenario)]))
sys.exit(not result.wasSuccessful())
