"""The bootstrap transport retains exact mutation identities across failures."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import common


class Identity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        token = Path(self.temporary.name) / "token"
        common.secure_write(token, "fixture-secret")
        self.api = bootstrap.API("http://127.0.0.1:1", token)
        self.open = MagicMock()
        self.open.return_value.__enter__.return_value.read.return_value = b"{}"
        self.open.return_value.__enter__.return_value.headers = {}
        self.api.opener.open = self.open

    def request(self, method="PUT", path="/binding", body=None, headers=None):
        self.api.request(method, path, body, headers)
        self.open.assert_called_with(self.open.call_args.args[0], timeout=30)
        return self.open.call_args.args[0]

    def key(self, *args, **kwargs):
        return self.request(*args, **kwargs).get_header("Idempotency-key")

    def test_complete_identity_distinguishes_later_cas_and_other_mutations(self):
        cases = [
            ("PUT", "/binding", {"a": 1}, {"If-None-Match": "*"}),
            ("PUT", "/binding", {"a": 1}, {"If-Match": '"1"'}),
            ("PUT", "/binding", {"a": 1}, {"If-Match": '"3"'}),
            ("POST", "/binding", {"a": 1}, {"If-Match": '"3"'}),
            ("PUT", "/other", {"a": 1}, {"If-Match": '"3"'}),
            ("PUT", "/binding", {"a": 2}, {"If-Match": '"3"'}),
            ("DELETE", "/binding", None, {"If-Match": '"3"'}),
        ]
        keys = [self.key(*case) for case in cases]
        self.assertEqual(len(set(keys)), len(cases))
        for case, key in zip(cases, keys, strict=True):
            self.assertEqual(self.key(*case), key)
            self.assertRegex(key, r"\A[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
            self.assertNotIn("fixture-secret", key)

    def test_header_case_method_case_and_canonical_body_keep_identity(self):
        first = self.key("put", body={"b": 2, "a": 1}, headers={"If-Match": '"2"'})
        second = self.key(body={"a": 1, "b": 2}, headers={"if-match": '"2"'})
        self.assertEqual(first, second)
        self.assertEqual(self.open.call_args.args[0].data, b'{"a":1,"b":2}')

    def test_explicit_key_is_respected_including_bodyless_mutation(self):
        for header in ("Idempotency-Key", "idempotency-key", "IDEMPOTENCY-KEY"):
            for method, body in (("POST", {"value": 1}), ("DELETE", None)):
                with self.subTest(header=header, method=method):
                    self.assertEqual(
                        self.key(method, body=body, headers={header: "chosen-key"}), "chosen-key"
                    )
        with self.assertRaisesRegex(ValueError, "duplicate API request header"):
            self.key(headers={"If-Match": '"1"', "if-match": '"2"'})
        with self.assertRaisesRegex(ValueError, "duplicate API request header"):
            self.key(headers={"Idempotency-Key": "one", "idempotency-key": "two"})

    def test_bodyless_reads_have_no_generated_key(self):
        for method in ("GET", "HEAD", "OPTIONS"):
            self.assertIsNone(self.key(method))

    def test_timeout_and_cancellation_do_not_retry_or_change_identity(self):
        expected = self.key(body={"a": 1}, headers={"If-Match": '"2"'})
        for error in (TimeoutError(), KeyboardInterrupt()):
            self.open.reset_mock()
            self.open.side_effect = error
            with self.assertRaises(type(error)):
                self.key(body={"a": 1}, headers={"If-Match": '"2"'})
            self.open.assert_called_once()
            self.assertEqual(self.open.call_args.args[0].get_header("Idempotency-key"), expected)
        self.open.side_effect = None
        self.assertEqual(self.key(body={"a": 1}, headers={"If-Match": '"2"'}), expected)

    def test_http_failures_remain_visible_without_secret_output_or_new_keys(self):
        expected = self.key(body={"material": "fixture-secret"})
        for code in (400, 409, 412, 503):
            keys = []
            for _ in range(2):
                self.open.reset_mock()
                self.open.side_effect = HTTPError(
                    self.api.url, code, "fixture-secret", {}, io.BytesIO(b"fixture-secret")
                )
                with self.assertRaisesRegex(RuntimeError, f"HTTP {code}") as failure:
                    self.key(body={"material": "fixture-secret"})
                self.open.assert_called_once()
                self.assertNotIn("fixture-secret", str(failure.exception))
                keys.append(self.open.call_args.args[0].get_header("Idempotency-key"))
            self.assertEqual(keys, [expected, expected])
