"""Exercise secret handling and real local transports with disposable fixtures."""

from __future__ import annotations

import base64
import json
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

DEPLOY = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(DEPLOY / "hybrid"), str(DEPLOY / "wireguard")]

import bootstrap  # noqa: E402 - import standalone scripts after adding their directories
import common  # noqa: E402
import egress  # noqa: E402
import ingress  # noqa: E402
import mesh  # noqa: E402


class Files(unittest.TestCase):
    def test_secure_creation_preserves_identity_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "private" / "key"
            common.secure_write(path, "first", replace=False)
            common.secure_write(path, "second", replace=False)
            self.assertEqual(common.secret(path), "first")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
            link = path.parent / "link"
            link.symlink_to(path)
            with self.assertRaises(RuntimeError):
                common.secure_write(link, "replacement")
            path.chmod(0o644)
            with self.assertRaises(RuntimeError):
                common.secret(path)

    def test_subprocess_failure_never_echoes_secret_output(self):
        with self.assertRaises(RuntimeError) as result:
            common.run([sys.executable, "-c", "import sys; print('secret-value'); sys.exit(1)"])
        self.assertNotIn("secret-value", str(result.exception))

    def test_mesh_render_retains_keys_and_only_routes_individual_peers(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            doc = {
                "controller_address": "10.77.0.10",
                "port": 51821,
                "hosts": [{"ssh": "fixture", "address": "10.77.0.11", "slots": 1}],
            }
            key = base64.b64encode(b"k" * 32).decode()
            with (
                patch.object(mesh, "ssh_hostname", return_value="192.0.2.1"),
                patch.object(mesh, "run", return_value=key),
            ):
                mesh.render(state, doc)
                before = (state / "wireguard/controller/private.key").read_bytes()
                mesh.render(state, doc)
            self.assertEqual((state / "wireguard/controller/private.key").read_bytes(), before)
            config = (state / "wireguard/controller/wg-mesh.conf").read_text()
            self.assertIn("AllowedIPs = 10.77.0.11/32", config)
            self.assertNotIn("0.0.0.0/0", config)

    def test_two_aliases_cannot_deploy_to_one_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            (state / "deployment.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "hosts": [{"ssh": "first"}, {"ssh": "second"}],
                    }
                )
            )
            with (
                patch.object(common, "ssh_hostname", return_value="192.0.2.1"),
                self.assertRaises(ValueError),
            ):
                common.load_state(state)


class ProviderProfile(unittest.TestCase):
    def parse(self, extra="", allowed="0.0.0.0/0, ::/0", key=None):
        key = key or base64.b64encode(b"k" * 32).decode()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "provider.conf"
            source.write_text(
                f"[Interface]\nPrivateKey={key}\nAddress=10.1.17.24/32\n"
                f"DNS=8.8.8.8\n{extra}\n[Peer]\nPublicKey={key}\n"
                f"AllowedIPs={allowed}\nEndpoint=192.0.2.1:51820\n"
            )
            return egress.parse_profile(source)

    def test_provider_profile_is_normalized_without_host_hooks(self):
        config, metadata = self.parse()
        self.assertEqual(metadata, {"address": "10.1.17.24/32", "dns": ["8.8.8.8"], "mtu": 1380})
        self.assertNotIn("DNS", config)
        self.assertNotIn("Address", config)
        self.assertIn("AllowedIPs = 0.0.0.0/0, ::/0", config)

    def test_hooks_partial_routes_and_bad_keys_fail(self):
        for values in (
            {"extra": "PostUp=echo secret"},
            {"allowed": "10.0.0.0/8"},
            {"key": "Zm9v"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.parse(**values)


class ModelSelection(unittest.TestCase):
    def test_cloud_policy_refs_follow_deployed_defaults_and_role_overrides(self):
        settings = {
            "configured": True,
            "models": [
                {
                    "id": "first",
                    "policies": {
                        "worker": "worker_first@2",
                        "summarizer": "summarizer_first@1",
                        "audit_completion_worker": "audit_first@1",
                    },
                },
                {"id": "second", "policies": {"summarizer": "summarizer_second@1"}},
            ],
            "selection": {"defaultModel": "first", "roles": {"summarizer": "second"}},
        }
        api = unittest.mock.Mock()
        api.request.return_value = (settings, {})
        self.assertEqual(
            bootstrap.selected_model_policies(api),
            [
                "worker_first@2",
                "summarizer_second@1",
                "audit_first@1",
            ],
        )
        api.request.assert_called_once_with("GET", "/v1/operations/settings/models")

    def test_no_compiled_model_catalog_requires_an_explicit_policy_selection(self):
        api = unittest.mock.Mock()
        api.request.return_value = ({"configured": False}, {})
        with self.assertRaises(ValueError):
            bootstrap.selected_model_policies(api)


class FixtureHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list] = []

    def log_message(self, *unused):
        pass

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/destination")
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ready" if self.path == "/readyz" else b"x" * (1024 * 1024 + 1))

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.requests.append((self.headers["Idempotency-Key"], body))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")


class Transport(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        token = Path(self.temporary.name) / "token"
        common.secure_write(token, "fixture-token")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.api = bootstrap.API(f"http://127.0.0.1:{self.server.server_port}", token)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def test_redirect_is_refused_and_response_size_is_bounded(self):
        for route in ("/redirect", "/large"):
            with self.subTest(route=route), self.assertRaises(RuntimeError):
                self.api.request("GET", route)
        self.api.request("GET", "/readyz")

    def test_mutation_retry_keeps_the_same_key(self):
        FixtureHandler.requests.clear()
        self.api.request("POST", "/mutate", {"b": 2, "a": 1})
        self.api.request("POST", "/mutate", {"a": 1, "b": 2})
        self.assertEqual(FixtureHandler.requests[0], FixtureHandler.requests[1])

    def test_tcp_relay_preserves_bytes_without_terminating_protocol(self):
        relay = ingress.Relay(("127.0.0.1", 0), self.server.server_address)
        thread = threading.Thread(target=relay.serve_forever)
        thread.start()
        try:
            with socket.create_connection(relay.server_address, timeout=5) as client:
                client.sendall(b"GET /readyz HTTP/1.0\r\nHost: localhost\r\n\r\n")
                chunks = []
                while data := client.recv(65536):
                    chunks.append(data)
            self.assertTrue(b"".join(chunks).endswith(b"ready"))
        finally:
            relay.shutdown()
            relay.server_close()
            thread.join()


class Vault(unittest.TestCase):
    def test_real_vault_round_trip_with_fixture_password(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, restored = root / "original", root / "restored"
            password, encrypted = root / "password", root / "backup.vault"
            common.secure_write(password, "disposable-fixture-password\n")
            common.secure_write(original / "secrets/key", "fixture-secret-value\n")
            argv = [
                sys.executable,
                str(DEPLOY / "hybrid/vault.py"),
                "--vault-id",
                "fixture@" + str(password),
            ]
            for args in (
                ["--state", str(original), "seal", str(encrypted)],
                ["--state", str(restored), "restore", str(encrypted)],
            ):
                result = subprocess.run(
                    argv + args, capture_output=True, text=True, timeout=60, check=False
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("fixture-secret-value", result.stdout + result.stderr)
            self.assertTrue(encrypted.read_text().startswith("$ANSIBLE_VAULT;"))
            self.assertNotIn("fixture-secret-value", encrypted.read_text())
            self.assertEqual(common.secret(restored / "secrets/key"), "fixture-secret-value")
            self.assertEqual(encrypted.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
