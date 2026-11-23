"""Exercise scope policy resolution against the real Runtime parser."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from contractor_runtime.sandbox.podman.settings import PodmanSettings

SOURCE = (
    Path(__file__).resolve().parents[1] / "roles/contractor_runtime/files/podman_scope_policy.py"
)
SPEC = importlib.util.spec_from_file_location("podman_scope_policy", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("scope policy resolver is missing")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ScopePolicy(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        key = Path(self.temporary.name) / "identity"
        key.write_text("test-only parser fixture")
        key.chmod(0o600)
        self.arguments = [
            "--control-plane-url",
            "https://127.0.0.1:8443",
            "--advertised-control-url",
            "https://127.0.0.1:9443",
            "--advertised-a2a-url",
            "https://127.0.0.1:9443",
            "--ca-file",
            str(key),
            "--certificate-file",
            str(key),
            "--private-key-file",
            str(key),
            "--workspace-storage",
            "local",
            "--workspace-work-root",
            self.temporary.name + "/workspaces",
            "--podman-enabled",
            "true",
            "--podman-owner",
            "test-owner",
            "--podman-image",
            "localhost/test@sha256:" + "a" * 64,
        ]

    def test_defaults_come_from_runtime_and_disabled_instances_are_ignored(self):
        defaults = PodmanSettings()
        policy = MODULE.scope_policy(
            [
                self.arguments,
                [*self.arguments, "--podman-enabled", "false", "--podman-cpus", "4"],
            ]
        )
        self.assertEqual(
            policy,
            {
                "cpu_quota": format(defaults.cpus * 100, ".12g"),
                "memory_bytes": defaults.memory_bytes,
                "pids": defaults.pids,
            },
        )

    def test_uniform_overrides_are_kept_including_fractional_cpu(self):
        args = [
            *self.arguments,
            "--podman-cpus",
            "0.5",
            "--podman-memory-bytes",
            "1073741824",
            "--podman-pids",
            "128",
        ]
        self.assertEqual(
            MODULE.scope_policy([args, args]),
            {
                "cpu_quota": "50",
                "memory_bytes": 1073741824,
                "pids": 128,
            },
        )

    def test_mixed_limits_fail_instead_of_silently_changing_an_instance(self):
        for option, value in (
            ("--podman-cpus", "4"),
            ("--podman-memory-bytes", "4294967296"),
            ("--podman-pids", "512"),
        ):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "identical"):
                MODULE.scope_policy([self.arguments, [*self.arguments, option, value]])

    def test_empty_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "enabled"):
            MODULE.scope_policy([])


if __name__ == "__main__":
    unittest.main()
