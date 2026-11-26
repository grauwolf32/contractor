"""Verify durable hybrid location declarations through the real Ansible role."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

DEPLOY = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(DEPLOY / "hybrid"), str(DEPLOY / "wireguard")]

import mesh  # noqa: E402
import runtime  # noqa: E402


class RuntimePlacement(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state = Path(self.temporary.name)
        binary = self.state / "contractor"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o700)
        pki = self.state / "pki"
        pki.mkdir()
        (pki / "ca.crt").touch()
        (pki / "ca.key").touch()
        self.args = SimpleNamespace(
            contractor=binary,
            pki_root=pki,
            version="HEAD",
            full_capabilities=False,
            defer_restarts=False,
        )
        self.document = {
            "version": 1,
            "controller_address": "10.77.0.10",
            "hosts": [
                {
                    "ssh": "cloud-fixture",
                    "address": "10.77.0.11",
                    "slots": 2,
                    "allocation_slots": 2,
                    "placement_labels": ["cloud", "cloud/aws"],
                },
                {
                    "ssh": "second-fixture",
                    "address": "10.77.0.12",
                    "slots": 1,
                    "placement_labels": ["cloud/hetzner"],
                },
            ],
        }

    def inventory(self, document=None):
        with patch.object(runtime, "run", return_value="0" * 40):
            return runtime.inventory(self.args, self.state, document or self.document)

    def test_init_explicitly_declares_cloud_without_inventing_a_provider(self):
        args = SimpleNamespace(
            state=self.state,
            hosts=["fixture"],
            network="10.77.0.0/24",
            port=51821,
            egress_network="172.25.18.0/30",
            egress_check_url="https://provider.example/models",
        )
        with (
            patch.object(mesh, "ssh_hostname", return_value="192.0.2.1"),
            patch.object(mesh, "render"),
        ):
            mesh.init(args)
        source = json.loads((self.state / "deployment.json").read_text())
        self.assertEqual(source["hosts"][0]["placement_labels"], ["cloud"])

    def test_inventory_keeps_each_hosts_source_declaration_on_reruns(self):
        for full in (False, True):
            for deferred in (False, True):
                with self.subTest(full=full, deferred=deferred):
                    self.args.full_capabilities, self.args.defer_restarts = (
                        full,
                        deferred,
                    )
                    path = self.inventory()
                    first = path.read_bytes()
                    path.write_text("discarded edit of generated inventory")
                    self.inventory()
                    self.assertEqual(path.read_bytes(), first)
                    config = json.loads(first)["contractor_runtime"]
                    self.assertNotIn("contractor_runtime_labels", config["vars"])
                    self.assertEqual(config["vars"]["contractor_runtime_restart"], not deferred)
                    for source in self.document["hosts"]:
                        host = config["hosts"][source["ssh"]]
                        self.assertEqual(
                            host["contractor_runtime_placement_labels"],
                            source["placement_labels"],
                        )
                        self.assertEqual(len(host["contractor_runtime_instances"]), source["slots"])
                        self.assertEqual(
                            host["contractor_runtime_allocation_slots"],
                            source.get("allocation_slots", 4),
                        )

    def test_invalid_capacity_preserves_existing_inventory(self):
        path = self.inventory()
        original = path.read_bytes()
        for value in (0, 5, True, "4", None):
            with self.subTest(value=value):
                source = copy.deepcopy(self.document)
                source["hosts"][0]["allocation_slots"] = value
                with self.assertRaisesRegex(ValueError, "allocation_slots"):
                    self.inventory(source)
                self.assertEqual(path.read_bytes(), original)

    def test_shared_grammar_rejects_invalid_sources_before_replacing_inventory(self):
        path = self.inventory()
        original = path.read_bytes()
        cases = json.loads(
            (DEPLOY.parent / "api/testdata/v1alpha1/placement-label-cases.json").read_text()
        )["cases"]
        cases += [
            {"name": "missing", "valid": False},
            {"name": "null", "labels": None, "valid": False},
            {"name": "string", "labels": "cloud", "valid": False},
            {"name": "non-string", "labels": [1], "valid": False},
        ]
        for case in cases:
            with self.subTest(case=case["name"]):
                source = copy.deepcopy(self.document)
                source["hosts"][0].pop("placement_labels")
                if "labels" in case:
                    source["hosts"][0]["placement_labels"] = case["labels"]
                if case["valid"]:
                    self.inventory(source)
                    original = path.read_bytes()
                else:
                    with (
                        patch.object(runtime, "run") as command,
                        self.assertRaises(ValueError),
                    ):
                        runtime.inventory(self.args, self.state, source)
                    command.assert_not_called()
                    self.assertEqual(path.read_bytes(), original)

    def test_all_cli_modes_carry_placement_without_remote_work_in_preview_modes(self):
        for mode in ("--inventory-only", "--syntax-check", "deploy"):
            with self.subTest(mode=mode):
                argv = [
                    "runtime.py",
                    "--state",
                    str(self.state),
                    "--contractor",
                    str(self.args.contractor),
                    "--pki-root",
                    str(self.args.pki_root),
                    "--version",
                    "HEAD",
                    "--full-capabilities",
                    "--defer-restarts",
                ]
                if mode != "deploy":
                    argv.append(mode)
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(runtime, "load_state", return_value=(self.state, self.document)),
                    patch.object(runtime, "run", return_value="0" * 40),
                    patch.object(runtime, "preflight") as preflight,
                    patch.object(runtime, "verify") as verify,
                    patch.object(
                        runtime.subprocess,
                        "run",
                        return_value=SimpleNamespace(returncode=0),
                    ) as ansible,
                ):
                    runtime.main()
                self.assertEqual(preflight.call_count, int(mode == "deploy"))
                self.assertEqual(verify.call_count, int(mode == "deploy"))
                self.assertEqual(ansible.call_count, int(mode != "--inventory-only"))
                if mode == "--syntax-check":
                    self.assertIn("--syntax-check", ansible.call_args.args[0])
                config = json.loads((self.state / "runtime-inventory.yml").read_text())[
                    "contractor_runtime"
                ]
                self.assertEqual(
                    config["hosts"]["cloud-fixture"]["contractor_runtime_placement_labels"],
                    ["cloud", "cloud/aws"],
                )

    def test_generated_inventory_renders_real_role_startup_arguments(self):
        for full in (False, True):
            with self.subTest(full=full):
                self.args.full_capabilities = full
                path = self.inventory()
                result = subprocess.run(
                    [
                        *runtime.ANSIBLE,
                        "-i",
                        str(path),
                        str(Path(__file__).with_name("resolve-runtime.yml")),
                    ],
                    cwd=DEPLOY / "ansible",
                    env=dict(os.environ, ANSIBLE_CONFIG=str(DEPLOY / "ansible/ansible.cfg")),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("Verify generated location arguments", result.stdout)


if __name__ == "__main__":
    unittest.main()
