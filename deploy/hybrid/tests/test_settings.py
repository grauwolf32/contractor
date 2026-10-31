"""Exercise portable deployment settings through generated payloads and shell scripts."""

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

import bootstrap  # noqa: E402
import common  # noqa: E402
import gateway  # noqa: E402
import mesh  # noqa: E402


class DeploymentSettings(unittest.TestCase):
    def document(self, network="172.25.18.0/30"):
        return {
            "version": 1,
            "network": "10.42.0.0/24",
            "controller_address": "10.42.0.10",
            "port": 51821,
            "hosts": [{"ssh": "runtime-fixture", "address": "10.42.0.11", "slots": 1}],
            "gateway_host": "runtime-fixture",
            "egress": {
                "network": network,
                "check_url": "https://provider.example/models",
            },
        }

    def test_missing_malformed_and_overlapping_inputs_fail_before_ssh(self):
        base = self.document()
        cases = [
            None,
            {},
            {"network": "172.25.18.0/30"},
            dict(base["egress"], extra=True),
        ]
        cases += [
            dict(base["egress"], network=value)
            for value in (
                "10.42.0.0/30",
                "172.25.18.0/24",
                "8.8.8.0/30",
                "fd00::/30",
                "bad",
                None,
                [],
            )
        ]
        cases += [
            dict(base["egress"], check_url=value)
            for value in (
                "http://provider.example/models",
                "https://secret@provider.example/models",
                "https://provider.example/?token=secret",
                "https://provider.example/#secret",
                "https://provider.example/\nmodels",
                None,
            )
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "deployment.json"
            for settings in cases:
                with self.subTest(settings=settings):
                    doc = dict(base, egress=settings)
                    path.write_text(json.dumps(doc))
                    with (
                        patch.object(common, "ssh_hostname") as ssh,
                        self.assertRaises(ValueError),
                    ):
                        common.load_state(path.parent)
                    ssh.assert_not_called()

    def test_init_requires_explicit_networks_without_writing_partial_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            args = SimpleNamespace(
                state=state,
                hosts=["runtime-fixture"],
                network="10.42.0.0/24",
                port=51821,
                egress_network="10.42.0.0/30",
                egress_check_url="https://provider.example/models",
            )
            with patch.object(mesh, "ssh_hostname"), self.assertRaises(ValueError):
                mesh.init(args)
            self.assertFalse((state / "deployment.json").exists())

    def test_networks_reach_gateway_payload_tunnel_and_real_shell_commands(self):
        for network in ("172.25.18.0/30", "192.168.233.4/30"):
            with (
                self.subTest(network=network),
                tempfile.TemporaryDirectory() as temporary,
            ):
                state = Path(temporary)
                doc = self.document(network)
                settings = common.egress_settings(doc)
                common.secure_write(
                    state / "egress.json",
                    json.dumps({"dns": ["192.0.2.53"], "egress": settings}),
                )
                common.secure_write(state / "upstream-key", "sk-or-v1-fixture")
                argv = [
                    "gateway.py",
                    "apply",
                    "--openrouter-key-file",
                    str(state / "upstream-key"),
                ]
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(gateway, "load_state", return_value=(state, doc)),
                    patch.object(gateway, "apply_remote", return_value="") as apply,
                ):
                    gateway.main()
                files = apply.call_args.args[1]
                env = files["/etc/contractor-hybrid/litellm.env"][0]
                self.assertIn("@" + settings["host_address"] + ":54329/litellm", env)
                public = json.loads(files["/etc/contractor-hybrid/gateway.json"][0])
                self.assertEqual(public["egress"], settings)
                with (
                    patch.object(
                        bootstrap.subprocess,
                        "run",
                        return_value=SimpleNamespace(returncode=1),
                    ),
                    patch.object(bootstrap, "run") as run,
                ):
                    bootstrap.tunnel(state, doc)
                self.assertIn(
                    f"127.0.0.1:14000:{settings['namespace_address']}:4000",
                    run.call_args.args[0],
                )
                self.run_shell_consumers(state, settings, public)

    def run_shell_consumers(self, state, settings, public):
        """Run real shell logic with fake OS commands and only temporary filesystem paths."""
        base = state / "etc"
        base.mkdir()
        (base / "gateway.json").write_text(json.dumps(public))
        (base / "egress.json").write_text(
            json.dumps(
                {
                    "address": "10.9.0.2/32",
                    "mtu": 1380,
                    "mesh_address": "10.42.0.11",
                    "mesh_network": "10.42.0.0/24",
                    "dns": ["192.0.2.53"],
                    "egress": settings,
                }
            )
        )
        bins = state / "bin"
        bins.mkdir()
        shim = bins / "shim"
        shim.write_text(
            "#!/usr/bin/env python3\n"
            "import json,os,pathlib,sys\n"
            "name=pathlib.Path(sys.argv[0]).name\n"
            "with open(os.environ['FIXTURE_LOG'],'a') as f:\n"
            " f.write(json.dumps([name,*sys.argv[1:]])+'\\n')\n"
            "if name=='id': print('0')\n"
        )
        shim.chmod(0o700)
        for name in ("id", "ip", "iptables", "sysctl", "podman"):
            (bins / name).symlink_to(shim)
        log = state / "commands.jsonl"
        for source, action in (
            (DEPLOY / "wireguard/linux-egress.sh", "up"),
            (DEPLOY / "hybrid/gateway.sh", "postgres"),
            (DEPLOY / "hybrid/gateway.sh", "litellm"),
        ):
            content = source.read_text()
            for original, replacement in (
                ("/etc/contractor-hybrid", str(base)),
                ("/var/lib/contractor-hybrid", str(state / "var")),
                ("/etc/netns", str(state / "netns")),
                ("/etc/sysctl.d", str(state / "sysctl")),
            ):
                content = content.replace(original, replacement)
            (state / "sysctl").mkdir(exist_ok=True)
            script = state / source.name
            script.write_text(content)
            result = subprocess.run(
                ["bash", str(script), action],
                env=dict(
                    os.environ,
                    PATH=str(bins) + os.pathsep + os.environ["PATH"],
                    FIXTURE_LOG=str(log),
                ),
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertIn(
            [
                "ip",
                "address",
                "add",
                settings["host_address"] + "/30",
                "dev",
                "hc-llm0",
            ],
            commands,
        )
        self.assertIn(
            [
                "ip",
                "-n",
                "contractor-llm",
                "address",
                "add",
                settings["namespace_address"] + "/30",
                "dev",
                "eth0",
            ],
            commands,
        )
        self.assertTrue(
            any("listen_addresses=" + settings["host_address"] in row for row in commands)
        )
        self.assertTrue(any(settings["namespace_address"] + ":4000" in row for row in commands))
        self.assertTrue(any(settings["host_address"] + "/32" in row for row in commands))

    def test_mirror_pair_is_explicit_and_invalid_input_does_not_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            args = SimpleNamespace(
                state=state,
                hosts=[],
                ubuntu_mirror="https://new-mirror.example/ubuntu",
                ubuntu_mirror_source="",
            )
            with (
                patch.object(mesh, "load_state", return_value=(state, self.document())),
                patch.object(mesh, "render") as render,
                patch.object(mesh, "apply_remote") as apply,
                self.assertRaises(ValueError),
            ):
                mesh.apply(args)
            render.assert_not_called()
            apply.assert_not_called()
            args.ubuntu_mirror_source = "https://old-mirror.example/ubuntu"
            config = state / "ubuntu.sources"
            original = "URIs: " + args.ubuntu_mirror_source + "\nSigned-By: fixture-keyring\n"
            config.write_text(original)
            bins = state / "bin"
            bins.mkdir()
            for name, body in (
                ("id", "echo 0"),
                ("wg-quick", "exit 0"),
                ("ip", "exit 1"),
                ("systemctl", "exit 0"),
            ):
                path = bins / name
                path.write_text("#!/bin/sh\n" + body + "\n")
                path.chmod(0o700)
            owned = state / "wg-mesh.conf"
            owned.write_text("# Managed by Contractor WireGuard mesh\n")
            script = state / "mesh.sh"
            script.write_text(
                (DEPLOY / "wireguard/linux-mesh.sh")
                .read_text()
                .replace("/etc/apt/sources.list.d/ubuntu.sources", str(config))
                .replace("/etc/wireguard/wg-mesh.conf", str(owned))
            )
            for _ in range(2):
                result = subprocess.run(
                    ["bash", str(script), "up", args.ubuntu_mirror, args.ubuntu_mirror_source],
                    env=dict(os.environ, PATH=str(bins) + os.pathsep + os.environ["PATH"]),
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(
                config.read_text(), original.replace(args.ubuntu_mirror_source, args.ubuntu_mirror)
            )
            self.assertEqual(
                config.with_name("ubuntu.sources.contractor-backup").read_text(), original
            )

    def test_changed_network_refuses_gateway_before_remote_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            common.secure_write(
                state / "egress.json", json.dumps({"dns": ["192.0.2.53"], "egress": {}})
            )
            common.secure_write(state / "upstream-key", "sk-or-v1-fixture")
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "gateway.py",
                        "apply",
                        "--openrouter-key-file",
                        str(state / "upstream-key"),
                    ],
                ),
                patch.object(
                    gateway,
                    "load_state",
                    return_value=(state, copy.deepcopy(self.document())),
                ),
                patch.object(gateway, "apply_remote") as apply,
                self.assertRaises(ValueError),
            ):
                gateway.main()
            apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
