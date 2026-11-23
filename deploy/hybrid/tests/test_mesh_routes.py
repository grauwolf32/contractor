"""Exercise local-up through disposable command processes; never touch host routes."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_deployment import mesh

TOOL = r"""
import json, os, subprocess, sys
from pathlib import Path
path = Path(os.environ["MESH_FIXTURE_STATE"])
s = json.loads(path.read_text())
name = Path(sys.argv[0]).name
args = sys.argv[1:]
s["calls"].append([name, *args])
path.write_text(json.dumps(s))
if name == "sudo":
    if s.get("permission_denied"):
        print("sudo: a password is required; fixture-secret", file=sys.stderr)
        sys.exit(1)
    # privileged() preserves PATH through env; execute the real fixture child.
    sys.exit(subprocess.run(args).returncode)
if name == "wg":
    if args == ["pubkey"]:
        if sys.stdin.read().strip() != "fixture-secret":
            sys.exit(1)
        print("fixture-public-key")
    elif s["up"]:
        print(s["interface"], "wrong-key" if s.get("wrong_key") else "fixture-public-key")
elif name == "ifconfig":
    print("utun4: flags=8051<UP,POINTOPOINT,RUNNING> mtu 1420")
    print("\tinet 10.1.17.23 --> 10.1.17.23 netmask 0xffffffff")
    if s["up"]:
        print(s["interface"] + ": flags=8051<UP,POINTOPOINT,RUNNING> mtu 1380")
        print("\tinet " + s["address"] + " --> " + s["address"] + " netmask 0xffffffff")
        if s.get("duplicate_address"):
            print("utun27: flags=8051<UP> mtu 1380")
            print("\tinet " + s["address"] + " --> " + s["address"] + " netmask 0xffffffff")
elif name == "wg-quick":
    if args[0] == "down":
        s["up"] = False
    elif s["up"]:
        print("fixture-secret: interface already exists", file=sys.stderr)
        sys.exit(1)
    else:
        s["up"] = not s.get("missing_interface_after_start", False)
        s["self_route_interface"] = s["interface"]
elif name == "route":
    if args[1] == "get":
        is_network = "-net" in args
        is_self = not is_network and args[-1] == s["address"]
        if is_self and s.get("self_read_failed"):
            sys.exit(1)
        exact = s.get("self_exact", True) if is_self else s["exact"] if is_network else True
        print("   route to: " + args[-1])
        if is_self:
            print("destination: " + (s["address"] if exact else s["network_address"]))
            print("      flags: " + ("<UP,HOST,DONE>" if exact else "<UP,DONE>"))
            print("    gateway: " + s.get("self_route_gateway", s["interface"]))
            print("  interface: " + s["self_route_interface"])
        else:
            destination = s["network_address"] if exact and is_network else "default"
            print("destination: " + (destination if is_network else args[-1]))
            print("       mask: " + ("255.255.255.0" if exact else "0.0.0.0"))
            captured = not is_network and s.get("peer_override")
            print("  interface: " + ("utun4" if captured else s["route_interface"]))
    else:
        is_self = "-host" in args
        if s.get("write_failed") or (is_self and s.get("self_write_failed")):
            print("fixture-secret: route write failed", file=sys.stderr)
            sys.exit(1)
        if not s.get("ignore_write") and not (is_self and s.get("self_ignore_write")):
            gateway = args[args.index("-interface") + 1]
            ifp = args[args.index("-ifp") + 1] if "-ifp" in args else None
            if is_self:
                s["self_exact"] = args[1] != "delete"
                s["self_route_gateway"] = gateway
                if args[1] == "delete":
                    s["self_route_interface"] = s["route_interface"]
                elif ifp and not s.get("ignore_ifp"):
                    s["self_route_interface"] = ifp
                elif args[1] == "add":
                    s["self_route_interface"] = gateway
            else:
                s["exact"] = True
                s["route_gateway"] = gateway
                if ifp and not s.get("ignore_ifp"):
                    s["route_interface"] = ifp
                elif args[1] == "add":
                    s["route_interface"] = gateway
path.write_text(json.dumps(s))
"""


class MacOSMeshRoutes(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for tool in ["wg", "wg-quick", "ifconfig", "route", "sudo"]:
            path = self.bin / tool
            path.write_text(f"#!{sys.executable}\n" + TOOL)
            path.chmod(0o700)
        key = self.root / "wireguard/controller/private.key"
        key.parent.mkdir(parents=True)
        key.write_text("fixture-secret")
        key.chmod(0o600)
        self.args = SimpleNamespace(state=self.root)
        self.document = {
            "network": "10.89.0.0/24",
            "controller_address": "10.89.0.10",
            "hosts": [{"address": f"10.89.0.{n}"} for n in [11, 12]],
        }
        self.state = self.root / "state.json"
        self.write_state()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.dict(
                os.environ,
                PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                MESH_FIXTURE_STATE=str(self.state),
            )
        )
        self.stack.enter_context(
            patch.object(mesh, "load_state", return_value=(self.root, self.document))
        )
        self.stack.enter_context(patch.object(mesh.sys, "platform", "darwin"))
        self.stack.enter_context(patch.object(mesh.os, "geteuid", return_value=0))

    def write_state(self, **values):
        self.state.write_text(
            json.dumps(
                {
                    "up": True,
                    "interface": "utun19",
                    "address": "10.89.0.10",
                    "network_address": "10.89.0.0",
                    "route_interface": "utun4",
                    "self_route_interface": "lo0",
                    "exact": False,
                    "calls": [],
                    **values,
                }
            )
        )

    def invoke(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            mesh.local_up(self.args)
        return output.getvalue()

    def calls(self, tool):
        return [c[1:] for c in json.loads(self.state.read_text())["calls"] if c[0] == tool]

    def mutations(self):
        return [c for c in self.calls("route") if c[1] != "get"]

    def test_missing_tunnel_starts_and_discovers_actual_interface(self):
        self.write_state(up=False, interface="utun23")
        output = self.invoke()
        self.assertEqual(len(self.calls("wg-quick")), 1)
        self.assertEqual(
            self.mutations(),
            [
                ["-n", "add", "-net", "10.89.0.0/24", "-interface", "utun23", "-ifp", "utun23"],
                ["-n", "change", "-host", "10.89.0.10", "-interface", "lo0", "-ifp", "lo0"],
            ],
        )
        self.assertIn("10.89.0.0/24 via utun23", output)
        self.assertIn("10.89.0.10 via lo0", output)

    def test_existing_tunnel_repairs_missing_route_without_restart(self):
        self.assertIn("via utun19", self.invoke())
        self.assertEqual(self.calls("wg-quick"), [])
        self.assertEqual(self.mutations()[0][1], "add")
        self.assertTrue(all("default" not in c and "0.0.0.0/0" not in c for c in self.mutations()))

    def test_exact_route_captured_by_personal_vpn_is_changed(self):
        self.write_state(exact=True)
        self.invoke()
        self.assertEqual(
            self.mutations(),
            [["-n", "change", "-net", "10.89.0.0/24", "-interface", "utun19", "-ifp", "utun19"]],
        )
        self.assertEqual(self.calls("wg-quick"), [])

    def test_repeated_correct_route_is_read_only(self):
        self.write_state(exact=True, route_interface="utun19")
        self.invoke()
        self.invoke()
        self.assertEqual(self.mutations(), [])
        self.assertEqual(self.calls("wg-quick"), [])

    def test_existing_mesh_repairs_only_the_self_route(self):
        self.write_state(
            exact=True,
            route_interface="utun19",
            self_route_interface="utun19",
            self_route_gateway="lo0",
        )
        self.assertIn("10.89.0.10 via lo0", self.invoke())
        self.assertEqual(
            self.mutations(),
            [["-n", "change", "-host", "10.89.0.10", "-interface", "lo0", "-ifp", "lo0"]],
        )
        self.assertEqual(self.calls("wg-quick"), [])
        before = self.mutations()
        self.invoke()
        self.assertEqual(self.mutations(), before)

    def test_missing_self_host_route_is_added_instead_of_changing_a_prefix(self):
        self.write_state(
            exact=True,
            route_interface="utun19",
            self_exact=False,
            self_route_interface="utun19",
        )
        self.invoke()
        self.assertEqual(
            self.mutations(),
            [["-n", "add", "-host", "10.89.0.10", "-interface", "lo0", "-ifp", "lo0"]],
        )

    def test_self_route_write_and_readback_failures_do_not_claim_readiness(self):
        for failure in ["self_write_failed", "self_ignore_write", "ignore_ifp"]:
            with self.subTest(failure=failure):
                self.write_state(
                    exact=True,
                    route_interface="utun19",
                    self_route_interface="utun19",
                    **{failure: True},
                )
                with (
                    contextlib.redirect_stdout(io.StringIO()) as output,
                    self.assertRaises(RuntimeError) as error,
                ):
                    mesh.local_up(self.args)
                self.assertNotIn("Local mesh ready", output.getvalue())
                self.assertNotIn("fixture-secret", str(error.exception))
                if failure == "ignore_ifp":
                    state = json.loads(self.state.read_text())
                    self.assertEqual(state["self_route_gateway"], "lo0")
                    self.assertEqual(state["self_route_interface"], "utun19")

    def test_wrong_identity_or_ambiguous_address_cannot_change_routes(self):
        for values in [
            {"wrong_key": True},
            {"duplicate_address": True},
            {"interface": "en0"},
        ]:
            with self.subTest(values=values):
                self.write_state(**values)
                with self.assertRaises(RuntimeError):
                    self.invoke()
                self.assertEqual(self.mutations(), [])
                self.assertEqual(self.calls("wg-quick"), [])

    def test_missing_interface_after_start_does_not_install_route(self):
        self.write_state(up=False, missing_interface_after_start=True)
        with self.assertRaisesRegex(RuntimeError, "without creating"):
            self.invoke()
        self.assertEqual(self.mutations(), [])

    def test_failed_write_and_unverified_result_never_claim_success_or_echo_key(self):
        for values in [
            {"write_failed": True},
            {"ignore_write": True},
            {"peer_override": True},
        ]:
            with self.subTest(values=values):
                self.write_state(**values)
                output = io.StringIO()
                with (
                    contextlib.redirect_stdout(output),
                    self.assertRaises(RuntimeError) as error,
                ):
                    mesh.local_up(self.args)
                self.assertNotIn("Local mesh ready", output.getvalue())
                self.assertNotIn("fixture-secret", str(error.exception) + output.getvalue())

    def test_administrator_denial_does_not_mutate_or_echo_captured_output(self):
        self.write_state(permission_denied=True)
        with (
            patch.object(mesh.os, "geteuid", return_value=501),
            self.assertRaises(RuntimeError) as error,
        ):
            self.invoke()
        self.assertTrue(self.calls("sudo"))
        self.assertEqual(self.mutations(), [])
        self.assertNotIn("fixture-secret", str(error.exception))

    def test_invalid_mesh_network_is_rejected_before_commands(self):
        for network in ["0.0.0.0/0", "192.0.2.0/24", "10.89.0.0/16"]:
            with self.subTest(network=network):
                self.document["network"] = network
                with self.assertRaises(ValueError):
                    self.invoke()
        self.assertEqual(json.loads(self.state.read_text())["calls"], [])

    def test_controller_and_duplicate_peer_addresses_are_rejected_before_commands(self):
        for addresses in [["10.89.0.10"], ["10.89.0.11", "10.89.0.11"]]:
            with self.subTest(addresses=addresses):
                self.document["hosts"] = [{"address": address} for address in addresses]
                with self.assertRaises(ValueError):
                    self.invoke()
                self.assertEqual(json.loads(self.state.read_text())["calls"], [])

    def test_all_sixteen_peers_are_checked_with_one_route_write(self):
        self.document["hosts"] = [{"address": f"10.89.0.{n}"} for n in range(11, 27)]
        self.invoke()
        peer_reads = [
            c
            for c in self.calls("route")
            if c[1] == "get" and "-net" not in c and c[-1] != self.document["controller_address"]
        ]
        self.assertEqual(len(peer_reads), 16)
        self.assertEqual(len(self.mutations()), 1)

    def test_non_macos_startup_remains_wg_quick_only(self):
        self.write_state(up=False)
        with patch.object(mesh.sys, "platform", "linux"):
            self.invoke()
        self.assertEqual(len(self.calls("wg-quick")), 1)
        self.assertEqual(self.calls("ifconfig"), [])
        self.assertEqual(self.calls("route"), [])

    def test_local_down_removes_self_route_before_stopping_the_mesh(self):
        self.write_state(exact=True, route_interface="utun19")
        mesh.local_down(self.args)
        calls = json.loads(self.state.read_text())["calls"]
        deletion = next(i for i, c in enumerate(calls) if c[0] == "route" and "delete" in c)
        teardown = next(i for i, c in enumerate(calls) if c[0] == "wg-quick")
        self.assertLess(deletion, teardown)
        self.assertEqual(
            self.mutations(),
            [["-n", "delete", "-host", "10.89.0.10", "-interface", "lo0", "-ifp", "lo0"]],
        )
        self.assertEqual(
            self.calls("wg-quick"),
            [["down", str(self.root / "wireguard/controller/wg-mesh.conf")]],
        )
        mesh.local_down(self.args)
        self.assertEqual(len(self.mutations()), 1)
        self.assertEqual(len(self.calls("wg-quick")), 1)

    def test_local_down_cleans_an_orphaned_self_route_without_starting_a_tunnel(self):
        self.write_state(up=False, exact=True, route_interface="utun19")
        mesh.local_down(self.args)
        self.assertEqual(self.calls("wg-quick"), [])
        self.assertEqual(self.mutations()[0][1], "delete")

    def test_failed_self_route_removal_leaves_the_tunnel_running(self):
        self.write_state(exact=True, route_interface="utun19", self_ignore_write=True)
        with self.assertRaises(RuntimeError):
            mesh.local_down(self.args)
        self.assertEqual(self.calls("wg-quick"), [])
        self.assertTrue(json.loads(self.state.read_text())["up"])

    def test_unavailable_self_route_lookup_prevents_teardown(self):
        self.write_state(exact=True, route_interface="utun19", self_read_failed=True)
        with self.assertRaises(RuntimeError):
            mesh.local_down(self.args)
        self.assertEqual(self.mutations(), [])
        self.assertEqual(self.calls("wg-quick"), [])

    def test_local_down_refuses_an_unverified_interface(self):
        self.write_state(wrong_key=True)
        with self.assertRaises(RuntimeError):
            mesh.local_down(self.args)
        self.assertEqual(self.mutations(), [])
        self.assertEqual(self.calls("wg-quick"), [])

    def test_non_macos_teardown_remains_wg_quick_only(self):
        with patch.object(mesh.sys, "platform", "linux"):
            mesh.local_down(self.args)
        self.assertEqual(self.calls("ifconfig"), [])
        self.assertEqual(self.calls("route"), [])
        self.assertEqual(
            self.calls("wg-quick"),
            [["down", str(self.root / "wireguard/controller/wg-mesh.conf")]],
        )
