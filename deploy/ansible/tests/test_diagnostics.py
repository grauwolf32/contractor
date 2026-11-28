"""Capture real Ansible callbacks, including verbose transport and old/new diffs.

All credentials/values are synthetic. Child receipts and unit content are read
privately by the test and are never printed as diagnostics.
"""

from __future__ import annotations

import getpass
import grp
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ANSIBLE = Path(__file__).resolve().parents[1]
ROLE = ANSIBLE / "roles/contractor_runtime"
HOST_VALUE = "synthetic-v416-015-host-value"
INSTANCE_VALUE = "synthetic-v416-015-instance-value"
OLD_VALUE = "synthetic-v416-015-old-unit-value"
ARG_VALUE = "synthetic-v416-015-argument-value"
VALUES = [HOST_VALUE, INSTANCE_VALUE, OLD_VALUE, ARG_VALUE]


def include(stage):
    return {
        "name": f"Exercise production {stage}",
        "ansible.builtin.include_role": {
            "name": "contractor_runtime",
            "tasks_from": stage,
            "public": True,
        },
    }


def private_fact(**facts):
    return {
        "name": "Prepare private test facts",
        "ansible.builtin.set_fact": facts,
        "no_log": True,
    }


def production_task(file, name):
    tasks = yaml.safe_load((ROLE / "tasks" / file).read_text())
    selected = [task for task in tasks if task["name"] == name]
    if len(selected) != 1:
        raise RuntimeError(
            f"production task selection must match exactly once: {file}: {name}"
        )
    return selected[0]


class Diagnostics(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="contractor-private-diagnostics-"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prefix = self.root / "prefix"
        self.home = self.root / "home"
        self.config = self.root / "config"
        self.tools = self.prefix / "tools"
        self.tools.mkdir(parents=True)
        (self.home / ".config/systemd/user").mkdir(parents=True)
        self.instance_root = self.home / "instances/safe-a"
        self.instance_root.mkdir(parents=True, mode=0o700)
        (self.instance_root / "work").mkdir(mode=0o700)
        (self.config / "pki").mkdir(parents=True)
        for name in ["ca.crt", "safe-a.crt", "safe-a.key"]:
            path = self.config / "pki" / name
            path.write_text("synthetic test-only certificate/key fixture")
            path.chmod(0o600)
        shutil.copyfile(
            ROLE / "files/private_command.py", self.tools / "private_command.py"
        )
        self.fakebin = self.root / "bin"
        self.fakebin.mkdir()
        self.variables = {
            "ansible_system": "Linux",
            "ansible_service_mgr": "systemd",
            "ansible_os_family": "Debian",
            "ansible_architecture": "x86_64",
            "ansible_python_interpreter": sys.executable,
            "ansible_facts": {"python": {"executable": sys.executable}},
            "contractor_runtime_control_plane_url": "https://127.0.0.1:8443",
            "contractor_runtime_prefix": str(self.prefix),
            "contractor_runtime_home": str(self.home),
            "contractor_runtime_config_dir": str(self.config),
            "contractor_runtime_user": getpass.getuser(),
            "contractor_runtime_group": grp.getgrgid(os.getgid()).gr_name,
            "contractor_runtime_capabilities": {
                "workspace_storage": "none",
                "openapi_validator": False,
                "likec4_validator": False,
                "scanners": [],
                "podman": False,
                "podman_network": False,
            },
            "contractor_runtime_instances": [
                {
                    "name": "safe-a",
                    "port": 9443,
                    "extra_environment": {"SYN_INSTANCE": INSTANCE_VALUE},
                }
            ],
            "contractor_runtime_extra_environment": {"SYN_HOST": HOST_VALUE},
            "__contractor_runtime_python": sys.executable,
            "__contractor_runtime_executable": sys.executable,
            "__contractor_runtime_tool_paths": {},
            "__contractor_runtime_podman_image": "test-only-unused",
        }

    def run_play(self, tasks, *, verbose=False):
        play = [
            {
                "name": "Private diagnostic boundary regression",
                "hosts": "localhost",
                "gather_facts": False,
                "vars": self.variables,
                "tasks": tasks,
            }
        ]
        path = self.root / "play.json"
        path.write_text(json.dumps(play))
        result = subprocess.run(
            [
                "uvx",
                "--from",
                "ansible-core==2.21.5",
                "ansible-playbook",
                "-i",
                "localhost,",
                "-c",
                "local",
                *(["-vvv", "--diff"] if verbose else []),
                str(path),
            ],
            cwd=ANSIBLE,
            env={
                **os.environ,
                "ANSIBLE_CONFIG": str(ANSIBLE / "ansible.cfg"),
                "ANSIBLE_NOCOLOR": "1",
                "PATH": str(self.fakebin) + os.pathsep + os.environ["PATH"],
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        return result.returncode, result.stdout + result.stderr

    def assert_private(self, output):
        for sentinel in VALUES:
            self.assertNotIn(sentinel, output)

    def resolved(self):
        return [
            include("resolve"),
            production_task("main.yml", "Complete instance settings with host values"),
            private_fact(
                __contractor_runtime_instance="{{ __contractor_runtime_ready | first }}"
            ),
        ]

    def script(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!{sys.executable}\n" + text)
        path.chmod(0o755)

    def semgrep_install_tasks(self):
        tasks = yaml.safe_load((ROLE / "tasks/semgrep.yml").read_text())
        # Exercise the actual production modules and templates in an owned
        # directory. Only Linux root ownership is adapted to the local user.
        def local_ownership(values):
            for task in values:
                for module in ["ansible.builtin.file", "ansible.builtin.copy"]:
                    if module in task:
                        args = task[module]
                        args["owner"] = getpass.getuser()
                        args["group"] = grp.getgrgid(os.getgid()).gr_name
                        if "src" in args:
                            args["src"] = str(ROLE / "files") + "/" + args["src"]
                for key in ["block", "rescue"]:
                    local_ownership(task.get(key, []))
        local_ownership(tasks)
        return tasks

    def semgrep_install_fixture(self, exit_code=0):
        destination = self.tools / "semgrep-1.187.0"
        self.variables.update({
            "__contractor_runtime_semgrep_dir": str(destination),
            "__contractor_runtime_tools": {
                "uv": {"version": "fixture", "path": "uv"},
                "semgrep": {"version": "1.187.0"},
            },
            "contractor_runtime_python_version": "3.13",
            "contractor_runtime_uv_environment": {"SYN_HOST": HOST_VALUE},
        })
        receipt = self.root / "semgrep-builds"
        self.script(self.tools / "uv-fixture/uv",
            "import sys\nfrom pathlib import Path\n"
            + "assert '--locked' in sys.argv and '--compile-bytecode' in sys.argv\n"
            + "root=Path(sys.argv[sys.argv.index('--project')+1])\n"
            + "assert (root/'pyproject.toml').is_file() and (root/'uv.lock').is_file()\n"
            + f"with Path({str(receipt)!r}).open('a') as out: out.write('compiled\\n')\n"
            + f"print({HOST_VALUE!r})\nsys.exit({exit_code})\n")
        return destination, receipt

    def test_semgrep_installer_preserves_lockfiles_and_builds_once(self):
        destination, receipt = self.semgrep_install_fixture()
        destination.mkdir()
        (destination / ".complete").write_text("old installation\n")
        tasks = self.semgrep_install_tasks()
        for _ in range(2):
            code, output = self.run_play(tasks, verbose=True)
            self.assertEqual(code, 0, output)
            self.assert_private(output)
            for name in ["pyproject.toml", "uv.lock"]:
                self.assertEqual((destination / name).read_bytes(),
                                 (ROLE / "files/semgrep" / name).read_bytes())
            self.assertEqual((destination / ".bytecode-complete").read_text(), "1.187.0\n")
        self.assertEqual(receipt.read_text(), "compiled\n")

    def test_semgrep_failed_build_does_not_mark_completion(self):
        destination, receipt = self.semgrep_install_fixture(exit_code=7)
        code, output = self.run_play(self.semgrep_install_tasks(), verbose=True)
        self.assertNotEqual(code, 0)
        self.assert_private(output)
        self.assertIn("Semgrep environment build failed", output)
        self.assertEqual(receipt.read_text(), "compiled\n")
        self.assertFalse((destination / ".bytecode-complete").exists())

    def test_invalid_environment_value_is_classified_without_value(self):
        self.variables["contractor_runtime_extra_environment"]["SYN_HOST"] = (
            HOST_VALUE + " invalid space"
        )
        code, output = self.run_play([include("resolve")])
        self.assertNotEqual(code, 0)
        self.assert_private(output)
        self.assertIn("safe-a: environment.SYN_HOST/unit_literal", output)

    def test_unrelated_argument_and_capability_failures_do_not_dump_environment(self):
        for field, value, category in [
            ("extra_args", [ARG_VALUE + " invalid space"], "args["),
            ("capabilities", {"podman": True}, "capabilities/combination"),
        ]:
            with self.subTest(field=field):
                entry = self.variables["contractor_runtime_instances"][0]
                entry[field] = value
                code, output = self.run_play([include("resolve")])
                self.assertNotEqual(code, 0)
                self.assert_private(output)
                self.assertIn(category, output)
                del entry[field]

    def test_malformed_raw_entries_and_newline_values_are_private_at_verbose_level(
        self,
    ):
        for entry in [
            {"extra_environment": {"SECRET": INSTANCE_VALUE}},
            {
                "name": "safe-a",
                "state": "absent",
                "extra_environment": {"SECRET": INSTANCE_VALUE},
                "port": ARG_VALUE,
                "state_invalid": True,
            },
            {
                "name": "safe-a",
                "port": 9443,
                "extra_environment": {"SECRET": INSTANCE_VALUE + "\nline"},
            },
        ]:
            with self.subTest(shape=list(entry)):
                self.variables["contractor_runtime_instances"] = [entry]
                code, output = self.run_play([include("resolve")], verbose=True)
                self.assertNotEqual(code, 0)
                self.assert_private(output)
                self.assertIn("Runtime configuration rejected", output)

    def test_malformed_field_types_cannot_escape_through_template_errors(self):
        for field in [
            "extra_environment",
            "capabilities",
            "extra_args",
            "workspace_limits",
            "podman_limits",
        ]:
            with self.subTest(field=field):
                entry = self.variables["contractor_runtime_instances"][0]
                original = entry.copy()
                entry[field] = INSTANCE_VALUE
                code, output = self.run_play([include("resolve")], verbose=True)
                self.assertNotEqual(code, 0)
                self.assert_private(output)
                self.assertIn(field + "/type", output)
                self.variables["contractor_runtime_instances"][0] = original
        self.variables["contractor_runtime_log_level"] = HOST_VALUE
        code, output = self.run_play([include("resolve")], verbose=True)
        self.assertNotEqual(code, 0)
        self.assert_private(output)
        self.assertIn("log_level/verification", output)

    def test_probe_valid_json_cannot_echo_unexpected_capability(self):
        self.script(
            self.tools / "probe_tools.py",
            "import json\n"
            + f"print(json.dumps({{'openapi@1': [], 'likec4@1': [], 'scan@1': [{INSTANCE_VALUE!r}]}}))\n",
        )
        code, output = self.run_play(
            self.resolved() + [include("preflight")], verbose=True
        )
        self.assertNotEqual(code, 0)
        self.assert_private(output)
        self.assertIn("tools/capability_probe", output)

    def test_parser_rejection_hides_unknown_argument_payload(self):
        tasks = self.resolved() + [
            private_fact(
                __contractor_runtime_instance="{{ __contractor_runtime_instance | combine({'argv': ['unused', '--unknown-argument', '"
                + ARG_VALUE
                + "']}) }}"
            ),
            include("preflight"),
        ]
        code, output = self.run_play(tasks, verbose=True)
        self.assertNotEqual(code, 0)
        self.assert_private(output)
        self.assertIn("args/runtime_parser", output)

    def test_probe_failure_and_malformed_json_are_private(self):
        for exit_code in [0, 1]:
            with self.subTest(exit_code=exit_code):
                self.script(
                    self.tools / "probe_tools.py",
                    f"import os, sys\nprint(os.environ['SYN_HOST'])\nprint(os.environ['SYN_INSTANCE'], file=sys.stderr)\nsys.exit({exit_code})\n",
                )
                code, output = self.run_play(
                    self.resolved() + [include("preflight")], verbose=True
                )
                self.assertNotEqual(code, 0)
                self.assert_private(output)
                self.assertIn("tools/capability_probe", output)

    def test_successful_probe_receives_private_environment(self):
        receipt = self.root / "probe-receipt.json"
        self.script(
            self.tools / "probe_tools.py",
            "import json, os\nfrom pathlib import Path\n"
            + f"Path({str(receipt)!r}).write_text(json.dumps({{name: os.environ[name] for name in ['SYN_HOST', 'SYN_INSTANCE']}}))\n"
            + "print(json.dumps({'openapi@1': [], 'likec4@1': [], 'scan@1': []}))\n",
        )
        code, output = self.run_play(
            self.resolved() + [include("preflight")], verbose=True
        )
        self.assertEqual(code, 0, output)
        self.assert_private(output)
        self.assertEqual(
            json.loads(receipt.read_text()),
            {"SYN_HOST": HOST_VALUE, "SYN_INSTANCE": INSTANCE_VALUE},
        )

    def test_uv_environment_success_and_failure_are_private(self):
        self.variables.update(
            {
                "__contractor_runtime_tools": {
                    "uv": {"version": "fixture", "path": "uv"}
                },
                "__contractor_runtime_release_dir": str(self.root / "release"),
                "contractor_runtime_uv_environment": {"SYN_HOST": HOST_VALUE},
            }
        )
        receipt = self.root / "uv-receipt"
        for exit_code in [0, 1]:
            with self.subTest(exit_code=exit_code):
                self.script(
                    self.tools / "uv-fixture/uv",
                    "import os, sys\nfrom pathlib import Path\n"
                    + f"Path({str(receipt)!r}).write_text(os.environ['SYN_HOST'])\n"
                    + f"print(os.environ['SYN_HOST'])\nsys.exit({exit_code})\n",
                )
                code, output = self.run_play(
                    [include("build_environment")], verbose=True
                )
                self.assertEqual(code == 0, exit_code == 0, output)
                self.assert_private(output)
                self.assertEqual(receipt.read_text(), HOST_VALUE)
                if exit_code:
                    self.assertIn("environment/uv_sync", output)

    def test_scope_parser_failure_hides_stdin_and_output(self):
        self.script(
            self.tools / "podman_scope_policy.py",
            "import sys\nprint(sys.stdin.read())\nsys.exit(1)\n",
        )
        tasks = self.resolved() + [
            private_fact(__contractor_runtime_ready=[{"argv": ["unused", ARG_VALUE]}]),
            include("scope_resolution"),
        ]
        code, output = self.run_play(tasks, verbose=True)
        self.assertNotEqual(code, 0)
        self.assert_private(output)
        self.assertIn("Podman scope/limits", output)

    def test_unit_old_and_new_diff_stays_private_and_restart_truth_survives(self):
        unit = self.home / ".config/systemd/user/contractor-runtime-safe-a.service"
        unit.write_text("[Service]\nEnvironment=OLD=" + OLD_VALUE + "\n")
        flags = {"changed": False}
        self.variables.update(
            {
                "__contractor_runtime_ca": flags,
                "__contractor_runtime_certificate_file": flags,
                "__contractor_runtime_private_key_file": flags,
            }
        )
        record = production_task(
            "instance.yml",
            "Record a needed restart of {{ __contractor_runtime_instance.name }}",
        )
        tasks = self.resolved() + [
            include("unit"),
            record,
            {
                "name": "Require restart after changed unit",
                "ansible.builtin.assert": {
                    "that": "__contractor_runtime_changed == ['safe-a']"
                },
            },
            private_fact(__contractor_runtime_changed=[]),
            include("unit"),
            record,
            {
                "name": "Require no restart after unchanged unit",
                "ansible.builtin.assert": {
                    "that": "__contractor_runtime_changed == []"
                },
            },
        ]
        code, output = self.run_play(tasks, verbose=True)
        self.assertEqual(code, 0, output)
        self.assert_private(output)
        contents = unit.read_text()
        self.assertIn(HOST_VALUE, contents)
        self.assertIn(INSTANCE_VALUE, contents)
        self.assertNotIn(OLD_VALUE, contents)

    def test_outer_instance_include_failure_does_not_serialize_raw_loop_item(self):
        self.variables.update(
            {
                "contractor_runtime_pki_issue": False,
                "contractor_runtime_pki_root": str(self.root / "missing-pki"),
                "contractor_runtime_pki_host": "localhost",
            }
        )
        configure = production_task("main.yml", "Configure instances")
        configure["ansible.builtin.include_tasks"] = str(ROLE / "tasks/instance.yml")
        # The include is the production full configure loop. Its certificate
        # read fails on purpose before any privileged host mutation.
        code, output = self.run_play(self.resolved() + [configure], verbose=True)
        self.assertNotEqual(code, 0)
        self.assert_private(output)

    def test_journal_failures_and_unknown_probe_payloads_are_classified(self):
        self.variables.update(
            {
                "__contractor_runtime_since": 0,
                "__contractor_runtime_uid": os.getuid(),
                "contractor_runtime_verify_timeout_seconds": 1,
                "__contractor_runtime_systemctl": [
                    sys.executable,
                    str(self.fakebin / "state"),
                ],
            }
        )
        self.script(self.fakebin / "state", "print('active')\n")
        for records, category in [
            (
                [{"message": "runtime agent failed: " + HOST_VALUE}],
                "registration/runtime_failed",
            ),
            (
                [
                    {"message": "runtime agent registered"},
                    {
                        "capabilityRef": "code-analysis@1",
                        "probeOutcome": INSTANCE_VALUE,
                    },
                ],
                "registration/missing_capabilities",
            ),
            (
                [{"message": "runtime agent registered"}, "{" + ARG_VALUE],
                "registration/malformed_events",
            ),
        ]:
            with self.subTest(category=category):
                lines = "\n".join(
                    record
                    if isinstance(record, str)
                    else json.dumps(record, separators=(",", ":"))
                    for record in records
                )
                self.script(self.fakebin / "journalctl", f"print({lines!r})\n")
                code, output = self.run_play(
                    self.resolved() + [include("verify")], verbose=True
                )
                self.assertNotEqual(code, 0)
                self.assert_private(output)
                self.assertIn(category, output)

    def test_latest_probe_outcome_controls_registration_in_both_orders(self):
        self.variables.update(
            {
                "__contractor_runtime_since": 0,
                "__contractor_runtime_uid": os.getuid(),
                "contractor_runtime_verify_timeout_seconds": 1,
                "__contractor_runtime_systemctl": [
                    sys.executable,
                    str(self.fakebin / "state"),
                ],
            }
        )
        self.script(self.fakebin / "state", "print('active')\n")
        for outcomes in [
            ["available", "invalid_result"],
            ["invalid_result", "available"],
        ]:
            with self.subTest(outcomes=outcomes):
                events = [
                    {
                        "capabilityRef": "code-analysis@1",
                        "probeOutcome": outcome,
                        "private": HOST_VALUE,
                    }
                    for outcome in outcomes
                ] + [
                    {
                        "capabilityRef": "taint-annotations@1",
                        "probeOutcome": "available",
                    },
                    {"message": "runtime agent registered"},
                ]
                lines = "\n".join(
                    json.dumps(event, separators=(",", ":")) for event in events
                )
                self.script(self.fakebin / "journalctl", f"print({lines!r})\n")
                code, output = self.run_play(
                    self.resolved() + [include("verify")], verbose=True
                )
                self.assertEqual(code == 0, outcomes[-1] == "available", output)
                self.assert_private(output)
                if outcomes[-1] != "available":
                    self.assertIn(
                        "registration/missing_capabilities code-analysis@1", output
                    )

    def test_registration_timeout_withholds_journal_values(self):
        self.variables.update(
            {
                "__contractor_runtime_since": 0,
                "__contractor_runtime_uid": os.getuid(),
                "contractor_runtime_verify_timeout_seconds": 1,
                "__contractor_runtime_systemctl": [
                    sys.executable,
                    str(self.fakebin / "state"),
                ],
            }
        )
        self.script(self.fakebin / "state", "print('active')\n")
        self.script(self.fakebin / "journalctl", f"print({HOST_VALUE!r})\n")
        code, output = self.run_play(
            self.resolved() + [include("verify")], verbose=True
        )
        self.assertNotEqual(code, 0)
        self.assert_private(output)
        self.assertIn("registration/", output)

    def test_invalid_uv_environment_shape_does_not_echo_value(self):
        self.variables["contractor_runtime_uv_environment"] = HOST_VALUE
        code, output = self.run_play([include("build_environment")], verbose=True)
        self.assertNotEqual(code, 0)
        self.assert_private(output)
        self.assertIn("environment/uv_sync", output)

    def test_registration_success_still_requires_every_expected_capability(self):
        self.variables.update(
            {
                "__contractor_runtime_since": 0,
                "__contractor_runtime_uid": os.getuid(),
                "contractor_runtime_verify_timeout_seconds": 1,
                "__contractor_runtime_systemctl": [
                    sys.executable,
                    str(self.fakebin / "state"),
                ],
            }
        )
        self.script(self.fakebin / "state", "print('active')\n")
        events = [{"message": "runtime agent registered", "private": HOST_VALUE}] + [
            {"capabilityRef": ref, "probeOutcome": "available"}
            for ref in ["code-analysis@1", "taint-annotations@1"]
        ]
        lines = "\n".join(json.dumps(event, separators=(",", ":")) for event in events)
        self.script(self.fakebin / "journalctl", f"print({lines!r})\n")
        code, output = self.run_play(
            self.resolved() + [include("verify")], verbose=True
        )
        self.assertEqual(code, 0, output)
        self.assert_private(output)
        self.assertIn("registered with code-analysis@1, taint-annotations@1", output)


if __name__ == "__main__":
    unittest.main()
