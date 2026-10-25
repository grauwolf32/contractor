"""Bounded ownership checks and real localhost Ansible pruning regressions."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ANSIBLE = Path(__file__).resolve().parents[1]
SOURCE = ANSIBLE / "roles/contractor_runtime/files/release_pruning.py"
SPEC = importlib.util.spec_from_file_location("release_pruning", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("release ownership checker is missing")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="contractor-pruning-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prefix = self.root / "prefix"
        self.home = self.root / "home"
        self.config_dir = self.root / "config"
        self.units = self.home / ".config/systemd/user"
        self.units.mkdir(parents=True)
        (self.prefix / "tools").mkdir(parents=True)
        (self.config_dir / "restart-pending").mkdir(parents=True)
        self.old = self.release("a")
        self.new = self.release("b")
        self.extra = self.release("c")
        self.config = {
            "prefix": str(self.prefix),
            "home": str(self.home),
            "config_dir": str(self.config_dir),
            "current": str(self.new),
            "keep": 1,
            "instances": ["a"],
            "systemctl": ["systemctl"],
        }
        self.unit("a", self.new)

    def release(self, char):
        release = self.prefix / "releases" / (char * 40)
        executable = release / "runtime/.venv/bin/contractor-runtime"
        executable.parent.mkdir(parents=True)
        executable.write_text("test-only executable")
        return release

    def unit(self, name, release):
        unit = self.units / f"contractor-runtime-{name}.service"
        unit.write_text(
            f"[Service]\nExecStart={release}/runtime/.venv/bin/contractor-runtime --port 9443\n"
        )
        return unit

    def plan(self, **kwargs):
        return MODULE.plan(self.config, state_reader=lambda *_: 0, **kwargs)


class Ownership(ReleaseFixture):
    def test_partial_inventory_refuses_before_selection(self):
        self.unit("b", self.old)
        with self.assertRaisesRegex(MODULE.UnsafePruning, "inventory omits"):
            self.plan()
        self.assertTrue(self.old.exists())

    def test_old_running_process_is_protected_even_after_unit_rewrite(self):
        proc = self.root / "proc"
        (proc / "42").mkdir(parents=True)
        (proc / "42/cmdline").write_bytes(
            os.fsencode(self.old / "runtime/.venv/bin/contractor-runtime") + b"\0"
        )
        result = MODULE.plan(self.config, state_reader=lambda *_: 42, proc=proc)
        self.assertEqual(result["remove"], [str(self.extra)])

    def test_stopped_unit_argument_and_environment_references_are_protected(self):
        unit = self.units / "contractor-runtime-a.service"
        for content in [
            f" --ca-file={self.old}/ca.pem",
            f"\nEnvironment=PYTHONPATH={self.old}/modules",
        ]:
            with self.subTest(content=content):
                self.unit("a", self.new)
                unit.write_text(unit.read_text().rstrip() + content + "\n")
                self.assertEqual(self.plan()["remove"], [str(self.extra)])

    def test_unidentifiable_or_disappeared_process_refuses(self):
        proc = self.root / "proc"
        (proc / "42").mkdir(parents=True)
        (proc / "42/cmdline").write_bytes(b"unowned-process\0")
        with self.assertRaisesRegex(MODULE.UnsafePruning, "no identifiable"):
            MODULE.plan(self.config, state_reader=lambda *_: 42, proc=proc)
        (proc / "42/cmdline").unlink()
        with self.assertRaises(OSError):
            MODULE.plan(self.config, state_reader=lambda *_: 42, proc=proc)

    def test_pending_marker_defers_all_cleanup(self):
        (self.config_dir / "restart-pending/a").touch()
        self.assertEqual(self.plan(), {"remove": [], "deferred": True})

    def test_protected_references_count_towards_total_retention(self):
        self.unit("b", self.old)
        self.config["instances"].append("b")
        for keep in [1, 2]:
            with self.subTest(keep=keep):
                self.config["keep"] = keep
                self.assertEqual(self.plan()["remove"], [str(self.extra)])
        self.config["keep"] = 3
        self.assertEqual(self.plan()["remove"], [])

    def test_unknown_unit_shape_and_dropin_refuse(self):
        for suffix, content in [
            ("service", "[Service]\nExecStart=/unknown\n"),
            ("service.d", ""),
        ]:
            with self.subTest(suffix=suffix):
                path = self.units / f"contractor-runtime-b.{suffix}"
                path.write_text(content)
                self.config["instances"] = ["a", "b"]
                with self.assertRaises(MODULE.UnsafePruning):
                    self.plan()
                path.unlink()

    def test_symlink_or_missing_reference_refuses(self):
        path = self.units / "contractor-runtime-a.service"
        path.unlink()
        path.symlink_to(self.root / "missing")
        with self.assertRaises(MODULE.UnsafePruning):
            self.plan()
        path.unlink()
        self.unit("a", self.prefix / "releases" / ("d" * 40))
        with self.assertRaisesRegex(MODULE.UnsafePruning, "missing release"):
            self.plan()

    def test_metadata_and_cardinality_limits_refuse(self):
        unit = self.units / "contractor-runtime-a.service"
        unit.write_bytes(b"x" * (MODULE.MAX_BYTES + 1))
        with self.assertRaisesRegex(MODULE.UnsafePruning, "byte bound"):
            self.plan()
        self.unit("a", self.new)
        with (
            patch.object(MODULE, "MAX_ENTRIES", 2),
            self.assertRaisesRegex(MODULE.UnsafePruning, "entry bound"),
        ):
            self.plan()
        self.unit("b", self.old)
        self.config["instances"].append("b")
        with (
            patch.object(MODULE, "MAX_UNITS", 1),
            self.assertRaisesRegex(MODULE.UnsafePruning, "unit bound"),
        ):
            self.plan()

    def test_actual_directory_and_unit_limits(self):
        bounded = self.root / "bounded"
        bounded.mkdir()
        for index in range(4097):
            (bounded / str(index)).touch()
        with self.assertRaisesRegex(MODULE.UnsafePruning, "entry bound"):
            MODULE.entries(bounded)
        for index in range(256):
            name = f"slot-{index}"
            self.unit(name, self.new)
            self.config["instances"].append(name)
        with self.assertRaisesRegex(MODULE.UnsafePruning, "unit bound"):
            self.plan()

    def test_malformed_marker_refuses(self):
        (self.config_dir / "restart-pending/a").write_text("unexpected content")
        with self.assertRaisesRegex(MODULE.UnsafePruning, "restart marker"):
            self.plan()

    def test_discovery_failure_timeout_and_cancellation_never_select_deletions(self):
        for error in [
            OSError("test-only unavailable manager"),
            subprocess.TimeoutExpired("test-only", 1),
            KeyboardInterrupt(),
        ]:
            with self.subTest(error=type(error).__name__):

                def fail(*_, failure=error):
                    raise failure

                with self.assertRaises(type(error)):
                    MODULE.plan(self.config, state_reader=fail)
                self.assertTrue(self.old.exists())
        with (
            patch.object(MODULE.time, "monotonic", side_effect=[0, 31]),
            self.assertRaisesRegex(MODULE.UnsafePruning, "deadline"),
        ):
            self.plan()

    def test_oversized_manager_streams_are_bounded_and_child_is_joined(self):
        pidfile = self.root / "manager.pid"
        for stream in ["stdout", "stderr"]:
            with self.subTest(stream=stream):
                script = (
                    "import os, sys, time; from pathlib import Path; "
                    + f"Path({str(pidfile)!r}).write_text(str(os.getpid())); "
                    + f"sys.{stream}.write('x' * 1000000); sys.{stream}.flush(); time.sleep(10)"
                )
                with self.assertRaisesRegex(MODULE.UnsafePruning, "byte bound"):
                    MODULE.manager_output([sys.executable, "-c", script], 5)
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pidfile.read_text()), 0)

    def test_real_manager_command_rejects_wrong_fragment_and_times_out(self):
        fake = self.root / "manager.py"
        fake.write_text("print('MainPID=0\\nFragmentPath=/unknown\\nDropInPaths=')\n")
        with self.assertRaisesRegex(MODULE.UnsafePruning, "unowned"):
            MODULE.manager_state(
                [sys.executable, str(fake)],
                self.units / "contractor-runtime-a.service",
                5,
            )
        fake.write_text("import time; time.sleep(10)\n")
        with self.assertRaises(subprocess.TimeoutExpired):
            MODULE.manager_state(
                [sys.executable, str(fake)],
                self.units / "contractor-runtime-a.service",
                0.01,
            )


class LocalhostRole(ReleaseFixture):
    def run_role(self):
        fake = self.root / "manager.py"
        fake.write_text(
            "import sys\nfrom pathlib import Path\n"
            + f"unit = Path({str(self.units)!r}) / sys.argv[2]\n"
            + "print(f'MainPID=0\\nFragmentPath={unit}\\nDropInPaths=')\n"
        )
        variables = {
            "contractor_runtime_prefix": str(self.prefix),
            "contractor_runtime_home": str(self.home),
            "contractor_runtime_config_dir": str(self.config_dir),
            "contractor_runtime_keep_releases": self.config["keep"],
            "__contractor_runtime_release_dir": str(self.new),
            "__contractor_runtime_python": sys.executable,
            "__contractor_runtime_ready": [
                {"name": name} for name in self.config["instances"]
            ],
            "__contractor_runtime_systemctl": [sys.executable, str(fake)],
            "__contractor_runtime_pending": [],
        }
        play = [
            {
                "name": "Disposable release ownership regression",
                "hosts": "localhost",
                "gather_facts": False,
                "vars": variables,
                "tasks": [
                    {
                        "name": "Run production pruning",
                        "ansible.builtin.include_role": {
                            "name": "contractor_runtime",
                            "tasks_from": os.environ.get("PRUNING_TASKS_FILE", "prune"),
                        },
                    }
                ],
            }
        ]
        if original := os.environ.get("PRUNING_TASKS_FILE"):
            play[0]["tasks"] = [
                {
                    "name": "Run reviewed pruning",
                    "ansible.builtin.include_tasks": original,
                }
            ]
        playbook = self.root / "play.json"
        playbook.write_text(json.dumps(play))
        return subprocess.run(
            [
                "uvx",
                "--from",
                "ansible-core==2.21.5",
                "ansible-playbook",
                "-i",
                "localhost,",
                "-c",
                "local",
                str(playbook),
            ],
            cwd=ANSIBLE,
            env={
                **os.environ,
                "ANSIBLE_CONFIG": str(ANSIBLE / "ansible.cfg"),
                "ANSIBLE_NOCOLOR": "1",
            },
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )

    def test_omitted_stopped_unit_keeps_its_executable(self):
        self.unit("b", self.old)
        result = self.run_role()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("inventory omits", result.stdout)
        self.assertTrue((self.old / "runtime/.venv/bin/contractor-runtime").is_file())
        self.assertTrue((self.units / "contractor-runtime-b.service").is_file())
        self.assertTrue(self.extra.is_dir())  # refusal precedes ALL deletions

    def test_deferred_rewritten_unit_preserves_previous_release(self):
        self.unit("b", self.new)
        self.config["instances"].append("b")
        (self.config_dir / "restart-pending/b").touch()
        result = self.run_role()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.old.is_dir())
        self.assertIn("cleanup deferred", result.stdout)

    def test_complete_inventory_protects_old_unit_and_cleanup_is_idempotent(self):
        self.unit("b", self.old)
        self.config["instances"].append("b")
        for _ in range(2):
            result = self.run_role()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(self.old.is_dir())
            self.assertTrue(self.new.is_dir())
            self.assertFalse(self.extra.exists())
        # The production absent step removes the unit before pruning. Its data
        # and keys are outside the release tree and remain after retirement.
        (self.units / "contractor-runtime-b.service").unlink()
        self.config["instances"].remove("b")
        retained = self.home / "b-key"
        retained.write_text("synthetic retained key")
        result = self.run_role()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.old.exists())
        self.assertTrue(retained.is_file())


if __name__ == "__main__":
    unittest.main()
