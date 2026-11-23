"""Behavioral maintenance checks, included in make lint's unittest discovery."""

import contextlib
import io
import os
import secrets
import selectors
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import test_storage as storage


class CacheTests(unittest.TestCase):
    def test_preview_then_oldest_first_pruning_retains_recent_and_unknown_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "aa"
            shard.mkdir()
            old = shard / ("aa" + "0" * 62 + "-d")
            middle = shard / ("aa" + "1" * 62 + "-d")
            recent = shard / ("aa" + "2" * 62 + "-d")
            unknown = shard / "unrecognized-d"
            for path, used in ((old, 10), (middle, 20), (recent, 1000), (unknown, 1)):
                path.write_bytes(b"x" * 10)
                os.utime(path, (used, used))
            symlink = shard / ("aa" + "3" * 62 + "-d")
            symlink.symlink_to(unknown)
            entries = storage.cache_entries(root)
            self.assertEqual(len(entries), 3)
            self.assertEqual(storage.prune_cache(entries, 20, 100, False), (10, 20))
            self.assertTrue(old.exists())
            self.assertEqual(storage.prune_cache(entries, 20, 100, True), (10, 20))
            self.assertFalse(old.exists())
            self.assertTrue(middle.exists())
            self.assertEqual(
                storage.prune_cache(storage.cache_entries(root), 0, 100, True), (10, 10)
            )
            self.assertTrue(recent.exists())
            self.assertTrue(unknown.exists())
            self.assertTrue(symlink.is_symlink())

    def test_changed_entries_and_symlink_shards_are_not_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "aa"
            shard.mkdir()
            path = shard / ("a" * 64 + "-d")
            path.write_bytes(b"old")
            entries = storage.cache_entries(root)
            path.write_bytes(b"new and in use")
            self.assertEqual(
                storage.prune_cache(entries, 0, float("inf"), True), (0, 3)
            )
            self.assertTrue(path.exists())
            (root / "bb").symlink_to(shard, target_is_directory=True)
            self.assertEqual(len(storage.cache_entries(root)), 1)

    def test_cached_executable_is_removed_without_recursive_unknown_layout_deletion(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "aa"
            shard.mkdir()
            executable = shard / ("a" * 64 + "-d")
            executable.mkdir()
            (executable / "tool").write_bytes(b"executable")
            unexpected = shard / ("aa" + "1" * 62 + "-d")
            unexpected.mkdir()
            (unexpected / "nested").mkdir()
            entries = storage.cache_entries(root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(
                storage.prune_cache(entries, 0, float("inf"), True), (10, 0)
            )
            self.assertFalse(executable.exists())
            self.assertTrue((unexpected / "nested").is_dir())

    def test_cache_inventory_limit_fails_before_any_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "aa"
            shard.mkdir()
            for number in range(5):
                (shard / ("aa" + f"{number:062x}" + "-d")).write_bytes(b"x")
            with (
                mock.patch.object(storage, "MAX_CACHE_ENTRIES", 4),
                self.assertRaisesRegex(RuntimeError, "inventory exceeds"),
            ):
                storage.cache_entries(root)
            self.assertEqual(len(list(shard.iterdir())), 5)


class CommandTests(unittest.TestCase):
    def test_subprocess_diagnostics_do_not_print_credentials(self):
        result = subprocess.CompletedProcess(
            [], 1, "", "postgresql://user:private@host/db"
        )
        with (
            mock.patch.object(storage.subprocess, "run", return_value=result),
            self.assertRaisesRegex(RuntimeError, "psql failed") as failure,
        ):
            storage.run(["psql"])
        self.assertNotIn("private", str(failure.exception))

    def test_report_without_database_and_prune_cli_default_to_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "aa").mkdir()
            entry = root / "aa" / ("a" * 64 + "-d")
            entry.write_bytes(b"payload")
            os.utime(entry, (1, 1))
            with (
                mock.patch.object(storage, "cache_root", return_value=root),
                mock.patch.dict(os.environ, {"CONTRACTOR_TEST_DATABASE_URL": ""}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(storage.main(["report"]), 0)
                self.assertEqual(storage.main(["prune-cache", "--max-gib", "0"]), 0)
                self.assertTrue(entry.exists())
                self.assertEqual(
                    storage.main(["prune-cache", "--max-gib", "0", "--apply"]), 0
                )
                self.assertFalse(entry.exists())

    def test_nonfinite_or_negative_pruning_options_fail(self):
        with (
            mock.patch.object(storage, "cache_root"),
            mock.patch.object(storage, "cache_entries", return_value=[]),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for option, value in (
                ("--max-gib", "nan"),
                ("--max-gib", "-1"),
                ("--min-age-hours", "0"),
            ):
                self.assertEqual(storage.main(["prune-cache", option, value]), 1)

    def test_known_test_names_are_eligible_but_not_protected_or_active_databases(self):
        for name in (
            "contractor_test",
            "contractor_test_lane",
            "contractor_v64026_cb11b933",
            "contractor_runservice_123456789abc",
        ):
            storage.require_disposable(
                {"name": name, "template": False, "current": False, "sessions": 0}
            )
        for name, template, current, sessions in (
            ("contractor_demo", False, False, 0),
            ("postgres", False, False, 0),
            ("contractor_test_lane", True, False, 0),
            ("contractor_test_lane", False, True, 0),
            ("contractor_test_lane", False, False, 1),
            ('contractor_test_x"; DROP DATABASE postgres;--', False, False, 0),
        ):
            with self.assertRaises(RuntimeError):
                storage.require_disposable(
                    {
                        "name": name,
                        "template": template,
                        "current": current,
                        "sessions": sessions,
                    }
                )


@unittest.skipUnless(
    os.environ.get("CONTRACTOR_TEST_STORAGE_PODMAN_IMAGE"),
    "Podman storage fixture image is unset",
)
class PodmanTests(unittest.TestCase):
    def test_real_preview_active_protection_and_exact_test_resource_cleanup(self):
        name = "contractor-test-storage-" + secrets.token_hex(8)
        image = os.environ["CONTRACTOR_TEST_STORAGE_PODMAN_IMAGE"]
        storage.run(["podman", "volume", "create", name])
        try:
            storage.run(["podman", "network", "create", name])
            storage.run(
                [
                    "podman",
                    "create",
                    "--pull=never",
                    "--name",
                    name,
                    "--network",
                    "none",
                    "--volume",
                    name + ":/fixture",
                    image,
                    "/bin/sleep",
                    "300",
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                storage.podman_clean("container", [name], False)
            self.assertEqual(
                storage.podman_select("container", [name])[0]["State"], "created"
            )
            with self.assertRaisesRegex(RuntimeError, "still attached"):
                storage.podman_clean("volume", [name], True)
            with self.assertRaisesRegex(RuntimeError, "protected Podman"):
                storage.podman_clean("container", ["contractor-litellm-postgres"], True)
            storage.run(["podman", "start", name])
            with self.assertRaisesRegex(RuntimeError, "container is active"):
                storage.podman_clean("container", [name], True)
            container = next(
                row
                for row in storage.podman_inventory("container")
                if name in row["Names"]
            )
            with self.assertRaisesRegex(RuntimeError, "missing, tagged or used"):
                storage.podman_clean("image", [container["ImageID"]], True)
            storage.run(["podman", "stop", "--time", "1", name])
            # Commit one private fixture image and untag it to exercise actual
            # dangling image removal without touching any preinstalled image.
            temporary_tag = "localhost/" + name
            committed = storage.run(["podman", "commit", name, temporary_tag]).strip()
            try:
                storage.run(["podman", "untag", temporary_tag])
                with contextlib.redirect_stdout(io.StringIO()):
                    storage.podman_clean("image", [committed], False)
                    storage.podman_clean("image", [committed], True)
            finally:
                storage.run(
                    ["podman", "image", "rm", "--ignore", "--no-prune", committed]
                )
            with contextlib.redirect_stdout(io.StringIO()):
                storage.podman_clean("container", [name], True)
                storage.podman_clean("volume", [name], True)
                storage.podman_clean("network", [name], True)
            self.assertFalse(
                any(
                    name in row["Names"]
                    for row in storage.podman_inventory("container")
                )
            )
            self.assertFalse(
                any(name == row["Name"] for row in storage.podman_inventory("volume"))
            )
            self.assertFalse(
                any(name == row["name"] for row in storage.podman_inventory("network"))
            )
        finally:
            # Only this test's exact resources may be stopped/removed.
            storage.run(["podman", "stop", "--ignore", "--time", "1", name])
            storage.run(["podman", "rm", "--ignore", "--volumes", name])
            if any(row["Name"] == name for row in storage.podman_inventory("volume")):
                storage.run(["podman", "volume", "rm", name])
            storage.run(["podman", "network", "rm", "--ignore", name])


@unittest.skipUnless(
    os.environ.get("CONTRACTOR_TEST_DATABASE_URL"), "test database URL is unset"
)
class PostgresTests(unittest.TestCase):
    def test_real_database_preview_active_identity_and_explicit_drop(self):
        postgres = storage.Postgres()
        name = "contractor_test_storage_" + secrets.token_hex(8)
        postgres.sql(f'CREATE DATABASE "{name}" TEMPLATE template0;')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(storage.main(["drop-databases", name]), 0)
            expected = postgres.inventory()[name]
            self.assertGreater(expected["bytes"], 0)
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                postgres.drop({**expected, "oid": expected["oid"] + 1})
            # A real psql connection stays open on stdin until this context exits.
            with subprocess.Popen(
                postgres.command,
                env=postgres.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ) as session:
                try:
                    session.stdin.write(f"\\connect {name}\nSELECT 1;\n".encode())
                    session.stdin.flush()
                    with selectors.DefaultSelector() as selector:
                        selector.register(session.stdout, selectors.EVENT_READ)
                        self.assertTrue(
                            selector.select(timeout=10),
                            "psql did not open fixture connection",
                        )
                    self.assertEqual(session.stdout.readline().strip(), b"1")
                    with self.assertRaisesRegex(RuntimeError, "active connections"):
                        postgres.drop(expected)
                    # Even bypassing the precheck, the SQL never forcibly disconnects a session.
                    with self.assertRaisesRegex(RuntimeError, "psql failed"):
                        postgres.sql(f'DROP DATABASE "{name}";')
                finally:
                    session.stdin.close()
                    session.wait(timeout=10)
            self.assertIn(name, postgres.inventory())
            with (
                mock.patch.object(storage, "MAX_DATABASES", 1),
                self.assertRaisesRegex(RuntimeError, "inventory exceeds"),
            ):
                postgres.inventory()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(storage.main(["drop-databases", name, "--apply"]), 0)
            self.assertNotIn(name, postgres.inventory())
        finally:
            # Only this test's exact, randomly named fixture is ever removed.
            postgres.sql(f'DROP DATABASE IF EXISTS "{name}";')

    def test_configured_unreachable_database_fails(self):
        with (
            mock.patch.dict(
                os.environ,
                {
                    "CONTRACTOR_TEST_DATABASE_URL": "postgresql://127.0.0.1:1/unavailable?connect_timeout=1"
                },
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                storage.main(["drop-databases", "contractor_test_storage_unavailable"]),
                1,
            )


if __name__ == "__main__":
    unittest.main()
