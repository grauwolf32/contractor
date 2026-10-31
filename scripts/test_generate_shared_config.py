"""Prove that changing one policy value updates bindings without rewriting history."""

import copy
import json
import os
import shutil
import subprocess
import uuid
import tempfile
import unittest
from pathlib import Path

import generate_shared_config as generator


def fixture_root(root):
    config = json.loads((generator.ROOT / "runtime/tests/testdata/shared-config.json").read_text())
    for name in [
        *generator.schema_bindings(),
        *(name for files in generator.input_bindings().values() for name in files),
        "api/testdata/public/utf8-byte-bound-cases.json",
        "api/testdata/v1alpha1/run-metadata-label-cases.json",
        "api/testdata/v1alpha1/placement-label-cases.json",
        *(name for files in generator.worker_bindings().values() for name in files),
        "api/testdata/v1alpha1/valid/artifact-metadata-result-boundary.json",
        "api/testdata/v1alpha1/artifact-history-cases.json",
        "api/testdata/http-capture-cases.json",
        "api/v1alpha1/shared-config.schema.json",
        "api/testdata/v1alpha1/artifact-list-cases.json",
        "api/testdata/v1alpha1/worker-result-size-cases.json",
        "api/testdata/v1alpha1/valid/worker-completion-empty-observations.json",
    ]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((generator.ROOT / name).read_text())
    migrations = root / "internal/persistence/migrations"
    migrations.mkdir(parents=True)
    baseline = migrations / "000001_baseline.sql"
    historical = "CONSTRAINT artifact_blobs_size_check CHECK (size_bytes <= 42);\n"
    historical += f"CONSTRAINT artifact_versions_git_repository_url_check CHECK (octet_length(git_repository_url) BETWEEN 1 AND {3 * config['limits']['gitImportUrlMaxBytes']});\n"
    sql = (
        generator.ROOT / "internal/persistence/migrations/000001_baseline.sql"
    ).read_text()
    historical += (
        generator.metadata_routine_with_limits(
            generator.re.search(
                r"CREATE(?: OR REPLACE)? FUNCTION contractor_valid_run_metadata_labels[\s\S]*?^END;",
                sql,
                generator.re.MULTILINE,
            )[0],
            config["limits"],
        )
        + "\n"
    )
    baseline.write_text(historical)
    return config, baseline, historical


class SharedConfigGenerationTests(unittest.TestCase):
    def test_changed_source_updates_bindings_and_emits_one_incremental_migration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, baseline, historical = fixture_root(root)
            migrations = root / "internal/persistence/migrations"
            changed = copy.deepcopy(config)
            changed["limits"]["artifactPayloadMaxBytes"] = 43
            changed["limits"]["httpRequestBodyMaxBytes"] = 11
            changed["limits"]["httpResponseBodyMaxBytes"] = 19
            changed["limits"]["httpHeaderBlockMaxBytes"] = 13
            changed["limits"]["workerResultMaxBytes"] = 1000
            changed["limits"]["workerCompletionMaxBytes"] = 8000
            changed["limits"]["workerFailureMessageMaxBytes"] = 23
            changed["limits"]["publicPageMaxItems"] = 7
            changed["limits"]["artifactListMaxItems"] = 11
            with self.assertRaisesRegex(ValueError, "stale shared config"):
                generator.generate(root, changed, check=True)
            generator.generate(root, changed)
            generator.generate(root, changed, check=True)
            self.assertEqual(baseline.read_text(), historical)
            self.assertEqual(len(list(migrations.glob("*.sql"))), 2)
            self.assertIn(
                "size_bytes <= 43",
                (migrations / "000002_shared_artifact_payload_limit.sql").read_text(),
            )
            public = generator.yaml.safe_load((root / generator.PUBLIC).read_text())
            self.assertEqual(
                public["components"]["schemas"]["ArtifactMetadata"]["properties"]["size"][
                    "maximum"
                ],
                43,
            )
            self.assertEqual(public["components"]["parameters"]["Limit"]["schema"]["maximum"], 7)
            self.assertEqual(public["components"]["parameters"]["Limit"]["schema"]["default"], 7)
            self.assertEqual(
                public["components"]["schemas"]["ProjectPage"]["properties"]["items"]["maxItems"], 7
            )
            self.assertEqual(
                public["components"]["schemas"]["FindingProposalReceipt"],
                generator.yaml.safe_load((generator.ROOT / generator.PUBLIC).read_text())[
                    "components"
                ]["schemas"]["FindingProposalReceipt"],
            )
            self.assertEqual(
                json.loads((root / "api/testdata/v1alpha1/artifact-list-cases.json").read_text())[
                    "maxLimit"
                ],
                11,
            )
            http = json.loads((root / "api/testdata/http-capture-cases.json").read_text())["limits"]
            self.assertEqual(
                [http["requestBodyBytes"], http["responseBodyBytes"], http["headerBytes"]],
                [11, 19, 13],
            )
            worker = json.loads((root / "api/v1alpha1/worker-completion.schema.json").read_text())
            self.assertEqual(
                worker["$defs"]["workerFailure"]["properties"]["message"]["x-maxBytes"], 23
            )
            cases = json.loads(
                (root / "api/testdata/v1alpha1/worker-result-size-cases.json").read_text()
            )
            self.assertEqual(next(c for c in cases if c["name"] == "text_at_limit")["repeat"], 1000)
            self.assertEqual(
                next(c for c in cases if c["name"] == "envelope_at_limit")["json_bytes"], 8000
            )
            # Equal-valued telemetry is a different policy, not an Artifact binding.
            original = generator.yaml.safe_load((generator.ROOT / generator.PUBLIC).read_text())
            self.assertEqual(
                public["components"]["schemas"]["WorkerTelemetryExportConfig"],
                original["components"]["schemas"]["WorkerTelemetryExportConfig"],
            )
            self.assertIn(
                "MAX_ARTIFACT_BYTES = 43",
                (root / "runtime/tests/shared_config_defaults.py").read_text(),
            )
            changed["limits"]["artifactPayloadMaxBytes"] = 44
            generator.generate(root, changed)
            self.assertEqual(len(list(migrations.glob("*.sql"))), 3)

    def test_input_policy_updates_schemas_and_sequential_incremental_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, baseline, historical = fixture_root(root)
            migrations = root / "internal/persistence/migrations"
            limits = config["limits"]
            limits["artifactPayloadMaxBytes"] = 42
            for key in (
                "gitImportUrlMaxBytes",
                "runMetadataMaxLabels",
                "runMetadataKeyMaxBytes",
                "runMetadataValueMaxBytes",
                "auditMaxLabels",
            ):
                limits[key] += 1
            generator.generate(root, config)
            generator.generate(root, config, check=True)
            migrations = root / "internal/persistence/migrations"
            self.assertEqual(baseline.read_text(), historical)
            self.assertEqual(len(list(migrations.glob("*.sql"))), 3)
            git = (migrations / "000002_shared_git_source_limit.sql").read_text()
            labels = (migrations / "000003_shared_run_metadata_limits.sql").read_text()
            self.assertIn(f"AND {3 * limits['gitImportUrlMaxBytes']}", git)
            self.assertIn(f"<= {limits['runMetadataMaxLabels']}", labels)
            self.assertIn(f"BETWEEN 1 AND {limits['runMetadataKeyMaxBytes']}", labels)
            self.assertIn(f"> {limits['runMetadataValueMaxBytes']}", labels)
            self.assertIn("ADD CONSTRAINT workflow_runs_metadata_labels_check", labels)
            public = generator.yaml.safe_load((root / generator.PUBLIC).read_text())
            self.assertEqual(
                public["components"]["schemas"]["GitSource"]["properties"]["repositoryUrl"][
                    "maxLength"
                ],
                3 * limits["gitImportUrlMaxBytes"],
            )
            self.assertEqual(
                public["components"]["schemas"]["AuditLabels"]["allOf"][1]["maxProperties"],
                limits["auditMaxLabels"],
            )
            private = json.loads((root / "api/v1alpha1/allocation.schema.json").read_text())
            self.assertEqual(
                private["$defs"]["runMetadataLabels"]["maxProperties"],
                limits["runMetadataMaxLabels"],
            )
            limits["gitImportUrlMaxBytes"] += 1
            generator.generate(root, config)
            self.assertEqual(len(list(migrations.glob("*.sql"))), 4)

    @unittest.skipUnless(os.environ.get("CONTRACTOR_TEST_DATABASE_URL"), "PostgreSQL URL is unset")
    def test_generated_input_checks_execute_and_reject_a_decrease_on_postgres(self):
        client = shutil.which("psql")
        self.assertIsNotNone(client, "psql is required for configured PostgreSQL verification")
        schema = "shared_policy_" + uuid.uuid4().hex
        database = os.environ["CONTRACTOR_TEST_DATABASE_URL"]

        def execute(sql, succeeds=True):
            # Do not let a subprocess exception expose a connection URL or environment value.
            result = subprocess.run(
                [client, "-X", "-q", "-v", "ON_ERROR_STOP=1", "--dbname", database],
                input=sql,
                text=True,
                capture_output=True,
            )
            if succeeds:
                self.assertEqual(
                    result.returncode, 0, "generated PostgreSQL policy verification failed"
                )
            else:
                self.assertTrue(
                    result.returncode != 0 and "check constraint" in result.stderr,
                    "expected a PostgreSQL CHECK rejection",
                )

        def json_sql(value):
            return "'" + json.dumps(value).replace("'", "''") + "'::jsonb"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _, historical = fixture_root(root)
            limits = config["limits"]
            routine = historical[historical.index("CREATE") :]
            old_git = 3 * limits["gitImportUrlMaxBytes"]
            old_count = limits["runMetadataMaxLabels"]
            setup = f"CREATE SCHEMA {schema}; SET search_path TO {schema};\n" + routine
            setup += f"CREATE TABLE artifact_versions (git_repository_url text CONSTRAINT artifact_versions_git_repository_url_check CHECK (octet_length(git_repository_url) BETWEEN 1 AND {old_git}));\n"
            setup += "CREATE TABLE workflow_runs (metadata_labels jsonb CONSTRAINT workflow_runs_metadata_labels_check CHECK (contractor_valid_run_metadata_labels(metadata_labels)));\n"
            setup += "INSERT INTO artifact_versions VALUES ('x'); INSERT INTO workflow_runs VALUES ('{}'::jsonb);\n"
            execute("BEGIN; " + setup + "COMMIT;")
            try:
                for key in (
                    "gitImportUrlMaxBytes",
                    "runMetadataMaxLabels",
                    "runMetadataKeyMaxBytes",
                    "runMetadataValueMaxBytes",
                ):
                    limits[key] += 1
                generated = {}
                generator.append_input_migrations(root, limits, generated)
                self.assertEqual(len(generated), 2)
                for name, sql in generated.items():
                    path = root / name
                    path.write_text(sql)
                growth = "\n".join(generated.values())
                prefix = f"BEGIN; SET search_path TO {schema};\n"
                execute(prefix + growth + "COMMIT;")
                boundary = {f"k{i}": "" for i in range(limits["runMetadataMaxLabels"])}
                oversized = {**boundary, "overflow": ""}
                key_value_boundary = {
                    "a" * limits["runMetadataKeyMaxBytes"]: "x" * limits["runMetadataValueMaxBytes"]
                }
                check = f"INSERT INTO artifact_versions VALUES (repeat('x', {3 * limits['gitImportUrlMaxBytes']}));\n"
                check += f"INSERT INTO workflow_runs VALUES ({json_sql(boundary)});\n"
                check += f"DO $$ BEGIN IF NOT contractor_valid_run_metadata_labels({json_sql(key_value_boundary)}) THEN RAISE EXCEPTION 'byte boundary rejected'; END IF; END $$;\n"
                execute(prefix + check + "COMMIT;")
                execute(
                    prefix + f"INSERT INTO workflow_runs VALUES ({json_sql(oversized)}); COMMIT;",
                    succeeds=False,
                )
                execute(
                    prefix
                    + f"INSERT INTO artifact_versions VALUES (repeat('x', {3 * limits['gitImportUrlMaxBytes'] + 1})); COMMIT;",
                    succeeds=False,
                )
                limits["runMetadataMaxLabels"] = old_count
                decrease = {}
                generator.append_input_migrations(root, limits, decrease)
                self.assertEqual(len(decrease), 1)
                execute(prefix + "\n".join(decrease.values()) + "COMMIT;", succeeds=False)
                execute(
                    prefix
                    + f"DO $$ BEGIN IF NOT contractor_valid_run_metadata_labels({json_sql(boundary)}) THEN RAISE EXCEPTION 'failed migration changed policy'; END IF; END $$; COMMIT;"
                )
            finally:
                execute(f"DROP SCHEMA {schema} CASCADE;")


if __name__ == "__main__":
    unittest.main()
