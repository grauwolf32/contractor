"""Prove that changing one policy value updates bindings without rewriting history."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import generate_shared_config as generator


class SharedConfigGenerationTests(unittest.TestCase):
    def test_changed_source_updates_bindings_and_emits_one_incremental_migration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in [
                *generator.schema_bindings(),
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
            baseline.write_text(historical)
            config = json.loads(
                (generator.ROOT / "runtime/tests/testdata/shared-config.json").read_text()
            )
            changed = copy.deepcopy(config)
            changed["limits"]["artifactPayloadMaxBytes"] = 43
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


if __name__ == "__main__":
    unittest.main()
