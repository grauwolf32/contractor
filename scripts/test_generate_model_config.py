"""Model configuration tests use synthetic sources, never the shipped configs tree."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from generate_model_config import ConfigError, MAX_BYTES, ROLES, generate, render


class ModelConfigurationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.large = {
            "model": "openrouter/example/large",
            "apiBase": "https://example.com/v1",
            "apiKeyEnv": "EXAMPLE_API_KEY",
            "contextWindowTokens": 1000000,
            "maxOutputTokens": 128000,
        }
        self.small = dict(
            self.large,
            model="other/example-small",
            apiBase="https://other.example.com/v1",
            apiKeyEnv="OTHER_API_KEY",
            contextWindowTokens=118000,
            maxOutputTokens=64000,
        )
        self.budgets = {
            "planner": {
                "policyVersion": "2",
                "gatewayAlias": "planner-model",
                "maxOutputTokens": 16384,
                "maxModelCalls": 200,
                "maxWorkerCalls": 200,
                "maxTotalTokens": 2500000,
                "temperature": 0.6,
            },
            "worker": {
                "policyVersion": "2",
                "gatewayAlias": "worker-model",
                "maxOutputTokens": 32768,
                "maxModelCalls": 200,
                "maxToolCalls": 200,
                "maxTotalTokens": 25000000,
                "temperature": 0.6,
            },
            "summarizer": {
                "policyVersion": "1",
                "gatewayAlias": "summarizer-model",
                "maxOutputTokens": 8192,
                "maxModelCalls": 1,
                "temperature": 0.6,
            },
            "audit_completion_worker": {
                "policyVersion": "1",
                "gatewayAlias": "audit-completion-model",
                "maxOutputTokens": 4096,
                "maxModelCalls": 16,
                "maxToolCalls": 32,
                "maxTotalTokens": 25000000,
                "temperature": 0.6,
            },
        }
        self.write("deploy/models/catalog/large.yaml", self.large)
        self.write("deploy/models/catalog/small.yaml", self.small)
        self.write("deploy/models/roles.yaml", self.budgets)
        self.select()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False))
        return path

    def select(self, default="large", **roles):
        self.write("deploy/models/selection.yaml", {"defaultModel": default, "roles": roles})

    def test_default_routes_context_and_role_budgets_are_compiled_together(self):
        output = render(self.root)
        self.assertEqual(len(output), 14)
        routes = yaml.safe_load(output["deploy/litellm/litellm_config.yaml"])["model_list"]
        for role in ROLES:
            policy = yaml.safe_load(output[f"configs/model-policies/{role}.yaml"])
            route = next(
                route for route in routes if route["model_name"] == policy["spec"]["model"]
            )
            budget = self.budgets[role]
            self.assertEqual(policy["metadata"], {"name": role, "version": budget["policyVersion"]})
            self.assertEqual(policy["spec"]["contextWindowTokens"], 1000000)
            self.assertEqual(policy["spec"]["model"], route["model_name"])
            self.assertEqual(route["litellm_params"]["model"], self.large["model"])
            self.assertEqual(route["litellm_params"]["api_key"], "os.environ/EXAMPLE_API_KEY")
            self.assertEqual(
                {
                    k: v
                    for k, v in policy["spec"].items()
                    if k not in {"model", "contextWindowTokens"}
                },
                {k: v for k, v in budget.items() if k not in {"policyVersion", "gatewayAlias"}},
            )

    def test_each_role_can_change_model_without_changing_others_or_budgets(self):
        baseline = render(self.root)
        for index, role in enumerate(ROLES):
            with self.subTest(role=role):
                self.select(**{role: "small"})
                output = render(self.root)
                changed = {name for name in output if output[name] != baseline[name]}
                self.assertEqual(
                    changed, {f"configs/model-policies/{role}.yaml", "configs/model-selection.json"}
                )
                policy = yaml.safe_load(output[f"configs/model-policies/{role}.yaml"])
                self.assertEqual(policy["spec"]["contextWindowTokens"], 118000)
                original = yaml.safe_load(baseline[f"configs/model-policies/{role}.yaml"])
                policy["spec"]["contextWindowTokens"] = 1000000
                alias = policy["spec"]["model"]
                policy["spec"]["model"] = original["spec"]["model"]
                self.assertEqual(policy, original)
                self.assertNotEqual(alias, original["spec"]["model"])
                routes = yaml.safe_load(output["deploy/litellm/litellm_config.yaml"])["model_list"]
                route = next(route for route in routes if route["model_name"] == alias)
                self.assertEqual(route["litellm_params"]["model"], self.small["model"])
                self.assertEqual(route["litellm_params"]["api_key"], "os.environ/OTHER_API_KEY")

    def test_default_change_updates_all_contexts_and_preserves_summary_and_spend(self):
        self.select("small")
        output = render(self.root)
        for role in ROLES:
            policy = yaml.safe_load(output[f"configs/model-policies/{role}.yaml"])["spec"]
            self.assertEqual(policy["contextWindowTokens"], 118000)
            self.assertEqual(policy.get("maxTotalTokens"), self.budgets[role].get("maxTotalTokens"))
        self.assertEqual(
            yaml.safe_load(output["configs/model-policies/summarizer.yaml"])["spec"][
                "maxModelCalls"
            ],
            1,
        )

    def test_generate_and_check_detect_drift_and_restore_it_without_touching_templates(self):
        template = self.write("configs/agent-templates/example.yaml", {"contextWindowRatio": 0.8})
        unchanged = template.read_bytes()
        self.assertEqual(len(generate(self.root)), 14)
        self.assertEqual(generate(self.root, check=True), [])
        target = self.root / "configs/model-policies/worker.yaml"
        target.write_text("drift\n")
        self.assertEqual(generate(self.root, check=True), ["configs/model-policies/worker.yaml"])
        self.assertEqual(target.read_text(), "drift\n")
        self.assertEqual(generate(self.root), ["configs/model-policies/worker.yaml"])
        self.assertEqual(template.read_bytes(), unchanged)

    def test_invalid_selection_or_budget_leaves_all_existing_outputs_untouched(self):
        generate(self.root)
        baseline = render(self.root)
        invalid = [
            ("deploy/models/selection.yaml", {"defaultModel": "missing", "roles": {}}),
            (
                "deploy/models/selection.yaml",
                {"defaultModel": "large", "roles": {"unknown": "small"}},
            ),
            ("deploy/models/selection.yaml", {"defaultModel": "large", "roles": {"worker": []}}),
        ]
        for field, value in (
            ("maxOutputTokens", 2000000),
            ("maxModelCalls", True),
            ("maxTotalTokens", 100000001),
            ("temperature", float("nan")),
            ("contextWindowTokens", 999),
        ):
            budgets = copy.deepcopy(self.budgets)
            budgets["worker"][field] = value
            invalid.append(("deploy/models/roles.yaml", budgets))
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                self.write(name, value)
                with self.assertRaises(ConfigError):
                    generate(self.root)
                for output, expected in baseline.items():
                    self.assertEqual((self.root / output).read_bytes(), expected)
                self.write("deploy/models/roles.yaml", self.budgets)
                self.select()

    def test_model_output_capacity_and_summary_contract_are_validated(self):
        self.write("deploy/models/catalog/large.yaml", dict(self.large, maxOutputTokens=16000))
        with self.assertRaisesRegex(ConfigError, "output capacity"):
            render(self.root)
        self.write("deploy/models/catalog/large.yaml", self.large)
        for change in ({"maxModelCalls": 2}, {"maxToolCalls": 1}, {"maxWorkerCalls": 1}):
            with self.subTest(change=change):
                budgets = copy.deepcopy(self.budgets)
                budgets["summarizer"].update(change)
                self.write("deploy/models/roles.yaml", budgets)
                with self.assertRaises(ConfigError):
                    render(self.root)

    def test_small_output_model_is_available_for_summary_without_reducing_worker_budgets(self):
        self.write("deploy/models/catalog/small.yaml", dict(self.small, maxOutputTokens=10000))
        self.select(summarizer="small")
        output = render(self.root)
        catalog = json.loads(output["configs/model-selection.json"])
        model = next(model for model in catalog["models"] if model["id"] == "small")
        self.assertEqual(set(model["policies"]), {"summarizer", "audit_completion_worker"})
        worker = yaml.safe_load(output["configs/model-policies/worker.yaml"])["spec"]
        summary = yaml.safe_load(output["configs/model-policies/summarizer.yaml"])["spec"]
        self.assertEqual(worker["maxOutputTokens"], 32768)
        self.assertEqual(worker["contextWindowTokens"], 1000000)
        self.assertEqual(summary["maxOutputTokens"], 8192)
        self.assertEqual(summary["contextWindowTokens"], 118000)
        self.select(worker="small")
        with self.assertRaisesRegex(ConfigError, "output capacity"):
            render(self.root)

    def test_profiles_reject_literal_credentials_duplicate_keys_and_yaml_indirection(self):
        path = self.root / "deploy/models/catalog/large.yaml"
        for suffix in (
            "apiKey: sk-test-secret\n",
            "model: duplicate\n",
            "unexpected: &anchor {}\n",
            "unexpected: [*missing]\n",
        ):
            with self.subTest(suffix=suffix):
                self.write("deploy/models/catalog/large.yaml", self.large)
                path.write_text(path.read_text() + suffix)
                with self.assertRaises(ConfigError):
                    render(self.root)
        for base in ("https://user:secret@example.com/v1", "https://example.com?key=secret"):
            self.write("deploy/models/catalog/large.yaml", dict(self.large, apiBase=base))
            with self.assertRaises(ConfigError):
                render(self.root)

    def test_cli_rejects_credentials_without_echoing_them_or_creating_outputs(self):
        self.write(
            "deploy/models/catalog/large.yaml", dict(self.large, apiKey="sk-test-do-not-log")
        )
        script = Path(__file__).with_name("generate_model_config.py")
        result = subprocess.run(
            [sys.executable, str(script), "--root", str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("sk-test-do-not-log", result.stdout + result.stderr)
        self.assertFalse((self.root / "configs").exists())

    def test_file_and_profile_count_bounds_accept_the_limit_and_reject_overflow(self):
        for index in range(30):
            self.write(f"deploy/models/catalog/fixture-{index}.yaml", self.small)
        self.assertEqual(len(render(self.root)), 134)
        self.write("deploy/models/catalog/overflow.yaml", self.small)
        with self.assertRaisesRegex(ConfigError, "32 profiles"):
            render(self.root)
        (self.root / "deploy/models/catalog/overflow.yaml").unlink()
        path = self.root / "deploy/models/catalog/large.yaml"
        path.write_bytes(path.read_bytes() + b"#" * MAX_BYTES)
        with self.assertRaisesRegex(ConfigError, "65536 bytes"):
            render(self.root)

    def test_symlink_target_is_rejected_before_any_other_file_changes(self):
        generate(self.root)
        before = (self.root / "configs/model-policies/planner.yaml").read_bytes()
        target = self.root / "deploy/litellm/litellm_config.yaml"
        target.unlink()
        outside = self.root / "outside.yaml"
        outside.write_text("preserve\n")
        target.symlink_to(outside)
        self.select("small")
        with self.assertRaisesRegex(ConfigError, "symlinks"):
            generate(self.root)
        self.assertEqual((self.root / "configs/model-policies/planner.yaml").read_bytes(), before)
        self.assertEqual(outside.read_text(), "preserve\n")

    def test_input_symlinks_and_duplicate_role_aliases_are_rejected(self):
        budgets = copy.deepcopy(self.budgets)
        budgets["summarizer"]["gatewayAlias"] = "worker-model"
        self.write("deploy/models/roles.yaml", budgets)
        with self.assertRaisesRegex(ConfigError, "distinct"):
            render(self.root)
        self.write("deploy/models/roles.yaml", self.budgets)
        source = self.root / "deploy/models/selection.yaml"
        other = self.root / "selection.yaml"
        source.rename(other)
        source.symlink_to(other)
        with self.assertRaisesRegex(ConfigError, "symlinks"):
            render(self.root)

    def test_selection_does_not_retarget_existing_routes_and_profile_edits_get_new_aliases(self):
        before = render(self.root)
        self.select("small")
        after = render(self.root)
        self.assertEqual(
            before["deploy/litellm/litellm_config.yaml"],
            after["deploy/litellm/litellm_config.yaml"],
        )
        catalog = json.loads(after["configs/model-selection.json"])
        self.assertEqual(len(catalog["models"]), 2)
        for model in catalog["models"]:
            self.assertEqual(set(model["policies"]), set(ROLES))
        generate(self.root)
        old_routes = {
            route["model_name"]
            for route in yaml.safe_load(after["deploy/litellm/litellm_config.yaml"])["model_list"]
        }
        self.write("deploy/models/catalog/large.yaml", dict(self.large, model="other/replacement"))
        changed = render(self.root)
        new_routes = {
            route["model_name"]
            for route in yaml.safe_load(changed["deploy/litellm/litellm_config.yaml"])["model_list"]
        }
        self.assertEqual(len(old_routes - new_routes), 4)
        self.assertEqual(len(new_routes - old_routes), 4)
        self.assertTrue(generate(self.root, check=True))
        generate(self.root)
        self.assertEqual(generate(self.root, check=True), [])
        self.assertEqual(len(list((self.root / "configs/model-policies/models").iterdir())), 8)

    def test_failed_replace_leaves_checkable_drift_and_no_temporary_files(self):
        generate(self.root)
        self.select("small")
        with mock.patch("generate_model_config.os.replace", side_effect=OSError("unavailable")):
            with self.assertRaises(OSError):
                generate(self.root)
        self.assertEqual(len(generate(self.root, check=True)), 5)
        self.assertEqual(len(list((self.root / "configs/model-policies").iterdir())), 5)
        generate(self.root)
        self.assertEqual(generate(self.root, check=True), [])


if __name__ == "__main__":
    unittest.main()
