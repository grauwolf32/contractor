"""Settings hygiene tests.

Covers the ``target_url`` / ``proxy`` fields (routed from the historical
``CONTRACTOR_TARGET_URL`` / ``CONTRACTOR_PROXY`` env vars) and the anchored
``cli/.env`` discovery, which must work from non-CLI entrypoints too.
"""

from __future__ import annotations

from contractor.utils import settings as settings_module
from contractor.utils.settings import Settings


class TestTargetSettings:
    def test_default_to_none_without_env(self, monkeypatch):
        monkeypatch.delenv("CONTRACTOR_TARGET_URL", raising=False)
        monkeypatch.delenv("CONTRACTOR_PROXY", raising=False)
        s = Settings(_env_file=None)
        assert s.target_url is None
        assert s.proxy is None

    def test_contractor_env_vars_route_to_fields(self, monkeypatch):
        # The pre-Settings callsites read CONTRACTOR_TARGET_URL / CONTRACTOR_PROXY
        # via os.environ — the aliases must keep those exact names working.
        monkeypatch.setenv("CONTRACTOR_TARGET_URL", "http://localhost:5002")
        monkeypatch.setenv("CONTRACTOR_PROXY", "http://127.0.0.1:8888")
        s = Settings(_env_file=None)
        assert s.target_url == "http://localhost:5002"
        assert s.proxy == "http://127.0.0.1:8888"

    def test_constructible_by_field_name(self, monkeypatch):
        # populate_by_name lets tests/programmatic callers bypass the alias.
        monkeypatch.delenv("CONTRACTOR_TARGET_URL", raising=False)
        monkeypatch.delenv("CONTRACTOR_PROXY", raising=False)
        s = Settings(_env_file=None, target_url="http://t", proxy="http://p")
        assert s.target_url == "http://t"
        assert s.proxy == "http://p"


class TestEnvFileAnchor:
    def test_cli_env_file_is_anchored_to_repo_cli_dir(self):
        # The documented config file is `cli/.env` next to the CLI entrypoint;
        # the anchor must resolve there regardless of the process CWD.
        env_file = settings_module._CLI_ENV_FILE
        assert env_file.name == ".env"
        assert env_file.parent.name == "cli"
        assert (env_file.parents[1] / "pyproject.toml").is_file()

    def test_settings_env_file_sources_include_anchor(self):
        env_files = Settings.model_config["env_file"]
        assert settings_module._CLI_ENV_FILE in tuple(env_files)
        # CWD-relative .env stays as the (higher-precedence) fallback.
        assert ".env" in tuple(env_files)
