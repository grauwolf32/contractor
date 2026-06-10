import json
from unittest.mock import AsyncMock

import pytest

from contractor.runners.artifacts import (
    ARTIFACT_KINDS,
    InvalidArtifactKeyError,
    _records_to_text,
    artifact_filename,
    artifact_key_slug,
    artifact_names_for_key,
    save_result_artifacts,
    validate_artifact_key,
)


class TestValidateArtifactKey:
    def test_strips_surrounding_whitespace_and_slashes(self):
        assert validate_artifact_key("  /a/b/  ") == "a/b"

    def test_returns_cleaned_key(self):
        assert validate_artifact_key("plan/v1") == "plan/v1"

    @pytest.mark.parametrize("value", ["", "   ", "/", "//", "  /  "])
    def test_rejects_empty(self, value):
        with pytest.raises(InvalidArtifactKeyError, match="must not be empty"):
            validate_artifact_key(value)

    @pytest.mark.parametrize(
        "value",
        [
            "..",
            "../a",
            "a/..",
            "a/../b",
            "a/b/../c",
        ],
    )
    def test_rejects_path_traversal(self, value):
        with pytest.raises(InvalidArtifactKeyError, match="path traversal"):
            validate_artifact_key(value)

    def test_does_not_reject_dotdot_inside_segment(self):
        # The check is on path *segments*, not substrings — `a..b` is one segment.
        assert validate_artifact_key("a..b") == "a..b"
        assert validate_artifact_key("foo/bar..baz") == "foo/bar..baz"


class TestArtifactKeySlug:
    def test_keeps_safe_chars(self):
        assert artifact_key_slug("sqli-list_2") == "sqli-list_2"

    def test_collapses_unsafe_runs_to_single_underscore(self):
        assert artifact_key_slug("trace-annotation:openapi:items") == (
            "trace-annotation_openapi_items"
        )
        assert artifact_key_slug("a / b") == "a_b"

    def test_stable_and_key_safe(self):
        slug = artifact_key_slug("../weird name!")
        assert slug == artifact_key_slug("../weird name!")
        # The slug is a single segment that passes key validation.
        assert validate_artifact_key(f"t/{slug}") == f"t/{slug}"

    @pytest.mark.parametrize("value", ["", "   ", "::", "__"])
    def test_degenerate_inputs_fall_back(self, value):
        assert artifact_key_slug(value) == "item"


class TestArtifactFilename:
    def test_appends_kind_to_cleaned_key(self):
        assert artifact_filename("plan", "result") == "plan/result"
        assert artifact_filename("plan", "summary") == "plan/summary"
        assert artifact_filename("plan", "records") == "plan/records"

    def test_cleans_key(self):
        assert artifact_filename("/plan/", "result") == "plan/result"

    def test_rejects_invalid_key(self):
        with pytest.raises(InvalidArtifactKeyError):
            artifact_filename("..", "result")


class TestArtifactNamesForKey:
    def test_returns_all_three_kinds(self):
        names = artifact_names_for_key("plan")
        assert names == {
            "result": "plan/result",
            "summary": "plan/summary",
            "records": "plan/records",
        }

    def test_kinds_cover_constant(self):
        names = artifact_names_for_key("anything")
        assert set(names.keys()) == set(ARTIFACT_KINDS)


class TestRecordsToText:
    def test_str_passthrough(self):
        assert _records_to_text("already a string") == "already a string"

    def test_empty_string_passthrough(self):
        # Strings are not JSON-encoded even when empty.
        assert _records_to_text("") == ""

    def test_none_becomes_empty_json_list(self):
        assert _records_to_text(None) == "[]"

    def test_list_serialized(self):
        assert _records_to_text([{"a": 1}, {"b": 2}]) == '[{"a": 1}, {"b": 2}]'

    def test_dict_serialized(self):
        assert _records_to_text({"key": "value"}) == '{"key": "value"}'

    def test_unicode_preserved(self):
        # ensure_ascii=False means non-ASCII chars are not escaped.
        assert _records_to_text(["café"]) == '["café"]'


class TestSaveResultArtifacts:
    @pytest.mark.asyncio
    async def test_saves_all_three_kinds(self):
        svc = AsyncMock()
        names = await save_result_artifacts(
            artifact_service=svc,
            app_name="app",
            user_id="u",
            key="plan/v1",
            result="R",
            summary="S",
            records=[{"id": 1}],
        )

        assert names == {
            "result": "plan/v1/result",
            "summary": "plan/v1/summary",
            "records": "plan/v1/records",
        }
        assert svc.save_artifact.await_count == 3

        saved = {
            call.kwargs["filename"]: call.kwargs["artifact"]
            for call in svc.save_artifact.await_args_list
        }
        assert saved["plan/v1/result"].text == "R"
        assert saved["plan/v1/summary"].text == "S"
        assert saved["plan/v1/records"].text == '[{"id": 1}]'

    @pytest.mark.asyncio
    async def test_defaults_are_empty_strings(self):
        svc = AsyncMock()
        await save_result_artifacts(
            artifact_service=svc,
            app_name="app",
            user_id="u",
            key="plan",
        )
        saved = {
            call.kwargs["filename"]: call.kwargs["artifact"].text
            for call in svc.save_artifact.await_args_list
        }
        assert saved["plan/result"] == ""
        assert saved["plan/summary"] == ""
        assert saved["plan/records"] == "[]"

    @pytest.mark.asyncio
    async def test_none_records_becomes_empty_list_json(self):
        svc = AsyncMock()
        await save_result_artifacts(
            artifact_service=svc,
            app_name="app",
            user_id="u",
            key="plan",
            records=None,
        )
        saved = {
            call.kwargs["filename"]: call.kwargs["artifact"].text
            for call in svc.save_artifact.await_args_list
        }
        assert json.loads(saved["plan/records"]) == []

    @pytest.mark.asyncio
    async def test_invalid_key_raises_before_any_save(self):
        svc = AsyncMock()
        with pytest.raises(InvalidArtifactKeyError):
            await save_result_artifacts(
                artifact_service=svc,
                app_name="app",
                user_id="u",
                key="../escape",
                result="R",
            )
        svc.save_artifact.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_session_id_is_none(self):
        # `session_id=None` is a contract requirement so artifacts are user-scoped,
        # not session-scoped. Regression guard.
        svc = AsyncMock()
        await save_result_artifacts(
            artifact_service=svc,
            app_name="app",
            user_id="u",
            key="plan",
        )
        for call in svc.save_artifact.await_args_list:
            assert call.kwargs["session_id"] is None
