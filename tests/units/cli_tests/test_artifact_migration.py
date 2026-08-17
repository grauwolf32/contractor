from pathlib import Path

from contractor.utils.adk_artifacts import migrate_legacy_artifact_layout


def test_migrates_legacy_adk_artifacts_without_removing_source(tmp_path: Path):
    legacy_payload = (
        tmp_path
        / "users"
        / "cli-user"
        / "artifacts"
        / "report"
        / "versions"
        / "0"
        / "report"
    )
    legacy_payload.parent.mkdir(parents=True)
    legacy_payload.write_text("finding", encoding="utf-8")

    assert migrate_legacy_artifact_layout(tmp_path, app_name="contractor") is True

    migrated_payload = (
        tmp_path
        / "apps"
        / "contractor"
        / "users"
        / legacy_payload.relative_to(tmp_path / "users")
    )
    assert migrated_payload.read_text(encoding="utf-8") == "finding"
    assert legacy_payload.read_text(encoding="utf-8") == "finding"


def test_does_not_merge_legacy_artifacts_into_existing_scoped_store(
    tmp_path: Path,
):
    (tmp_path / "users").mkdir()
    scoped_users = tmp_path / "apps" / "contractor" / "users"
    scoped_users.mkdir(parents=True)
    marker = scoped_users / "keep"
    marker.write_text("new-layout", encoding="utf-8")

    assert migrate_legacy_artifact_layout(tmp_path, app_name="contractor") is False
    assert marker.read_text(encoding="utf-8") == "new-layout"


def test_noop_when_legacy_layout_is_absent(tmp_path: Path):
    assert migrate_legacy_artifact_layout(tmp_path, app_name="contractor") is False
