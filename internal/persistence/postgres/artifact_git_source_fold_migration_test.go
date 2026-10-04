package postgres

import (
	"context"
	"slices"
	"testing"
	"time"
)

func TestPostgresArtifactGitSourcesFoldMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 88)

	_, err := pool.Exec(ctx, `
INSERT INTO artifact_blobs (sha256, size_bytes, payload)
VALUES (sha256('source'::bytea), 6, 'source'::bytea);
INSERT INTO artifact_versions (version_id, blob_sha256, media_type, created_at)
VALUES ('imported', sha256('source'::bytea), 'application/zip', now()),
       ('written', sha256('source'::bytea), 'application/zip', now());
INSERT INTO artifact_git_sources (version_id, repository_url, requested_ref, resolved_commit, imported_at)
VALUES ('imported', 'https://example.test/repo.git', 'main', repeat('a',40), '2026-10-01T00:00:00Z');
`)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 89) {
		t.Fatalf("fold artifact Git sources = %+v, %v", result, err)
	}
	var importedURL, writtenURL *string
	var sourcesGone bool
	if err := pool.QueryRow(ctx, `
SELECT imported.git_repository_url, written.git_repository_url, to_regclass('artifact_git_sources') IS NULL
FROM artifact_versions AS imported, artifact_versions AS written
WHERE imported.version_id = 'imported' AND written.version_id = 'written'`,
	).Scan(&importedURL, &writtenURL, &sourcesGone); err != nil {
		t.Fatal(err)
	}
	if importedURL == nil || *importedURL != "https://example.test/repo.git" || writtenURL != nil || !sourcesGone {
		t.Fatalf("folded origins = (%v, %v), table dropped %t", importedURL, writtenURL, sourcesGone)
	}
	for _, statement := range []string{
		`UPDATE artifact_versions SET git_resolved_commit = repeat('b',40) WHERE version_id = 'imported'`,
		`UPDATE artifact_versions SET media_type = 'text/plain' WHERE version_id = 'written'`,
		`DELETE FROM artifact_versions WHERE version_id = 'written'`,
	} {
		if _, err := pool.Exec(ctx, statement); err == nil {
			t.Fatalf("immutable version changed by %s", statement)
		}
	}
	if _, err := pool.Exec(ctx, `
UPDATE artifact_versions
SET git_repository_url = 'https://example.test/other.git', git_resolved_commit = repeat('c',40), git_imported_at = now()
WHERE version_id = 'written'`); err != nil {
		t.Fatalf("first Git origin rejected: %v", err)
	}
}
