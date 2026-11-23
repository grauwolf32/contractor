package postgres_test

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func TestSharedGitSourceURLBoundsCountUTF8Bytes(t *testing.T) {
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	pool := isolatedArtifactPool(t, t.Context())
	ctx := t.Context()
	artifactStorage := testBlobStorage(t, ctx, pool)
	scope, _ := UserScope("user-1")
	user, _ := NewService(artifactStorage.Repository(pool)).User("user-1")
	checked := false
	for _, field := range fields {
		if field.Rule != "LIM-GIT-24" {
			continue
		}
		checked = true
		if field.MaxBytes != contracts.MaxGitSourceURLBytes {
			t.Fatalf("Git source shared bound %d, Go bound %d", field.MaxBytes, contracts.MaxGitSourceURLBytes)
		}
		for index, value := range field.EmbeddedValues(characters, "https://example.test:443/", "") {
			for _, direct := range []bool{false, true} {
				name := fmt.Sprintf("git-url-%d-%t", index, direct)
				source := GitSource{RepositoryURL: value.Value, ResolvedCommit: strings.Repeat("a", 40), ImportedAt: time.Now().UTC().Truncate(time.Microsecond)}
				err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
					repo := artifactStorage.TxRepository(tx)
					store, _ := NewService(repo).User("user-1")
					written, err := store.Write(ctx, ArtifactRef{Namespace: "source", Name: name}, Payload{MediaType: "application/zip", Data: []byte("fixture")}, nil)
					if err != nil {
						return err
					}
					if !direct {
						return repo.RecordGitSource(ctx, scope, written.Ref, source)
					}
					// Bypass only the Go validation to exercise the real CHECK
					// independently, on a newly created version without provenance.
					_, err = tx.Exec(ctx, `UPDATE artifact_versions
SET git_repository_url=$1, git_resolved_commit=$2, git_imported_at=$3
WHERE version_id=(SELECT version_id FROM artifact_binding_revisions
WHERE scope_kind='user' AND scope_id='user-1' AND namespace='source' AND name=$4 AND revision=$5)`,
						source.RepositoryURL, source.ResolvedCommit, source.ImportedAt, name, *written.Ref.Revision)
					return err
				})
				if (err == nil) != value.Within {
					t.Fatalf("direct=%t %q URL bytes=%d: error=%v within=%t", direct, value.Character, len(value.Value), err, value.Within)
				}
				if !value.Within {
					if direct && (postgres.Classify(err) != postgres.ClassInvalid || postgres.ConstraintName(err) != "artifact_versions_git_repository_url_check") {
						t.Fatalf("expected URL CHECK rejection, got %v", err)
					}
					if !direct && !errors.Is(err, blob.ErrIntegrity) {
						t.Fatalf("expected artifact integrity rejection, got %v", err)
					}
				}
				metadata, readErr := user.Metadata(ctx, ArtifactRef{Namespace: "source", Name: name})
				if value.Within {
					if readErr != nil || metadata.GitSource == nil || !sameGitSource(metadata.GitSource, source) {
						t.Fatalf("Git URL did not round-trip: %v", readErr)
					}
				} else if !errors.Is(readErr, ErrArtifactNotFound) {
					t.Fatalf("rejected Git source published metadata: %v", readErr)
				}
			}
		}
	}
	if !checked {
		t.Fatal("shared Git provenance bound is missing")
	}
}

func TestPostgresGitProvenanceRetainsExactRunInputAndRollsBack(t *testing.T) {
	ctx := t.Context()
	pool := isolatedArtifactPool(t, ctx)
	artifactStorage := testBlobStorage(t, ctx, pool)
	createArtifactRun(t, ctx, pool, "run-git", false)
	service := NewService(artifactStorage.Repository(pool))
	scope, _ := UserScope("user-1")
	user, _ := service.User("user-1")
	var first WriteResult
	source := GitSource{Analysis: &GitAnalysis{Policy: GitAnalysisPolicy, ReportPath: GitAnalysisReportPath, IncludedFiles: 1, ExcludedEntries: 3}, RepositoryURL: "https://example.test:443/source.git", ResolvedCommit: strings.Repeat("a", 40), ImportedAt: time.Now().UTC().Truncate(time.Microsecond)}
	write := func(name string, source GitSource, abort bool) error {
		payload, err := artifactStorage.PreparePayload(ctx, pool, Payload{MediaType: "application/zip", Data: []byte("same bytes")})
		if err != nil {
			return err
		}
		return postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			repo := artifactStorage.TxRepository(tx)
			store, _ := NewService(repo).User("user-1")
			written, err := store.Write(ctx, ArtifactRef{Namespace: "source", Name: name}, payload, nil)
			if err != nil {
				return err
			}
			if err := repo.RecordGitSource(ctx, scope, written.Ref, source); err != nil {
				return err
			}
			if abort {
				return errors.New("abort fixture")
			}
			first = written
			return nil
		})
	}
	invalid := source
	invalid.Analysis = &GitAnalysis{Policy: GitAnalysisPolicy, ReportPath: GitAnalysisReportPath, IncludedFiles: 0, ExcludedEntries: 3}
	if err := write("invalid", invalid, false); !errors.Is(err, blob.ErrIntegrity) {
		t.Fatalf("invalid analysis: %v", err)
	}
	if err := write("cancelled", source, true); err == nil {
		t.Fatal("expected rollback")
	}
	if _, err := user.Metadata(ctx, ArtifactRef{Namespace: "source", Name: "cancelled"}); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("rollback metadata: %v", err)
	}
	if err := write("original", source, false); err != nil {
		t.Fatal(err)
	}
	fork, err := artifactstest.ForkInput(ctx, service, "user-1", first.Ref, "run-git", "source")
	if err != nil {
		t.Fatal(err)
	}
	other := source
	other.RepositoryURL = "https://example.test:443/other.git"
	other.ResolvedCommit = strings.Repeat("b", 40)
	if err := write("other", other, false); err != nil {
		t.Fatal(err)
	}
	var blobs, origins int
	if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM artifact_blobs),(SELECT count(*) FROM artifact_versions WHERE git_resolved_commit IS NOT NULL)`).Scan(&blobs, &origins); err != nil {
		t.Fatal(err)
	}
	if blobs != 1 || origins != 2 {
		t.Fatalf("dedup: %d blobs, %d origins", blobs, origins)
	}
	run, _ := service.Run("run-git")
	metadata, err := run.Metadata(ctx, fork.TargetRef)
	if err != nil || metadata.GitSource == nil || !sameGitSource(metadata.GitSource, source) {
		t.Fatalf("Run origin: %+v %v", metadata.GitSource, err)
	}
	versions, err := run.ListVersions(ctx, ArtifactRef{Namespace: "inputs", Name: "source"}, VersionPageQuery{Limit: 10})
	if err != nil || len(versions) != 1 || versions[0].GitSource == nil || !sameGitSource(versions[0].GitSource, source) {
		t.Fatalf("version origin: %+v %v", versions, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE artifact_versions SET git_resolved_commit=$1 WHERE git_resolved_commit IS NOT NULL`, strings.Repeat("c", 40)); err == nil {
		t.Fatal("origin mutation accepted")
	}
	if _, err := pool.Exec(ctx, `UPDATE artifact_versions SET media_type='text/plain' WHERE git_resolved_commit IS NOT NULL`); err == nil {
		t.Fatal("version mutation accepted")
	}
	for _, value := range []string{`{"policy":"bad"}`, `{"policy":"source-analysis-v1","reportPath":".contractor-import-report.json","includedFiles":1,"excludedEntries":3}`} {
		if _, err := pool.Exec(ctx, `UPDATE artifact_versions SET git_analysis=$1 WHERE git_resolved_commit IS NOT NULL`, value); err == nil {
			t.Fatal("analysis provenance mutation accepted")
		}
	}
	// Remove only the source binding. The Run revision still references the
	// immutable version and keeps its origin.
	if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, query := range []string{
			`SELECT set_config('contractor.lifecycle_purge', 'project', true)`,
			`DELETE FROM artifact_lineage WHERE source_scope_kind='user' AND source_namespace='source' AND source_name='original'`,
			`DELETE FROM artifact_bindings WHERE scope_kind='user' AND namespace='source' AND name='original'`,
			`DELETE FROM artifact_binding_revisions WHERE scope_kind='user' AND namespace='source' AND name='original'`,
		} {
			if _, err := tx.Exec(ctx, query); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	metadata, err = run.Metadata(ctx, fork.TargetRef)
	if err != nil || metadata.GitSource == nil || !sameGitSource(metadata.GitSource, source) {
		t.Fatalf("retained origin: %+v %v", metadata.GitSource, err)
	}
}

func sameGitSource(got *GitSource, want GitSource) bool {
	return reflect.DeepEqual(got.Analysis, want.Analysis) && got.RepositoryURL == want.RepositoryURL && got.ResolvedCommit == want.ResolvedCommit && got.RequestedRef == nil && want.RequestedRef == nil && got.ImportedAt.Equal(want.ImportedAt)
}
