package artifacts_test

import (
	"errors"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func TestPostgresGitProvenanceRetainsExactRunInputAndRollsBack(t *testing.T) {
	ctx := t.Context()
	pool := isolatedArtifactPool(t, ctx)
	ctx = testBlobContext(t, ctx, pool)
	createArtifactRun(t, ctx, pool, "run-git", false)
	service := NewService(NewPostgresRepository(pool))
	scope, _ := UserScope("user-1")
	user, _ := service.User("user-1")
	var first WriteResult
	source := GitSource{RepositoryURL: "https://example.test:443/source.git", ResolvedCommit: strings.Repeat("a", 40), ImportedAt: time.Now().UTC().Truncate(time.Microsecond)}
	write := func(name string, source GitSource, abort bool) error {
		payload, err := PreparePayload(ctx, Payload{MediaType: "application/zip", Data: []byte("same bytes")})
		if err != nil {
			return err
		}
		return postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			repo := NewPostgresRepository(tx)
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
	if err := write("cancelled", source, true); err == nil {
		t.Fatal("expected rollback")
	}
	if _, err := user.Metadata(ctx, ArtifactRef{Namespace: "source", Name: "cancelled"}); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("rollback metadata: %v", err)
	}
	if err := write("original", source, false); err != nil {
		t.Fatal(err)
	}
	fork, err := service.ForkInput(ctx, "user-1", first.Ref, "run-git", "source")
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
	if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM artifact_blobs),(SELECT count(*) FROM artifact_git_sources)`).Scan(&blobs, &origins); err != nil {
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
	if _, err := pool.Exec(ctx, `UPDATE artifact_git_sources SET resolved_commit=$1`, strings.Repeat("c", 40)); err == nil {
		t.Fatal("origin mutation accepted")
	}
	// Remove only the source binding after dropping its source-side pin. The
	// Run revision still references the immutable version and keeps its origin.
	if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, query := range []string{
			`SELECT set_config('contractor.lifecycle_purge', 'project', true)`,
			`DELETE FROM artifact_pins WHERE scope_kind='user' AND namespace='source' AND name='original'`,
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
	return got.RepositoryURL == want.RepositoryURL && got.ResolvedCommit == want.ResolvedCommit && got.RequestedRef == nil && want.RequestedRef == nil && got.ImportedAt.Equal(want.ImportedAt)
}
