package postgres_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

// Every inserted revision charges its binding one retained revision and its
// full encoded size, also when it repeats content another revision already
// stores; a refused CAS write charges nothing (ART-R123).
func TestPostgresHistoryAccountingChargesEveryRetainedRevision(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	artifactStorage := testBlobStorage(t, ctx, pool)
	user, err := artifacts.NewService(artifactStorage.Repository(pool)).User("history-owner")
	if err != nil {
		t.Fatal(err)
	}
	ref := artifacts.ArtifactRef{Namespace: "notes", Name: "history"}
	first, err := user.Write(ctx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("abc")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := user.Write(ctx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("defgh")}, first.Ref.Revision)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := user.Write(ctx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("stale")}, first.Ref.Revision); !errors.Is(err, artifacts.ErrArtifactConflict) {
		t.Fatalf("stale CAS write error = %v", err)
	}
	if _, err := user.Write(ctx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("abc")}, second.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	var revisions, bytes int64
	if err := pool.QueryRow(ctx, `SELECT retained_revisions, retained_bytes FROM artifact_bindings
WHERE scope_kind = 'user' AND scope_id = 'history-owner' AND namespace = 'notes' AND name = 'history'`,
	).Scan(&revisions, &bytes); err != nil {
		t.Fatal(err)
	}
	if revisions != 3 || bytes != 3+5+3 {
		t.Fatalf("retained counters = %d revisions, %d bytes; want 3 and 11", revisions, bytes)
	}
}
