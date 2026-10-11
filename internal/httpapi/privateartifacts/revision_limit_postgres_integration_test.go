//go:build integration

package privateartifacts

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

// A Run that has used up its revision limit (LIM-ART-6) gets the permanent
// artifact_limit_exceeded for every further write, and nothing is stored
// (ART-R114). The counter is set directly: filling 50,000 revisions is the
// artifacts package's test.
func TestPostgresPrivateRunRevisionLimitIsPermanent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	pool := isolatedPrivateArtifactPool(t, ctx)
	createPrivateArtifactRun(t, ctx, pool, "run-a")
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	store, err := service.Run("run-a")
	if err != nil {
		t.Fatal(err)
	}
	written, err := store.Write(ctx, artifacts.ArtifactRef{Namespace: "builder", Name: "report"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("first")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	tag, err := pool.Exec(ctx, `UPDATE artifact_scopes SET revision_count = 50000
WHERE scope_kind = 'run' AND scope_id = 'run-a' AND revision_count = 1`)
	if err != nil || tag.RowsAffected() != 1 {
		t.Fatalf("exhaust the Run revision limit = %v, %v", tag, err)
	}
	handler, err := NewHandler(Dependencies{
		Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: service,
		NewRequestID: func() (string, error) { return "artifact-limit", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	response := putArtifact(t, handler, "builder", "other", "*", []byte("second"))
	assertPrivateErrorEnvelope(t, response, http.StatusConflict, "artifact_limit_exceeded", false)
	var revisions, counted int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind = 'run' AND scope_id = 'run-a'),
       (SELECT revision_count FROM artifact_scopes WHERE scope_kind = 'run' AND scope_id = 'run-a')`,
	).Scan(&revisions, &counted); err != nil {
		t.Fatal(err)
	}
	if revisions != 1 || counted != 50000 {
		t.Fatalf("rejected write left %d revisions and counter %d", revisions, counted)
	}
	current, err := store.Read(ctx, artifacts.ArtifactRef{Namespace: "builder", Name: "report"})
	if err != nil || *current.Ref.Revision != *written.Ref.Revision {
		t.Fatalf("binding after the rejected write = (%+v, %v)", current, err)
	}
}
