//go:build integration

package privateartifacts

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

// PostgresRepository rejects every write to a frozen binding, and the private
// API reports it as the permanent artifact_frozen (ART-R59).
func TestPostgresPrivateFrozenArtifactWriteIsPermanent(t *testing.T) {
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
	tag, err := pool.Exec(ctx, `UPDATE artifact_bindings SET frozen = true
WHERE scope_kind = 'run' AND scope_id = 'run-a' AND namespace = 'builder' AND name = 'report'`)
	if err != nil || tag.RowsAffected() != 1 {
		t.Fatalf("freeze binding = %v, %v", tag, err)
	}
	handler, err := NewHandler(Dependencies{
		Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: service,
		NewRequestID: func() (string, error) { return "artifact-frozen", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, precondition := range [][2]string{
		{"If-Match", strconv.Quote(*written.Ref.Revision)}, {"If-None-Match", "*"}, {"If-Match", `"stale"`},
	} {
		request := trustedRequest(http.MethodPut, "/private/v1/allocations/allocation-1/artifacts/builder/report",
			strings.NewReader("second"))
		request.Header.Set("Content-Type", "text/plain")
		request.Header.Set(precondition[0], precondition[1])
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		assertPrivateErrorEnvelope(t, response, http.StatusConflict, "artifact_frozen", false)
	}
	current, err := store.Read(ctx, artifacts.ArtifactRef{Namespace: "builder", Name: "report"})
	if err != nil || string(current.Payload.Data) != "first" || *current.Ref.Revision != *written.Ref.Revision {
		t.Fatalf("frozen binding after rejected writes = (%+v, %v)", current, err)
	}
}
