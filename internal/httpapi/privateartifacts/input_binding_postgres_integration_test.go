//go:build integration

package privateartifacts

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

// Only the trusted Workflow input fork creates inputs/<slot> (ART-R76). A
// Worker advances its working copy by exact CAS but never adds or recreates a
// name in inputs (ART-R16, ART-R37).
func TestPostgresPrivateWorkerAdvancesButNeverCreatesInputs(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	pool := isolatedPrivateArtifactPool(t, ctx)
	createPrivateArtifactRun(t, ctx, pool, "run-a")
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	user, err := service.User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	source, err := user.Write(ctx, artifacts.ArtifactRef{Namespace: "sources", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	fork, err := service.ForkInput(ctx, "user-1", source.Ref, "run-a", "source")
	if err != nil {
		t.Fatal(err)
	}
	handler, err := NewHandler(Dependencies{
		Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: service,
		NewRequestID: func() (string, error) { return "artifact-inputs", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	put := func(name, header, value string) *httptest.ResponseRecorder {
		request := trustedRequest(http.MethodPut, "/private/v1/allocations/allocation-1/artifacts/inputs/"+name,
			strings.NewReader("worker edit"))
		request.Header.Set("Content-Type", "text/plain")
		request.Header.Set(header, value)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}
	if updated := put("source", "If-Match", strconv.Quote(*fork.TargetRef.Revision)); updated.Code != http.StatusOK {
		t.Fatalf("working-copy update = %d %s", updated.Code, updated.Body.String())
	}
	for _, name := range []string{"invented", "source"} {
		assertPrivateErrorEnvelope(t, put(name, "If-None-Match", "*"), http.StatusForbidden, "artifact_access_denied", false)
	}
	store, err := service.Run("run-a")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Metadata(ctx, artifacts.ArtifactRef{Namespace: "inputs", Name: "invented"}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("Worker-created input = %v, want absent", err)
	}
	current, err := store.Read(ctx, artifacts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil || string(current.Payload.Data) != "worker edit" {
		t.Fatalf("working copy after rejected creates = (%q, %v)", current.Payload.Data, err)
	}
}
