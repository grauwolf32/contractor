package public

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

// A frozen binding never advances, so a write to it is the permanent 409
// artifact_frozen whatever its precondition (ART-R59). The fake rejects the
// write like PostgresRepository; TestPostgresPublicFrozenArtifactWriteIsPermanent
// covers the real store.
func TestPublicFrozenArtifactWriteIsPermanent(t *testing.T) {
	fixture := newHandlerFixture(t)
	created := putPublicArtifact(fixture.handler, "notes", "report", "If-None-Match", "*", "first")
	if created.Code != http.StatusCreated {
		t.Fatalf("create = %d %s", created.Code, created.Body.String())
	}
	fixture.repository.mu.Lock()
	frozen := 0
	for key := range fixture.repository.current {
		if key.namespace == "notes" && key.name == "report" {
			fixture.repository.frozen[key] = true
			frozen++
		}
	}
	fixture.repository.mu.Unlock()
	if frozen != 1 {
		t.Fatalf("froze %d bindings", frozen)
	}
	assertFrozenWritesRejected(t, fixture.handler, created.Header().Get("ETag"))
}

func TestPostgresPublicFrozenArtifactWriteIsPermanent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t),
		mustTestOrigins(t), false, nil, func(dependencies *Dependencies) { dependencies.Artifacts = service })
	created := putPublicArtifact(fixture.handler, "notes", "report", "If-None-Match", "*", "first")
	if created.Code != http.StatusCreated {
		t.Fatalf("create = %d %s", created.Code, created.Body.String())
	}
	tag, err := pool.Exec(ctx, `UPDATE artifact_bindings SET frozen = true
WHERE scope_kind = 'user' AND namespace = 'notes' AND name = 'report'`)
	if err != nil || tag.RowsAffected() != 1 {
		t.Fatalf("freeze binding = %v, %v", tag, err)
	}
	assertFrozenWritesRejected(t, fixture.handler, created.Header().Get("ETag"))
}

func assertFrozenWritesRejected(t *testing.T, handler http.Handler, etag string) {
	t.Helper()
	for _, precondition := range [][2]string{{"If-Match", etag}, {"If-None-Match", "*"}, {"If-Match", `"stale"`}} {
		response := putPublicArtifact(handler, "notes", "report", precondition[0], precondition[1], "second")
		assertErrorEnvelope(t, response, http.StatusConflict, "artifact_frozen", false)
	}
	read := httptest.NewRecorder()
	handler.ServeHTTP(read, authenticatedRequest(http.MethodGet, "/v1/artifacts/notes/report", bytes.NewReader(nil)))
	if read.Code != http.StatusOK || read.Body.String() != "first" || read.Header().Get("ETag") != etag {
		t.Fatalf("frozen binding after rejected writes = %d %q %s", read.Code, read.Body.String(), read.Header().Get("ETag"))
	}
}

// Drift of the configuration roots is permanent for a publication until the
// roots match the loaded snapshot again; then publication resumes without a
// restart (CAT-R74, FAIL-CAT-2).
func TestPublishConfigurationReportsRootDriftAsPermanent(t *testing.T) {
	operator := t.TempDir()
	if err := os.CopyFS(operator, os.DirFS("../../config/testdata/valid")); err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithConfig(t, operator)
	added := filepath.Join(operator, "model-policies", "operator-added.yaml")
	if err := os.WriteFile(added, []byte("apiVersion: contractor/v1alpha1\nkind: ModelPolicy\n"+
		"metadata: {name: operator-added, version: \"1\"}\nspec: {model: added-model}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	publish := func(key string) *httptest.ResponseRecorder {
		request := authenticatedRequest(http.MethodPost, "/v1/configurations/model-policies",
			bytes.NewReader([]byte(`{"name":"after-drift","version":"1","modelPolicy":{"model":"m"}}`)))
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Idempotency-Key", key)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	assertErrorEnvelope(t, publish("during-drift"), http.StatusConflict, "configuration_drift", false)
	if _, err := fixture.configs.Configuration("model-policies", "operator-added@1"); err == nil {
		t.Fatal("drift leaked into the current snapshot")
	}
	if err := os.Remove(added); err != nil {
		t.Fatal(err)
	}
	if restored := publish("after-restore"); restored.Code != http.StatusCreated {
		t.Fatalf("publication after restoring the roots = %d %s", restored.Code, restored.Body.String())
	}
}

func putPublicArtifact(handler http.Handler, namespace, name, header, value, body string) *httptest.ResponseRecorder {
	request := authenticatedRequest(http.MethodPut, "/v1/artifacts/"+namespace+"/"+name, bytes.NewReader([]byte(body)))
	request.Header.Set("Content-Type", "text/plain")
	request.Header.Set(header, value)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func assertErrorEnvelope(t *testing.T, response *httptest.ResponseRecorder, status int, code string, retryable bool) {
	t.Helper()
	var envelope errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode error envelope %q: %v", response.Body.String(), err)
	}
	if response.Code != status || envelope.Code != code || envelope.Retryable != retryable ||
		strings.TrimSpace(envelope.Message) == "" {
		t.Fatalf("error = %d %+v, want %d %s retryable=%t", response.Code, envelope, status, code, retryable)
	}
}
