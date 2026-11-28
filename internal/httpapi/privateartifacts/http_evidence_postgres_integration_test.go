//go:build integration

package privateartifacts

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestPostgresPrivateSelectedHTTPArtifactIsFrozenScopedAndReplayable(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPrivateArtifactPool(t, ctx)
	createPrivateArtifactRun(t, ctx, pool, "run-a")
	storage := artifactpgtest.NewStorage()
	service := artifacts.NewService(storage.Repository(pool))
	grant := testGrant("run-a")
	grant.HTTPResultEvidence = true
	registry := &fakeRegistry{grant: grant}
	handler, err := newPayloadNativeRecorder(Dependencies{Transfers: storage, Registry: registry, Artifacts: service})
	if err != nil {
		t.Fatal(err)
	}
	var capture auditdomain.CapturedHTTPExchange
	if err := strictjson.Decode([]byte(`{"source":"http","request_id":"1","request_tag":"tag","observed_at":"2026-11-06T18:20:00Z","attempts":[{"request":{"method":"GET","url":"https://example.test/","message":{"kind":"structured","headers":[],"body_base64":""}},"error":"timeout"}]}`), &capture); err != nil {
		t.Fatal(err)
	}
	payload, err := auditdomain.EncodeHTTPExchangeArtifact(capture, "run-a", "invocation")
	if err != nil {
		t.Fatal(err)
	}
	name := auditdomain.HTTPExchangeArtifactName("invocation", "http", "1")
	put := func(data []byte, target string) *httptest.ResponseRecorder {
		request := trustedRequest(http.MethodPut, "/private/v1/allocations/allocation-1/artifacts/http-exchanges/"+target, bytes.NewReader(data))
		request.Header.Set("Content-Type", auditdomain.HTTPExchangeMediaType)
		request.Header.Set("If-None-Match", "*")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}
	first := put(payload, name)
	if first.Code != http.StatusCreated {
		t.Fatalf("first publication: %d %s", first.Code, first.Body.String())
	}
	for range 2 {
		replay := put(payload, name)
		if replay.Code != http.StatusCreated || replay.Header().Get("ETag") != first.Header().Get("ETag") {
			t.Fatalf("lost reply replay: %d %s", replay.Code, replay.Body.String())
		}
	}
	changed := capture
	changed.RequestTag = "changed-tag"
	changedPayload, err := auditdomain.EncodeHTTPExchangeArtifact(changed, "run-a", "invocation")
	if err != nil {
		t.Fatal(err)
	}
	assertPrivateErrorEnvelope(t, put(changedPayload, name), http.StatusConflict, "artifact_frozen", false)
	foreign, _ := auditdomain.EncodeHTTPExchangeArtifact(capture, "other-run", "invocation")
	assertPrivateErrorEnvelope(t, put(foreign, name), http.StatusBadRequest, "invalid_request", false)
	assertPrivateErrorEnvelope(t, put(payload, "wrong-identity"), http.StatusBadRequest, "invalid_request", false)
	var old map[string]any
	if err := json.Unmarshal(payload, &old); err != nil {
		t.Fatal(err)
	}
	old["finding"] = map[string]string{"proposal_id": "proposal", "receipt_id": "receipt"}
	invalid, _ := json.Marshal(old)
	assertPrivateErrorEnvelope(t, put(invalid, name), http.StatusBadRequest, "invalid_request", false)
	registry.grant.HTTPResultEvidence = false
	assertPrivateErrorEnvelope(t, put(payload, name), http.StatusForbidden, "artifact_access_denied", false)
	var count int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind='run' AND scope_id='run-a' AND namespace='http-exchanges'`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("publication count=%d %v", count, err)
	}
	store, _ := service.Run("run-a")
	read, err := store.Read(ctx, artifacts.ArtifactRef{Namespace: "http-exchanges", Name: name})
	if err != nil || !bytes.Equal(read.Payload.Data, payload) {
		t.Fatalf("retained content changed: %v", err)
	}
	metadata, err := store.Metadata(ctx, read.Ref)
	if err != nil || !metadata.Frozen {
		t.Fatalf("mutable evidence: %v", err)
	}
}
