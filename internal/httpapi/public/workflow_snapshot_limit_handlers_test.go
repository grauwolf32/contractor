package public

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// TestCreateRunRejectsAnOversizedWorkflowSnapshotBeforeAcceptance is the
// regression for a Run whose resolved Workflow snapshot exceeded its bound:
// it was accepted with 202 and failed later in the Scheduler. A Run override
// that enlarges a Workflow already at the bound is now a permanent 400 that
// stores nothing, while the override-free Run at the bound is accepted
// (CAT-R76).
func TestCreateRunRejectsAnOversizedWorkflowSnapshotBeforeAcceptance(t *testing.T) {
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	root := configtest.CopyWithPolicies(t, "../../../testdata/configs")
	padArtifactCopySnapshotToBound(t, root)
	fixture := newHandlerFixtureWithConfig(t, root)
	user, err := fixture.artifacts.User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	create := func(key, executionConfig string) *http.Request {
		body := `{"workflow":"artifact-copy@1","parameters":{},` +
			`"artifacts":{"source":{"namespace":"projects","name":"source"}}` + executionConfig + `}`
		request := newPublicContractRequest(http.MethodPost, "/v1/runs", []byte(body))
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set(idempotencyKeyHeader, key)
		return request
	}

	response := serveAndValidatePublicContract(t, router, fixture.handler, create(
		"snapshot-over-bound", `,"executionConfig":{"workers":{"modelPolicy":"test-domain-worker@1"}}`,
	), true)
	var envelope errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode error response %q: %v", response.Body.String(), err)
	}
	if response.Code != http.StatusBadRequest || envelope.Code != "invalid_request" || envelope.Retryable {
		t.Fatalf("oversized Run snapshot = %d %+v, want permanent 400 invalid_request", response.Code, envelope)
	}
	if len(fixture.runs.runs) != 0 || len(fixture.runs.idempotencyClaims) != 0 {
		t.Fatalf("rejected Run stored state: runs=%d keys=%d", len(fixture.runs.runs), len(fixture.runs.idempotencyClaims))
	}

	accepted := serveAndValidatePublicContract(t, router, fixture.handler, create("snapshot-at-bound", ""), true)
	if accepted.Code != http.StatusAccepted {
		t.Fatalf("Run at the snapshot bound = %d %s", accepted.Code, accepted.Body.String())
	}
	if stored := fixture.runs.runs["run_fixed"]; len(stored.WorkflowSnapshot) != config.MaxWorkflowSnapshotBytes {
		t.Fatalf("stored snapshot = %d bytes, want exactly %d", len(stored.WorkflowSnapshot), config.MaxWorkflowSnapshotBytes)
	}
}

// padArtifactCopySnapshotToBound pads the entry Stage instruction of
// artifact-copy@1 until its override-free snapshot has exactly
// MaxWorkflowSnapshotBytes.
func padArtifactCopySnapshotToBound(t *testing.T, root string) {
	t.Helper()
	instruction := filepath.Join(root, "instructions", "copy-planner.md")
	encodedSize := func(text string) int {
		if err := os.WriteFile(instruction, []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
		snapshot, err := configload.Load(root)
		if err != nil {
			t.Fatal(err)
		}
		workflow, err := snapshot.Workflow("artifact-copy@1")
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := config.EncodeResolvedWorkflowSnapshot(workflow)
		if err != nil {
			t.Fatal(err)
		}
		return len(encoded)
	}
	// The ASCII instruction is embedded once, so its length adds exactly.
	padded := strings.Repeat("a", 1+config.MaxWorkflowSnapshotBytes-encodedSize("a"))
	if size := encodedSize(padded); size != config.MaxWorkflowSnapshotBytes {
		t.Fatalf("padded snapshot = %d bytes, want exactly %d", size, config.MaxWorkflowSnapshotBytes)
	}
}
