package privateartifacts

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

// The repository can commit a write and then lose its database acknowledgement.
// This exercises the real handler's HTTP mapping, rather than disconnecting HTTP.
func TestPrivateArtifactCommittedWriteCanReturnHTTP500(t *testing.T) {
	repository := &acknowledgementFailureRepository{memoryRepository: newMemoryRepository()}
	handler, err := NewHandler(Dependencies{
		Registry:     &fakeRegistry{grant: testGrant("run-a")},
		Artifacts:    artifacts.NewService(repository),
		NewRequestID: func() (string, error) { return "artifact-ack-failure", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte("committed-memory-content-canary")
	response := putArtifact(t, handler, "builder", "memory.note", "*", payload)
	if response.Code != http.StatusInternalServerError || !bytes.Contains(response.Body.Bytes(), []byte(`"code":"internal_error"`)) || bytes.Contains(response.Body.Bytes(), payload) {
		t.Fatalf("ambiguous response = %d %s", response.Code, response.Body.String())
	}
	read := httptest.NewRecorder()
	handler.ServeHTTP(read, trustedRequest(http.MethodGet, "/private/v1/allocations/allocation-1/artifacts/builder/memory.note", nil))
	if read.Code != http.StatusOK || !bytes.Equal(read.Body.Bytes(), payload) {
		t.Fatalf("committed read = %d %q", read.Code, read.Body.Bytes())
	}
	replay := putArtifact(t, handler, "builder", "memory.note", "*", payload)
	if replay.Code != http.StatusConflict || repository.next != 1 {
		t.Fatalf("exact replay = %d, revisions = %d", replay.Code, repository.next)
	}
}

type acknowledgementFailureRepository struct{ *memoryRepository }

func (r *acknowledgementFailureRepository) Write(ctx context.Context, scope artifacts.Scope, target artifacts.ArtifactRef, payload artifacts.Payload, expected *string) (artifacts.WriteResult, error) {
	result, err := r.memoryRepository.Write(ctx, scope, target, payload, expected)
	if err != nil {
		return result, err
	}
	return artifacts.WriteResult{}, errors.New("injected database acknowledgement loss")
}
