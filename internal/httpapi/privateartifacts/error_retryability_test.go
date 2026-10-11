package privateartifacts

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

// The private API keeps the retryability of the store's answer: a frozen
// binding is permanent (ART-R59) and a definite transaction conflict is the
// retryable 503 the public API returns (FAIL-ART-1).
func TestPrivateArtifactWriteErrorsReportRetryability(t *testing.T) {
	for _, test := range []struct {
		name      string
		err       error
		status    int
		code      string
		retryable bool
	}{
		{"frozen binding", artifacts.ErrArtifactFrozen, http.StatusConflict, "artifact_frozen", false},
		{"stale revision", &artifacts.ConflictError{Ref: artifacts.ArtifactRef{Namespace: "builder", Name: "report"}},
			http.StatusConflict, "artifact_conflict", true},
		{"serialization failure", fmt.Errorf("write artifact: %w", &pgconn.PgError{Code: "40001"}),
			http.StatusServiceUnavailable, "storage_transaction_conflict", true},
		{"deadlock", fmt.Errorf("write artifact: %w", &pgconn.PgError{Code: "40P01"}),
			http.StatusServiceUnavailable, "storage_transaction_conflict", true},
		{"other storage failure", fmt.Errorf("write artifact: %w", &pgconn.PgError{Code: "53300"}),
			http.StatusInternalServerError, "internal_error", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			repository := &rejectingWriteRepository{memoryRepository: newMemoryRepository(), err: test.err}
			handler, err := NewHandler(Dependencies{
				Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(repository),
				NewRequestID: func() (string, error) { return "artifact-retryability", nil },
			})
			if err != nil {
				t.Fatal(err)
			}
			response := putArtifact(t, handler, "builder", "report", "*", []byte("report"))
			assertPrivateErrorEnvelope(t, response, test.status, test.code, test.retryable)
		})
	}
}

type rejectingWriteRepository struct {
	*memoryRepository
	err error
}

func (r *rejectingWriteRepository) Write(
	context.Context, artifacts.Scope, artifacts.ArtifactRef, artifacts.Payload, *string,
) (artifacts.WriteResult, error) {
	return artifacts.WriteResult{}, r.err
}

func assertPrivateErrorEnvelope(
	t *testing.T, response *httptest.ResponseRecorder, status int, code string, retryable bool,
) {
	t.Helper()
	var envelope errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode error envelope %q: %v", response.Body.String(), err)
	}
	if response.Code != status || envelope.Code != code || envelope.Retryable != retryable {
		t.Fatalf("error = %d %+v, want %d %s retryable=%t", response.Code, envelope, status, code, retryable)
	}
}
