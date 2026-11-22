package privateartifacts

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type judgmentFailure struct {
	err   error
	calls int
}

func (f *judgmentFailure) RecordJudgments(context.Context, controlplane.AllocationGrant, auditcontract.JudgmentSubmission) (auditcontract.JudgmentReceipt, error) {
	f.calls++
	return auditcontract.JudgmentReceipt{}, f.err
}

func TestPrivateJudgmentClassifiesPermanentDomainRejections(t *testing.T) {
	for _, test := range []struct {
		name   string
		err    error
		status int
		code   string
	}{
		{"quota", auditstore.ErrEvidenceBudgetExhausted, http.StatusConflict, "audit_result_limit_exceeded"},
		{"identity", auditstore.ErrConflict, http.StatusConflict, "audit_result_conflict"},
		{"closed", auditstore.ErrPrecondition, http.StatusConflict, "audit_result_closed"},
		{"run-gone", runstore.ErrNotFound, http.StatusConflict, "audit_result_closed"},
		{"foreign-item", auditstore.ErrNotFound, http.StatusBadRequest, "invalid_request"},
		{"invalid", auditstore.ErrInvalid, http.StatusBadRequest, "invalid_request"},
	} {
		t.Run(test.name, func(t *testing.T) {
			intake := &judgmentFailure{err: test.err}
			handler, err := NewHandler(Dependencies{Transfers: artifactpgtest.NewStorage(), Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(newMemoryRepository()), AuditJudgments: intake})
			if err != nil {
				t.Fatal(err)
			}
			request := trustedRequest(http.MethodPost, "/private/v1/allocations/allocation-1/audit-results", strings.NewReader(`{"apiVersion":"contractor/v1alpha1","invocationId":"invocation","submissionId":"call","mode":"single","items":[{"baseGeneration":0,"conclusion":{"item_key":"item-00","rationale":"Scoped judgment","tested_conditions":[],"evidence":[],"limitations":[]}}]}`))
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.status || !strings.Contains(response.Body.String(), `"code":"`+test.code+`"`) || !strings.Contains(response.Body.String(), `"retryable":false`) || intake.calls != 1 {
				t.Fatalf("classification: calls=%d HTTP=%d %s", intake.calls, response.Code, response.Body.String())
			}
		})
	}
}

func TestPrivateJudgmentStrictInputNeverReachesTheOwner(t *testing.T) {
	for _, body := range []string{
		`{"apiVersion":"contractor/v1alpha1","audit_id":"forged"}`,
		`{"apiVersion":"contractor/v1alpha1","apiVersion":"contractor/v1alpha1"}`,
		`{"apiVersion":"contractor/v1alpha1","invocationId":"invocation","submissionId":"call","mode":"single","items":[{"conclusion":{"item_key":"item-00","rationale":"Scoped judgment","tested_conditions":[],"evidence":[],"limitations":[]}}]}`,
		`{"apiVersion":"contractor/v1alpha1","invocationId":"invocation","submissionId":"call","mode":"single","items":[{"baseGeneration":0,"conclusion":{"item_key":"item-00","rationale":"Scoped judgment","tested_conditions":[],"evidence":[],"limitations":[],"author":"system"}}]}`,
	} {
		intake := &judgmentFailure{err: errors.New("unexpected owner call")}
		handler, err := NewHandler(Dependencies{Transfers: artifactpgtest.NewStorage(), Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(newMemoryRepository()), AuditJudgments: intake})
		if err != nil {
			t.Fatal(err)
		}
		request := trustedRequest(http.MethodPost, "/private/v1/allocations/allocation-1/audit-results", strings.NewReader(body))
		request.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest || intake.calls != 0 {
			t.Fatalf("strict input: calls=%d HTTP=%d %s", intake.calls, response.Code, response.Body.String())
		}
	}
}
