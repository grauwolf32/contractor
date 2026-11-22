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

type evidenceFailure struct {
	err   error
	calls int
}

func (f *evidenceFailure) RecordEvidence(context.Context, controlplane.AllocationGrant, auditcontract.EvidenceSubmission) (auditcontract.EvidenceReceipt, error) {
	f.calls++
	return auditcontract.EvidenceReceipt{}, f.err
}

func TestPrivateEvidenceClassifiesPermanentDomainRejections(t *testing.T) {
	for _, test := range []struct {
		name   string
		err    error
		status int
		code   string
	}{
		{"quota", auditstore.ErrEvidenceBudgetExhausted, http.StatusConflict, "audit_evidence_limit_exceeded"},
		{"identity", auditstore.ErrConflict, http.StatusConflict, "audit_evidence_conflict"},
		{"closed", auditstore.ErrPrecondition, http.StatusConflict, "audit_evidence_closed"},
		{"run-gone", runstore.ErrNotFound, http.StatusConflict, "audit_evidence_closed"},
		{"foreign-item", auditstore.ErrNotFound, http.StatusBadRequest, "invalid_request"},
		{"invalid", auditstore.ErrInvalid, http.StatusBadRequest, "invalid_request"},
	} {
		t.Run(test.name, func(t *testing.T) {
			intake := &evidenceFailure{err: test.err}
			handler, err := NewHandler(Dependencies{Transfers: artifactpgtest.NewStorage(), Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(newMemoryRepository()), AuditEvidence: intake})
			if err != nil {
				t.Fatal(err)
			}
			request := trustedRequest(http.MethodPost, "/private/v1/allocations/allocation-1/audit-evidence", strings.NewReader(`{"apiVersion":"contractor/v1alpha1","invocationId":"invocation","submissionId":"call","itemKey":"item-00","sources":[{"namespace":"inputs","name":"source","revision":"exact"}],"interpretation":"Selected source supports only this claim","testedConditions":[],"limitations":[]}`))
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.status || !strings.Contains(response.Body.String(), `"code":"`+test.code+`"`) || !strings.Contains(response.Body.String(), `"retryable":false`) || intake.calls != 1 {
				t.Fatalf("classification: calls=%d HTTP=%d %s", intake.calls, response.Code, response.Body.String())
			}
		})
	}
}

func TestPrivateEvidenceStrictInputNeverReachesTheOwner(t *testing.T) {
	for _, body := range []string{
		`{"apiVersion":"contractor/v1alpha1","audit_id":"forged"}`,
		`{"apiVersion":"contractor/v1alpha1","apiVersion":"contractor/v1alpha1"}`,
		`{"apiVersion":"contractor/v1alpha1","invocationId":"invocation","submissionId":"call","itemKey":"item-00","sources":[{"namespace":"inputs","name":"source","revision":null}],"interpretation":"Claim","testedConditions":[],"limitations":[]}`,
	} {
		intake := &evidenceFailure{err: errors.New("unexpected owner call")}
		handler, err := NewHandler(Dependencies{Transfers: artifactpgtest.NewStorage(), Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(newMemoryRepository()), AuditEvidence: intake})
		if err != nil {
			t.Fatal(err)
		}
		request := trustedRequest(http.MethodPost, "/private/v1/allocations/allocation-1/audit-evidence", strings.NewReader(body))
		request.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest || intake.calls != 0 {
			t.Fatalf("strict input: calls=%d HTTP=%d %s", intake.calls, response.Code, response.Body.String())
		}
	}
}
