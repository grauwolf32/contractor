package public

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
)

type auditArtifactManagement struct {
	fakeAuditManagement
	read                        artifacts.ReadResult
	owner, audit, key, revision string
}

func (f *auditArtifactManagement) ReadArtifact(_ context.Context, owner, audit, key, revision string) (artifacts.ReadResult, error) {
	f.owner, f.audit, f.key, f.revision = owner, audit, key, revision
	return f.read, f.err
}

func TestAuditArtifactReadsExactReceiptAndValidatedMember(t *testing.T) {
	data, _, err := auditdomain.BuildPackage("accepted", auditdomain.PackageKindCheckResults, "", []auditdomain.PackageInput{
		{ID: "scanner-report", Path: "report.json", MediaType: "application/json", Data: []byte(`{"observations":[]}`)},
	})
	if err != nil {
		t.Fatal(err)
	}
	management := &auditArtifactManagement{read: artifacts.ReadResult{Payload: artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: data}}}
	h := auditHandlers{AuditDependencies: AuditDependencies{Audits: management}}
	for _, tc := range []struct {
		query  string
		status int
		body   string
	}{
		{"key=result%2Fexecution&revision=r1&member=scanner-report", http.StatusOK, `{"observations":[]}`},
		{"key=result%2Fexecution&revision=r1&member=absent", http.StatusNotFound, ""},
		{"key=result%2Fexecution", http.StatusBadRequest, ""},
		{"key=result%2Fexecution&revision=r1&revision=r2", http.StatusBadRequest, ""},
		{"key=result%2Fexecution&revision=r1&namespace=other", http.StatusBadRequest, ""},
	} {
		r := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/audit-one/artifacts?"+tc.query, nil)
		r.SetPathValue("auditId", "audit-one")
		response := httptest.NewRecorder()
		h.getAuditArtifact(response, r)
		if response.Code != tc.status || (tc.body != "" && response.Body.String() != tc.body) {
			t.Fatalf("%s: %d %s", tc.query, response.Code, response.Body.String())
		}
	}
	if management.owner != "user-1" || management.audit != "audit-one" || management.key != "result/execution" || management.revision != "r1" {
		t.Fatalf("lost owner or exact receipt: %+v", management)
	}
}
