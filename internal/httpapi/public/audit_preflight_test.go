package public

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestAuditPreflightChecksRevisionAndOwnerWithoutMutationKey(t *testing.T) {
	management := &fakeAuditManagement{
		audit: auditstore.Audit{AuditID: "audit-one", OwnerID: "user-1", Revision: 7, State: auditstore.AuditDraft},
		preflight: auditservice.PreflightResult{AuditID: "audit-one", Revision: 7,
			BaselineInputDigest: auditHandlerDigest("exact-inputs"), ServerCompatible: true,
			CompatibleCapabilities: []string{}, MissingCapabilities: []string{}, Reasons: []auditservice.CompatibilityReason{}},
	}
	h := auditTestHandler(management)
	for _, tc := range []struct {
		name, match, body, query string
		status                   int
		calls                    int
	}{
		{"valid", `"7"`, "", "", http.StatusOK, 1},
		{"missing revision", "", "", "", http.StatusBadRequest, 0},
		{"weak revision", `W/"7"`, "", "", http.StatusBadRequest, 0},
		{"stale revision", `"6"`, "", "", http.StatusPreconditionFailed, 1},
		{"body", `"7"`, `{}`, "", http.StatusBadRequest, 0},
		{"query", `"7"`, "", "?projectId=foreign", http.StatusBadRequest, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			management.preflightCalls = 0
			r := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/audit-one/preflight"+tc.query, []byte(tc.body))
			r.SetPathValue("auditId", "audit-one")
			if tc.match != "" {
				r.Header.Set("If-Match", tc.match)
			}
			w := httptest.NewRecorder()
			h.preflightAudit(w, r)
			if w.Code != tc.status || management.preflightCalls != tc.calls {
				t.Fatalf("preflight = %d, calls %d, body %s", w.Code, management.preflightCalls, w.Body.String())
			}
			if tc.status == http.StatusOK {
				if management.preflightOwner != "user-1" || management.preflightAudit != "audit-one" || management.preflightRevision != 7 ||
					w.Header().Get("ETag") != `"7"` || w.Header().Get("Cache-Control") != "no-store" ||
					!strings.Contains(w.Body.String(), `"baselineInputDigest"`) {
					t.Fatalf("preflight lost owner/revision or response metadata: %s", w.Body.String())
				}
			}
		})
	}
	for _, tc := range []struct {
		name, owner string
		state       auditstore.AuditState
		want        int
	}{
		{"foreign", "another-user", auditstore.AuditDraft, http.StatusNotFound},
		{"active", "user-1", auditstore.AuditActive, http.StatusConflict},
	} {
		t.Run(tc.name, func(t *testing.T) {
			management.audit.OwnerID, management.audit.State = tc.owner, tc.state
			r := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/audit-one/preflight", nil)
			r.SetPathValue("auditId", "audit-one")
			r.Header.Set("If-Match", `"7"`)
			w := httptest.NewRecorder()
			h.preflightAudit(w, r)
			if w.Code != tc.want {
				t.Fatalf("preflight = %d: %s", w.Code, w.Body.String())
			}
		})
	}
}
