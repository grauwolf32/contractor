package public

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/auditservice"
)

func TestPublicAuditReportPreservesProposedReview(t *testing.T) {
	document := loadPublicOpenAPI(t)
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 9, 19, 12, 0, 0, 0, time.UTC)
	review := &auditservice.ReviewRequest{
		RequestID: "review-report", AuditID: "audit-report",
		SubjectKind: auditservice.ReviewSubjectReport, SubjectID: "audit-report",
		Kind: auditservice.ReportAcceptanceReviewKind, SubjectRevision: 7,
		SubjectDigest:    auditHandlerDigest("report-candidate"),
		RequestedActions: []auditservice.ReviewRequestedAction{"approve", "reject"},
		State:            auditservice.ReviewPending, Revision: 3, CreatedAt: now, UpdatedAt: now,
	}
	management := &fakeAuditManagement{}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid",
		newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(dependencies *Dependencies) { dependencies.Audits = management },
	)
	for _, status := range []auditservice.ReportStatus{
		auditservice.ReportPending, auditservice.ReportProposed,
		auditservice.ReportReady, auditservice.ReportUnavailable,
	} {
		t.Run(string(status), func(t *testing.T) {
			management.report = auditservice.ReportProjection{Status: status}
			if status == auditservice.ReportProposed {
				management.report.Review = review
			}
			response := serveAndValidatePublicContract(t, router, fixture.handler,
				newPublicContractRequest(http.MethodGet, "/v1/audits/audit-report/report", nil), true)
			if response.Code != http.StatusOK || response.Header().Get("Cache-Control") != "no-store" {
				t.Fatalf("report = %d headers=%v body=%s", response.Code, response.Header(), response.Body.String())
			}
			var actual auditReportResponse
			if err := json.Unmarshal(response.Body.Bytes(), &actual); err != nil {
				t.Fatal(err)
			}
			if actual.Status != status || !reflect.DeepEqual(actual.Review, management.report.Review) {
				t.Fatalf("report projection changed: got %+v, want %+v", actual, management.report)
			}
		})
	}
	request := newPublicContractRequest(http.MethodGet, "/v1/audits/audit-report/report", nil)
	request.Header.Del("Authorization")
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated report = %d", response.Code)
	}
	if err := document.Components.Schemas["AuditReport"].Value.VisitJSON(
		map[string]any{"status": "proposed"}, openapi3.EnableJSONSchema2020(),
	); err == nil {
		t.Fatal("proposed report without a review passed the response schema")
	}
}
