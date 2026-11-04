package public

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/auditservice"
)

func TestPublicAuditReportStatusesRejectApprovals(t *testing.T) {
	document := loadPublicOpenAPI(t)
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatal(err)
	}
	management := &fakeAuditManagement{}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid",
		newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(dependencies *testServices) { dependencies.Audits = management },
	)
	for _, status := range []auditservice.ReportStatus{
		auditservice.ReportPending,
		auditservice.ReportReady, auditservice.ReportUnavailable,
	} {
		t.Run(string(status), func(t *testing.T) {
			management.report = auditservice.ReportProjection{Status: status}
			response := serveAndValidatePublicContract(t, router, fixture.handler,
				newPublicContractRequest(http.MethodGet, "/v1/audits/audit-report/report", nil), true)
			if response.Code != http.StatusOK || response.Header().Get("Cache-Control") != "no-store" {
				t.Fatalf("report = %d headers=%v body=%s", response.Code, response.Header(), response.Body.String())
			}
			var actual auditReportResponse
			if err := json.Unmarshal(response.Body.Bytes(), &actual); err != nil {
				t.Fatal(err)
			}
			if actual.Status != status {
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
		t.Fatal("removed proposed report status passed the response schema")
	}
}
