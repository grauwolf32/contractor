package public

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

type hypothesisAuditManagement struct {
	fakeAuditManagement
	write auditservice.HypothesisMutationParams
	list  auditstore.ListHypothesesParams
	item  auditstore.Hypothesis
}

func (f *hypothesisAuditManagement) WriteHypothesis(_ context.Context, p auditservice.HypothesisMutationParams) (auditstore.HypothesisAcceptance, bool, error) {
	f.write = p
	return auditstore.HypothesisAcceptance{Hypotheses: []auditcontract.HypothesisRef{{ID: f.item.Document.ID, Revision: f.item.Document.Revision}}, AuditRevision: 3}, false, nil
}
func (f *hypothesisAuditManagement) Hypotheses(_ context.Context, p auditstore.ListHypothesesParams) (auditstore.HypothesisPage, error) {
	f.list = p
	page := auditstore.HypothesisPage{FixedDenominator: 2, PendingFixedCount: 1, Items: []auditstore.Hypothesis{f.item}}
	if p.AfterOrdinal == 0 {
		n := f.item.Ordinal
		page.NextOrdinal = &n
	}
	return page, nil
}
func (f *hypothesisAuditManagement) Hypothesis(_ context.Context, owner, audit, id string, revision uint64) (auditstore.Hypothesis, error) {
	if owner != "user-1" || audit != "audit-one" || id != f.item.Document.ID || revision != 1 {
		return auditstore.Hypothesis{}, auditstore.ErrNotFound
	}
	return f.item, nil
}
func TestAuditHypothesisOperationsConformAndBindCursors(t *testing.T) {
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	management := &hypothesisAuditManagement{item: auditstore.Hypothesis{Document: auditcontract.Hypothesis{Schema: auditcontract.HypothesisSchema, ID: "hypothesis-one", Revision: 1, Statement: "A scoped assertion", Scope: map[string]string{"actor": "reader"}, Grounds: "Analyst observation", Origin: auditcontract.HypothesisOrigin{Kind: "manual", AuditID: "audit-one"}}, SourceKey: "manual", Ordinal: 7}}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(services *testServices) { services.Audits = management })
	path := "/v1/audits/audit-one/hypotheses"
	body := `{"key":"manual","statement":"A scoped assertion","scope":{"actor":"reader"},"grounds":"Analyst observation"}`
	for _, method := range []string{http.MethodPost, http.MethodPatch} {
		target := path
		if method == http.MethodPatch {
			target += "/hypothesis-one"
		}
		response := contractCall(t, router, fixture.handler, jsonContractRequest(method, target, body, "Idempotency-Key", "write", "If-Match", `"2"`), http.StatusOK)
		if response.Header().Get("ETag") != `"3"` || management.write.OwnerID != "user-1" || management.write.ExpectedRevision != 2 || management.write.Content.Statement != "A scoped assertion" {
			t.Fatalf("lost mutation authority: %+v", management.write)
		}
	}
	first := contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, path+"?limit=1", ""), http.StatusOK)
	var page struct{ Page pageInfoResponse }
	if json.Unmarshal(first.Body.Bytes(), &page) != nil || page.Page.NextCursor == nil {
		t.Fatal("missing continuation")
	}
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, path+"?limit=1&cursor="+*page.Page.NextCursor, ""), http.StatusOK)
	if management.list.AfterOrdinal != 7 || management.list.OwnerID != "user-1" {
		t.Fatal("cursor lost scope")
	}
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, "/v1/audits/other/hypotheses?cursor="+*page.Page.NextCursor, ""), http.StatusBadRequest)
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, path+"/hypothesis-one?revision=1", ""), http.StatusOK)
	for _, query := range []string{"", "?revision=01", "?revision=1&revision=2", "?revision=0"} {
		contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, path+"/hypothesis-one"+query, ""), http.StatusBadRequest)
	}
	for _, invalid := range []string{`{"key":"manual","statement":"x","scope":{},"grounds":"x","origin":{}}`, `{"key":"a","key":"b"}`} {
		contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodPost, path, invalid, "Idempotency-Key", "bad", "If-Match", `"2"`), http.StatusBadRequest)
	}
}
