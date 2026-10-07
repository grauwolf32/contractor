package public

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
)

func TestAuditEventPaginationAndCursorBinding(t *testing.T) {
	fake := &fakeAuditManagement{events: auditservice.EventPage{ThroughSequence: 201, Total: 201}}
	for sequence := uint64(201); sequence > 0; sequence-- {
		fake.events.Items = append(fake.events.Items, auditservice.Event{AuditID: "audit-1", Sequence: sequence, Kind: "audit.created", EntityID: "audit-1", Summary: map[string]any{}, CreatedAt: time.Now().UTC()})
	}
	h := auditTestHandler(fake)
	read := func(path, auditID string) *httptest.ResponseRecorder {
		r := auditAuthenticatedRequest(http.MethodGet, path, nil)
		r.SetPathValue("auditId", auditID)
		w := httptest.NewRecorder()
		h.listAuditEvents(w, r)
		return w
	}
	w := read("/v1/audits/audit-1/events?limit=200", "audit-1")
	var body struct {
		Items []auditservice.Event `json:"items"`
		Page  pageInfoResponse     `json:"page"`
		Total int                  `json:"total"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if w.Code != http.StatusOK || w.Header().Get("Cache-Control") != "no-store" || len(body.Items) != 200 || !body.Page.HasMore || body.Page.NextCursor == nil || body.Total != 201 || fake.eventParams.Limit != 201 || fake.eventParams.OwnerID != "user-1" {
		t.Fatalf("page = %d %s params=%+v", w.Code, w.Body, fake.eventParams)
	}
	cursor := url.QueryEscape(*body.Page.NextCursor)
	w = read("/v1/audits/audit-1/events?limit=1&cursor="+cursor, "audit-1")
	if w.Code != http.StatusOK || fake.eventParams.ThroughSequence == nil || *fake.eventParams.ThroughSequence != 201 || *fake.eventParams.BeforeSequence != 2 {
		t.Fatalf("continuation = %d params=%+v", w.Code, fake.eventParams)
	}
	for _, path := range []string{"?limit=201", "?limit=0", "?limit=1&limit=2", "?state=active", "?cursor=", "?cursor=invalid"} {
		if w := read("/v1/audits/audit-1/events"+path, "audit-1"); w.Code != http.StatusBadRequest {
			t.Fatalf("%s = %d", path, w.Code)
		}
	}
	if w := read("/v1/audits/audit-2/events?cursor="+cursor, "audit-2"); w.Code != http.StatusBadRequest {
		t.Fatalf("foreign Audit cursor = %d", w.Code)
	}
	r := auditAuthenticatedRequest(http.MethodHead, "/v1/audits/audit-1/events", nil)
	w = httptest.NewRecorder()
	h.listAuditEvents(w, r)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("HEAD = %d", w.Code)
	}
}
