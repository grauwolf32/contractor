package public

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

type researchPageFixture struct {
	AuditManagement
	materials     auditstore.MaterialPage
	attempts      auditstore.TaskAttemptPage
	materialQuery auditstore.MaterialListParams
	attemptQuery  auditstore.TaskAttemptListParams
	err           error
}

func (f *researchPageFixture) Materials(_ context.Context, p auditstore.MaterialListParams) (auditstore.MaterialPage, error) {
	f.materialQuery = p
	return f.materials, f.err
}
func (f *researchPageFixture) TaskAttempts(_ context.Context, p auditstore.TaskAttemptListParams) (auditstore.TaskAttemptPage, error) {
	f.attemptQuery = p
	return f.attempts, f.err
}

func TestSharedAuditResearchCursorsAndContract(t *testing.T) {
	revision := "r1"
	ordinal := int64(900000)
	attempt := 1
	run := "run-one"
	f := &researchPageFixture{materials: auditstore.MaterialPage{AuditRevision: 7, NextOrdinal: &ordinal, Items: []auditstore.MaterialEntry{{Key: strings.Repeat("é", 256), CreatedAt: time.Date(2026, 11, 6, 0, 0, 0, 0, time.UTC), Material: auditcontract.Material{ScopeKind: "audit", ScopeID: "a", Ref: contracts.ArtifactRef{Namespace: "results", Name: "proposal", Revision: &revision}, Digest: auditHandlerDigest("proposal"), MediaType: "application/json", SizeBytes: 7}}}}, attempts: auditstore.TaskAttemptPage{AuditRevision: 7, NextAttempt: &attempt, Items: []auditstore.TaskAttempt{{Attempt: 1, ExecutionID: "execution-one", RunID: &run, RunDeleted: true, State: "settled", CreatedAt: time.Date(2026, 11, 6, 0, 0, 0, 0, time.UTC)}}}}
	h := auditTestHandler(f)
	document := loadPublicOpenAPI(t)
	attemptBound := document.Components.Schemas["AuditTaskAttempt"].Value.Properties["attempt"].Value.Max
	if attemptBound == nil || *attemptBound != float64(auditprofile.MaxAuditRunAttempts) {
		t.Fatal("public attempt ceiling differs from the owning Audit policy")
	}
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatal(err)
	}
	read := func(path, audit, task string, handler func(http.ResponseWriter, *http.Request) error, want int) *httptest.ResponseRecorder {
		t.Helper()
		r := auditAuthenticatedRequest("GET", "http://127.0.0.1:8080"+path, nil)
		r.Header.Set("Authorization", "Bearer "+testBearerToken)
		r.SetPathValue("auditId", audit)
		r.SetPathValue("taskId", task)
		return contractCall(t, router, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("X-Request-ID", "research-request")
			serve(handler)(w, r)
		}), r, want)
	}
	first := read("/v1/audits/a/materials?limit=1", "a", "", h.listAuditMaterials, 200)
	var page struct {
		Page pageInfoResponse `json:"page"`
	}
	if err := json.Unmarshal(first.Body.Bytes(), &page); err != nil || page.Page.NextCursor == nil {
		t.Fatalf("page: %s %v", first.Body, err)
	}
	cursor := *page.Page.NextCursor
	if len(cursor) > 512 {
		t.Fatalf("oversized cursor: %d", len(cursor))
	}
	if first.Header().Get("Cache-Control") != "no-store" {
		t.Fatal("cached live page")
	}
	read("/v1/audits/a/materials?cursor="+url.QueryEscape(cursor), "a", "", h.listAuditMaterials, 200)
	if f.materialQuery.OwnerID != "user-1" || f.materialQuery.AfterOrdinal != ordinal || f.materialQuery.AuditRevision == nil || *f.materialQuery.AuditRevision != 7 {
		t.Fatalf("lost cursor authority: %+v", f.materialQuery)
	}
	read("/v1/audits/b/materials?cursor="+url.QueryEscape(cursor), "b", "", h.listAuditMaterials, 400)
	read("/v1/audits/a/materials?cursor="+url.QueryEscape(cursor)+"&auditRevision=8", "a", "", h.listAuditMaterials, 400)
	f.err = auditstore.ErrConflict
	read("/v1/audits/a/materials?cursor="+url.QueryEscape(cursor), "a", "", h.listAuditMaterials, 409)
	f.err = nil
	first = read("/v1/audits/a/tasks/t/attempts?limit=1", "a", "t", h.listAuditTaskAttempts, 200)
	if err := json.Unmarshal(first.Body.Bytes(), &page); err != nil || page.Page.NextCursor == nil {
		t.Fatal(err)
	}
	cursor = *page.Page.NextCursor
	read("/v1/audits/a/tasks/t/attempts?cursor="+url.QueryEscape(cursor), "a", "t", h.listAuditTaskAttempts, 200)
	if f.attemptQuery.AfterAttempt != 1 || f.attemptQuery.AuditRevision == nil || *f.attemptQuery.AuditRevision != 7 {
		t.Fatalf("lost attempt pin: %+v", f.attemptQuery)
	}
	read("/v1/audits/a/tasks/other/attempts?cursor="+url.QueryEscape(cursor), "a", "other", h.listAuditTaskAttempts, 400)
	read("/v1/audits/a/materials?limit=201", "a", "", h.listAuditMaterials, 400)
}
