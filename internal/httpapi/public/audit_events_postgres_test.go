package public

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestPublicAuditEventHistory(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	store := auditstore.NewPostgresStore(pool)
	for _, auditID := range []string{"audit-events", "audit-other"} {
		_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "project-" + auditID, OwnerID: "user-1", Kind: projectstore.KindProject, Name: auditID, IdempotencyKey: auditID, RequestDigest: auditHandlerDigest(auditID)})
		if err != nil {
			t.Fatal(err)
		}
		_, _, err = store.CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: auditID, OwnerID: "user-1", ProjectID: "project-" + auditID, Profile: auditstore.ProfileIdentity{Name: "page-profile", Version: "1", Digest: auditHandlerDigest("profile")}, ProfileSnapshot: json.RawMessage(`{}`), InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1, MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024}, IdempotencyKey: auditID, RequestDigest: auditHandlerDigest(auditID)})
		if err != nil {
			t.Fatal(err)
		}
	}
	appendEvent := func(sequence int) {
		t.Helper()
		err := store.AppendReviewEvent(ctx, auditstore.ReviewEventParams{AuditID: "audit-events", Kind: "review.decided", EntityID: fmt.Sprintf("decision-%d", sequence), Summary: map[string]any{"action": "approve", "kind": "active-check-approval", "message": "secret-must-not-appear", "previousStopReason": map[string]any{"Message": "secret"}}})
		if err != nil {
			t.Fatal(err)
		}
	}
	for i := 2; i <= 201; i++ {
		appendEvent(i)
	}
	var service *auditservice.Service
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *Dependencies) {
		credentials := newFakeManagedCredentials()
		var err error
		service, err = auditservice.New(auditservice.Options{Pool: pool, Profiles: d.Config.(*config.Manager), CredentialGuard: credentials, TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (config.CredentialLookup, error) { return credentials, nil })})
		if err != nil {
			t.Fatal(err)
		}
		d.Audits = service
	})
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	type eventPage struct {
		Items   []auditservice.Event `json:"items"`
		Page    pageInfoResponse     `json:"page"`
		Total   int                  `json:"total"`
		Through uint64               `json:"throughSequence"`
	}
	read := func(path string) eventPage {
		t.Helper()
		w := serveAndValidatePublicContract(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, path, nil), true)
		if w.Code != http.StatusOK {
			t.Fatalf("%s: %d %s", path, w.Code, w.Body)
		}
		var page eventPage
		if err := json.Unmarshal(w.Body.Bytes(), &page); err != nil {
			t.Fatal(err)
		}
		return page
	}
	first := read("/v1/audits/audit-events/events?limit=200")
	if len(first.Items) != 200 || first.Total != 201 || first.Through != 201 || !first.Page.HasMore || first.Page.NextCursor == nil {
		t.Fatalf("full event page=%+v", first)
	}
	for i, event := range first.Items {
		if event.Sequence != uint64(201-i) || len(event.Summary) != 2 || event.Summary["action"] != "approve" {
			t.Fatalf("order or safe projection=%+v", event)
		}
	}
	// New events cannot shift the frozen continuation; stored events cannot be edited.
	appendEvent(202)
	_, err = pool.Exec(ctx, `UPDATE audit_events SET created_at='2026-10-07T10:00:00Z' WHERE audit_id='audit-events'`)
	var constraint *pgconn.PgError
	if !errors.As(err, &constraint) || constraint.Code != "23514" {
		t.Fatalf("event immutability = %v", err)
	}
	cursor := url.QueryEscape(*first.Page.NextCursor)
	last := read("/v1/audits/audit-events/events?limit=200&cursor=" + cursor)
	if len(last.Items) != 1 || last.Items[0].Sequence != 1 || last.Page.HasMore || last.Total != 201 || last.Through != 201 {
		t.Fatalf("immutable continuation=%+v", last)
	}
	refreshed := read("/v1/audits/audit-events/events?limit=1")
	if refreshed.Items[0].Sequence != 202 || refreshed.Total != 202 || refreshed.Through != 202 {
		t.Fatalf("fresh head=%+v", refreshed)
	}
	if _, err := service.ListEventsPage(ctx, auditservice.EventListParams{OwnerID: "foreign", AuditID: "audit-events", Limit: 1}); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner=%v", err)
	}
	for _, path := range []string{"/v1/audits/missing/events", "/v1/audits/audit-other/events?cursor=" + cursor} {
		w := httptest.NewRecorder()
		fixture.handler.ServeHTTP(w, newPublicContractRequest(http.MethodGet, path, nil))
		want := http.StatusBadRequest
		if path == "/v1/audits/missing/events" {
			want = http.StatusNotFound
		}
		if w.Code != want {
			t.Fatalf("%s=%d %s", path, w.Code, w.Body)
		}
	}
	for _, method := range []string{http.MethodHead, http.MethodPost} {
		w := httptest.NewRecorder()
		fixture.handler.ServeHTTP(w, newPublicContractRequest(method, "/v1/audits/audit-events/events", nil))
		if w.Code != http.StatusMethodNotAllowed {
			t.Fatalf("%s=%d", method, w.Code)
		}
	}
	request := httptest.NewRequest(http.MethodGet, "/v1/audits/audit-events/events", nil)
	w := httptest.NewRecorder()
	fixture.handler.ServeHTTP(w, request)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated=%d", w.Code)
	}
}
