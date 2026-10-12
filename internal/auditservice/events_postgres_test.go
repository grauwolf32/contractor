package auditservice

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestAuditEventPagesReadOnlyTheirEvents covers V370-027: the event total is
// the dense sequence (AUD-R223), so reading all N events in pages reads each
// event once and counts none of them.
func TestAuditEventPagesReadOnlyTheirEvents(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	store := auditstoretest.NewStore(pool)
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-events", OwnerID: "owner-events", Kind: projectstore.KindProject,
		Name: "Events", IdempotencyKey: "project-events", RequestDigest: serviceTestDigest("project-events"),
	}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-events", OwnerID: "owner-events", ProjectID: "project-events",
		Profile:         auditstore.ProfileIdentity{Name: "events", Version: "1", Digest: serviceTestDigest("events-profile")},
		ProfileSnapshot: json.RawMessage(`{}`), InputSelection: json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
		},
		IdempotencyKey: "audit-events", RequestDigest: serviceTestDigest("audit-events"),
	}); err != nil {
		t.Fatal(err)
	}
	const events = 250
	appended := make([]auditstore.ReviewEventParams, 0, events-1)
	for index := 2; index <= events; index++ {
		appended = append(appended, auditstore.ReviewEventParams{
			AuditID: "audit-events", Kind: "review.decided", EntityID: fmt.Sprintf("decision-%d", index),
			Summary: map[string]any{"action": "approve"},
		})
	}
	if err := store.AppendReviewEvents(ctx, appended); err != nil {
		t.Fatal(err)
	}

	tracer := &eventQueryTracer{}
	tracedConfig := pool.Config()
	tracedConfig.ConnConfig.Tracer = tracer
	traced, err := pgxpool.NewWithConfig(ctx, tracedConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer traced.Close()
	service := &Service{pool: traced, now: time.Now}

	params := auditstore.EventListParams{OwnerID: "owner-events", AuditID: "audit-events", Limit: 101}
	read := 0
	for {
		page, err := service.ListEventsPage(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		if page.ThroughSequence != events || page.Total != events {
			t.Fatalf("event page basis = through %d, total %d, want %d", page.ThroughSequence, page.Total, events)
		}
		read += len(page.Items)
		if len(page.Items) < params.Limit {
			break
		}
		through, before := page.ThroughSequence, page.Items[len(page.Items)-1].Sequence
		params.ThroughSequence, params.BeforeSequence = &through, &before
	}
	if read != events {
		t.Fatalf("read %d events, want %d", read, events)
	}
	eventRows, eventStatements := tracer.eventReads()
	if eventRows != events || eventStatements != 3 {
		t.Fatalf("event pages read %d rows in %d statements (%v), want %d rows in 3", eventRows, eventStatements, tracer.queries, events)
	}
}

// eventQueryTracer records the statements that read audit_events and the
// rows they returned.
type eventQueryTracer struct {
	mu      sync.Mutex
	queries []string
	rows    int64
}

type eventQueryKey struct{}

func (r *eventQueryTracer) TraceQueryStart(
	ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData,
) context.Context {
	if !strings.Contains(data.SQL, "audit_events") {
		return ctx
	}
	r.mu.Lock()
	r.queries = append(r.queries, data.SQL)
	r.mu.Unlock()
	return context.WithValue(ctx, eventQueryKey{}, true)
}

func (r *eventQueryTracer) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryEndData) {
	if ctx.Value(eventQueryKey{}) == nil {
		return
	}
	r.mu.Lock()
	r.rows += data.CommandTag.RowsAffected()
	r.mu.Unlock()
}

func (r *eventQueryTracer) eventReads() (int, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return int(r.rows), len(r.queries)
}
