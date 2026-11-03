package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// auditEventDensityViolationsSQL lists the Audits whose events are not
// numbered 1..N with N = next_event_sequence - 1.
const auditEventDensityViolationsSQL = `
SELECT audit.audit_id, events.count, events.first_sequence, events.last_sequence,
       audit.next_event_sequence - 1
  FROM audits AS audit
  CROSS JOIN LATERAL (
      SELECT count(*) AS count,
             COALESCE(min(event.sequence_number), 1) AS first_sequence,
             COALESCE(max(event.sequence_number), 0) AS last_sequence
        FROM audit_events AS event
       WHERE event.audit_id = audit.audit_id
  ) AS events
 WHERE events.count <> audit.next_event_sequence - 1
    OR events.last_sequence <> audit.next_event_sequence - 1
    OR events.first_sequence <> 1
 ORDER BY audit.audit_id`

// requireDenseAuditEvents fails the test for every Audit whose event
// sequence has a gap, a duplicate count or a reservation without events.
func requireDenseAuditEvents(t *testing.T, ctx context.Context, db persistencepostgres.DBTX) {
	t.Helper()
	rows, err := db.Query(ctx, auditEventDensityViolationsSQL)
	if err != nil {
		t.Errorf("check Audit event density: %v", err)
		return
	}
	defer rows.Close()
	for rows.Next() {
		var auditID string
		var count, first, last, reserved int64
		if err := rows.Scan(&auditID, &count, &first, &last, &reserved); err != nil {
			t.Errorf("scan Audit event density: %v", err)
			return
		}
		t.Errorf("Audit %s has %d events numbered %d..%d, want 1..%d", auditID, count, first, last, reserved)
	}
	if err := rows.Err(); err != nil {
		t.Errorf("iterate Audit event density: %v", err)
	}
}

func TestPostgresAuditEventAllocatorRejectsBrokenReservations(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-allocator", OwnerID: "owner-allocator", Kind: projectstore.KindProject,
		Name: "Allocator", IdempotencyKey: "project-create", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	audit, _, err := newTestStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-allocator", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: "checklist", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"checklist"}`), InputSelection: json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1,
		},
		IdempotencyKey: "audit-create", RequestDigest: testDigest("3"),
	})
	if err != nil || audit.EventSequence != 1 {
		t.Fatalf("create Audit = (%+v, %v), want event sequence 1", audit, err)
	}
	const appendSQL = `
WITH changed AS (
    UPDATE audits AS audit
       SET next_event_sequence = audit.next_event_sequence + $2
     WHERE audit.audit_id = $1
    RETURNING contractor_append_audit_events(audit.audit_id, $3::jsonb) AS last_event_sequence
)
SELECT last_event_sequence FROM changed`
	event := `{"kind":"review.requested","entityId":"review-allocator","entityRevision":1,"summary":{}}`
	for _, broken := range []struct {
		name     string
		reserved int
		events   string
	}{
		{"reserved more than appended", 2, "[" + event + "]"},
		{"reserved fewer than appended", 1, "[" + event + "," + event + "]"},
		{"appended without a reservation", 0, "[" + event + "]"},
		{"not an array", 1, event},
	} {
		var last int64
		err := pool.QueryRow(ctx, appendSQL, audit.AuditID, broken.reserved, broken.events).Scan(&last)
		var postgresError *pgconn.PgError
		if !errors.As(err, &postgresError) || postgresError.Code != persistencepostgres.SQLStateAssertFailure {
			t.Errorf("%s = (%d, %v), want SQLSTATE P0004", broken.name, last, err)
		}
	}
	var direct int64
	err = pool.QueryRow(ctx, `SELECT contractor_append_audit_events($1, $2::jsonb)`,
		audit.AuditID, "["+event+"]").Scan(&direct)
	var postgresError *pgconn.PgError
	if !errors.As(err, &postgresError) || postgresError.Code != persistencepostgres.SQLStateAssertFailure {
		t.Errorf("unreserved direct append = (%d, %v), want SQLSTATE P0004", direct, err)
	}
	var last int64
	if err := pool.QueryRow(ctx, appendSQL, audit.AuditID, 2, "["+event+","+event+"]").Scan(&last); err != nil || last != 3 {
		t.Fatalf("reserved append = (%d, %v), want last sequence 3", last, err)
	}
	stored, err := newTestStore(pool).Get(ctx, audit.OwnerID, audit.AuditID)
	if err != nil || stored.EventSequence != 3 {
		t.Fatalf("Audit after appends = (%+v, %v), want event sequence 3", stored, err)
	}
	requireDenseAuditEvents(t, ctx, pool)
}

// TestPostgresAuditEventAllocatorIsTheOnlyEventWriter checks the migrated
// schema: one routine inserts Audit events, and the only other routine that
// advances an Audit's event counter hands its events to it.
func TestPostgresAuditEventAllocatorIsTheOnlyEventWriter(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	routines := func(pattern string) []string {
		rows, err := pool.Query(ctx, `
SELECT routine.proname
  FROM pg_proc AS routine
  JOIN pg_namespace AS namespace ON namespace.oid = routine.pronamespace
 WHERE namespace.nspname = current_schema() AND `+explaintest.RoutineBodySQL("routine")+` ~* $1
 ORDER BY routine.proname`, pattern)
		if err != nil {
			t.Fatal(err)
		}
		names, err := pgx.CollectRows(rows, pgx.RowTo[string])
		if err != nil {
			t.Fatal(err)
		}
		return names
	}
	if writers := routines(`insert\s+into\s+audit_events\M`); !slices.Equal(writers, []string{"contractor_append_audit_events"}) {
		t.Errorf("routines inserting Audit events = %q, want only the allocator", writers)
	}
	reservations := routines(`next_event_sequence\s*=`)
	if len(reservations) != 0 {
		t.Errorf("routines advancing Audit event counters = %q", reservations)
	}
	callsAllocatorSQL := `SELECT ` + explaintest.RoutineBodySQL("") + ` ~ 'contractor_append_audit_events\(' FROM pg_proc
 WHERE proname = $1 AND pronamespace = current_schema()::regnamespace`
	for _, name := range reservations {
		var callsAllocator bool
		if err := pool.QueryRow(ctx, callsAllocatorSQL, name).Scan(&callsAllocator); err != nil || !callsAllocator {
			t.Errorf("%s reserves Audit events without the allocator (%t, %v)", name, callsAllocator, err)
		}
	}
	// The same three inspections must also see a SQL-standard routine.
	// It is parsed and stored, never invoked, so no Audit/event is mutated.
	if _, err := pool.Exec(ctx, `CREATE FUNCTION contractor_test_event_source() RETURNS void LANGUAGE sql
BEGIN ATOMIC
    SELECT contractor_append_audit_events(NULL::text, '[]'::jsonb);
    INSERT INTO audit_events SELECT * FROM audit_events WHERE false;
    UPDATE audits SET next_event_sequence = next_event_sequence WHERE false;
END`); err != nil {
		t.Fatal(err)
	}
	for _, pattern := range []string{`insert\s+into\s+audit_events\M`, `next_event_sequence\s*=`} {
		if !slices.Contains(routines(pattern), "contractor_test_event_source") {
			t.Errorf("SQL-standard routine is invisible to %s", pattern)
		}
	}
	var callsAllocator bool
	if err := pool.QueryRow(ctx, callsAllocatorSQL, "contractor_test_event_source").Scan(&callsAllocator); err != nil || !callsAllocator {
		t.Fatalf("SQL-standard allocator call = %t, %v", callsAllocator, err)
	}
}

// Common Task acceptance, collection and owner transitions allocate dense events.
func TestSharedPostgresCommonTaskEventsStayDense(t *testing.T) {
	f := newCollectingAuditFixture(t, "dense-task-events", 1<<20)
	gap := "fixture-failure"
	_, _, err := f.store.Collect(f.ctx, auditstore.CollectParams{Claim: f.claim, ExecutionID: f.execution.ExecutionID, ReceiptID: "dense-task-receipt", RequestDigest: testDigest("e"), Disposition: auditstore.CollectionMissingOutput, ErrorCode: &gap, Tasks: []auditstore.TaskCollection{{ExecutionItemID: f.memberID, TaskID: f.itemID, GapCode: &gap}}})
	if err != nil {
		t.Fatal(err)
	}
	current, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = f.store.Transition(f.ctx, auditstore.TransitionParams{OwnerID: current.OwnerID, AuditID: current.AuditID, ExpectedRevision: current.Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused, IdempotencyKey: "pause", RequestDigest: testDigest("d")})
	if err != nil {
		t.Fatal(err)
	}
	current, err = f.store.Get(f.ctx, current.OwnerID, current.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	var count, minimum, maximum int64
	if err := f.pool.QueryRow(f.ctx, "SELECT count(*),min(sequence_number),max(sequence_number) FROM audit_events WHERE audit_id=$1", current.AuditID).Scan(&count, &minimum, &maximum); err != nil {
		t.Fatal(err)
	}
	if minimum != 1 || maximum != count || uint64(maximum) != current.EventSequence {
		t.Fatalf("event range=%d..%d count=%d parent=%d", minimum, maximum, count, current.EventSequence)
	}
}
