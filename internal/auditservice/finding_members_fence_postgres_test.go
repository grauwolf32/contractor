package auditservice

import (
	"context"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Statement markers of the correlation fence load, the scalar fence recheck
// and the bounded preview batch (auditstore pgstore finding comparison SQL).
const (
	memberFenceLoadMarker  = `ORDER BY audit.audit_id COLLATE "C"`
	memberFenceCheckMarker = `unnest($4::text[],$5::bigint[])`
	memberPreviewMarker    = `receipt.comparison_projection->'preview'`
)

// memberFenceTrace records each statement of one reader connection and runs
// an optional interleaving immediately before or after the preview batch.
type memberFenceTrace struct {
	mu            sync.Mutex
	statements    []string
	beforePreview func(ids []string)
	afterPreview  func()
}

type memberPreviewKey struct{}

func (q *memberFenceTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	q.mu.Lock()
	q.statements = append(q.statements, data.SQL)
	before := q.beforePreview
	q.mu.Unlock()
	preview := strings.Contains(data.SQL, memberPreviewMarker)
	if preview && before != nil && len(data.Args) == 3 {
		ids, _ := data.Args[2].([]string)
		before(ids)
	}
	return context.WithValue(ctx, memberPreviewKey{}, preview)
}

func (q *memberFenceTrace) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryEndData) {
	preview, _ := ctx.Value(memberPreviewKey{}).(bool)
	q.mu.Lock()
	after := q.afterPreview
	q.mu.Unlock()
	if preview && after != nil {
		after()
	}
}

// take returns the recorded statements and clears both interleavings.
func (q *memberFenceTrace) take() []string {
	q.mu.Lock()
	defer q.mu.Unlock()
	out := q.statements
	q.statements, q.beforePreview, q.afterPreview = nil, nil, nil
	return out
}

func (q *memberFenceTrace) interleave(before func([]string), after func()) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.statements, q.beforePreview, q.afterPreview = nil, before, after
}

func countStatements(statements []string, marker string) int {
	count := 0
	for _, statement := range statements {
		if strings.Contains(statement, marker) {
			count++
		}
	}
	return count
}

// A member page loads the fence once and rechecks it once, after its last
// store read. A revision change after the preview batch still makes the page
// stale, and a stale fence outranks the preview error a concurrent member
// deletion causes.
func TestFindingGroupMembersRecheckFenceOnceAfterLastRead(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, url)
	const owner, project, audit = "member-fence-owner", "member-fence-project", "member-fence-a"
	createCorrelationAudit(t, ctx, pool, owner, project, audit)
	fp := correlationFingerprint(t)
	ids := make([]string, 0, 5)
	for _, suffix := range []string{"member-fence-1", "member-fence-2", "member-fence-3", "member-fence-4", "member-fence-5"} {
		ids = append(ids, seedAuditFindingDocument(t, ctx, pool, project, owner, audit, suffix, auditFindingDocument(suffix), fp))
	}
	var group string
	if err := pool.QueryRow(ctx, "SELECT correlation_group_key FROM audit_findings WHERE finding_id=$1", ids[0]).Scan(&group); err != nil || group == "" {
		t.Fatalf("seeded group=%q %v", group, err)
	}
	trace := &memberFenceTrace{}
	config := pool.Config()
	config.MaxConns, config.MinConns = 1, 0
	config.ConnConfig.Tracer = trace
	reader, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	service := &Service{pool: reader, now: time.Now}
	projectScope := FindingMembersParams{OwnerID: owner, ProjectID: project, GroupKey: group, Limit: 2}
	auditScope := projectScope
	auditScope.AuditIDs = []string{audit}

	for name, params := range map[string]FindingMembersParams{"project": projectScope, "audit": auditScope} {
		trace.take()
		page, err := service.FindingGroupMembers(ctx, params)
		statements := trace.take()
		if err != nil || len(page.Items) != 2 || page.Next == nil {
			t.Fatalf("%s page=%+v %v", name, page, err)
		}
		if loads, checks := countStatements(statements, memberFenceLoadMarker), countStatements(statements, memberFenceCheckMarker); loads != 1 || checks != 1 {
			t.Fatalf("%s page loaded the fence %d times and rechecked it %d times, want 1 and 1", name, loads, checks)
		}
		if previews := countStatements(statements, memberPreviewMarker); previews != 1 {
			t.Fatalf("%s page ran %d preview batches", name, previews)
		}
		if last := statements[len(statements)-1]; !strings.Contains(last, memberFenceCheckMarker) {
			t.Fatalf("%s page read after its fence recheck: %s", name, last)
		}
	}

	// A fence change after the last read is detected by the one recheck.
	for name, params := range map[string]FindingMembersParams{"project": projectScope, "audit": auditScope} {
		victim := ids[len(ids)-1]
		ids = ids[:len(ids)-1]
		fired := false
		trace.interleave(nil, func() {
			if fired {
				return
			}
			fired = true
			if _, err := pool.Exec(ctx, "DELETE FROM audit_findings WHERE finding_id=$1", victim); err != nil {
				t.Fatal(err)
			}
		})
		if _, err := service.FindingGroupMembers(ctx, params); !errors.Is(err, auditstore.ErrCorrelationStale) {
			t.Fatalf("%s fence change after the preview batch=%v", name, err)
		}
		if !fired {
			t.Fatalf("%s preview interleaving did not run", name)
		}
	}

	// Deleting a selected member between the member seek and its preview
	// makes the preview batch incomplete; the changed fence is reported.
	var removed []string
	trace.interleave(func(selected []string) {
		if removed != nil || len(selected) == 0 {
			return
		}
		removed = append([]string{}, selected...)
		if _, err := pool.Exec(ctx, "DELETE FROM audit_findings WHERE finding_id=$1", selected[0]); err != nil {
			t.Fatal(err)
		}
	}, nil)
	if _, err := service.FindingGroupMembers(ctx, auditScope); !errors.Is(err, auditstore.ErrCorrelationStale) {
		t.Fatalf("member deletion before the preview batch=%v", err)
	}
	trace.take()
	if len(removed) != 2 {
		t.Fatalf("preview interleaving selected %v", removed)
	}
	if _, err := service.reads().ComparisonPreviews(ctx, owner, project, removed); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("preview batch without the deleted member=%v, want the read error the fence outranked", err)
	}
}
