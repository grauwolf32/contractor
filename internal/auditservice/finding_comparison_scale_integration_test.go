//go:build integration

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

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type correlationScaleQuery struct {
	sql  string
	args []any
}
type correlationScaleTrace struct {
	mu      sync.Mutex
	queries []correlationScaleQuery
}

func (q *correlationScaleTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.queries = append(q.queries, correlationScaleQuery{data.SQL, append([]any{}, data.Args...)})
	return ctx
}
func (*correlationScaleTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}
func (q *correlationScaleTrace) take() []correlationScaleQuery {
	q.mu.Lock()
	defer q.mu.Unlock()
	out := q.queries
	q.queries = nil
	return out
}

func TestFindingComparisonMillionRowScopeUsesBoundedIndexedReads(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 8*time.Minute)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, url)
	const owner, project = "comparison-scale-owner", "comparison-scale-project"
	for n := range 100 {
		createCorrelationAudit(t, ctx, pool, owner, project, fmt.Sprintf("scale-audit-%03d", n))
	}
	source := seedAuditFindingDocument(t, ctx, pool, project, owner, "scale-audit-000", "million-source", auditFindingDocument("million-source"), correlationFingerprint(t))
	// Fixture loading is separate from the read budget. Real production tables,
	// constraints, primary derivation and index maintenance remain enabled. The
	// small intake tests separately prove receipt/evidence admission semantics.
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_receipts
 SELECT copied.* FROM finding_proposal_receipts AS original CROSS JOIN generate_series(1,99999) AS n
 CROSS JOIN LATERAL jsonb_populate_record(NULL::finding_proposal_receipts,to_jsonb(original)||jsonb_build_object(
 'receipt_id','million-receipt-'||n,'proposal_id','million-proposal-'||n,'invocation_id','million-invocation-'||n,
 'submission_id','million-submission-'||n,'client_key','million-client-'||n,
 'fingerprint_projection',jsonb_set(original.fingerprint_projection::jsonb,'{value}',to_jsonb(CASE WHEN n<=50000
 THEN repeat('a',64) ELSE encode(sha256(convert_to(n::text,'UTF8')),'hex') END))::text)) AS copied
 WHERE original.receipt_id='receipt-million-source'`); err != nil {
		t.Fatal(err)
	}
	t.Log("loaded 100000 compact retained receipts")
	if _, err := pool.Exec(ctx, `INSERT INTO audit_findings(finding_id,audit_id,first_receipt_id,first_proposal_ref)
 SELECT 'million-finding-'||lpad(n::text,7,'0'),
 CASE WHEN n<99999 THEN 'scale-audit-000' ELSE 'scale-audit-'||lpad((1+(n-99999)%99)::text,3,'0') END,
 'million-receipt-'||(CASE WHEN n<99999 THEN n+1 ELSE 1+(n-99999)/99 END)::text,
 original.first_proposal_ref FROM generate_series(0,999998) AS n
 CROSS JOIN audit_findings AS original WHERE original.finding_id=$1`, source); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "ANALYZE audit_findings"); err != nil {
		t.Fatal(err)
	}
	var audits, findings, largest int
	if err := pool.QueryRow(ctx, `SELECT count(DISTINCT audit_id),count(*),max(n) FROM
 (SELECT audit_id,count(*) OVER(PARTITION BY audit_id) AS n FROM audit_findings) AS selected`).Scan(&audits, &findings, &largest); err != nil {
		t.Fatal(err)
	}
	if audits != 100 || findings != 1000000 || largest != 100000 {
		t.Fatalf("wrong scale fixture: audits=%d findings=%d largest=%d", audits, findings, largest)
	}
	t.Log("loaded 100 Audits / 1000000 findings; largest Audit has 100000 findings")
	trace := &correlationScaleTrace{}
	config := pool.Config()
	config.MaxConns, config.MinConns = 1, 0
	config.ConnConfig.Tracer = trace
	config.ConnConfig.RuntimeParams["plan_cache_mode"] = "force_generic_plan"
	reader, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	service := &Service{pool: reader, now: time.Now}
	trace.take()
	comparison, err := service.CompareFindings(ctx, FindingComparisonParams{OwnerID: owner, ProjectID: project, LeftAuditID: "scale-audit-001", RightAuditID: "scale-audit-002", Limit: 100})
	if err != nil || len(comparison.Items) != 1 || comparison.Items[0].Kind != "both" {
		t.Fatalf("large pair=%+v %v", comparison, err)
	}
	queries := trace.take()
	if len(queries) > 6 {
		t.Fatalf("pair read grew with parents: %d queries", len(queries))
	}
	assertCorrelationScalePlans(t, ctx, pool, queries, 1000)
	group := comparison.Items[0].GroupKey
	members, err := service.FindingGroupMembers(ctx, FindingMembersParams{OwnerID: owner, ProjectID: project, GroupKey: group, AuditIDs: []string{"scale-audit-001", "scale-audit-002"}, Limit: 100})
	if err != nil || len(members.Items) != 100 || !members.Page.HasMore || members.Representative.AuditID == "scale-audit-000" {
		t.Fatalf("large pair members=%d rep=%+v %v", len(members.Items), members.Representative, err)
	}
	assertCorrelationScalePlans(t, ctx, pool, trace.take(), 1000)
	candidates, err := service.FindingCandidates(ctx, FindingCandidatesParams{OwnerID: owner, AuditID: "scale-audit-000", FindingID: source, Scope: "project", Limit: 100})
	if err != nil || len(candidates.Items) != 100 || candidates.RowsExamined != auditstore.FindingCandidateWindowRows || !candidates.WorkLimited {
		t.Fatalf("large candidate page=%d examined=%d %v", len(candidates.Items), candidates.RowsExamined, err)
	}
	assertCorrelationScalePlans(t, ctx, pool, trace.take(), 1000)
	// Deep continuations must seek from the cursor even under generic plans,
	// rather than filtering hundreds of thousands of earlier index entries.
	var deepTime time.Time
	var deepID string
	if err := pool.QueryRow(ctx, `SELECT created_at,finding_id FROM audit_findings WHERE correlation_project_id=$1 AND correlation_group_key=$2
 ORDER BY created_at DESC,finding_id COLLATE "C" DESC OFFSET 1500 LIMIT 1`, project, group).Scan(&deepTime, &deepID); err != nil {
		t.Fatal(err)
	}
	deep, err := service.FindingCandidates(ctx, FindingCandidatesParams{OwnerID: owner, AuditID: "scale-audit-000", FindingID: source, Scope: "project", Limit: 100, Cursor: correlationPosition(candidates.Fence, 1, deepID, deepTime.UTC().Format(time.RFC3339Nano))})
	if err != nil || len(deep.Items) != 100 {
		t.Fatalf("deep candidate page=%d %v", len(deep.Items), err)
	}
	assertCorrelationScalePlans(t, ctx, pool, trace.take(), 1000)
	weakDocument := auditFindingDocument("million-weak-source")
	weakDocument.StandardRefs = []auditdomain.StandardReference{{Scheme: "CWE", Version: "4.20", RequirementID: "CWE-89"}}
	weak := seedAuditFindingDocument(t, ctx, pool, project, owner, "scale-audit-001", "million-weak-source", weakDocument)
	progress, err := service.FindingCandidates(ctx, FindingCandidatesParams{OwnerID: owner, AuditID: "scale-audit-001", FindingID: weak, Scope: "project", Limit: 100})
	if err != nil || len(progress.Items) != 0 || !progress.Page.HasMore || progress.Next == nil || progress.RowsExamined != auditstore.FindingCandidateWindowRows {
		t.Fatalf("empty bounded progress=%+v %v", progress, err)
	}
	assertCorrelationScalePlans(t, ctx, pool, trace.take(), 1000)
}

func assertCorrelationScalePlans(t *testing.T, ctx context.Context, pool *pgxpool.Pool, queries []correlationScaleQuery, budget int) {
	t.Helper()
	examined := 0
	proved := 0
	for _, query := range queries {
		if !strings.Contains(query.sql, "audit_findings") {
			continue
		}
		tx, err := pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tx.Exec(ctx, "SET LOCAL plan_cache_mode=force_generic_plan"); err != nil {
			_ = tx.Rollback(ctx)
			t.Fatal(err)
		}
		var raw []byte
		err = tx.QueryRow(ctx, "EXPLAIN (ANALYZE,FORMAT JSON) "+query.sql, query.args...).Scan(&raw)
		rollbackErr := tx.Rollback(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if rollbackErr != nil {
			t.Fatal(rollbackErr)
		}
		var roots []struct{ Plan map[string]any }
		if err := json.Unmarshal(raw, &roots); err != nil {
			t.Fatal(err)
		}
		var visit func(map[string]any)
		visit = func(node map[string]any) {
			if node["Relation Name"] == "audit_findings" {
				kind, _ := node["Node Type"].(string)
				if kind != "Index Scan" && kind != "Index Only Scan" {
					t.Fatalf("unbounded finding access %s: %s", kind, raw)
				}
				rows, _ := node["Actual Rows"].(float64)
				loops, _ := node["Actual Loops"].(float64)
				removed, _ := node["Rows Removed by Filter"].(float64)
				examined += int((rows + removed) * loops)
				proved++
			}
			children, _ := node["Plans"].([]any)
			for _, child := range children {
				visit(child.(map[string]any))
			}
		}
		visit(roots[0].Plan)
	}
	if proved == 0 || examined > budget {
		t.Fatalf("indexed work proof failed: %d probes, %d finding rows (budget %d)", proved, examined, budget)
	}
	t.Logf("production query plans examined %d indexed finding rows, budget %d", examined, budget)
}
