//go:build integration

package auditservice

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type findingPageTrace struct{ reads []correlationQueryTrace }

func (t *findingPageTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	t.reads = append(t.reads, correlationQueryTrace{sql: data.SQL, args: append([]any{}, data.Args...)})
	return ctx
}
func (*findingPageTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// Cardinality is deliberately independent of Audits and one Finding's source
// fanout. Every query below also runs under a generic prepared plan.
func TestProjectFindingPagesStayBoundedAtTenThousandFindingsAndFiftyThousandSources(t *testing.T) {
	f := newProjectFindingFixture(t)
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Minute)
	defer cancel()
	// Seed immutable rows from a real retained, validated source. The canonical
	// state stays unassessed; no legacy Finding or review machinery participates.
	var columns []string
	if err := f.pool.QueryRow(ctx, `SELECT array_agg(attname::text ORDER BY attnum) FROM pg_attribute WHERE attrelid='finding_proposal_receipts'::regclass AND attnum>0 AND NOT attisdropped AND attgenerated=''`).Scan(&columns); err != nil {
		t.Fatal(err)
	}
	projection := make([]string, len(columns))
	for i, column := range columns {
		columns[i] = pgx.Identifier{column}.Sanitize()
		projection[i] = "cloned." + columns[i]
	}
	seedSQL := strings.NewReplacer("{{columns}}", strings.Join(columns, ","), "{{projection}}", strings.Join(projection, ",")).Replace(projectFindingScaleReceiptsSQL)
	if _, err := f.pool.Exec(ctx, seedSQL, f.project); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, projectFindingScaleFindingsSQL, f.project, f.finding); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, projectFindingScaleSourcesSQL, f.project, f.finding); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, `INSERT INTO project_finding_audits(audit_id,finding_id,project_id,finding_created_at,correlation_group_key)
 SELECT $1,finding_id,project_id,created_at,correlation_group_key FROM project_findings WHERE project_id=$2 ON CONFLICT DO NOTHING`, f.first, f.project); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"project_findings", "project_finding_sources", "project_finding_audits", "finding_proposal_receipts"} {
		if _, err := f.pool.Exec(ctx, "ANALYZE "+table); err != nil {
			t.Fatal(err)
		}
	}
	var findings, sources, fanout int
	if err := f.pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM project_findings WHERE project_id=$1),(SELECT count(*) FROM project_finding_sources WHERE project_id=$1),(SELECT count(*) FROM project_finding_sources WHERE finding_id=$2)`, f.project, f.finding).Scan(&findings, &sources, &fanout); err != nil || findings != 10000 || sources != 50000 || fanout != 10000 {
		t.Fatalf("scale=%d/%d/%d: %v", findings, sources, fanout, err)
	}

	// A different owner has a larger, newer history. Owner pagination must
	// seek its own immutable owner index before looking up Project visibility.
	createCorrelationAudit(t, ctx, f.pool, "foreign-owner", "foreign-project", "foreign-audit")
	foreignFinding := seedAuditFinding(t, ctx, f.pool, "foreign-project", "foreign-owner", "foreign-audit", "foreign-base")
	foreignReceipts := strings.NewReplacer("{{columns}}", strings.Join(columns, ","), "{{projection}}", strings.Join(projection, ",")).Replace(projectFindingForeignScaleReceiptsSQL)
	if _, err := f.pool.Exec(ctx, foreignReceipts); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, projectFindingForeignScaleFindingsSQL, foreignFinding); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, `INSERT INTO project_finding_sources(receipt_id,project_id,finding_id,created_at)
 SELECT first_receipt_id,project_id,finding_id,created_at FROM project_findings WHERE project_id='foreign-project' ON CONFLICT DO NOTHING`); err != nil {
		t.Fatal(err)
	}

	if _, err := f.pool.Exec(ctx, `ANALYZE project_findings`); err != nil {
		t.Fatal(err)
	}
	trace := &findingPageTrace{}
	cfg := f.pool.Config()
	cfg.MaxConns = 1
	cfg.MinConns = 0
	cfg.ConnConfig.Tracer = trace
	cfg.ConnConfig.RuntimeParams["plan_cache_mode"] = "force_generic_plan"
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	intake, err := findingintake.New(pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{storage: artifactpgtest.NewStorage(), pool: pool, findings: intake, now: time.Now}
	var after *time.Time
	var afterID string
	seen := map[string]bool{}
	var revision *uint64
	var summaryProbes, sourceProbes []correlationQueryTrace
	for {
		trace.reads = nil
		page, err := service.ListProjectFindings(ctx, ProjectFindingListParams{OwnerFindingListParams: auditstore.OwnerFindingListParams{OwnerID: f.owner, ProjectID: f.project, BeforeCreatedAt: after, BeforeFindingID: afterID, Limit: 129}, ProjectRevision: revision})
		if err != nil {
			t.Fatal(err)
		}
		if len(trace.reads) != 3 {
			t.Fatalf("canonical list ran %d statements; expected fence/page/fence without per-row hydration", len(trace.reads))
		}
		if len(seen) == 0 || len(page.Items) < 128 {
			for _, q := range trace.reads {
				if len(q.args) == 9 {
					summaryProbes = append(summaryProbes, q)
				}
			}
		}
		payload, _ := json.Marshal(page)
		if len(payload) > 1<<20 || len(page.Items) > 128 {
			t.Fatalf("oversized summary page: %d bytes/%d rows", len(payload), len(page.Items))
		}
		if revision == nil {
			v := page.ProjectRevision
			revision = &v
		}
		for _, row := range page.Items {
			if seen[row.FindingID] || row.State != nil || row.Revision != 1 {
				t.Fatalf("repeated or invented canonical state: %+v", row)
			}
			seen[row.FindingID] = true
		}
		if len(page.Items) < 128 {
			break
		}
		last := page.Items[len(page.Items)-1]
		after, afterID = &last.CreatedAt, last.FindingID
	}
	if len(seen) != 10000 {
		t.Fatalf("reached %d of 10000 Findings", len(seen))
	}
	// One source page never loads proposals/evidence, all 10000 source records
	// of the largest Finding remain reachable, and work stays three statements.
	visited := map[string]bool{}
	var sourceTime *time.Time
	sourceID := ""
	for {
		trace.reads = nil
		page, err := intake.ListProjectSources(ctx, f.owner, f.project, f.finding, findingintake.ProjectSourceQuery{ListQuery: findingintake.ListQuery{Limit: 64, AfterCreatedAt: sourceTime, AfterReceiptID: sourceID}, ProjectRevision: revision})
		if err != nil {
			t.Fatal(err)
		}
		if len(trace.reads) != 3 {
			t.Fatalf("source page ran %d statements", len(trace.reads))
		}
		if len(visited) == 0 || len(page.Items) < 64 {
			for _, q := range trace.reads {
				if len(q.args) == 6 {
					sourceProbes = append(sourceProbes, q)
				}
			}
		}
		payload, _ := json.Marshal(page)
		if len(payload) > 1<<20 || len(page.Items) > 64 {
			t.Fatalf("oversized source page %d/%d", len(payload), len(page.Items))
		}
		for _, source := range page.Items {
			if visited[source.ReceiptID] || source.FindingID == nil || *source.FindingID != f.finding {
				t.Fatal("source cursor repeated or crossed canonical identity")
			}
			visited[source.ReceiptID] = true
		}
		if len(page.Items) < 64 {
			break
		}
		last := page.Items[len(page.Items)-1]
		sourceTime, sourceID = &last.CreatedAt, last.ReceiptID
	}
	if len(visited) != 10000 {
		t.Fatalf("reached %d of 10000 sources", len(visited))
	}
	// Physical work is bounded for first and deep pages under generic plans.
	if len(summaryProbes) != 2 || len(sourceProbes) != 2 {
		t.Fatalf("missing production probes: %d/%d", len(summaryProbes), len(sourceProbes))
	}
	for _, q := range summaryProbes {
		plan, err := explaintest.ExplainPrepared(ctx, f.pool, explaintest.Generic, q.sql, q.args...)
		if err != nil {
			t.Fatal(err)
		}
		if reads := plan.Reads.Of("project_findings"); reads > 160 {
			t.Fatalf("summary page read %d Findings, budget 160\n%s", reads, plan.Plan.String())
		}
		for _, table := range []string{"project_finding_sources", "finding_proposal_receipts"} {
			if reads := plan.Reads.Of(table); reads != 0 {
				t.Fatalf("summary hydrated %d %s rows", reads, table)
			}
		}
	}
	for _, q := range sourceProbes {
		plan, err := explaintest.ExplainPrepared(ctx, f.pool, explaintest.Generic, q.sql, q.args...)
		if err != nil {
			t.Fatal(err)
		}
		for _, table := range []string{"project_finding_sources", "finding_proposal_receipts"} {
			if reads := plan.Reads.Of(table); reads > 80 {
				t.Fatalf("source page read %d %s rows, budget 80\n%s", reads, table, plan.Plan.String())
			}
		}
	}
	// A rare filter must not make PostgreSQL search all 10,000 memberships
	// to fill a page. Empty windows expose progress through the same cursor.
	confirmed := auditstore.FindingConfirmed
	for _, filter := range []struct {
		name       string
		state      *auditstore.FindingState
		unreviewed bool
		want       int
	}{
		{"unassessed", nil, true, 10000}, {"no confirmed matches", &confirmed, false, 0},
	} {
		var positionTime *time.Time
		positionID := ""
		reached := map[string]bool{}
		progressed := map[string]bool{}
		var firstProbe, lastProbe *correlationQueryTrace
		for pass := 0; ; pass++ {
			if pass > 80 {
				t.Fatalf("%s did not finish its bounded traversal", filter.name)
			}
			trace.reads = nil
			page, err := service.ListFindingsPage(ctx, auditstore.FindingListParams{OwnerID: f.owner, AuditID: f.first, State: filter.state, Unreviewed: filter.unreviewed, AfterCreatedAt: positionTime, AfterFindingID: positionID, Limit: 129})
			if err != nil {
				t.Fatal(err)
			}
			if len(page.Items) > 128 || len(trace.reads) > 4 {
				t.Fatalf("oversized filtered window: %d rows/%d statements", len(page.Items), len(trace.reads))
			}
			for _, q := range trace.reads {
				if len(q.args) == 9 {
					q := q
					if firstProbe == nil {
						firstProbe = &q
					}
					lastProbe = &q
				}
			}
			for _, row := range page.Items {
				if reached[row.FindingID] {
					t.Fatal("filtered cursor repeated a Finding")
				}
				reached[row.FindingID] = true
			}
			if page.ContinueAfter == nil {
				break
			}
			if progressed[page.ContinueAfter.FindingID] {
				t.Fatal("empty filtered window did not progress")
			}
			progressed[page.ContinueAfter.FindingID] = true
			positionTime, positionID = &page.ContinueAfter.CreatedAt, page.ContinueAfter.FindingID
		}
		if len(reached) != filter.want || len(progressed) != 78 {
			t.Fatalf("%s reached=%d progress=%d", filter.name, len(reached), len(progressed))
		}
		for _, probe := range []*correlationQueryTrace{firstProbe, lastProbe} {
			if probe == nil {
				t.Fatal("missing filtered production query")
			}
			plan, err := explaintest.ExplainPrepared(ctx, f.pool, explaintest.Generic, probe.sql, probe.args...)
			if err != nil {
				t.Fatal(err)
			}
			for _, table := range []string{"project_finding_audits", "project_findings"} {
				if reads := plan.Reads.Of(table); reads > 160 {
					t.Fatalf("%s read %d %s rows, budget 160\n%s", filter.name, reads, table, plan.Plan.String())
				}
			}
		}
	}

	// Owner-wide windows are live keysets, with no global revision. Rare filters
	// still expose real progress, without scanning foreign owners to fill a page.
	for _, filter := range []struct {
		name  string
		state *auditstore.FindingState
		want  int
	}{
		{"owner unassessed", nil, 10000}, {"owner rare filter", &confirmed, 0},
	} {
		var before *time.Time
		beforeID := ""
		reached, progressed := map[string]bool{}, map[string]bool{}
		var first, last *correlationQueryTrace
		for pass := 0; ; pass++ {
			if pass > 80 {
				t.Fatal("owner window did not finish")
			}
			trace.reads = nil
			window, err := service.ListOwnerFindings(ctx, auditstore.OwnerFindingListParams{OwnerID: f.owner, State: filter.state, BeforeCreatedAt: before, BeforeFindingID: beforeID, Limit: 129})
			if err != nil {
				t.Fatal(err)
			}
			if len(trace.reads) != 1 {
				t.Fatalf("owner window used %d statements", len(trace.reads))
			}
			probe := trace.reads[0]
			if first == nil {
				first = &probe
			}
			last = &probe
			payload, _ := json.Marshal(window.Items)
			if len(payload) > 1<<20 || len(window.Items) > 128 {
				t.Fatal("oversized owner window")
			}
			for _, row := range window.Items {
				if row.ProjectID != f.project || reached[row.FindingID] {
					t.Fatal("foreign or repeated owner Finding")
				}
				reached[row.FindingID] = true
			}
			if window.ContinueBefore == nil {
				break
			}
			position := window.ContinueBefore
			if position.ProjectID != f.project || progressed[position.FindingID] {
				t.Fatal("owner continuation crossed scope or repeated")
			}
			progressed[position.FindingID] = true
			before, beforeID = &position.CreatedAt, position.FindingID
		}
		if len(reached) != filter.want || len(progressed) != 78 {
			t.Fatalf("%s reached=%d progress=%d", filter.name, len(reached), len(progressed))
		}
		for _, probe := range []*correlationQueryTrace{first, last} {
			plan, err := explaintest.ExplainPrepared(ctx, f.pool, explaintest.Generic, probe.sql, probe.args...)
			if err != nil {
				t.Fatal(err)
			}
			for _, table := range []string{"project_findings", "projects"} {
				if reads := plan.Reads.Of(table); reads > 160 {
					t.Fatalf("%s read %d %s rows, budget 160\n%s", filter.name, reads, table, plan.Plan.String())
				}
			}
		}
	}

	// Report assembly uses the same bounded membership windows even when a
	// section has no matching current states. Check real generic query plans.
	for _, section := range []auditstore.ReportSection{auditstore.ReportSectionProposed, auditstore.ReportSectionConfirmed} {
		var after *auditstore.ReportFindingCursor
		seen := map[string]bool{}
		var first, last *correlationQueryTrace
		windows := 0
		for {
			trace.reads = nil
			window, err := querystore.New(pool, artifactpgtest.NewStorage()).ListReportFindings(ctx, auditstore.ReportFindingPageParams{AuditID: f.first, Section: section, After: after, Limit: 128, ProjectionBytes: 1 << 20})
			if err != nil {
				t.Fatal(err)
			}
			windows++
			if windows > 80 {
				t.Fatal("report window did not converge")
			}
			if len(trace.reads) != 1 {
				t.Fatal("report window hydrated by multiple statements")
			}
			probe := trace.reads[0]
			if first == nil {
				first = &probe
			}
			last = &probe
			for _, row := range window.Items {
				if seen[row.FindingID] {
					t.Fatal("report repeated a Finding")
				}
				seen[row.FindingID] = true
			}
			if window.ContinueAfter == nil {
				break
			}
			after = window.ContinueAfter
		}
		want := 0
		if section == auditstore.ReportSectionProposed {
			want = 10000
		}
		if len(seen) != want || windows != 79 {
			t.Fatalf("report %s rows=%d windows=%d", section, len(seen), windows)
		}
		for _, probe := range []*correlationQueryTrace{first, last} {
			plan, err := explaintest.ExplainPrepared(ctx, f.pool, explaintest.Generic, probe.sql, probe.args...)
			if err != nil {
				t.Fatal(err)
			}
			for _, table := range []string{"project_finding_audits", "project_findings", "finding_proposal_receipts"} {
				if reads := plan.Reads.Of(table); reads > 160 {
					t.Fatalf("report %s read %d %s rows, budget 160\n%s", section, reads, table, plan.Plan.String())
				}
			}
		}
	}
	old := *revision
	if _, err := f.service.EditProjectFinding(ctx, f.edit(t, 1, "scale-edit", auditstore.VerdictFalsePositive)); err != nil {
		t.Fatal(err)
	}
	if _, err := service.ListProjectFindings(ctx, ProjectFindingListParams{OwnerFindingListParams: auditstore.OwnerFindingListParams{OwnerID: f.owner, ProjectID: f.project, Limit: 128}, ProjectRevision: &old}); err == nil {
		t.Fatal("a canonical edit did not invalidate the paginated Project snapshot")
	}

	// History uses ordinary Artifact metadata pages. Replicate a real retained
	// edit version as immutable aliases to stress listing independently of fact
	// production, whose operation identity/replay is covered by edit tests.
	historySeed := auditdomain.FindingSourceHistoryName("receipt-shared-a", "analyst-edit", f.finding+":2")
	prefix := auditdomain.FindingSourceHistoryPrefix("receipt-shared-a")
	if _, err := f.pool.Exec(ctx, projectFindingScaleHistorySQL, f.project, historySeed, prefix); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"artifact_bindings", "artifact_binding_revisions"} {
		if _, err := f.pool.Exec(ctx, "ANALYZE "+table); err != nil {
			t.Fatal(err)
		}
	}
	historySeen := map[string]bool{}
	afterName := ""
	var historyRevision *uint64
	var firstHistory, lastHistory *correlationQueryTrace
	for {
		trace.reads = nil
		page, err := intake.ListProjectSourceHistory(ctx, f.owner, f.project, "receipt-shared-a", afterName, 64, historyRevision)
		if err != nil {
			t.Fatal(err)
		}
		if len(trace.reads) != 4 {
			t.Fatalf("history page used %d statements", len(trace.reads))
		}
		historyRevision = &page.ProjectRevision
		for _, probe := range trace.reads {
			if strings.Contains(probe.sql, "artifact_bindings") && strings.Contains(probe.sql, "LIMIT") {
				copy := probe
				if firstHistory == nil {
					firstHistory = &copy
				}
				lastHistory = &copy
			}
		}
		encoded, _ := json.Marshal(page)
		if len(page.Items) > 64 || len(encoded) > 1<<20 {
			t.Fatal("oversized history page")
		}
		for _, record := range page.Items {
			if historySeen[record.Ref.Name] || !record.Frozen || !strings.HasPrefix(record.Ref.Name, prefix) {
				t.Fatal("history cursor repeated or crossed source")
			}
			historySeen[record.Ref.Name] = true
		}
		if len(page.Items) < 64 {
			break
		}
		afterName = page.Items[len(page.Items)-1].Ref.Name
	}
	if len(historySeen) != 10001 {
		t.Fatalf("reached %d of 10001 history artifacts", len(historySeen))
	}
	for _, probe := range []*correlationQueryTrace{firstHistory, lastHistory} {
		if probe == nil {
			t.Fatal("missing ordinary history query")
		}
		plan, err := explaintest.ExplainPrepared(ctx, f.pool, explaintest.Generic, probe.sql, probe.args...)
		if err != nil {
			t.Fatal(err)
		}
		for _, table := range []string{"artifact_bindings", "artifact_binding_revisions"} {
			if reads := plan.Reads.Of(table); reads > 80 {
				t.Fatalf("history page read %d %s rows, budget 80\n%s", reads, table, plan.Plan.String())
			}
		}
	}
	t.Log(fmt.Sprintf("traversed %d canonical Findings and %d independent sources at fixed page cost", len(seen), len(visited)))
}

const projectFindingScaleReceiptsSQL = `WITH selected AS (
 SELECT n,CASE WHEN n>=10000 AND n<=19997 THEN NULL
 ELSE 1+((CASE WHEN n>=19998 THEN n-9998 ELSE n END)-1)%9998 END AS group_id
 FROM generate_series(1,49997) AS n
)
INSERT INTO finding_proposal_receipts({{columns}})
 SELECT {{projection}} FROM finding_proposal_receipts AS seed CROSS JOIN selected
 CROSS JOIN LATERAL jsonb_populate_record(NULL::finding_proposal_receipts,to_jsonb(seed)||jsonb_build_object(
 'receipt_id','scale-receipt-'||lpad(n::text,6,'0'),'proposal_id','scale-proposal-'||n,
 'run_id','scale-run-'||(n/1000),'allocation_id','scale-allocation-'||n,'invocation_id','scale-invocation-'||n,
 'submission_id','scale-submission-'||n,
 'fingerprint_projection',CASE WHEN group_id IS NULL THEN seed.fingerprint_projection
 ELSE jsonb_set(seed.fingerprint_projection::jsonb,'{value}',to_jsonb(encode(sha256(convert_to('scale-group-'||group_id,'UTF8')),'hex')))::text END,
 'created_at',timestamptz '2026-11-01 00:00:00+00'+n*interval '1 microsecond')) AS cloned
 WHERE seed.project_id=$1 AND seed.receipt_id='receipt-shared-a'`
const projectFindingScaleFindingsSQL = `INSERT INTO project_findings
 SELECT (jsonb_populate_record(NULL::project_findings,to_jsonb(seed)||jsonb_build_object(
 'finding_id','scale-finding-'||lpad(n::text,6,'0'),'first_receipt_id','scale-receipt-'||lpad(n::text,6,'0'),
 'correlation_value',encode(sha256(convert_to('scale-group-'||n,'UTF8')),'hex'),
 'correlation_group_key',contractor_finding_group_key($1,'contractor/producer-issue/v1',encode(sha256(convert_to('scale-group-'||n,'UTF8')),'hex')),
 'created_at',timestamptz '2026-11-01 00:00:00+00'+n*interval '1 microsecond',
 'updated_at',timestamptz '2026-11-01 00:00:00+00'+n*interval '1 microsecond'))).*
 FROM project_findings AS seed CROSS JOIN generate_series(1,9998) AS n
 WHERE seed.project_id=$1 AND seed.finding_id=$2`
const projectFindingScaleSourcesSQL = `INSERT INTO project_finding_sources(receipt_id,project_id,finding_id,created_at)
 SELECT receipt_id,$1,CASE WHEN n>=10000 AND n<=19997 THEN $2
 ELSE 'scale-finding-'||lpad((1+((CASE WHEN n>=19998 THEN n-9998 ELSE n END)-1)%9998)::text,6,'0') END,
 created_at FROM finding_proposal_receipts CROSS JOIN LATERAL
 (SELECT substring(receipt_id FROM 15)::integer AS n) AS selected
 WHERE project_id=$1 AND receipt_id LIKE 'scale-receipt-%'`

const projectFindingForeignScaleReceiptsSQL = `INSERT INTO finding_proposal_receipts({{columns}})
 SELECT {{projection}} FROM finding_proposal_receipts AS seed CROSS JOIN generate_series(1,10000) AS n
 CROSS JOIN LATERAL jsonb_populate_record(NULL::finding_proposal_receipts,to_jsonb(seed)||jsonb_build_object(
 'receipt_id','foreign-receipt-'||n,'proposal_id','foreign-proposal-'||n,
 'allocation_id','foreign-allocation-'||n,'invocation_id','foreign-invocation-'||n,
 'submission_id','foreign-submission-'||n,
 'created_at',timestamptz '2027-01-01 00:00:00+00'+n*interval '1 microsecond')) AS cloned
 WHERE seed.receipt_id='receipt-foreign-base'`
const projectFindingForeignScaleFindingsSQL = `INSERT INTO project_findings
 SELECT (jsonb_populate_record(NULL::project_findings,to_jsonb(seed)||jsonb_build_object(
 'finding_id','foreign-finding-'||n,'first_receipt_id','foreign-receipt-'||n,
 'created_at',timestamptz '2027-01-01 00:00:00+00'+n*interval '1 microsecond',
 'updated_at',timestamptz '2027-01-01 00:00:00+00'+n*interval '1 microsecond'))).*
 FROM project_findings AS seed CROSS JOIN generate_series(1,10000) AS n
 WHERE seed.finding_id=$1`

const projectFindingScaleHistorySQL = `WITH copied AS (
 INSERT INTO artifact_bindings
 SELECT (jsonb_populate_record(NULL::artifact_bindings,to_jsonb(seed)||jsonb_build_object(
 'name',$3::text||'scale.'||lpad(n::text,6,'0')))).*
 FROM artifact_bindings AS seed CROSS JOIN generate_series(1,10000) AS n
 WHERE scope_kind='project' AND scope_id=$1 AND namespace='finding-history' AND name=$2
 RETURNING name
)
 INSERT INTO artifact_binding_revisions
 SELECT (jsonb_populate_record(NULL::artifact_binding_revisions,to_jsonb(seed)||jsonb_build_object('name',copied.name))).*
 FROM artifact_binding_revisions AS seed CROSS JOIN copied
 WHERE seed.scope_kind='project' AND seed.scope_id=$1 AND seed.namespace='finding-history' AND seed.name=$2`
