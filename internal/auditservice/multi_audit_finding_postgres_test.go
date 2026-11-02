package auditservice

import (
	"context"
	"errors"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5/pgxpool"
)

type projectFindingFixture struct {
	ctx                                            context.Context
	pool                                           *pgxpool.Pool
	service                                        *Service
	owner, project, first, second, finding, target string
}

func newProjectFindingFixture(t *testing.T) projectFindingFixture {
	t.Helper()
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	t.Cleanup(cancel)
	pool := isolatedAuditServicePool(t, ctx, dsn)
	f := projectFindingFixture{ctx: ctx, pool: pool, owner: "project-finding-owner", project: "project-finding-project", first: "finding-audit-a", second: "finding-audit-b"}
	for _, audit := range []string{f.first, f.second} {
		createCorrelationAudit(t, ctx, pool, f.owner, f.project, audit)
	}
	fp := correlationFingerprint(t)
	f.finding = seedAuditFindingDocument(t, ctx, pool, f.project, f.owner, f.first, "shared-a", auditFindingDocument("shared-a"), fp)
	same := seedAuditFindingDocument(t, ctx, pool, f.project, f.owner, f.second, "shared-b", auditFindingDocument("shared-b"), fp)
	if same != f.finding {
		t.Fatal("qualified sources in different Audits did not use one Finding identity")
	}
	f.target = seedAuditFinding(t, ctx, pool, f.project, f.owner, f.first, "manual-target")
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	f.service = &Service{pool: pool, findings: intake, now: time.Now}
	return f
}
func (f projectFindingFixture) edit(t *testing.T, revision uint64, key string, verdict auditstore.AnalystVerdict) ProjectFindingEditParams {
	t.Helper()
	p := ProjectFindingEditParams{ProjectFindingEdit: auditstore.ProjectFindingEdit{OwnerID: f.owner, ProjectID: f.project, FindingID: f.finding, ExpectedRevision: revision, Verdict: verdict, Rationale: "Explicit Project decision " + key}, IdempotencyKey: key}
	if verdict == auditstore.VerdictTruePositive {
		severity := auditstore.SeverityHigh
		p.Severity = &severity
	}
	if verdict == auditstore.VerdictDuplicate {
		p.DuplicateTargetID = &f.target
	}
	var err error
	p.RequestDigest, err = ProjectFindingEditRequestDigest(p)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
func TestProjectFindingEditsAreSharedAcrossAuditsAndReplayOriginalSnapshots(t *testing.T) {
	f := newProjectFindingFixture(t)
	initial, err := f.service.GetProjectFinding(f.ctx, f.owner, f.project, f.finding)
	if err != nil || initial.State != nil || initial.Revision != 1 || initial.Claim == nil {
		t.Fatalf("initial canonical claim: %+v %v", initial, err)
	}
	var firstParams ProjectFindingEditParams
	var firstResult ProjectFindingEditResult
	for index, verdict := range []auditstore.AnalystVerdict{auditstore.VerdictTruePositive, auditstore.VerdictFalsePositive, auditstore.VerdictNeedsEvidence, auditstore.VerdictReopen, auditstore.VerdictDuplicate} {
		p := f.edit(t, uint64(index+1), "verdict-"+string(verdict), verdict)
		result, err := f.service.EditProjectFinding(f.ctx, p)
		if err != nil || result.Replayed || result.Finding.Revision != uint64(index+2) || result.Finding.AnalystVerdict == nil || *result.Finding.AnalystVerdict != verdict {
			t.Fatalf("direct verdict %s: %+v %v", verdict, result, err)
		}
		if result.Finding.ActorID == nil || *result.Finding.ActorID != f.owner || result.Finding.AssessedAt == nil {
			t.Fatal("actor/time lost")
		}
		replay, err := f.service.EditProjectFinding(f.ctx, p)
		if err != nil || !replay.Replayed || !reflect.DeepEqual(replay.Finding, result.Finding) {
			t.Fatalf("assessment receipt replay: %+v %v", replay, err)
		}
		project, err := f.service.GetProjectFinding(f.ctx, f.owner, f.project, f.finding)
		if err != nil {
			t.Fatal(err)
		}
		for _, audit := range []string{f.first, f.second} {
			current, err := f.service.GetAuditProjectFinding(f.ctx, f.owner, audit, f.finding)
			if err != nil || !reflect.DeepEqual(current, project) {
				t.Fatalf("Audit %s is not a filtered view of the same Project state: %+v %v", audit, current, err)
			}
		}
		if project.FirstReceiptID != initial.FirstReceiptID || !reflect.DeepEqual(project.Claim, initial.Claim) || project.FirstProposal.Verify(initial.FirstProposal) != nil {
			t.Fatal("an analyst edit changed immutable source identity or claim")
		}
		if index == 0 {
			firstParams, firstResult = p, result
		}
	}
	replay, err := f.service.EditProjectFinding(f.ctx, firstParams)
	if err != nil || !replay.Replayed || !reflect.DeepEqual(replay.Finding, firstResult.Finding) {
		t.Fatalf("a later edit replaced an older mutation receipt: %+v %v", replay, err)
	}
	conflict := firstParams
	conflict.Rationale = "Different intent"
	conflict.RequestDigest, _ = ProjectFindingEditRequestDigest(conflict)
	if _, err = f.service.EditProjectFinding(f.ctx, conflict); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("changed body under same key: %v", err)
	}
	stale := f.edit(t, 1, "stale", auditstore.VerdictFalsePositive)
	if _, err = f.service.EditProjectFinding(f.ctx, stale); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("stale CAS: %v", err)
	}
	cycle := f.edit(t, 1, "cycle", auditstore.VerdictDuplicate)
	cycle.FindingID = f.target
	cycle.DuplicateTargetID = &f.finding
	cycle.RequestDigest, _ = ProjectFindingEditRequestDigest(cycle)
	if _, err = f.service.EditProjectFinding(f.ctx, cycle); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("duplicate cycle: %v", err)
	}
	foreign := f.edit(t, 6, "foreign-owner", auditstore.VerdictFalsePositive)
	foreign.OwnerID = "another-owner"
	if _, err = f.service.EditProjectFinding(f.ctx, foreign); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign edit: %v", err)
	}
	createCorrelationAudit(t, f.ctx, f.pool, f.owner, "other-project", "other-audit")
	foreignTarget := seedAuditFinding(t, f.ctx, f.pool, "other-project", f.owner, "other-audit", "foreign-target")
	foreign = f.edit(t, 6, "foreign-target", auditstore.VerdictDuplicate)
	foreign.DuplicateTargetID = &foreignTarget
	foreign.RequestDigest, _ = ProjectFindingEditRequestDigest(foreign)
	if _, err = f.service.EditProjectFinding(f.ctx, foreign); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("cross-Project duplicate: %v", err)
	}
	var sources, reviews int
	if err = f.pool.QueryRow(f.ctx, `SELECT (SELECT count(*) FROM project_finding_sources WHERE finding_id=$1),(SELECT count(*) FROM audit_review_requests WHERE audit_id IN ($2,$3))`, f.finding, f.first, f.second).Scan(&sources, &reviews); err != nil || sources != 2 || reviews != 0 {
		t.Fatalf("sources=%d invented reviews=%d err=%v", sources, reviews, err)
	}
	for _, receiptID := range []string{"receipt-shared-a", "receipt-shared-b"} {
		source, err := f.service.findings.GetProjectReceipt(f.ctx, f.owner, f.project, receiptID)
		if err != nil || source.ProjectProposal == nil || len(source.ProjectEvidence) != 1 {
			t.Fatalf("independent retained source: %+v %v", source, err)
		}
		project, _ := artifacts.NewService(artifacts.NewPostgresRepository(f.pool)).Project(f.project)
		if _, err = project.Read(f.ctx, source.ProjectEvidence[0].Ref); err != nil {
			t.Fatal(err)
		}
	}
}
func TestProjectFindingEditRollsBackWithItsHistoryAndReceipt(t *testing.T) {
	f := newProjectFindingFixture(t)
	before, err := f.service.GetProjectFinding(f.ctx, f.owner, f.project, f.finding)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.pool.Exec(f.ctx, `CREATE FUNCTION refuse_finding_receipt() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'injected finding receipt failure'; END $$`); err != nil {
		t.Fatal(err)
	}
	if _, err = f.pool.Exec(f.ctx, `CREATE TRIGGER refuse_finding_receipt BEFORE INSERT ON mutation_receipts FOR EACH ROW WHEN(NEW.operation='finding.assess') EXECUTE FUNCTION refuse_finding_receipt()`); err != nil {
		t.Fatal(err)
	}
	var countBefore int
	if err = f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind='project' AND scope_id=$1 AND namespace='finding-history'`, f.project).Scan(&countBefore); err != nil {
		t.Fatal(err)
	}
	p := f.edit(t, 1, "atomic", auditstore.VerdictTruePositive)
	if _, err = f.service.EditProjectFinding(f.ctx, p); err == nil {
		t.Fatal("injected receipt failure was ignored")
	}
	after, err := f.service.GetProjectFinding(f.ctx, f.owner, f.project, f.finding)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("failed keyed mutation changed canonical state: %+v %v", after, err)
	}
	var history, receipts int
	if err = f.pool.QueryRow(f.ctx, `SELECT (SELECT count(*) FROM artifact_bindings WHERE scope_kind='project' AND scope_id=$1 AND namespace='finding-history'),(SELECT count(*) FROM mutation_receipts WHERE operation='finding.assess')`, f.project).Scan(&history, &receipts); err != nil || history != countBefore || receipts != 0 {
		t.Fatalf("failed mutation leaked history=%d/%d receipts=%d: %v", history, countBefore, receipts, err)
	}
	if _, err = f.pool.Exec(f.ctx, `DROP TRIGGER refuse_finding_receipt ON mutation_receipts`); err != nil {
		t.Fatal(err)
	}
	result, err := f.service.EditProjectFinding(f.ctx, p)
	if err != nil || result.Replayed || result.Finding.Revision != 2 {
		t.Fatalf("retry after rollback: %+v %v", result, err)
	}
	replay, err := f.service.EditProjectFinding(f.ctx, p)
	if err != nil || !replay.Replayed {
		t.Fatalf("receipt reconciliation: %+v %v", replay, err)
	}
}
func TestProjectFindingEditUsesExistingFenceAndRejectsConcurrentStaleWrites(t *testing.T) {
	f := newProjectFindingFixture(t)
	contender, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer contender.Rollback(f.ctx)
	if err = hookstore.NewFindingEventsTx(contender).LockFindingCorrelationFence(f.ctx, f.project); err != nil {
		t.Fatal(err)
	}
	p := f.edit(t, 1, "contention", auditstore.VerdictTruePositive)
	if _, err = f.service.EditProjectFinding(f.ctx, p); !persistencepostgres.IsTransactionConflict(err) {
		t.Fatalf("persistent Project fence conflict: %v", err)
	}
	if err = contender.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
	result, err := f.service.EditProjectFinding(f.ctx, p)
	if err != nil || result.Replayed || result.Finding.Revision != 2 {
		t.Fatalf("same key after definite abort: %+v %v", result, err)
	}
	start := make(chan struct{})
	var wg sync.WaitGroup
	results := make([]ProjectFindingEditResult, 2)
	failures := make([]error, 2)
	for index, verdict := range []auditstore.AnalystVerdict{auditstore.VerdictFalsePositive, auditstore.VerdictNeedsEvidence} {
		wg.Add(1)
		go func(index int, verdict auditstore.AnalystVerdict) {
			defer wg.Done()
			<-start
			key := "concurrent-" + string(verdict)
			params := f.edit(t, 2, key, verdict)
			results[index], failures[index] = f.service.EditProjectFinding(f.ctx, params)
		}(index, verdict)
	}
	close(start)
	wg.Wait()
	succeeded := 0
	for index, err := range failures {
		if err == nil {
			succeeded++
			if results[index].Finding.Revision != 3 {
				t.Fatal("unexpected winning revision")
			}
		} else if !errors.Is(err, auditstore.ErrPrecondition) && !persistencepostgres.IsTransactionConflict(err) {
			t.Fatalf("concurrent edit: %v", err)
		}
	}
	if succeeded != 1 {
		t.Fatalf("CAS allowed %d concurrent winners", succeeded)
	}
	cancelled, cancel := context.WithCancel(f.ctx)
	cancel()
	if _, err = f.service.EditProjectFinding(cancelled, f.edit(t, 3, "cancelled", auditstore.VerdictReopen)); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled edit: %v", err)
	}
}

func TestFindingComparisonRecognizesSharedCanonicalIdentityWithoutQualifiedKey(t *testing.T) {
	f := newProjectFindingFixture(t)
	// The explicit membership represents the same already-published Project
	// identity selected by another Audit; no qualified-key inference occurs.
	if _, err := f.pool.Exec(f.ctx, `INSERT INTO project_finding_audits(audit_id,finding_id,project_id,finding_created_at,correlation_group_key)
 SELECT $1,finding_id,project_id,created_at,correlation_group_key FROM project_findings WHERE finding_id=$2`, f.second, f.target); err != nil {
		t.Fatal(err)
	}
	page, err := f.service.CompareFindings(f.ctx, FindingComparisonParams{OwnerID: f.owner, ProjectID: f.project, LeftAuditID: f.first, RightAuditID: f.second, Limit: 20})
	if err != nil {
		t.Fatal(err)
	}
	for pages := 0; pages < 3; pages++ {
		for _, row := range page.Items {
			if row.FindingID == f.target {
				if row.Kind != "both" || row.Left == nil || row.Right == nil || row.Left.FindingID != row.Right.FindingID {
					t.Fatalf("shared identity presented as independent problems: %+v", row)
				}
				return
			}
		}
		if page.Next == nil {
			break
		}
		page, err = f.service.CompareFindings(f.ctx, FindingComparisonParams{OwnerID: f.owner, ProjectID: f.project, LeftAuditID: f.first, RightAuditID: f.second, Limit: 20, Cursor: page.Next})
		if err != nil {
			t.Fatal(err)
		}
	}
	t.Fatal("shared unqualified identity was unreachable")
}
