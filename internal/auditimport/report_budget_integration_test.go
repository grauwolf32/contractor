//go:build integration

package auditimport

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	largeReportItems    = 10_000
	largeReportFindings = 2_000
)

// largeReportFindingStates assigns each seeded finding its report section.
// The proposed section spans several pages; the others fit in one.
var largeReportFindingStates = []struct {
	state string
	count int
}{
	{"confirmed", 300}, {"proposed", 1_400}, {"rejected", 100}, {"duplicate", 100}, {"needs-evidence", 100},
}

// TestImporterFinalizesLargeAuditUnderPageBudgetAndResumes finalizes 10,000
// items and 2,000 findings on PostgreSQL under a per-attempt budget of four
// pages, which the report's 27 pages exceed, and forces one attempt to time
// out. Every page is read exactly once across attempts: the timed-out
// attempt's successor resumes from the recorded pages instead of restarting.
func TestImporterFinalizesLargeAuditUnderPageBudgetAndResumes(t *testing.T) {
	pool := completionPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	seed := seedLargeReportAudit(t, ctx, pool)
	access, err := NewArtifactAccess(artifacts.NewService(artifacts.NewPostgresRepository(pool)))
	mustCompletion(t, err)
	store := &budgetedReportStore{Store: auditstoretest.NewStore(pool), reads: map[string]int{}}
	importer, err := New(store, store, runstore.NewPostgresStore(pool), access)
	mustCompletion(t, err)
	importer.budget = reportBudget{pageRows: 500, pageBytes: 256 << 10, pagesPerAttempt: 4}
	const totalPages = largeReportItems/500 + 1 + 3 + 1 + 1 + 1

	attempts, timedOut := 0, false
	for {
		attempts++
		if attempts > totalPages {
			t.Fatal("report finalization did not converge")
		}
		attemptCtx := newForcedDeadline(ctx)
		before := store.recordedPages()
		if attempts == 3 {
			// Time the third attempt out after it records two pages.
			store.expireAfter(before+2, attemptCtx)
		}
		worked, err := importer.Finalize(attemptCtx, seed.claim, seed.snapshot)
		recorded := store.recordedPages() - before
		if recorded > importer.budget.pagesPerAttempt {
			t.Fatalf("attempt %d recorded %d pages, budget %d", attempts, recorded, importer.budget.pagesPerAttempt)
		}
		if attempts == 3 {
			if worked || !errors.Is(err, context.DeadlineExceeded) || recorded != 2 {
				t.Fatalf("timed-out attempt = (%t, %v), recorded %d", worked, err, recorded)
			}
			timedOut = true
			continue
		}
		if err != nil || !worked {
			t.Fatalf("attempt %d = (%t, %v)", attempts, worked, err)
		}
		if store.recordedPages() < totalPages {
			if _, err := store.GetArtifactLink(ctx, seed.snapshot.Audit.AuditID, auditstore.ReportMachineLogicalKey); !errors.Is(err, auditstore.ErrNotFound) {
				t.Fatalf("attempt %d published a report before its build completed: %v", attempts, err)
			}
			continue
		}
		break
	}
	if !timedOut || store.recordedPages() != totalPages || attempts != 8 {
		t.Fatalf("attempts=%d pages=%d timedOut=%t", attempts, store.recordedPages(), timedOut)
	}
	if duplicated := store.repeatedReads(); len(duplicated) != 0 {
		t.Fatalf("resumed build reread recorded pages: %v", duplicated)
	}
	if store.successfulReads() != totalPages || store.failedReads != 1 {
		t.Fatalf("page reads = %d successful, %d failed", store.successfulReads(), store.failedReads)
	}
	audit, err := store.Get(ctx, seed.snapshot.Audit.OwnerID, seed.snapshot.Audit.AuditID)
	mustCompletion(t, err)
	if audit.State != auditstore.AuditCompleted {
		t.Fatalf("Audit state = %s", audit.State)
	}
	var leftover int
	mustCompletion(t, pool.QueryRow(ctx, `SELECT count(*) FROM audit_report_pages WHERE audit_id = $1`,
		seed.snapshot.Audit.AuditID).Scan(&leftover))
	if leftover != 0 {
		t.Fatalf("completed Audit keeps %d report pages", leftover)
	}
	assertLargeReport(t, ctx, store, access, seed)
}

type largeReportSeed struct {
	claim       auditstore.ControllerClaim
	snapshot    auditstore.ReconcileSnapshot
	findingIDs  map[string][]string
	statusCount map[auditstore.CoverageStatus]int
}

// seedLargeReportAudit starts one Round of largeReportItems items through the
// Audit store, settles them, admits largeReportFindings findings through
// receipts and Audit holds, decides some of them and closes the Audit into
// finalizing the way the Controller does.
func seedLargeReportAudit(t *testing.T, ctx context.Context, pool *pgxpool.Pool) largeReportSeed {
	t.Helper()
	const owner, id = "owner", "large-report"
	_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: id, OwnerID: owner, Kind: projectstore.KindProject, Name: "Large report",
		IdempotencyKey: id, RequestDigest: completionDigest([]byte(id)),
	})
	mustCompletion(t, err)
	profile := loadResultProfile(t)
	profileSnapshot, err := json.Marshal(profile)
	mustCompletion(t, err)
	audits := auditstoretest.NewStore(pool)
	draft, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: id, OwnerID: owner, ProjectID: id,
		Profile:         auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest},
		ProfileSnapshot: profileSnapshot, InputSelection: json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 64, MaxItemsPerRound: largeReportItems, MaxItemsTotal: largeReportItems,
			MaxSubmittedRuns: 1_000, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20,
		},
		IdempotencyKey: id, RequestDigest: completionDigest([]byte("audit")),
	})
	mustCompletion(t, err)
	exact := func(name string) auditstore.ExactArtifact {
		revision := name + "-r1"
		return auditstore.ExactArtifact{
			Ref:    contracts.ArtifactRef{Namespace: "fixture", Name: name, Revision: &revision},
			Digest: completionDigest([]byte(name)),
		}
	}
	source := exact("checklist")
	items := make([]auditstore.MaterializedItem, largeReportItems)
	for index := range items {
		key := fmt.Sprintf("check-%05d", index)
		items[index] = auditstore.MaterializedItem{
			ItemID: id + "-" + key, ItemKey: key, Ordinal: index, Kind: "checklist", SubjectKey: key,
			Task: exact("task-" + key), WorkflowRole: "check", InitialState: auditstore.ItemReady,
			Origin: auditstore.ItemOrigin{
				Schema: auditstore.ItemOriginSchema, SourceRef: &source.Ref, SourceContentDigest: source.Digest,
				SourceMediaType: "application/json", CanonicalInventoryDigest: completionDigest([]byte("inventory")),
				EntryKey: key, EntryVersion: "1",
			},
			Coverage: auditstore.Coverage{Status: auditstore.CoverageNotTested, Requested: []string{"source-trace"}, Completed: []string{}, Gaps: []string{}},
		}
	}
	_, _, err = audits.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: owner, AuditID: id, ExpectedRevision: draft.Revision, RoundID: id, RoundOrdinal: 1,
		Manifest:         exact("worklist"),
		BaselineSnapshot: json.RawMessage(`{"schema":"contractor.audit.baseline.v1","inventory":{"gaps":[]}}`),
		DeadlineAt:       time.Now().Add(time.Hour), Items: items, IdempotencyKey: id, RequestDigest: completionDigest([]byte("round")),
	})
	mustCompletion(t, err)
	claims, err := audits.Claim(ctx, auditstore.ClaimParams{HolderID: id, Lease: 5 * time.Minute, Limit: 8})
	mustCompletion(t, err)
	if len(claims) != 1 || claims[0].AuditID != id {
		t.Fatalf("claims = %+v", claims)
	}
	seed := largeReportSeed{claim: claims[0], findingIDs: map[string][]string{}, statusCount: map[auditstore.CoverageStatus]int{}}
	round, err := audits.GetRound(ctx, id, id)
	mustCompletion(t, err)
	round, err = audits.TransitionRound(ctx, auditstore.RoundTransitionParams{
		Claim: seed.claim, RoundID: id, ExpectedRevision: round.Revision,
		ExpectedState: auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
	})
	mustCompletion(t, err)

	// Every item settles with one of the ten coverage statuses in turn.
	statuses := []auditstore.CoverageStatus{
		auditstore.CoverageNotTested, auditstore.CoverageInconclusive, auditstore.CoverageSatisfied,
		auditstore.CoverageViolated, auditstore.CoverageNotApplicable, auditstore.CoverageBlocked,
		auditstore.CoverageExcluded, auditstore.CoverageTracedComplete, auditstore.CoverageTracedPartial,
		auditstore.CoverageUnmapped,
	}
	names := make([]string, len(statuses))
	for index, status := range statuses {
		names[index] = string(status)
		seed.statusCount[status] = largeReportItems / len(statuses)
	}
	_, err = pool.Exec(ctx, `
UPDATE audit_items
   SET state = 'settled', final_disposition = 'execution-failed',
       coverage_status = ($2::text[])[ordinal % 10 + 1],
       coverage_completed = '["source-trace"]', coverage_gaps = '[]',
       coverage_rationale = 'Settled by the large report fixture.'
 WHERE audit_id = $1`, id, names)
	mustCompletion(t, err)

	// Ordinary-Run receipts carry the report projection stored at intake.
	// Each first receipt of a proposed finding; the fixture writes the
	// finding rows directly, since admission is not under test here.
	projection, err := auditdomain.EncodeFindingReportProjection(auditdomain.FindingReportProjection{
		Title: "Large report candidate", Description: "A finding admitted by the large report fixture.",
		Subject:      &auditdomain.FindingSubject{Kind: "component", Key: "handler"},
		StandardRefs: []auditdomain.StandardReference{}, SeveritySuggestion: "low",
		Limitations: []string{"Static review only."},
	})
	mustCompletion(t, err)
	_, err = pool.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id, runtime_instance_id,
    stage_execution_id, logical_agent_name, invocation_id, submission_id, client_key,
    request_digest, run_id, owner_id, project_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type, proposal_size_bytes, evidence,
    report_projection
)
SELECT 'receipt-' || n, 'proposal-' || n, 'allocation-large', 'runtime', 'instance',
       'stage', 'worker', 'invocation-large', 'submission-' || n, 'candidate-' || n,
       $1, 'run-large', $2, $3, 'source', '1', 'contractor/v1alpha1',
       '{"name":"source","version":"1"}'::jsonb, $1,
       jsonb_build_object('namespace', 'finding-proposals', 'name', 'candidate-' || n, 'revision', 'r1'),
       $1, 'application/json', 64, '[]'::jsonb, $4
  FROM generate_series(1, $5::integer) AS n`,
		completionDigest([]byte("large")), owner, id, string(projection), largeReportFindings)
	mustCompletion(t, err)
	_, err = pool.Exec(ctx, `
INSERT INTO audit_findings (finding_id, audit_id, first_receipt_id, first_proposal_ref, created_at)
SELECT 'finding-' || lpad(n::text, 5, '0'), $1, 'receipt-' || n,
       jsonb_build_object('ref', jsonb_build_object('namespace', $2::text, 'name', 'candidate-' || n, 'revision', 'r1'),
                          'digest', $3::text, 'mediaType', 'application/json', 'sizeBytes', 64),
       timestamptz '2026-10-01 00:00:00+00' + (n % 7) * interval '1 second'
  FROM generate_series(1, $4::integer) AS n`, id, auditdomain.ArtifactNamespace(id), completionDigest([]byte("large")), largeReportFindings)
	mustCompletion(t, err)
	rows, err := pool.Query(ctx, `SELECT finding_id FROM audit_findings WHERE audit_id = $1 ORDER BY created_at, finding_id`, id)
	mustCompletion(t, err)
	var ordered []string
	for rows.Next() {
		var findingID string
		mustCompletion(t, rows.Scan(&findingID))
		ordered = append(ordered, findingID)
	}
	mustCompletion(t, rows.Err())
	if len(ordered) != largeReportFindings {
		t.Fatalf("admitted %d findings", len(ordered))
	}
	// Assign states in contiguous runs of creation order, with the review
	// ledger the decided states require.
	next := 0
	for _, group := range largeReportFindingStates {
		selected := ordered[next : next+group.count]
		next += group.count
		seed.findingIDs[group.state] = selected
		switch group.state {
		case "confirmed", "rejected":
			verdict, severity, reason := "true_positive", "high", ""
			if group.state == "rejected" {
				verdict, severity, reason = "false_positive", "", "false-positive"
			}
			_, err = pool.Exec(ctx, `
WITH requests AS (
    INSERT INTO audit_review_requests (
        request_id, audit_id, finding_id, subject_kind, subject_id, kind,
        subject_revision, subject_digest, requested_actions, state,
        idempotency_key, request_digest
    )
    SELECT 'request-' || finding_id, $1, finding_id, 'finding', finding_id, 'finding-triage',
           1, $3, jsonb_build_array($4::text), 'decided', 'request-' || substr(finding_id, 9, 100), $3
      FROM unnest($2::text[]) AS finding_id
    RETURNING request_id, finding_id
), decisions AS (
    INSERT INTO audit_review_decisions (
        decision_id, request_id, audit_id, finding_id, actor_id, action, verdict,
        severity, rationale, subject_revision, subject_digest, idempotency_key, request_digest
    )
    SELECT 'decision-' || finding_id, request_id, $1, finding_id, $5, $4, $4,
           NULLIF($6, ''), 'Decided by the large report fixture.', 1, $3,
           'decision-' || substr(finding_id, 9, 100), $3
      FROM requests
    RETURNING decision_id, finding_id
)
UPDATE audit_findings AS finding
   SET state = $7, current_decision_id = decisions.decision_id,
       rejection_reason = NULLIF($8, ''), revision = finding.revision + 1
  FROM decisions
 WHERE finding.finding_id = decisions.finding_id`,
				id, selected, completionDigest([]byte("subject")), verdict, owner, severity, group.state, reason)
		case "duplicate":
			_, err = pool.Exec(ctx, `
UPDATE audit_findings SET state = 'duplicate', duplicate_target_id = $2, revision = revision + 1
 WHERE finding_id = ANY($1::text[])`, selected, ordered[0])
		case "needs-evidence":
			_, err = pool.Exec(ctx, `
UPDATE audit_findings SET state = 'needs-evidence', revision = revision + 1
 WHERE finding_id = ANY($1::text[])`, selected)
		}
		mustCompletion(t, err)
	}

	audit, err := audits.Get(ctx, owner, id)
	mustCompletion(t, err)
	_, err = audits.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
		Claim: seed.claim, ExpectedRevision: audit.Revision,
		ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing,
	})
	mustCompletion(t, err)
	_, err = audits.TransitionRound(ctx, auditstore.RoundTransitionParams{
		Claim: seed.claim, RoundID: id, ExpectedRevision: round.Revision,
		ExpectedState: auditstore.RoundExecuting, TargetState: auditstore.RoundClosed,
	})
	mustCompletion(t, err)
	_, _, err = audits.ReleaseDispatchHold(ctx, seed.claim)
	mustCompletion(t, err)
	seed.snapshot, err = audits.GetReconcileSnapshot(ctx, seed.claim)
	mustCompletion(t, err)
	if seed.snapshot.Audit.State != auditstore.AuditFinalizing || len(seed.snapshot.Items) != 0 ||
		seed.snapshot.Round == nil || seed.snapshot.Round.State != auditstore.RoundClosed {
		t.Fatalf("fixture snapshot = %+v", seed.snapshot.Audit)
	}
	return seed
}

// assertLargeReport checks the committed report against the seeded Audit and
// checks that its paged entries are byte-identical to one json.Marshal of the
// complete item and finding lists.
func assertLargeReport(
	t *testing.T, ctx context.Context, store *budgetedReportStore,
	access *ServiceArtifactAccess, seed largeReportSeed,
) {
	t.Helper()
	link, err := store.GetArtifactLink(ctx, seed.snapshot.Audit.AuditID, auditstore.ReportMachineLogicalKey)
	mustCompletion(t, err)
	data, err := access.ReadProjectExact(ctx, seed.snapshot.Audit.ProjectID, link.Artifact)
	mustCompletion(t, err)
	var raw struct {
		Coverage reportCoverageSummary      `json:"coverage"`
		Items    json.RawMessage            `json:"items"`
		Findings map[string]json.RawMessage `json:"findings"`
	}
	mustCompletion(t, json.Unmarshal(data, &raw))
	var items []reportItem
	mustCompletion(t, json.Unmarshal(raw.Items, &items))
	if encoded, err := json.Marshal(items); err != nil || !bytes.Equal(encoded, raw.Items) {
		t.Fatal("paged report items differ from one encoding of the complete list")
	}
	if len(items) != largeReportItems || raw.Coverage.SelectedItems != largeReportItems {
		t.Fatalf("report items = %d, selected = %d", len(items), raw.Coverage.SelectedItems)
	}
	for index, item := range items {
		if item.Ordinal != index || item.ItemKey != fmt.Sprintf("check-%05d", index) {
			t.Fatalf("report item %d = %+v", index, item)
		}
	}
	counts := raw.Coverage.Counts
	for status, want := range seed.statusCount {
		for _, entry := range counts.byStatus() {
			if entry.status == status && entry.count != want {
				t.Fatalf("coverage %s = %d, want %d", status, entry.count, want)
			}
		}
	}
	sections := map[string]string{
		"confirmed": "confirmed", "proposed": "proposed", "rejected": "rejected",
		"duplicate": "duplicates", "needs-evidence": "needsEvidence",
	}
	for state, key := range sections {
		var findings []reportFinding
		mustCompletion(t, json.Unmarshal(raw.Findings[key], &findings))
		if encoded, err := json.Marshal(findings); err != nil || !bytes.Equal(encoded, raw.Findings[key]) {
			t.Fatalf("paged %s findings differ from one encoding of the complete list", state)
		}
		ids := make([]string, len(findings))
		for index, finding := range findings {
			ids[index] = finding.FindingID
			if finding.State != state || finding.Title != "Large report candidate" ||
				!slices.Equal(finding.Limitations, []string{"Static review only."}) {
				t.Fatalf("%s finding %d = %+v", state, index, finding)
			}
			if (state == "confirmed" || state == "rejected") != (finding.AnalystDecision != nil) {
				t.Fatalf("%s finding decision = %+v", state, finding.AnalystDecision)
			}
		}
		if !slices.Equal(ids, seed.findingIDs[state]) {
			t.Fatalf("%s findings are not the seeded findings in creation order", state)
		}
	}
	summaryLink, err := store.GetArtifactLink(ctx, seed.snapshot.Audit.AuditID, auditstore.ReportSummaryLogicalKey)
	mustCompletion(t, err)
	summary, err := access.ReadProjectExact(ctx, seed.snapshot.Audit.ProjectID, summaryLink.Artifact)
	mustCompletion(t, err)
	if !strings.Contains(string(summary), "| confirmed | 300 |\n| proposed | 1400 |\n| rejected | 100 |\n| duplicate | 100 |\n| needs-evidence | 100 |") ||
		!strings.Contains(string(summary), "- Selected items: 10000\n") {
		t.Fatalf("report summary counts:\n%s", summary)
	}
}

// budgetedReportStore records every report page read and can expire an
// attempt's context once a given number of pages is recorded.
type budgetedReportStore struct {
	*auditstoretest.Store
	mu          sync.Mutex
	reads       map[string]int
	failedReads int
	recorded    int
	expireAt    int
	expire      *forcedDeadline
}

func (s *budgetedReportStore) ListReportItems(
	ctx context.Context, auditID, roundID string, after, limit int,
) ([]auditstore.ReportItem, error) {
	rows, err := s.Store.ListReportItems(ctx, auditID, roundID, after, limit)
	s.noteRead(fmt.Sprintf("items/%s/%d", roundID, after), err)
	return rows, err
}

func (s *budgetedReportStore) ListReportFindings(
	ctx context.Context, params auditstore.ReportFindingPageParams,
) ([]auditstore.ReportFinding, bool, error) {
	rows, more, err := s.Store.ListReportFindings(ctx, params)
	cursor := "start"
	if params.After != nil {
		cursor = params.After.FindingID
	}
	s.noteRead(fmt.Sprintf("%s/%s", params.Section, cursor), err)
	return rows, more, err
}

func (s *budgetedReportStore) RecordReportPage(ctx context.Context, params auditstore.RecordReportPageParams) error {
	err := s.Store.RecordReportPage(ctx, params)
	s.mu.Lock()
	defer s.mu.Unlock()
	if err == nil {
		s.recorded++
		if s.expire != nil && s.recorded == s.expireAt {
			s.expire.expire()
			s.expire = nil
		}
	}
	return err
}

func (s *budgetedReportStore) noteRead(key string, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err != nil {
		s.failedReads++
		return
	}
	s.reads[key]++
}

func (s *budgetedReportStore) expireAfter(recorded int, ctx *forcedDeadline) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.expireAt, s.expire = recorded, ctx
}

func (s *budgetedReportStore) recordedPages() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.recorded
}

func (s *budgetedReportStore) successfulReads() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	total := 0
	for _, count := range s.reads {
		total += count
	}
	return total
}

func (s *budgetedReportStore) repeatedReads() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var repeated []string
	for key, count := range s.reads {
		if count > 1 {
			repeated = append(repeated, key)
		}
	}
	return repeated
}

// forcedDeadline is a context whose deadline the test reaches at a chosen
// point of an attempt instead of by wall-clock time.
type forcedDeadline struct {
	context.Context
	done chan struct{}
	once sync.Once
}

func newForcedDeadline(parent context.Context) *forcedDeadline {
	return &forcedDeadline{Context: parent, done: make(chan struct{})}
}

func (c *forcedDeadline) Done() <-chan struct{} { return c.done }

func (c *forcedDeadline) Err() error {
	select {
	case <-c.done:
		return context.DeadlineExceeded
	default:
		return c.Context.Err()
	}
}

func (c *forcedDeadline) expire() { c.once.Do(func() { close(c.done) }) }
