package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestFindingCorrelationKeepsIndependentReviewsAndProjectScope(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, url)
	const owner, project = "correlation-review-owner", "correlation-review-project"
	for _, selected := range []struct{ project, owner, audit string }{{project, owner, "review-a"}, {project, owner, "review-b"}, {"other-project", owner, "review-other"}, {"foreign-project", "foreign-owner", "review-foreign"}} {
		createCorrelationAudit(t, ctx, pool, selected.owner, selected.project, selected.audit)
	}
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{pool: pool, findings: intake, now: time.Now}
	fingerprint := correlationFingerprint(t)
	document := auditFindingDocument("correlation-first")
	document.SeveritySuggestion = "low"
	first := seedAuditFindingDocument(t, ctx, pool, project, owner, "review-a", "correlation-first", document, fingerprint)
	pending, err := service.CreateFindingReview(ctx, CreateFindingReviewParams{OwnerID: owner, AuditID: "review-a", FindingID: first, ExpectedRevision: 1, RequestID: "correlation-pending", IdempotencyKey: "correlation-pending", RequestDigest: serviceTestDigest("correlation-pending")})
	if err != nil {
		t.Fatal(err)
	}
	before, err := service.GetFinding(ctx, owner, "review-a", first)
	if err != nil {
		t.Fatal(err)
	}
	second := seedAuditFindingDocument(t, ctx, pool, project, owner, "review-b", "correlation-second", auditFindingDocument("correlation-second"), fingerprint)
	rejected := decideFindingForTest(t, ctx, service, owner, "review-b", second, 1, "correlation-fp", auditstore.VerdictFalsePositive, nil, nil)
	high := auditstore.SeverityHigh
	accepted, err := service.DecideFinding(ctx, DecideFindingParams{OwnerID: owner, AuditID: "review-a", RequestID: pending.Request.RequestID, ExpectedRequestRevision: pending.Request.Revision, DecisionID: "correlation-tp", Verdict: auditstore.VerdictTruePositive, Severity: &high, Rationale: "Independent verification.", IdempotencyKey: "correlation-tp", RequestDigest: serviceTestDigest("correlation-tp")})
	if err != nil {
		t.Fatalf("correlation expired or invalidated pending review: %v", err)
	}
	after, err := service.GetFinding(ctx, owner, "review-a", first)
	if err != nil {
		t.Fatal(err)
	}
	peer, err := service.GetFinding(ctx, owner, "review-b", second)
	if err != nil {
		t.Fatal(err)
	}
	if before.Correlation == nil || after.Correlation == nil || peer.Correlation == nil || after.Correlation.GroupKey != peer.Correlation.GroupKey || !after.Correlation.HasOtherObservations || after.Correlation.ProjectRevision <= before.Correlation.ProjectRevision {
		t.Fatalf("group/fence missing: %+v %+v", after.Correlation, peer.Correlation)
	}
	if after.State != auditstore.FindingConfirmed || after.AnalystVerdict == nil || *after.AnalystVerdict != auditstore.VerdictTruePositive || after.AnalystSeverity == nil || *after.AnalystSeverity != high || peer.State != auditstore.FindingRejected || peer.AnalystVerdict == nil || *peer.AnalystVerdict != auditstore.VerdictFalsePositive || peer.AnalystSeverity != nil || peer.Revision != rejected.Revision || after.Revision != accepted.Finding.Revision || after.FirstProposal.Document.SeveritySuggestion != "low" {
		t.Fatalf("independent ratings changed: %+v %+v", after, peer)
	}
	third := seedAuditFindingDocument(t, ctx, pool, project, owner, "review-a", "correlation-manual", auditFindingDocument("correlation-manual"), fingerprint)
	manual := decideFindingForTest(t, ctx, service, owner, "review-a", third, 1, "correlation-manual", auditstore.VerdictDuplicate, nil, &first)
	if manual.State != auditstore.FindingDuplicate || manual.DuplicateTargetID == nil || *manual.DuplicateTargetID != first {
		t.Fatalf("manual link changed: %+v", manual)
	}
	for _, selected := range []struct{ project, owner, audit, suffix string }{{"other-project", owner, "review-other", "other"}, {"foreign-project", "foreign-owner", "review-foreign", "foreign"}} {
		id := seedAuditFindingDocument(t, ctx, pool, selected.project, selected.owner, selected.audit, selected.suffix, auditFindingDocument(selected.suffix), fingerprint)
		row, err := service.GetFinding(ctx, selected.owner, selected.audit, id)
		if err != nil || row.Correlation == nil || row.Correlation.GroupKey == after.Correlation.GroupKey || row.Correlation.HasOtherObservations || row.Correlation.AutomaticDuplicate {
			t.Fatalf("cross-Project grouping: %+v %v", row.Correlation, err)
		}
		if selected.owner != owner {
			if _, err := service.GetFinding(ctx, owner, selected.audit, id); !errors.Is(err, auditstore.ErrNotFound) {
				t.Fatalf("foreign visibility: %v", err)
			}
		}
	}
	// A later contribution with a different primary cannot bridge groups or
	// rewrite the first-receipt choice. Existing contribution retention owns
	// its separate receipt; it is not an additional automatic membership.
	separate := fingerprint
	separate.Value = strings.Repeat("b", 64)
	contributor := seedAuditFindingDocument(t, ctx, pool, project, owner, "review-a", "later-key", auditFindingDocument("later-key"), separate)
	if _, err := pool.Exec(ctx, "DELETE FROM audit_findings WHERE finding_id=$1", contributor); err != nil {
		t.Fatal(err)
	}
	beforeContribution, err := service.GetFinding(ctx, owner, "review-a", first)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO audit_finding_contributions(finding_id,audit_id,receipt_id,relation,proposal_ref)
 SELECT $1,'review-a',receipt_id,'contributing',proposal_ref FROM finding_proposal_audit_holds WHERE receipt_id='receipt-later-key' AND audit_id='review-a'`, first); err != nil {
		t.Fatal(err)
	}
	afterContribution, err := service.GetFinding(ctx, owner, "review-a", first)
	if err != nil {
		t.Fatal(err)
	}
	if afterContribution.Correlation.GroupKey != beforeContribution.Correlation.GroupKey || afterContribution.Correlation.ProjectRevision <= beforeContribution.Correlation.ProjectRevision || afterContribution.Revision != beforeContribution.Revision || *afterContribution.AnalystVerdict != auditstore.VerdictTruePositive {
		t.Fatalf("later contribution regrouped result: %+v", afterContribution)
	}
	// No key is not an equality group, even when the same ordinary component repeats.
	weak := seedAuditFinding(t, ctx, pool, project, owner, "review-a", "weak")
	weakRow, err := service.GetFinding(ctx, owner, "review-a", weak)
	if err != nil || weakRow.Correlation == nil || weakRow.Correlation.Availability != "unavailable" || weakRow.Correlation.AutomaticDuplicate {
		t.Fatalf("weak evidence grouped: %+v %v", weakRow.Correlation, err)
	}
}

func createCorrelationAudit(t *testing.T, ctx context.Context, pool *pgxpool.Pool, owner, project, audit string) {
	t.Helper()
	_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Correlation", IdempotencyKey: project, RequestDigest: serviceTestDigest(project)})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = auditstoretest.NewStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: audit, OwnerID: owner, ProjectID: project, Profile: auditstore.ProfileIdentity{Name: "correlation", Version: "1", Digest: serviceTestDigest("profile")}, ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`), InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 2, MaxItemsTotal: 2, MaxSubmittedRuns: 2, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20}, IdempotencyKey: audit, RequestDigest: serviceTestDigest(audit)})
	if err != nil {
		t.Fatal(err)
	}
}

func correlationFingerprint(t *testing.T) auditdomain.FindingFingerprint {
	t.Helper()
	revision := "1"
	source := contracts.ExactContent(contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &revision}, "application/zip", []byte("source"))
	report := contracts.ExactContent(contracts.ArtifactRef{Namespace: "inputs", Name: "report", Revision: &revision}, "application/sarif+json", []byte("report"))
	value := auditdomain.FindingFingerprint{Schema: auditdomain.FindingFingerprintSchema, Availability: "available", Algorithm: "contractor/producer-issue/v1", Value: strings.Repeat("a", 64), Adapter: "pinned-sarif-native/v1", RuleID: "rule/test", Repository: "https://example.test/repo.git", Commit: strings.Repeat("a", 40), SourceDigest: source.Digest, Source: &source, Report: &report, RecordLocator: "/runs/0/results/0"}
	if _, err := auditdomain.EncodeFindingFingerprint(value); err != nil {
		t.Fatal(err)
	}
	return value
}

type correlationQueryTrace struct {
	sql  string
	args []any
}

func (q *correlationQueryTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	q.sql, q.args = data.SQL, data.Args
	return ctx
}
func (*correlationQueryTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func TestFindingCorrelationLargeGroupUsesBoundedIndexProbes(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, url)
	const owner, project, audit = "scale-owner", "scale-project", "scale-audit"
	createCorrelationAudit(t, ctx, pool, owner, project, audit)
	first := seedAuditFindingDocument(t, ctx, pool, project, owner, audit, "scale", auditFindingDocument("scale"), correlationFingerprint(t))
	// Retained independent rows exercise the production query at 10001 peers;
	// their proposal artifacts need not be rehydrated by this row projection.
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_receipts
 SELECT copied.* FROM finding_proposal_receipts AS original CROSS JOIN generate_series(1,10000) AS n
 CROSS JOIN LATERAL jsonb_populate_record(NULL::finding_proposal_receipts,to_jsonb(original)||jsonb_build_object(
 'receipt_id','scale-receipt-'||n,'proposal_id','scale-proposal-'||n,'invocation_id','scale-invocation-'||n,
 'submission_id','scale-submission-'||n,'client_key','scale-client-'||n)) AS copied WHERE original.receipt_id='receipt-scale'`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO audit_findings SELECT copied.* FROM audit_findings AS original
 CROSS JOIN generate_series(1,10000) AS n
 CROSS JOIN LATERAL jsonb_populate_record(NULL::audit_findings,to_jsonb(original)||jsonb_build_object(
 'finding_id','scale-finding-'||lpad(n::text,5,'0'),'first_receipt_id','scale-receipt-'||n)) AS copied
 WHERE original.finding_id=$1`, first); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "ANALYZE audit_findings"); err != nil {
		t.Fatal(err)
	}
	trace := &correlationQueryTrace{}
	config := pool.Config()
	config.MaxConns, config.MinConns = 1, 0
	config.ConnConfig.Tracer = trace
	reader, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	row, err := querystore.New(reader).GetFinding(ctx, owner, audit, first)
	if err != nil || row.Correlation == nil || !row.Correlation.HasOtherObservations {
		t.Fatalf("large group=%+v %v", row.Correlation, err)
	}
	var raw []byte
	if err := pool.QueryRow(ctx, "EXPLAIN (ANALYZE, FORMAT JSON) "+trace.sql, trace.args...).Scan(&raw); err != nil {
		t.Fatal(err)
	}
	var plan []struct{ Plan map[string]any }
	if err := json.Unmarshal(raw, &plan); err != nil {
		t.Fatal(err)
	}
	probes := 0
	var visit func(map[string]any)
	visit = func(node map[string]any) {
		if node["Index Name"] == "audit_findings_correlation_group_idx" {
			probes++
			if node["Actual Rows"].(float64) > 1 || node["Actual Loops"].(float64) > 1 {
				t.Fatalf("unbounded probe: %s", raw)
			}
		}
		children, _ := node["Plans"].([]any)
		for _, child := range children {
			visit(child.(map[string]any))
		}
	}
	visit(plan[0].Plan)
	if probes != 2 {
		t.Fatalf("expected two group index probes, got %d: %s", probes, raw)
	}
}
