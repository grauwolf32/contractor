package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestAuditSameReceiptHasIndependentAnalystDecisions(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	const ownerID, projectID, firstAudit, secondAudit = "owner-shared", "project-shared", "audit-first", "audit-second"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject, Name: "Independent finding reviews",
		IdempotencyKey: projectID, RequestDigest: serviceTestDigest(projectID),
	}); err != nil {
		t.Fatal(err)
	}
	for _, auditID := range []string{firstAudit, secondAudit} {
		if _, _, err := auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
			AuditID: auditID, OwnerID: ownerID, ProjectID: projectID,
			Profile:         auditstore.ProfileIdentity{Name: "finding-review", Version: "1", Digest: serviceTestDigest("profile")},
			ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`), InputSelection: json.RawMessage(`{}`),
			Limits:         auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1, MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
			IdempotencyKey: auditID, RequestDigest: serviceTestDigest(auditID),
		}); err != nil {
			t.Fatal(err)
		}
	}
	firstFinding := seedAuditFinding(t, ctx, pool, projectID, ownerID, firstAudit, "shared")
	// Reuse the exact immutable receipt, exercising admission of a separate
	// finding rather than fabricating two unrelated findings for this review.
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence)
 SELECT receipt_id,$2,project_id,proposal_ref,evidence FROM finding_proposal_audit_holds
 WHERE audit_id=$1 AND receipt_id='receipt-shared'`, firstAudit, secondAudit); err != nil {
		t.Fatal(err)
	}
	var secondFinding string
	if err := pool.QueryRow(ctx, `SELECT finding_id FROM audit_findings WHERE audit_id=$1 AND first_receipt_id='receipt-shared'`, secondAudit).Scan(&secondFinding); err != nil {
		t.Fatal(err)
	}
	if firstFinding == secondFinding {
		t.Fatal("Audits share finding identity")
	}
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{pool: pool, findings: intake, now: time.Now}
	first := decideFindingForTest(t, ctx, service, ownerID, firstAudit, firstFinding, 1, "first", VerdictTruePositive, reviewStringPointer("high"), nil)
	untouched, err := service.GetFinding(ctx, ownerID, secondAudit, secondFinding)
	if err != nil || untouched.State != FindingProposed || untouched.Revision != 1 || untouched.AnalystVerdict != nil {
		t.Fatalf("first decision changed second Audit: %+v %v", untouched, err)
	}
	second := decideFindingForTest(t, ctx, service, ownerID, secondAudit, secondFinding, 1, "second", VerdictFalsePositive, nil, nil)
	after, err := service.GetFinding(ctx, ownerID, firstAudit, firstFinding)
	if err != nil || after.State != FindingConfirmed || after.Revision != first.Revision || after.AnalystVerdict == nil || *after.AnalystVerdict != VerdictTruePositive || second.State != FindingRejected {
		t.Fatalf("decisions are not independent: first=%+v second=%+v err=%v", after, second, err)
	}
	if _, err := service.GetFinding(ctx, ownerID, secondAudit, firstFinding); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("cross-Audit finding lookup: %v", err)
	}
	var decisions int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM audit_review_decisions WHERE audit_id IN ($1,$2)`, firstAudit, secondAudit).Scan(&decisions); err != nil || decisions != 2 {
		t.Fatalf("independent immutable decisions=%d err=%v", decisions, err)
	}
}
