package auditservice

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestOwnerFindingPageQueriesAreConstantInAudits lists one owner's findings
// spread over forty Audits and another owner's findings of one Audit, with
// the same page shape, and finds the same statements for both: receipts,
// holds, decisions and assessments are each read once for the whole page,
// whatever the number of Audits on it. A receipt held by two Audits of one
// page still exposes only each finding's own Audit hold.
func TestOwnerFindingPageQueriesAreConstantInAudits(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	writer := &Service{pool: pool, findings: intake, now: time.Now}
	createProject := func(ownerID, projectID string) {
		t.Helper()
		if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
			ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject, Name: projectID,
			IdempotencyKey: projectID, RequestDigest: serviceTestDigest(projectID),
		}); err != nil {
			t.Fatal(err)
		}
	}
	createAudit := func(ownerID, projectID, auditID string) {
		t.Helper()
		if _, _, err := auditstoretest.NewStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
			AuditID: auditID, OwnerID: ownerID, ProjectID: projectID,
			Profile:         auditstore.ProfileIdentity{Name: "owner-page", Version: "1", Digest: serviceTestDigest("owner-page")},
			ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`),
			InputSelection:  json.RawMessage(`{}`),
			Limits:          auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100, MaxSubmittedRuns: 100, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
			IdempotencyKey:  auditID, RequestDigest: serviceTestDigest(auditID),
		}); err != nil {
			t.Fatal(err)
		}
	}
	// review decides a finding and gives it a current assessment, so every
	// related read of the page has rows.
	review := func(ownerID, auditID, findingID, receiptID, suffix string) {
		t.Helper()
		severity := "high"
		decideFindingForTest(t, ctx, writer, ownerID, auditID, findingID, 1, "decision-"+suffix, VerdictTruePositive, &severity, nil)
		if _, err := pool.Exec(ctx, `
INSERT INTO audit_finding_assessments (
    assessment_id, finding_id, audit_id, receipt_id, semantic_assessment,
    result_ref, result_digest, direct_verification, contract_ref, contract_digest
) VALUES ($1, $2, $3, $4, 'supported',
    '{"namespace":"audit-results","name":"direct","revision":"r1"}'::jsonb, $5, true,
    '{"namespace":"audit-contracts","name":"direct","revision":"r1"}'::jsonb, $5)`,
			"assessment-"+suffix, findingID, auditID, receiptID, serviceTestDigest(suffix)); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `UPDATE audit_findings SET current_assessment_id = $1 WHERE finding_id = $2`,
			"assessment-"+suffix, findingID); err != nil {
			t.Fatal(err)
		}
	}

	const audits = 40
	const spreadOwner, spreadProject = "owner-spread", "project-spread"
	createProject(spreadOwner, spreadProject)
	for index := range audits {
		auditID, suffix := fmt.Sprintf("audit-spread-%02d", index), fmt.Sprintf("spread-%02d", index)
		createAudit(spreadOwner, spreadProject, auditID)
		review(spreadOwner, auditID, seedAuditFinding(t, ctx, pool, spreadProject, spreadOwner, auditID, suffix), "receipt-"+suffix, suffix)
	}
	// The second Audit also holds the first Audit's receipt, so one page reads
	// one receipt through two Audits.
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence)
 SELECT receipt_id,'audit-spread-01',project_id,proposal_ref,evidence FROM finding_proposal_audit_holds
 WHERE audit_id='audit-spread-00' AND receipt_id='receipt-spread-00'`); err != nil {
		t.Fatal(err)
	}
	var sharedFinding string
	if err := pool.QueryRow(ctx, `SELECT finding_id FROM audit_findings WHERE audit_id='audit-spread-01' AND first_receipt_id='receipt-spread-00'`).Scan(&sharedFinding); err != nil {
		t.Fatal(err)
	}
	review(spreadOwner, "audit-spread-01", sharedFinding, "receipt-spread-00", "spread-shared")

	const singleOwner, singleProject, singleAudit = "owner-single", "project-single", "audit-single"
	createProject(singleOwner, singleProject)
	createAudit(singleOwner, singleProject, singleAudit)
	for index := range audits + 1 {
		suffix := fmt.Sprintf("single-%02d", index)
		review(singleOwner, singleAudit, seedAuditFinding(t, ctx, pool, singleProject, singleOwner, singleAudit, suffix), "receipt-"+suffix, suffix)
	}

	trace := &findingQueryTrace{}
	configuration := pool.Config()
	configuration.MaxConns, configuration.MinConns = 1, 0
	configuration.ConnConfig.Tracer = trace
	readerPool, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	defer readerPool.Close()
	readerIntake, err := findingintake.New(readerPool)
	if err != nil {
		t.Fatal(err)
	}
	reader := &Service{pool: readerPool, findings: readerIntake, now: time.Now}
	page := func(ownerID string) []Finding {
		t.Helper()
		trace.take()
		findings, err := reader.ListOwnerFindings(ctx, OwnerFindingListParams{OwnerID: ownerID, Limit: audits + 1})
		if err != nil || len(findings) != audits+1 {
			t.Fatalf("%s page = (%d findings, %v), want %d", ownerID, len(findings), err, audits+1)
		}
		// Base findings, receipts, holds, decisions and assessments, plus one
		// exact Artifact read per bounded group of 32 proposals.
		want := 5 + (audits+1+artifacts.MaxExactReadBatchSize-1)/artifacts.MaxExactReadBatchSize
		if got := trace.take(); got != want {
			t.Fatalf("%s page of %d findings issued %d statements, want %d", ownerID, len(findings), got, want)
		}
		return findings
	}
	spread := page(spreadOwner)
	page(singleOwner)

	auditsOnPage := make(map[string]bool)
	sharedReads := 0
	for index, finding := range spread {
		auditsOnPage[finding.AuditID] = true
		if finding.AnalystDecision == nil || finding.AnalystDecision.FindingID != finding.FindingID ||
			finding.CurrentAssessment == nil || finding.CurrentAssessment.ReceiptID != finding.FirstProposal.ReceiptID {
			t.Fatalf("finding %s has another finding's related records: %+v", finding.FindingID, finding)
		}
		holds := finding.FirstProposal.AuditHolds
		if len(holds) != 1 || holds[0].AuditID != finding.AuditID {
			t.Fatalf("finding %s of %s exposes holds %+v, want only its own Audit's", finding.FindingID, finding.AuditID, holds)
		}
		if finding.FirstProposal.ReceiptID == "receipt-spread-00" {
			sharedReads++
		}
		if index > 0 && finding.CreatedAt.After(spread[index-1].CreatedAt) {
			t.Fatal("owner page is not newest first")
		}
		single, err := writer.GetFinding(ctx, spreadOwner, finding.AuditID, finding.FindingID)
		if err != nil || !reflect.DeepEqual(single, finding) {
			t.Fatalf("page hydration of %s differs from a single read: %v", finding.FindingID, err)
		}
	}
	if len(auditsOnPage) != audits || sharedReads != 2 {
		t.Fatalf("page covers %d Audits and reads the shared receipt %d times, want %d and 2", len(auditsOnPage), sharedReads, audits)
	}
}
