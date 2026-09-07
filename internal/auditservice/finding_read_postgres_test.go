package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type findingQueryTrace struct {
	mu    sync.Mutex
	count int
}

func (q *findingQueryTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryStartData) context.Context {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.count++
	return ctx
}
func (*findingQueryTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}
func (q *findingQueryTrace) take() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	count := q.count
	q.count = 0
	return count
}

func TestFindingPagesBatchRelatedQueriesWithSingleConnection(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	const owner, auditID, projectID = "owner-batch", "audit-batch", "project-batch"
	_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: owner, Kind: projectstore.KindProject,
		Name: "Batch findings", IdempotencyKey: "batch-project", RequestDigest: serviceTestDigest("batch-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: owner, ProjectID: projectID,
		Profile:         auditstore.ProfileIdentity{Name: "batch-profile", Version: "1", Digest: serviceTestDigest("batch-profile")},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits:          auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100, MaxSubmittedRuns: 100, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey:  "batch-audit", RequestDigest: serviceTestDigest("batch-audit"),
	})
	if err != nil {
		t.Fatal(err)
	}
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	writer := &Service{pool: pool, findings: intake, now: time.Now}
	for index := range 65 {
		suffix := fmt.Sprintf("batch-%03d", index)
		findingID := seedAuditFinding(t, ctx, pool, projectID, owner, auditID, suffix)
		severity := "high"
		decideFindingForTest(t, ctx, writer, owner, auditID, findingID, 1, "decision-"+suffix, VerdictTruePositive, &severity, nil)
		_, err := pool.Exec(ctx, `
INSERT INTO audit_finding_assessments (
    assessment_id, finding_id, audit_id, receipt_id, semantic_assessment,
    result_ref, result_digest, direct_verification, contract_ref, contract_digest
) VALUES ($1, $2, $3, $4, 'supported',
    '{"namespace":"audit-results","name":"direct","revision":"r1"}'::jsonb, $5, true,
    '{"namespace":"audit-contracts","name":"direct","revision":"r1"}'::jsonb, $5)`,
			"assessment-"+suffix, findingID, auditID, "receipt-"+suffix, serviceTestDigest(suffix))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `UPDATE audit_findings SET current_assessment_id = $1 WHERE finding_id = $2`, "assessment-"+suffix, findingID); err != nil {
			t.Fatal(err)
		}
	}
	trace := &findingQueryTrace{}
	config := pool.Config()
	config.MaxConns, config.MinConns = 1, 0
	config.ConnConfig.Tracer = trace
	readerPool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer readerPool.Close()
	readerIntake, err := findingintake.New(readerPool)
	if err != nil {
		t.Fatal(err)
	}
	reader := &Service{pool: readerPool, findings: readerIntake, now: time.Now}
	var firstPage []Finding
	for _, count := range []int{1, 5, 30, 65} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			readCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
			defer cancel()
			trace.take()
			findings, err := reader.ListFindings(readCtx, FindingListParams{OwnerID: owner, AuditID: auditID, Limit: count})
			if err != nil || len(findings) != count {
				t.Fatalf("page length=%d error=%v", len(findings), err)
			}
			// Base findings, authorized receipts, holds, decisions, assessments,
			// plus one exact artifact query per bounded group of 32 documents.
			want := 5 + (count+artifacts.MaxExactReadBatchSize-1)/artifacts.MaxExactReadBatchSize
			if got := trace.take(); got != want {
				t.Fatalf("queries=%d want=%d for %d findings", got, want, count)
			}
			for index, finding := range findings {
				if finding.AnalystDecision == nil || finding.AnalystVerdict == nil || *finding.AnalystVerdict != VerdictTruePositive || finding.CurrentAssessment == nil {
					t.Fatalf("missing related finding data: %+v", finding)
				}
				if finding.AnalystDecision.FindingID != finding.FindingID || finding.CurrentAssessment.ReceiptID != finding.FirstProposal.ReceiptID {
					t.Fatalf("related record belongs to another finding: %+v", finding)
				}
				single, err := intake.GetAuditReceipt(ctx, owner, auditID, finding.FirstProposal.ReceiptID)
				if err != nil || !reflect.DeepEqual(single, finding.FirstProposal) {
					t.Fatalf("batch receipt differs from single read: %v", err)
				}
				if index > 0 && finding.CreatedAt.Before(findings[index-1].CreatedAt) {
					t.Fatal("page order changed")
				}
			}
			if count == 5 {
				firstPage = findings
			}
		})
	}
	last := firstPage[len(firstPage)-1]
	following, err := reader.ListFindings(ctx, FindingListParams{OwnerID: owner, AuditID: auditID, Limit: 5, AfterCreatedAt: &last.CreatedAt, AfterFindingID: last.FindingID})
	if err != nil || len(following) != 5 {
		t.Fatalf("next page: %v", err)
	}
	for _, item := range following {
		for _, previous := range firstPage {
			if item.FindingID == previous.FindingID {
				t.Fatal("pagination repeated a finding")
			}
		}
	}
	if _, err := reader.ListFindings(ctx, FindingListParams{OwnerID: "other-owner", AuditID: auditID, Limit: 5}); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign page error=%v", err)
	}
	receiptID := firstPage[0].FirstProposal.ReceiptID
	if _, err := readerIntake.GetAuditReceipts(ctx, "other-owner", auditID, []string{receiptID}); !errors.Is(err, findingintake.ErrNotFound) {
		t.Fatalf("foreign receipt error=%v", err)
	}
	if _, err := readerIntake.GetAuditReceipts(ctx, owner, auditID, []string{receiptID, "missing"}); !errors.Is(err, findingintake.ErrNotFound) {
		t.Fatalf("partial receipt batch error=%v", err)
	}
}
