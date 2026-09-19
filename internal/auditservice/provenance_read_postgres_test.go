package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The callback commits through a different pool at a selected real SQL read.
// It introduces a deterministic interleaving without sleeps or production hooks.
type provenanceReadTrace struct {
	point         string
	mutate        func()
	sawProvenance bool
	called        bool
}

func (trace *provenanceReadTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	provenance := strings.Contains(data.SQL, "WITH anchors AS")
	trace.sawProvenance = trace.sawProvenance || provenance
	hydration := trace.sawProvenance && strings.Contains(data.SQL, "SELECT input.ordinality")
	if !trace.called && ((trace.point == "provenance-query" && provenance) || (trace.point == "receipt-hydration" && hydration)) {
		trace.called = true
		trace.mutate()
	}
	return ctx
}
func (*provenanceReadTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func TestFindingProvenanceConsistentReadsWithSingleConnection(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	const owner, auditID, projectID = "owner-provenance", "audit-provenance", "project-provenance"
	_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: owner, Kind: projectstore.KindProject, Name: "Provenance", IdempotencyKey: "project", RequestDigest: serviceTestDigest("project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: owner, ProjectID: projectID,
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: serviceTestDigest("profile")},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`), InputSelection: json.RawMessage(`{}`),
		Limits:         auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100, MaxSubmittedRuns: 100, MaxItemRunAttempts: 2, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey: "audit", RequestDigest: serviceTestDigest("audit"),
	})
	if err != nil {
		t.Fatal(err)
	}
	findingID := seedAuditFinding(t, ctx, pool, projectID, owner, auditID, "provenance")
	seedAuditFindingAttemptHistory(t, ctx, pool, auditID, findingID, "receipt-provenance")
	setCurrent := func(assessment string) {
		_, err := pool.Exec(ctx, `WITH changed AS (
   UPDATE audit_findings SET current_assessment_id=$3, revision=revision+1
   WHERE audit_id=$1 AND finding_id=$2 RETURNING audit_id
  ) UPDATE audits SET revision=revision+1 WHERE audit_id IN (SELECT audit_id FROM changed)`, auditID, findingID, assessment)
		if err != nil {
			t.Fatal(err)
		}
	}
	setCurrent("assessment-attempt-two")
	newReader := func(t *testing.T, trace pgx.QueryTracer) *Service {
		t.Helper()
		configuration := pool.Config()
		configuration.MaxConns, configuration.MinConns = 1, 0
		configuration.ConnConfig.Tracer = trace
		readerPool, err := pgxpool.NewWithConfig(ctx, configuration)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(readerPool.Close)
		intake, err := findingintake.New(readerPool)
		if err != nil {
			t.Fatal(err)
		}
		return &Service{pool: readerPool, findings: intake, now: time.Now}
	}
	params := ProvenanceListParams{OwnerID: owner, AuditID: auditID, FindingID: findingID, Limit: 201}
	reader := newReader(t, nil)
	var stable []FindingProvenance
	t.Run("retained exact history and pagination", func(t *testing.T) {
		readCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		finding, err := reader.GetFinding(readCtx, owner, auditID, findingID)
		if err != nil {
			t.Fatal(err)
		}
		stable, err = reader.ListFindingProvenance(readCtx, params)
		if err != nil || len(stable) != 4 {
			t.Fatalf("provenance count=%d err=%v", len(stable), err)
		}
		for _, item := range stable {
			if !reflect.DeepEqual(item.Origin, finding.FirstProposal.Origin) || !item.Origin.RunDeleted ||
				!reflect.DeepEqual(item.Proposal.Ref, finding.FirstProposal.Proposal.Ref) || item.Proposal.Digest != finding.FirstProposal.Proposal.Digest {
				t.Fatalf("retained receipt mismatch: %+v", item)
			}
			wantCurrent := item.Assessment != nil && item.Assessment.AssessmentID == "assessment-attempt-two"
			if item.SupportsCurrent != wantCurrent {
				t.Fatalf("current assessment mismatch: %+v", item)
			}
		}
		if stable[0].Kind != ProvenanceSourceProposal || stable[1].Attempt == nil || stable[1].Attempt.ItemAttempt != 1 ||
			stable[2].Attempt == nil || stable[2].Attempt.ItemAttempt != 2 || stable[3].Kind != ProvenanceDirect || stable[3].Assessment.Contract == nil {
			t.Fatalf("lost attempt/direct history: %+v", stable)
		}
		pageParams := params
		pageParams.Limit = 2
		var traversed []FindingProvenance
		for range 2 {
			page, err := reader.ListFindingProvenance(readCtx, pageParams)
			if err != nil || len(page) != 2 {
				t.Fatalf("page count=%d err=%v", len(page), err)
			}
			traversed = append(traversed, page...)
			last := page[len(page)-1]
			pageParams.AfterCreatedAt = &last.CreatedAt
			pageParams.AfterRecordID = last.RecordID
		}
		if !reflect.DeepEqual(traversed, stable) {
			t.Fatal("pagination changed order, identity or exact history")
		}
	})
	if len(stable) != 4 {
		t.FailNow()
	}
	t.Run("owner finding and explicit revision pins", func(t *testing.T) {
		revisions, err := reader.readProvenanceRevisions(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		pinned := params
		pinned.AuditRevision = &revisions.audit
		pinned.FindingRevision = &revisions.finding
		if _, err := reader.ListFindingProvenance(ctx, pinned); err != nil {
			t.Fatal(err)
		}
		zero := uint64(0)
		for _, field := range []string{"audit", "finding"} {
			stale := pinned
			if field == "audit" {
				stale.AuditRevision = &zero
			} else {
				stale.FindingRevision = &zero
			}
			if items, err := reader.ListFindingProvenance(ctx, stale); !errors.Is(err, auditstore.ErrConflict) || items != nil {
				t.Fatalf("%s zero pin: items=%v err=%v", field, items, err)
			}
		}
		foreign := params
		foreign.OwnerID = "other-owner"
		if _, err := reader.ListFindingProvenance(ctx, foreign); !errors.Is(err, auditstore.ErrNotFound) {
			t.Fatalf("foreign owner: %v", err)
		}
		foreign = params
		foreign.AuditID = "other-audit"
		if _, err := reader.ListFindingProvenance(ctx, foreign); !errors.Is(err, auditstore.ErrNotFound) {
			t.Fatalf("foreign audit: %v", err)
		}
	})
	for _, point := range []string{"provenance-query", "receipt-hydration"} {
		t.Run("changed-current-assessment-"+point, func(t *testing.T) {
			setCurrent("assessment-attempt-two")
			trace := &provenanceReadTrace{point: point, mutate: func() { setCurrent("assessment-direct") }}
			changing := newReader(t, trace)
			readCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			items, err := changing.ListFindingProvenance(readCtx, params)
			if !trace.called || !errors.Is(err, auditstore.ErrConflict) || items != nil {
				t.Fatalf("mutation called=%t items=%v err=%v", trace.called, items, err)
			}
			refreshed, err := reader.ListFindingProvenance(readCtx, params)
			if err != nil {
				t.Fatal(err)
			}
			for _, item := range refreshed {
				want := item.Assessment != nil && item.Assessment.AssessmentID == "assessment-direct"
				if item.SupportsCurrent != want {
					t.Fatalf("refreshed current assessment mismatch: %+v", item)
				}
			}
		})
	}
	t.Run("empty page also fenced", func(t *testing.T) {
		last := stable[len(stable)-1]
		empty := params
		empty.AfterCreatedAt = &last.CreatedAt
		empty.AfterRecordID = last.RecordID
		trace := &provenanceReadTrace{point: "provenance-query", mutate: func() { setCurrent("assessment-attempt-two") }}
		if items, err := newReader(t, trace).ListFindingProvenance(ctx, empty); !trace.called || !errors.Is(err, auditstore.ErrConflict) || items != nil {
			t.Fatalf("called=%t items=%v err=%v", trace.called, items, err)
		}
	})

	t.Run("201 unique receipts plus repeated history and unavailable receipt", func(t *testing.T) {
		// Match the existing contributing-source fixture used by finding
		// collection tests: remove each auxiliary finding, retaining its exact
		// Audit hold, and attach that receipt to the reviewed finding.
		for index := range 200 {
			suffix := fmt.Sprintf("contribution-%03d", index)
			auxiliary := seedAuditFinding(t, ctx, pool, projectID, owner, auditID, suffix)
			if _, err := pool.Exec(ctx, `DELETE FROM audit_findings WHERE finding_id=$1`, auxiliary); err != nil {
				t.Fatal(err)
			}
			if _, err := pool.Exec(ctx, `
INSERT INTO audit_finding_contributions (finding_id,audit_id,receipt_id,relation,proposal_ref,created_at)
SELECT $1,$2,receipt_id,'contributing',proposal_ref,'2000-01-01'::timestamptz
  FROM finding_proposal_audit_holds WHERE receipt_id=$3 AND audit_id=$2`, findingID, auditID, "receipt-"+suffix); err != nil {
				t.Fatal(err)
			}
		}
		readCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		page, err := reader.ListFindingProvenance(readCtx, params)
		if err != nil || len(page) != 201 {
			t.Fatalf("unique receipt page count=%d err=%v", len(page), err)
		}
		for index, item := range page {
			suffix := fmt.Sprintf("contribution-%03d", index)
			if index == 200 {
				suffix = "provenance"
			}
			if item.Kind != ProvenanceSourceProposal || item.RecordID != "proposal:receipt-"+suffix || item.ReceiptID != "receipt-"+suffix ||
				item.Origin.RunID != "deleted-run-"+suffix || !item.Origin.RunDeleted || item.Proposal.Ref.Name != "candidate-"+suffix ||
				item.Proposal.Ref.ValidateExact() != nil || item.Proposal.Digest == "" {
				t.Fatalf("source %d identity/exact receipt mismatch: %+v", index, item)
			}
		}
		last := page[len(page)-1]
		next := params
		next.AfterCreatedAt = &last.CreatedAt
		next.AfterRecordID = last.RecordID
		repeated, err := reader.ListFindingProvenance(readCtx, next)
		if err != nil || len(repeated) != 3 {
			t.Fatalf("repeated history count=%d err=%v", len(repeated), err)
		}
		for index, item := range repeated {
			if item.RecordID != stable[index+1].RecordID || item.ReceiptID != last.ReceiptID || !reflect.DeepEqual(item.Origin, last.Origin) {
				t.Fatalf("repeated history %d changed: %+v", index, item)
			}
		}
		// Missing authorization for one contributing receipt must reject the
		// entire page, not return a partial page with a blank origin.
		if _, err := pool.Exec(ctx, `DELETE FROM finding_proposal_audit_holds WHERE audit_id=$1 AND receipt_id=$2`, auditID, "receipt-contribution-199"); err != nil {
			t.Fatal(err)
		}
		if items, err := reader.ListFindingProvenance(readCtx, params); !errors.Is(err, findingintake.ErrNotFound) || items != nil {
			t.Fatalf("unavailable contributing receipt: items=%v err=%v", items, err)
		}
		if _, err := reader.GetFinding(readCtx, owner, auditID, findingID); err != nil {
			t.Fatalf("connection not reusable after rejected page: %v", err)
		}
	})
}
