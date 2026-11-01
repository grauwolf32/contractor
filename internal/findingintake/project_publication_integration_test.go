//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

func TestProjectFindingExplicitPublicationRetainsIndependentSources(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	const owner, project = "publication-owner", "publication-project"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Publication",
		IdempotencyKey: "publication", RequestDigest: auditdomain.DigestBytes([]byte("publication")),
	}); err != nil {
		t.Fatal(err)
	}
	service, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	grants := make([]controlplane.AllocationGrant, 2)
	inputs := make([]Submission, 2)
	for index, run := range []string{"publication-run-a", "publication-run-b"} {
		var refs []contracts.ArtifactRef
		grants[index] = quotaSubmissionGrantWithInputs(t, ctx, pool, owner, project, run, func() map[string]runstore.PinnedContextArtifact {
			refs = writeFingerprintInputs(t, ctx, pool, run)
			return map[string]runstore.PinnedContextArtifact{"source": {Artifact: &refs[0]}, "report": {Artifact: &refs[1]}}
		})
		inputs[index] = testSubmission("publication-invocation", "problem", refs)
		inputs[index].PublishProblem = true
		inputs[index].Proposal.Title = "rule/sql"
		inputs[index].Proposal.Description = "Native issue message"
		inputs[index].Proposal.Subject = nil
		line := int64(3)
		inputs[index].Proposal.Locations = []auditdomain.FindingLocation{{File: "src/a.py", Line: &line}}
	}

	// Evidence and a qualified key alone express no publication intent.
	observation := inputs[0]
	observation.PublishProblem = false
	observation.Proposal.ClientKey = "observation"
	observation.SubmissionID = stableSubmissionID(observation.InvocationID, observation.Proposal.ClientKey)
	ordinary, _, err := service.Submit(ctx, grants[0], observation)
	if err != nil || ordinary.FindingID != nil || ordinary.Fingerprint.Availability != "available" {
		t.Fatalf("ordinary evidenced observation: finding=%v err=%v", ordinary.FindingID, err)
	}
	invalid := testSubmission("publication-empty", "problem", nil)
	invalid.PublishProblem = true
	rejected, _, err := service.Submit(ctx, grants[0], invalid)
	if err != nil || rejected.FindingID != nil || rejected.PublicationError != "problem_evidence_required" || rejected.ReceiptID == "" {
		t.Fatalf("invalid publication lost ordinary proposal: %+v err=%v", rejected, err)
	}

	// Independent actual producers contend on the existing Project fence.
	var group sync.WaitGroup
	start := make(chan struct{})
	receipts := make([]Receipt, 2)
	submissionErrors := make([]error, 2)
	for index := range inputs {
		group.Go(func() {
			<-start
			receipts[index], _, submissionErrors[index] = service.Submit(ctx, grants[index], inputs[index])
		})
	}
	close(start)
	group.Wait()
	for index, receipt := range receipts {
		if submissionErrors[index] != nil || receipt.FindingID == nil || receipt.ProjectProposal == nil || len(receipt.ProjectEvidence) != 2 {
			t.Fatalf("publication %d: %+v err=%v", index, receipt, submissionErrors[index])
		}
	}
	if *receipts[0].FindingID != *receipts[1].FindingID || receipts[0].ReceiptID == receipts[1].ReceiptID {
		t.Fatal("qualified independent discoveries did not share exactly one canonical Finding")
	}
	for index, input := range inputs {
		replay, replayed, err := service.Submit(ctx, grants[index], input)
		if err != nil || !replayed || replay.FindingID == nil || *replay.FindingID != *receipts[index].FindingID {
			t.Fatalf("publication replay %d: %+v replayed=%v err=%v", index, replay, replayed, err)
		}
	}
	var count, sources, assessed int
	if err := pool.QueryRow(ctx, `SELECT count(*), count(*) FILTER(WHERE state IS NOT NULL) FROM project_findings`).Scan(&count, &assessed); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM project_finding_sources`).Scan(&sources); err != nil {
		t.Fatal(err)
	}
	if count != 1 || sources != 2 || assessed != 0 {
		t.Fatalf("canonical findings=%d independent sources=%d inferred assessments=%d", count, sources, assessed)
	}
	for index, auditID := range []string{"publication-audit-a", "publication-audit-b"} {
		if _, _, err := auditstoretest.NewStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
			AuditID: auditID, OwnerID: owner, ProjectID: project,
			Profile:         auditstore.ProfileIdentity{Name: "publication", Version: "1", Digest: auditdomain.DigestBytes([]byte("profile"))},
			ProfileSnapshot: json.RawMessage(`{}`), InputSelection: json.RawMessage(`{}`),
			Limits:         auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1, MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
			IdempotencyKey: auditID, RequestDigest: auditdomain.DigestBytes([]byte(auditID)),
		}); err != nil {
			t.Fatal(err)
		}
		if _, _, err := service.ImportIntoAudit(ctx, ImportRequest{OwnerID: owner, AuditID: auditID,
			RunID: grants[index].RunID, Proposal: receipts[index].Proposal.Ref}); err != nil {
			t.Fatal(err)
		}
	}
	read := querystore.New(pool)
	findingID := *receipts[0].FindingID
	var edited auditstore.ProjectFinding
	for index, auditID := range []string{"publication-audit-a", "publication-audit-b"} {
		fromAudit, err := read.GetAuditProjectFinding(ctx, owner, auditID, findingID)
		if err != nil || fromAudit.Revision != uint64(index+1) {
			t.Fatalf("Audit view before edit: %+v %v", fromAudit, err)
		}
		severity := auditstore.SeverityHigh
		err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var editErr error
			edited, editErr = ownerstore.NewTx(tx).AssessProjectFinding(ctx, auditstore.ProjectFindingEdit{
				OwnerID: owner, ProjectID: project, FindingID: findingID, ExpectedRevision: fromAudit.Revision,
				Verdict: auditstore.VerdictTruePositive, Severity: &severity, Rationale: "Explicit Project assessment",
			})
			return editErr
		})
		if err != nil {
			t.Fatal(err)
		}
		for _, view := range []string{"", "publication-audit-a", "publication-audit-b"} {
			var current auditstore.ProjectFinding
			if view == "" {
				current, err = read.GetProjectFinding(ctx, owner, project, findingID)
			} else {
				current, err = read.GetAuditProjectFinding(ctx, owner, view, findingID)
			}
			if err != nil || !reflect.DeepEqual(current, edited) {
				t.Fatalf("view %q did not read the canonical edited record: %+v err=%v", view, current, err)
			}
		}
	}
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := ownerstore.NewTx(tx).AssessProjectFinding(ctx, auditstore.ProjectFindingEdit{
			OwnerID: owner, ProjectID: project, FindingID: findingID, ExpectedRevision: 1,
			Verdict: auditstore.VerdictFalsePositive, Rationale: "Stale edit",
		})
		return err
	})
	if !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("stale Project edit: %v", err)
	}
	var auditCopies int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM information_schema.tables WHERE table_schema=current_schema() AND table_name='audit_findings'`).Scan(&auditCopies); err != nil || auditCopies != 0 {
		t.Fatalf("Audit import recreated mutable Findings: %d err=%v", auditCopies, err)
	}
}
