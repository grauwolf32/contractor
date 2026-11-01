//go:build integration

package findingintake

import (
	"bytes"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresFindingChecksIndexIntakeImmutabilityAndRetention(t *testing.T) {
	for _, backend := range []string{"postgresql", "filesystem"} {
		t.Run(backend, func(t *testing.T) {
			ctx := t.Context()
			pool := isolatedFindingPool(t, ctx)
			if backend == "filesystem" {
				files, err := artifacts.OpenFilesystemBlobStore(ctx, t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				defer files.Close()
				ctx = artifacts.WithBlobRuntime(ctx, artifacts.NewBlobRuntime(files, nil))
			}
			const owner, project, run = "owner", "delete-project", "run"
			if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project,
				OwnerID: owner, Kind: projectstore.KindProject, Name: "Check indexes", IdempotencyKey: "project",
				RequestDigest: auditdomain.DigestBytes([]byte("project"))}); err != nil {
				t.Fatal(err)
			}
			grant := quotaSubmissionGrant(t, ctx, pool, owner, project, run)
			service, _ := New(pool)
			input := testSubmission("checks", "checks", nil)
			input.Proposal.ProposedChecks = []auditdomain.ProposedCheck{
				{Objective: "Verify UTF-8 世界, braces }{ and a quoted \"proposed_checks\" key.", Method: "static-trace"},
				{Objective: "Preserve an ordinary unsupported method.", Method: "operator-custom-method"},
			}
			receipt, _, err := service.Submit(ctx, grant, input)
			if err != nil {
				t.Fatal(err)
			}
			readIndex := func() []byte {
				t.Helper()
				var data []byte
				if err := pool.QueryRow(ctx, `SELECT checks_projection FROM finding_proposal_receipts WHERE receipt_id=$1`, receipt.ReceiptID).Scan(&data); err != nil {
					t.Fatal(err)
				}
				return data
			}
			initial := readIndex()
			projection, err := auditdomain.DecodeFindingChecksProjection(initial)
			if err != nil || len(projection.Checks) != 2 || projection.Checks[0].Status != "registered" || projection.Checks[1].Status != "unsupported" {
				t.Fatalf("trusted index: %+v %v", projection, err)
			}
			store, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Run(run)
			source, err := store.ReadExact(ctx, receipt.Proposal)
			if err != nil {
				t.Fatal(err)
			}
			check, plan, err := auditdomain.ExtractFindingCheck(source.Payload.Data, projection.Checks[0])
			if err != nil || check != input.Proposal.ProposedChecks[0] || plan.Method != check.Method {
				t.Fatalf("selected exact fragment: %+v %+v %v", check, plan, err)
			}
			if _, err := pool.Exec(ctx, `UPDATE finding_proposal_receipts SET checks_projection=NULL WHERE receipt_id=$1`, receipt.ReceiptID); err == nil {
				t.Fatal("published private index was mutable")
			}
			// The column constraint independently rejects absent indexes even if an
			// administrative operation temporarily disables immutability triggers.
			if _, err := pool.Exec(ctx, `ALTER TABLE finding_proposal_receipts DISABLE TRIGGER finding_proposal_receipts_protect_immutable`); err != nil {
				t.Fatal(err)
			}
			if _, err := pool.Exec(ctx, `UPDATE finding_proposal_receipts SET checks_projection=NULL WHERE receipt_id=$1`, receipt.ReceiptID); err == nil {
				t.Fatal("absent private index was accepted")
			}
			if _, err := pool.Exec(ctx, `ALTER TABLE finding_proposal_receipts ENABLE TRIGGER finding_proposal_receipts_protect_immutable`); err != nil {
				t.Fatal(err)
			}
			f := deletionImportFixture{ctx: ctx, pool: pool, intake: service, receiptID: receipt.ReceiptID,
				request: ImportRequest{OwnerID: owner, RunID: run, Proposal: receipt.Proposal.Ref}}
			createFixtureAudit(t, f, "audit")
			request := f.request
			request.AuditID = "audit"
			if _, _, err := service.ImportIntoAudit(ctx, request); err != nil {
				t.Fatal(err)
			}
			runs := runstore.NewPostgresStore(pool)
			if err := runs.MarkStageAllocationReleased(ctx, grant.AllocationID); err != nil {
				t.Fatal(err)
			}
			if _, err := runs.TransitionRun(ctx, run, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "fixture_finished"}); err != nil {
				t.Fatal(err)
			}
			if err := deleteRun(ctx, pool, owner, run); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(initial, readIndex()) {
				t.Fatal("source Run deletion changed the immutable check index")
			}
		})
	}
}
