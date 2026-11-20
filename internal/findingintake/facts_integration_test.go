//go:build integration

package findingintake

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func fixtureEvidence(t *testing.T, ctx context.Context, pool *pgxpool.Pool, grant controlplane.AllocationGrant) []contracts.ArtifactRef {
	t.Helper()
	run, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(grant.RunID)
	if err != nil {
		t.Fatal(err)
	}
	saved, err := run.Write(ctx, contracts.ArtifactRef{Namespace: grant.Namespace, Name: "selected-proof"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("Observed Server: nginx. Reported source, not independently verified.")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return []contracts.ArtifactRef{saved.Ref}
}

func TestProjectFactsOptionalTaskOriginAndProjectRequired(t *testing.T) {
	for _, members := range []int{1, 2} {
		t.Run(fmt.Sprint(members), func(t *testing.T) {
			ctx := t.Context()
			pool := isolatedFindingPool(t, ctx)
			const owner, project = "facts-owner", "facts-project"
			if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Facts", IdempotencyKey: "facts", RequestDigest: auditdomain.DigestBytes([]byte("facts"))}); err != nil {
				t.Fatal(err)
			}
			grant, origin := auditSubmissionGrant(t, ctx, pool, owner, project, "facts-run", members)
			service, err := New(pool, artifactpgtest.NewStorage())
			if err != nil {
				t.Fatal(err)
			}
			evidence := fixtureEvidence(t, ctx, pool, grant)
			input := testSubmission("facts-invocation", "general", evidence)
			input.Incidental = true
			input.Proposal.SeveritySuggestion = "informational"
			general, replayed, err := service.Submit(ctx, grant, input)
			if err != nil || replayed || general.FindingID == nil || general.Origin.Audit == nil || general.Origin.Audit.AuditID != origin.AuditID || general.Origin.Audit.TaskID != "" || general.Origin.Audit.Hypothesis != nil {
				t.Fatalf("general: %+v %v", general, err)
			}
			read, err := service.GetProjectReceipt(ctx, owner, project, general.ReceiptID)
			if err != nil || read.Origin.Audit.TaskID != "" {
				t.Fatal(read, err)
			}
			finding, err := querystore.New(pool, artifactpgtest.NewStorage()).GetProjectFinding(ctx, owner, project, *general.FindingID)
			if err != nil || finding.AnalystSeverity != nil || finding.AnalystVerdict != nil {
				t.Fatal(finding, err)
			}
			requests := []ImportRequest{{OwnerID: owner, AuditID: origin.AuditID, RunID: grant.RunID, Proposal: general.Proposal.Ref}}
			for index, key := range []string{"item-00", "item-63"} {
				selected := testSubmission("facts-invocation", fmt.Sprint("selected-", index), evidence)
				selected.Incidental = true
				selected.ItemKey = &key
				selected.Proposal.SeveritySuggestion = "informational"
				receipt, _, err := service.Submit(ctx, grant, selected)
				if index == 1 {
					if !errors.Is(err, ErrAccessDenied) {
						t.Fatalf("foreign member: %v", err)
					}
					continue
				}
				requests = append(requests, ImportRequest{OwnerID: owner, AuditID: origin.AuditID, RunID: grant.RunID, Proposal: receipt.Proposal.Ref})
				if err != nil || receipt.Origin.Audit.TaskID != origin.TaskID || receipt.Origin.Audit.Hypothesis != nil {
					t.Fatal(receipt, err)
				}
			}

			for index, severity := range []string{"low", ""} {
				other := testSubmission("facts-invocation", fmt.Sprint("issue-", index), evidence)
				other.Incidental = true
				other.Proposal.SeveritySuggestion = severity
				receipt, _, err := service.Submit(ctx, grant, other)
				if err != nil {
					t.Fatal(err)
				}
				requests = append(requests, ImportRequest{OwnerID: owner, AuditID: origin.AuditID, RunID: grant.RunID, Proposal: receipt.Proposal.Ref})
			}
			if err := service.RetainAuditCollectionBatch(ctx, requests); err != nil {
				t.Fatal(err)
			}

			reader := querystore.New(pool, artifactpgtest.NewStorage())
			workspace, err := reader.GetWorkspace(ctx, owner, origin.AuditID)
			if err != nil || workspace.NeedsReviewFindings != 2 {
				t.Fatalf("workspace count=%+v %v", workspace, err)
			}
			rows, err := reader.ListFindings(ctx, auditstore.FindingListParams{OwnerID: owner, AuditID: origin.AuditID, NeedsReview: true, Limit: 64})
			if err != nil || len(rows) != workspace.NeedsReviewFindings {
				t.Fatalf("needs-review list/count mismatch: %+v %v", rows, err)
			}
			informational := auditstore.SeverityInformational
			facts, err := reader.ListFindings(ctx, auditstore.FindingListParams{OwnerID: owner, AuditID: origin.AuditID, Severity: &informational, Limit: 64})
			if err != nil || len(facts) == 0 {
				t.Fatalf("informational severity view=%+v %v", facts, err)
			}
			for _, row := range facts {
				if row.AnalystSeverity != nil || row.AnalystVerdict != nil || row.NeedsReview() {
					t.Fatalf("informational assessment inferred: %+v", row)
				}
			}
			refs, err := service.AuditExecutionReceiptReferences(ctx, owner, origin.AuditID, origin.ExecutionID, grant.RunID)
			if err != nil || len(refs) != 1 || len(refs[origin.TaskID].IDs) != 1 {
				t.Fatalf("Run-level receipt assigned to a Task: %+v %v", refs, err)
			}
			// A committed replay remains exactly the same result.
			replay, ok, err := service.Submit(ctx, grant, input)
			if err != nil || !ok || replay.ReceiptID != general.ReceiptID {
				t.Fatal(replay, ok, err)
			}
			// The database itself also forbids a hypothesis without Task provenance.
			if _, err := pool.Exec(ctx, `UPDATE finding_proposal_receipts SET audit_hypothesis='{"id":"h","revision":1}' WHERE receipt_id=$1`, general.ReceiptID); err == nil {
				t.Fatal("hypothesis without Task accepted")
			}
		})
	}
	t.Run("project-required-before-payload", func(t *testing.T) {
		ctx := t.Context()
		pool := isolatedFindingPool(t, ctx)
		const owner, project = "ordinary-owner", "ordinary-project"
		if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Ordinary", IdempotencyKey: "ordinary", RequestDigest: auditdomain.DigestBytes([]byte("ordinary"))}); err != nil {
			t.Fatal(err)
		}
		grant := quotaSubmissionGrant(t, ctx, pool, owner, "", "ordinary-run")
		evidence := fixtureEvidence(t, ctx, pool, grant)
		// A projectless ordinary Run is legal; direct Finding intake is not.
		service, err := New(pool, artifactpgtest.NewStorage())
		if err != nil {
			t.Fatal(err)
		}
		prepared := false
		service.afterPrepare = func(context.Context) { prepared = true }
		for _, severity := range []string{"", "informational"} {
			input := testSubmission("ordinary-invocation", "required", evidence)
			input.Incidental = true
			input.Proposal.SeveritySuggestion = severity
			if _, _, err := service.Submit(ctx, grant, input); !errors.Is(err, ErrProjectRequired) {
				t.Fatalf("projectless %q: %v", severity, err)
			}
		}
		var count int
		if err := pool.QueryRow(ctx, `SELECT count(*) FROM finding_proposal_receipts`).Scan(&count); err != nil || count != 0 || prepared {
			t.Fatalf("projectless call staged payload or receipt: %t %d %v", prepared, count, err)
		}
	})
}
