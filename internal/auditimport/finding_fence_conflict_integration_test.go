//go:build integration

package auditimport

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// Project fence contention that outlasts finding intake's finite transaction
// retries (FND-R177) is a transient collection failure (AUX-R96): the
// execution stays collecting without a receipt, a consumed item attempt or a
// retained proposal, and the next collection, after the contender rolled
// back, retains the proposal and accepts the result.
func TestPostgresProjectFenceConflictKeepsExecutionCollecting(t *testing.T) {
	f := newCompletionFixtureWithOptions(t, completionPool(t), completionOptions{
		acceptance: "automatic",
	})
	receiptID := f.proposal(t, "candidate", auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate",
		Title: "Candidate", Description: "Check the access control of the handler",
		Subject:        &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
		Preconditions:  []string{},
		StandardRefs:   []auditdomain.StandardReference{},
		EvidenceIDs:    []string{},
		ProposedChecks: []auditdomain.ProposedCheck{},
		Limitations:    []string{},
	})
	f.writeResult(t, map[string][]auditdomain.ProposalSelection{
		"first": {{InvocationID: f.proposalInvocation(), ClientKey: "candidate"}},
	})
	execution, snapshot := f.terminal(t, runstore.RunSucceeded)
	access, err := NewArtifactAccess(f.artifacts, hookstore.NewClaimedArtifacts(f.pool, artifactpgtest.NewStorage()))
	mustCompletion(t, err)
	intake, err := findingintake.New(f.pool, artifactpgtest.NewStorage())
	mustCompletion(t, err)
	importer, err := New(f.audits, f.audits, f.runs, access, intake)
	mustCompletion(t, err)

	// A concurrent Project metadata update holds the Project row exclusively
	// through every retention attempt. The finding correlation fence shares
	// that row without waiting (FND-R177), so each attempt aborts.
	projects := projectstore.NewPostgresStore(f.pool)
	project, err := projects.Get(f.ctx, "owner", f.id)
	mustCompletion(t, err)
	contender, err := f.pool.Begin(f.ctx)
	mustCompletion(t, err)
	defer func() { _ = contender.Rollback(context.Background()) }()
	_, err = projectstore.NewPostgresStore(contender).Update(f.ctx, projectstore.UpdateParams{
		ProjectID: f.id, OwnerID: "owner", ExpectedRevision: project.Revision,
		Name: "Renamed during collection", Description: project.Description,
	})
	mustCompletion(t, err)
	changed, err := importer.Collect(f.ctx, f.claim, snapshot, execution)
	if changed || !persistencepostgres.IsTransactionConflict(err) || errors.Is(err, ErrPermanent) {
		t.Fatalf("collection under persistent Project contention = (%t, %v)", changed, err)
	}
	f.requireCollecting(t, execution.ExecutionID)
	var retention string
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT retention_state FROM finding_proposal_receipts WHERE receipt_id = $1`, receiptID).Scan(&retention))
	if retention != string(findingintake.RetentionSourceHeld) {
		t.Fatalf("aborted retention left the proposal %q", retention)
	}

	mustCompletion(t, contender.Rollback(f.ctx))
	changed, err = importer.Collect(f.ctx, f.claim, snapshot, execution)
	if err != nil || !changed {
		t.Fatalf("collection after the contention ended = (%t, %v)", changed, err)
	}
	f.requireAcceptedAssessment(t, execution.ExecutionID, receiptID)
}
