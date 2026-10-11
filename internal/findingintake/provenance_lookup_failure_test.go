//go:build integration

package findingintake

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// A storage failure while reading the submitting Run is not a denied grant:
// the submission keeps the database cause, commits nothing and succeeds once
// the store answers again.
func TestPostgresFindingSubmissionKeepsProvenanceLookupFailure(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	const ownerID, projectID, runID = "lookup-owner", "lookup-project", "lookup-run"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject,
		Name: "Lookup project", IdempotencyKey: "lookup-project",
		RequestDigest: auditdomain.DigestBytes([]byte("lookup-project")),
	}); err != nil {
		t.Fatal(err)
	}
	grant := quotaSubmissionGrant(t, ctx, pool, ownerID, projectID, runID)
	service, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	input := testSubmission("lookup-invocation", "lookup-candidate", []contracts.ArtifactRef{})

	if _, err := pool.Exec(ctx, `ALTER TABLE workflow_runs RENAME TO lookup_unavailable_runs`); err != nil {
		t.Fatal(err)
	}
	_, _, err = service.Submit(ctx, grant, input)
	if errors.Is(err, ErrAccessDenied) || errors.Is(err, ErrToolNotSelected) ||
		persistencepostgres.SQLState(err) != persistencepostgres.SQLStateUndefinedTable {
		t.Fatalf("Run lookup failure = %v, want the undefined-table cause", err)
	}
	if _, err := pool.Exec(ctx, `ALTER TABLE lookup_unavailable_runs RENAME TO workflow_runs`); err != nil {
		t.Fatal(err)
	}

	receipt, replayed, err := service.Submit(ctx, grant, input)
	if err != nil || replayed || receipt.Origin.RunID != runID {
		t.Fatalf("submission after recovery = (%+v, %t, %v)", receipt, replayed, err)
	}
}
