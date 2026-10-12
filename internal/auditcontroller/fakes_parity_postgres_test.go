//go:build integration

package auditcontroller

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// TestPostgresStoreRefusesTheSameClaimedTransitions drives the PostgreSQL
// store with the table TestFakeControllerStoreRefusesInvalidClaimedTransitions
// applies to the fake: both refuse every row the same way and leave the Audit
// unchanged.
func TestPostgresStoreRefusesTheSameClaimedTransitions(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 1)
	claims, err := harness.audits.Claim(ctx, auditstore.ClaimParams{HolderID: "parity", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim = (%+v, %v)", claims, err)
	}
	before, err := harness.audits.GetReconcileSnapshot(ctx, claims[0])
	if err != nil {
		t.Fatal(err)
	}
	if before.Audit.State != auditstore.AuditActive {
		t.Fatalf("started Audit is %s, want active", before.Audit.State)
	}
	checkClaimedTransitionRefusals(t, "PostgreSQL",
		refuseClaimedTransitions(ctx, harness.audits, claims[0], before.Audit.Revision))
	after, err := harness.audits.GetReconcileSnapshot(ctx, claims[0])
	if err != nil {
		t.Fatal(err)
	}
	if after.Audit.State != before.Audit.State || after.Audit.Revision != before.Audit.Revision {
		t.Fatalf("refused transitions changed the Audit to %s at revision %d", after.Audit.State, after.Audit.Revision)
	}
}
