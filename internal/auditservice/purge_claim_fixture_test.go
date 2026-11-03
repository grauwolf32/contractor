package auditservice

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
)

func claimAuditForPurge(t *testing.T, ctx context.Context, store *auditstoretest.Store, auditID string) auditstore.ControllerClaim {
	t.Helper()
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "purge-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 || claims[0].AuditID != auditID {
		t.Fatalf("claim Audit %s = (%+v, %v)", auditID, claims, err)
	}
	return claims[0]
}
