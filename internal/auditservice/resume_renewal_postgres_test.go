package auditservice

import (
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
)

func TestAuditResumeKeepsAcceptedTasksWithoutReviewRenewal(t *testing.T) {
	ctx, pool, service, started, now := newTimeControlAudit(t, 7200)
	store := auditstoretest.NewStore(pool, artifactpgtest.NewStorage())
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "resume-fixture", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim=%+v %v", claims, err)
	}
	defer store.ReleaseClaim(ctx, claims[0])
	snapshot, err := store.GetReconcileSnapshot(ctx, claims[0])
	if err != nil {
		t.Fatal(err)
	}
	result, err := service.AcceptInitialTasks(ctx, claims[0], snapshot)
	if err != nil || !result.Changed {
		t.Fatalf("initial work=%+v %v", result, err)
	}
	query := auditstore.ListTasksParams{OwnerID: started.Audit.OwnerID, AuditID: started.Audit.AuditID, Limit: 10}
	before, err := service.Tasks(ctx, query)
	if err != nil || len(before.Tasks) != 1 {
		t.Fatalf("accepted Tasks=%+v %v", before, err)
	}
	current, err := service.Get(ctx, query.OwnerID, query.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	paused, err := service.Pause(ctx, MutationParams{OwnerID: query.OwnerID, AuditID: query.AuditID, ExpectedRevision: current.Revision, IdempotencyKey: "pause", RequestDigest: serviceTestDigest("pause")})
	if err != nil {
		t.Fatal(err)
	}
	*now = now.Add(time.Hour)
	resumed, err := service.Resume(ctx, MutationParams{OwnerID: query.OwnerID, AuditID: query.AuditID, ExpectedRevision: paused.Audit.Revision, IdempotencyKey: "resume", RequestDigest: serviceTestDigest("resume")})
	if err != nil || resumed.Audit.State != auditstore.AuditActive || resumed.Audit.Revision != paused.Audit.Revision+1 {
		t.Fatalf("resume=%+v %v", resumed, err)
	}
	after, err := service.Tasks(ctx, query)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("Resume changed accepted work", err)
	}
	var reviews int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM audit_review_requests WHERE audit_id=$1`, query.AuditID).Scan(&reviews); err != nil || reviews != 0 {
		t.Fatalf("Resume invented review requests=%d %v", reviews, err)
	}
}
