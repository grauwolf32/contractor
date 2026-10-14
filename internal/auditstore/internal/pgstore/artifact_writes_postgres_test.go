package pgstore

import (
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestClaimedArtifactWritesFenceTakeoverAndPurge(t *testing.T) {
	f := newCollectingAuditFixture(t, "artifact-fence", 1<<20)
	writes := NewClaimedArtifacts(f.pool)
	namespace := auditdomain.ArtifactNamespace(f.audit.AuditID)
	project, _ := artifacts.ProjectScope(f.project.ProjectID)
	payload := artifacts.Payload{MediaType: "application/json", Data: []byte(`{"stable":true}`)}
	write := func(claim auditstore.ControllerClaim, name string) error {
		return writes.WithArtifactWrite(f.ctx, claim, f.project.ProjectID, namespace, func(service *artifacts.Service) error {
			_, err := service.CreateFrozen(f.ctx, project, contracts.ArtifactRef{Namespace: namespace, Name: name}, payload)
			return err
		})
	}
	for range 2 {
		if err := write(f.claim, "round-manifest"); err != nil {
			t.Fatalf("live write/replay: %v", err)
		}
	}
	stale := f.claim
	if _, err := f.pool.Exec(f.ctx, `UPDATE audit_controller_claims
SET claimed_at = clock_timestamp() - interval '2 seconds', expires_at = clock_timestamp() - interval '1 second'
WHERE audit_id = $1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	claims, err := f.store.Claim(f.ctx, auditstore.ClaimParams{HolderID: "replacement", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 || claims[0].Epoch <= stale.Epoch {
		t.Fatalf("takeover: %v, %v", claims, err)
	}
	f.claim = claims[0]
	if err := write(stale, "task-package"); !errors.Is(err, auditstore.ErrClaimLost) {
		t.Errorf("stale write after takeover: %v", err)
	}
	collectPurgeFinding(t, f, "artifact-fence-receipt")
	claim := drainDeletedAuditForPurge(t, f)
	if err := write(claim, "deleting"); !errors.Is(err, auditstore.ErrClaimLost) {
		t.Errorf("write into deleting Audit: %v", err)
	}
	if err := f.store.PurgeClaimed(f.ctx, claim, namespace); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"round-manifest", "task-package", "preparation-descriptor", "report"} {
		if err := write(stale, name); !errors.Is(err, auditstore.ErrClaimLost) {
			t.Errorf("stale %s write after purge: %v", name, err)
		}
	}
	var bindings int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_bindings
WHERE scope_kind = 'project' AND scope_id = $1 AND namespace = $2`, f.project.ProjectID, namespace).Scan(&bindings); err != nil || bindings != 0 {
		t.Fatalf("bindings after purge and stale writes: %d, %v", bindings, err)
	}
}
