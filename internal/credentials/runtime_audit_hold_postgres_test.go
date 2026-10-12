package credentials

import (
	"bytes"
	"context"
	"errors"
	"os"
	"slices"
	"testing"
	"time"
)

// recordingRuntimeAuditHolds is an Audit dispatch-hold lookup with fixed
// holds per Runtime credential that records the limits it was asked for.
type recordingRuntimeAuditHolds struct {
	held   map[string][]string
	err    error
	limits []int
}

func (h *recordingRuntimeAuditHolds) ListHeldAuditIDsByRuntimeCredential(
	_ context.Context, credentialID string, limit int,
) ([]string, error) {
	h.limits = append(h.limits, limit)
	if h.err != nil {
		return nil, h.err
	}
	return append([]string(nil), h.held[credentialID]...), nil
}

// TestRuntimeCredentialDeletionRejectsAuditDispatchHold checks AUD-R16 for a
// Runtime credential: deletion asks the Audit dispatch holds through the
// service's lookup, which auditstore implements, refuses a credential that a
// held Audit keeps and names those Audits in order, and deletes it once no
// Audit holds it. A failed lookup deletes nothing.
func TestRuntimeCredentialDeletionRejectsAuditDispatchHold(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedCredentialPool(t, ctx, databaseURL)
	cipher, err := NewTokenCipher(bytes.Repeat([]byte{0x74}, 32))
	if err != nil {
		t.Fatal(err)
	}
	holds := &recordingRuntimeAuditHolds{held: map[string][]string{"held-auth": {"audit-z", "audit-a"}}}
	service, err := NewRuntimeCredentialService(RuntimeCredentialServiceOptions{
		Pool: pool, Cipher: cipher, Usage: NewRuntimeCredentialRepository(pool), Audits: holds,
		Barrier: NewLifecycleBarrier(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewRuntimeCredentialService(RuntimeCredentialServiceOptions{
		Pool: pool, Cipher: cipher, Usage: NewRuntimeCredentialRepository(pool), Barrier: NewLifecycleBarrier(),
	}); err == nil {
		t.Fatal("a Runtime credential service without the Audit hold lookup was configured")
	}
	material, err := NewOTLPHeadersCredential(map[string]string{"Authorization": "Bearer held-secret"})
	if err != nil {
		t.Fatal(err)
	}
	defer material.Destroy()
	if _, err := service.Create(ctx, RuntimeCredentialCreateRequest{
		CredentialID: "held-auth", Material: material, IdempotencyKey: "create-held-auth", ActorID: "operator",
	}); err != nil {
		t.Fatal(err)
	}

	_, err = service.Delete(ctx, "held-auth", "operator")
	var inUse *RuntimeCredentialInUseError
	if !errors.As(err, &inUse) || !slices.Equal(inUse.Usage.AuditIDs, []string{"audit-a", "audit-z"}) {
		t.Fatalf("delete of an Audit-held Runtime credential = %v", err)
	}
	if !slices.Equal(holds.limits, []int{maximumRuntimeCredentialReferences}) {
		t.Fatalf("Audit hold lookup limits = %v", holds.limits)
	}
	if _, err := service.Get(ctx, "held-auth"); err != nil {
		t.Fatalf("Audit-held credential after the refused delete: %v", err)
	}

	delete(holds.held, "held-auth")
	holds.err = errors.New("Audit dispatch holds are unavailable")
	if _, err := service.Delete(ctx, "held-auth", "operator"); err == nil || errors.Is(err, ErrRuntimeCredentialInUse) {
		t.Fatalf("delete without the Audit hold answer = %v", err)
	}
	if _, err := service.Get(ctx, "held-auth"); err != nil {
		t.Fatalf("credential after the failed Audit hold lookup: %v", err)
	}

	holds.err = nil
	if deleted, err := service.Delete(ctx, "held-auth", "operator"); err != nil || deleted.Replayed {
		t.Fatalf("delete after the Audit hold ended = (%+v, %v)", deleted, err)
	}
	if _, err := service.Get(ctx, "held-auth"); !errors.Is(err, ErrRuntimeCredentialNotFound) {
		t.Fatalf("deleted credential lookup = %v", err)
	}
}
