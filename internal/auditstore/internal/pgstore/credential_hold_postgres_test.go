package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// TestPostgresCredentialHoldsListHeldAuditsByCredentialKind checks the Audit
// dispatch holds that credential deletion asks for (AUD-R16): a started Audit
// holds the LLM and the Runtime credentials its baseline names, each only
// under its own kind.
func TestPostgresCredentialHoldsListHeldAuditsByCredentialKind(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	common := setupCommonTaskFixture(t, pool, "credential-hold", false)
	store := newTestStore(pool)
	audit := common.audit
	project := struct{ OwnerID, ProjectID string }{audit.OwnerID, audit.ProjectID}
	for _, kind := range []struct {
		name string
		list func(context.Context, string, int) ([]string, error)
	}{
		{"LLM", store.ListHeldAuditIDsByLLMCredential},
		{"Runtime", store.ListHeldAuditIDsByRuntimeCredential},
	} {
		if held, err := kind.list(ctx, "llm-held", 10); err != nil || len(held) != 0 {
			t.Fatalf("%s credential holds of a draft Audit = (%v, %v)", kind.name, held, err)
		}
	}
	started, _, err := store.Activate(ctx, auditstore.ActivateParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		BaselineSnapshot: json.RawMessage(`{"schema":"contractor.audit.baseline.v1","inputs":` + string(commonTaskBaselineInputs(auditcontract.Material{Ref: common.source, Digest: common.material.Digest, MediaType: common.material.MediaType, SizeBytes: common.material.SizeBytes})) + `,"skills":[],` +
			`"llmCredentialIds":["llm-held"],"runtimeCredentialIds":["runtime-held"]}`),
		DeadlineAt:     time.Now().Add(time.Hour),
		IdempotencyKey: "start-credential-hold", RequestDigest: testDigest("4"),
	})
	if err != nil || started.Hold != auditstore.HoldHeld {
		t.Fatalf("start Audit = (%+v, %v)", started, err)
	}
	for _, check := range []struct {
		name string
		list func(context.Context, string, int) ([]string, error)
		id   string
		want []string
	}{
		{"LLM credential", store.ListHeldAuditIDsByLLMCredential, "llm-held", []string{audit.AuditID}},
		{"Runtime credential as an LLM credential", store.ListHeldAuditIDsByLLMCredential, "runtime-held", nil},
		{"Runtime credential", store.ListHeldAuditIDsByRuntimeCredential, "runtime-held", []string{audit.AuditID}},
		{"LLM credential as a Runtime credential", store.ListHeldAuditIDsByRuntimeCredential, "llm-held", nil},
	} {
		held, err := check.list(ctx, check.id, 10)
		if err != nil || !slices.Equal(held, check.want) {
			t.Errorf("%s holds = (%v, %v), want %v", check.name, held, err, check.want)
		}
	}
	if _, err := store.ListHeldAuditIDsByRuntimeCredential(ctx, "runtime-held", 129); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("Runtime credential hold query above the limit = %v", err)
	}
}
