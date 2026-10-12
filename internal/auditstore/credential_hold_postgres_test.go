package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
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
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-credential-hold", OwnerID: "owner-credential-hold", Kind: projectstore.KindProject,
		Name: "Credential hold", IdempotencyKey: "project-credential-hold", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	store := NewPostgresStore(pool)
	audit, _, err := store.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-credential-hold", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         ProfileIdentity{Name: "profile", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"profile"}`),
		InputSelection:  json.RawMessage(`{"inputs":{}}`),
		Limits: Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
		},
		IdempotencyKey: "audit-credential-hold", RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
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
	started, _, err := store.MaterializeRound(ctx, MaterializeRoundParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: "round-credential-hold", RoundOrdinal: 1, Manifest: testExact("audits", "round-credential-hold", "r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":{},"skills":[],` +
			`"llmCredentialIds":["llm-held"],"runtimeCredentialIds":["runtime-held"]}`),
		DeadlineAt: time.Now().Add(time.Hour), Items: []MaterializedItem{},
		IdempotencyKey: "start-credential-hold", RequestDigest: testDigest("4"),
	})
	if err != nil || started.Hold != HoldHeld {
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
	if _, err := store.ListHeldAuditIDsByRuntimeCredential(ctx, "runtime-held", 129); !errors.Is(err, ErrInvalid) {
		t.Fatalf("Runtime credential hold query above the limit = %v", err)
	}
}
