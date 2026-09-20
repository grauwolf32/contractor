package auditservice

import (
	"bytes"
	"context"
	"errors"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	managedcredentials "github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"os"
	"strings"
	"testing"
	"time"
)

type mcpAuditCredentialGuard struct {
	barrier *managedcredentials.LifecycleBarrier
}

func (g mcpAuditCredentialGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	return g.barrier.WithCredentialReferences(ctx, fn)
}
func TestAuditMCPConfigurationBaselinePinsCredentialAfterUnbind(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	profileSnapshot := loadAuditServiceProfiles(t)
	profiles := &switchableProfileCatalog{snapshot: profileSnapshot, available: true}
	gateway, err := profileSnapshot.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	credentials := &switchableCredentialLookup{available: true, gateway: gateway.Ref}
	barrier := managedcredentials.NewLifecycleBarrier()
	guard := mcpAuditCredentialGuard{barrier}
	cipher, err := managedcredentials.NewTokenCipher(bytes.Repeat([]byte{0x66}, 32))
	if err != nil {
		t.Fatal(err)
	}
	repository := managedcredentials.NewRuntimeCredentialRepository(pool)
	runtimeCredentials, err := managedcredentials.NewRuntimeCredentialService(managedcredentials.RuntimeCredentialServiceOptions{Pool: pool, Cipher: cipher, Usage: repository, Barrier: barrier})
	if err != nil {
		t.Fatal(err)
	}
	material, err := managedcredentials.NewMCPHeadersCredential(map[string]string{"Authorization": "mcp-audit-baseline-secret-canary"})
	if err != nil {
		t.Fatal(err)
	}
	defer material.Destroy()
	if _, err := runtimeCredentials.Create(ctx, managedcredentials.RuntimeCredentialCreateRequest{CredentialID: "audit-knowledge", Material: material, IdempotencyKey: "create-audit-key", ActorID: "operator"}); err != nil {
		t.Fatal(err)
	}
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{Pool: pool, RuntimeCredentials: runtimeCredentials, PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(string) bool { return true })})
	if err != nil {
		t.Fatal(err)
	}
	published, err := publisher.Publish(ctx, []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"audit-knowledge","version":"1"},"spec":{"worker":{"toolsets":{"mcp@1":{"endpoint":"https://knowledge.example/mcp","credential":"audit-knowledge"}}}}}`), "publish-audit-knowledge", "operator")
	if err != nil {
		t.Fatal(err)
	}
	bindings, err := runtimeconfig.NewBindingService(pool, runtimeCredentials)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := bindings.Create(ctx, "knowledge", published.Version.Ref, "operator", time.Now()); err != nil {
		t.Fatal(err)
	}
	service, err := New(Options{
		Pool: pool, Profiles: profiles,
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (config.CredentialLookup, error) { return credentials, nil },
		),
		CredentialGuard: guard,
		Now:             time.Now,
	})
	if err != nil {
		t.Fatal(err)
	}

	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-audit-api", OwnerID: "owner-audit-api", Kind: projectstore.KindProject,
		Name: "Audit API project", IdempotencyKey: "create-project",
		RequestDigest: serviceTestDigest("project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	automatic := writeChecklist(t, ctx, projectArtifacts, "automatic", "automatic")

	draftParams := CreateDraftParams{
		AuditID: "audit-api-one", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:       ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:        map[string]contracts.ArtifactRef{"checklist": automatic.Ref},
		RuntimeLabels: []string{"knowledge"}, Scope: Scope{Objective: "Review the service"},
		IdempotencyKey: "create-audit", RequestDigest: serviceTestDigest("create-audit"),
	}
	draft, created, err := service.CreateDraft(ctx, draftParams)
	if err != nil || !created || draft.State != auditstore.AuditDraft || draft.Revision != 1 {
		t.Fatalf("create draft = (%+v, %t, %v)", draft, created, err)
	}
	profiles.setAvailable(false)
	replayedDraft, created, err := service.CreateDraft(ctx, draftParams)
	if err != nil || created || replayedDraft.AuditID != draft.AuditID {
		t.Fatalf("create replay after catalog removal = (%+v, %t, %v)", replayedDraft, created, err)
	}

	started, err := service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-audit", RequestDigest: serviceTestDigest("start-audit"),
	})
	if err != nil || started.Replayed || started.Audit.State != auditstore.AuditActive ||
		started.Audit.Hold != auditstore.HoldHeld || started.Audit.Revision != 2 ||
		len(started.Items) != 1 || started.Round.ExpectedItemCount != 1 {
		t.Fatalf("start = (%+v, %v)", started, err)
	}
	baseline, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil || len(baseline.LLMCredentialIDs) != 1 || baseline.LLMCredentialIDs[0] != "development-worker" ||
		baseline.Inventory.Worklist.Ref.Revision == nil || len(baseline.Skills) != 0 {
		t.Fatalf("baseline = (%+v, %v)", baseline, err)
	}
	if len(baseline.RuntimeCredentialIDs) != 1 || baseline.RuntimeCredentialIDs[0] != "audit-knowledge" {
		t.Fatal("Audit baseline lost MCP credential hold")
	}
	if strings.Contains(string(started.Audit.BaselineSnapshot), "mcp-audit-baseline-secret-canary") {
		t.Fatal("Audit baseline persisted header plaintext")
	}
	if err := bindings.Delete(ctx, "knowledge", 1); err != nil {
		t.Fatal(err)
	}
	if _, err := runtimeCredentials.Delete(ctx, "audit-knowledge", "operator"); !errors.Is(err, managedcredentials.ErrRuntimeCredentialInUse) {
		t.Fatalf("Audit did not protect MCP credential after unbind: %v", err)
	}
	usage, err := repository.InspectRuntimeCredentialUsage(ctx, "audit-knowledge", 128)
	if err != nil || len(usage.AuditIDs) != 1 || usage.AuditIDs[0] != draft.AuditID || len(usage.BindingLabels) != 0 {
		t.Fatalf("MCP Audit usage: %+v, %v", usage, err)
	}
	replayed, err := service.Start(ctx, StartParams{OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision, IdempotencyKey: "start-audit", RequestDigest: serviceTestDigest("start-audit")})
	if err != nil || !replayed.Replayed || !bytes.Equal(replayed.Audit.BaselineSnapshot, started.Audit.BaselineSnapshot) {
		t.Fatalf("Audit replay lost pinned MCP baseline after unbind: %v", err)
	}
}
