package config

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestDeploymentModelSelectionAuditDraftStartsWithCapturedProfile(t *testing.T) {
	pool := modelAuditPool(t)
	root, _ := modelSelectionFixture(t)
	modelAuditProfileFixture(t, root)
	manager := configuredModelManager(t, root, &memoryModelSettings{})
	credentialLookup := developmentCredentialLookup(t, manager.Snapshot())
	service, err := auditservice.New(auditservice.Options{
		Pool: pool, Profiles: manager,
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (resolved.CredentialLookup, error) { return credentialLookup, nil },
		),
		CredentialGuard: modelAuditGuard{},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-model-audit", OwnerID: "owner-model-audit", Kind: projectstore.KindProject,
		Name: "Captured model", IdempotencyKey: "project-model-audit", RequestDigest: contentdigest.Bytes([]byte("project")),
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	inputs := map[string]contracts.ArtifactRef{}
	for name, payload := range map[string]artifacts.Payload{
		"source":    {MediaType: "text/plain", Data: []byte("fixture source")},
		"checklist": {MediaType: "application/json", Data: []byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check-auth","version":"1","statement":"Verify authentication.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`)},
	} {
		written, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: name}, payload, nil)
		if err != nil {
			t.Fatal(err)
		}
		inputs[name] = written.Ref
	}
	draft, created, err := service.CreateDraft(ctx, auditservice.CreateDraftParams{
		AuditID: "audit-model-one", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile: auditservice.ProfileSelector{Name: "model-checklist", Version: "1"},
		Inputs:  inputs, RuntimeLabels: []string{}, IdempotencyKey: "draft-model-one",
		RequestDigest: contentdigest.Bytes([]byte("draft-model-one")),
	})
	if err != nil || !created {
		t.Fatalf("create draft: created=%t err=%v", created, err)
	}
	// Read through PostgreSQL JSONB before start, just like the reported check.
	stored, err := service.Get(ctx, project.OwnerID, draft.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	info, err := manager.GetModelSelection(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.ReplaceModelSelection(ctx, settingsstore.ModelSelectionValue{
		DefaultModel: "fixture-b", Roles: map[string]string{},
	}, 1, info.CatalogDigest); err != nil {
		t.Fatal(err)
	}
	params := auditservice.StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-model-one", RequestDigest: contentdigest.Bytes([]byte("start-model-one")),
	}
	started, err := service.Start(ctx, params)
	if err != nil || started.Audit.State != auditstore.AuditActive || len(started.Items) != 1 {
		t.Fatalf("start retained draft: state=%s items=%d err=%v", started.Audit.State, len(started.Items), err)
	}
	profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(started.Audit.ProfileSnapshot)
	if err != nil || profile.Ref.Digest != stored.Profile.Digest ||
		profile.Workflows["check"].Workflow.Stages["copy"].ModelDefaults["worker"].ContextWindowTokens != 131072 {
		t.Fatalf("start lost captured model or digest: %v", err)
	}
	if !reflect.DeepEqual(started.Audit.ProfileSnapshot, stored.ProfileSnapshot) {
		t.Fatal("start rewrote the retained profile")
	}
	replayed, err := service.Start(ctx, params)
	if err != nil || !replayed.Replayed || replayed.Audit.Revision != started.Audit.Revision {
		t.Fatalf("start replay: replayed=%t err=%v", replayed.Replayed, err)
	}
	// Eval dispatch creates a new draft from the retained effective profile.
	// It must not substitute the now-selected fixture-b model.
	expected, err := contentdigest.JSON(profile)
	if err != nil {
		t.Fatal(err)
	}
	evalDraft, created, err := service.CreateDraft(ctx, auditservice.CreateDraftParams{
		AuditID: "audit-model-eval", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:       auditservice.ProfileSelector{Name: "model-checklist", Version: "1"},
		PinnedProfile: &profile, ExpectedProfileSHA256: expected,
		Inputs: inputs, RuntimeLabels: []string{}, IdempotencyKey: "draft-model-eval",
		RequestDigest: contentdigest.Bytes([]byte("draft-model-eval")),
	})
	if err != nil || !created || evalDraft.Profile.Digest != draft.Profile.Digest {
		t.Fatalf("reuse prepared Eval profile: created=%t err=%v", created, err)
	}
	if _, err := service.Start(ctx, auditservice.StartParams{
		OwnerID: project.OwnerID, AuditID: evalDraft.AuditID, ExpectedRevision: evalDraft.Revision,
		IdempotencyKey: "start-model-eval", RequestDigest: contentdigest.Bytes([]byte("start-model-eval")),
	}); err != nil {
		t.Fatalf("start reused Eval profile: %v", err)
	}
}

type modelAuditGuard struct{}

func (modelAuditGuard) WithRunCreation(_ context.Context, fn func() error) error { return fn() }

func modelAuditPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(admin.Close)
	suffix := make([]byte, 8)
	if _, err := rand.Read(suffix); err != nil {
		t.Fatal(err)
	}
	schema := "model_audit_" + hex.EncodeToString(suffix)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, "CREATE SCHEMA "+identifier); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, "DROP SCHEMA "+identifier+" CASCADE"); err != nil {
			t.Errorf("drop test schema: %v", err)
		}
	})
	options, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	options.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}
