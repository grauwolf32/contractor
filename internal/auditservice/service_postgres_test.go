package auditservice

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	managedcredentials "github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestAuditDraftStartReplayAndAtomicUnsupportedRollback(t *testing.T) {
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
	guard := &countingCredentialGuard{}
	service, err := New(Options{
		Pool: pool, Profiles: profiles,
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (config.CredentialLookup, error) { return credentials, nil },
		),
		CredentialGuard: guard,
		Now:             func() time.Time { return time.Date(2026, 9, 5, 18, 0, 0, 0, time.UTC) },
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
	manual := writeChecklist(t, ctx, projectArtifacts, "manual", "manual")

	draftParams := CreateDraftParams{
		AuditID: "audit-api-one", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:       ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:        map[string]contracts.ArtifactRef{"checklist": automatic.Ref},
		RuntimeLabels: []string{}, Scope: Scope{Objective: "Review the service"},
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
	holds, err := auditstore.NewPostgresStore(pool).ListHeldAuditIDsByLLMCredential(ctx, "development-worker", 10)
	if err != nil || len(holds) != 1 || holds[0] != draft.AuditID {
		t.Fatalf("credential holds = (%v, %v)", holds, err)
	}
	pauseParams := MutationParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: started.Audit.Revision,
		IdempotencyKey: "pause-audit", RequestDigest: serviceTestDigest("pause-audit"),
	}
	paused, err := service.Pause(ctx, pauseParams)
	if err != nil || paused.Replayed || paused.Audit.State != auditstore.AuditPaused || paused.Audit.Revision != 3 {
		t.Fatalf("pause = (%+v, %v)", paused, err)
	}
	replayedPause, err := service.Pause(ctx, pauseParams)
	if err != nil || !replayedPause.Replayed || replayedPause.Audit.State != auditstore.AuditPaused {
		t.Fatalf("pause replay = (%+v, %v)", replayedPause, err)
	}
	resumed, err := service.Resume(ctx, MutationParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: paused.Audit.Revision,
		IdempotencyKey: "resume-audit", RequestDigest: serviceTestDigest("resume-audit"),
	})
	if err != nil || resumed.Audit.State != auditstore.AuditActive || resumed.Audit.Revision != 4 {
		t.Fatalf("resume = (%+v, %v)", resumed, err)
	}
	cancelled, err := service.Cancel(ctx, MutationParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: resumed.Audit.Revision,
		IdempotencyKey: "cancel-audit", RequestDigest: serviceTestDigest("cancel-audit"),
	})
	if err != nil || cancelled.Audit.State != auditstore.AuditCancelling || cancelled.Audit.Dispatch != auditstore.DispatchClosed {
		t.Fatalf("cancel = (%+v, %v)", cancelled, err)
	}
	deleting, err := service.Delete(ctx, MutationParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: cancelled.Audit.Revision,
		IdempotencyKey: "delete-audit", RequestDigest: serviceTestDigest("delete-audit"),
	})
	if err != nil || deleting.Audit.State != auditstore.AuditCancelling || deleting.Audit.DeletionRequestedAt == nil {
		t.Fatalf("delete intent = (%+v, %v)", deleting, err)
	}
	coverage, err := service.ListCoverage(ctx, project.OwnerID, draft.AuditID, started.Round.RoundID, -1, 10)
	if err != nil || len(coverage) != 1 || coverage[0].Ordinal != 0 || coverage[0].Coverage.Status != auditstore.CoverageNotTested {
		t.Fatalf("initial coverage = (%+v, %v)", coverage, err)
	}
	credentials.setAvailable(false)
	guardCalls := guard.callsCount()
	replayedStart, err := service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-audit", RequestDigest: serviceTestDigest("start-audit"),
	})
	if err != nil || !replayedStart.Replayed || replayedStart.Audit.AuditID != draft.AuditID || guard.callsCount() != guardCalls {
		t.Fatalf("start replay after credential removal = (%+v, %v), guard=%d", replayedStart, err, guard.callsCount())
	}
	if _, err := service.Get(ctx, "other-owner", draft.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner get = %v", err)
	}
	foreignProjectID := project.ProjectID
	if listed, err := service.List(ctx, auditstore.ListParams{
		OwnerID: "other-owner", ProjectID: &foreignProjectID, Limit: 10,
	}); err != nil || len(listed) != 0 {
		t.Fatalf("foreign owner list = (%+v, %v)", listed, err)
	}
	if _, err := service.ListItems(ctx, auditstore.ListItemsParams{
		OwnerID: "other-owner", AuditID: draft.AuditID, Limit: 10,
	}); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner item list = %v", err)
	}
	if _, err := service.ListCoverage(
		ctx, "other-owner", draft.AuditID, started.Round.RoundID, -1, 10,
	); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner coverage = %v", err)
	}
	if _, err := service.Start(ctx, StartParams{
		OwnerID: "other-owner", AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "foreign-start", RequestDigest: serviceTestDigest("foreign-start"),
	}); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner start = %v", err)
	}

	profiles.setAvailable(true)
	credentials.setAvailable(true)
	manualDraft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-api-manual", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:       ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:        map[string]contracts.ArtifactRef{"checklist": manual.Ref},
		RuntimeLabels: []string{}, Scope: Scope{},
		IdempotencyKey: "create-manual", RequestDigest: serviceTestDigest("create-manual"),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: manualDraft.AuditID, ExpectedRevision: manualDraft.Revision,
		IdempotencyKey: "start-manual", RequestDigest: serviceTestDigest("start-manual"),
	})
	var unsupportedError *UnsupportedError
	if !errors.As(err, &unsupportedError) || len(unsupportedError.Reasons) != 1 || unsupportedError.Reasons[0] != ReasonManualItemUnsupported {
		t.Fatalf("manual start error = %#v", err)
	}
	storedManual, err := service.Get(ctx, project.OwnerID, manualDraft.AuditID)
	if err != nil || storedManual.State != auditstore.AuditDraft || storedManual.BaselineSnapshot != nil || storedManual.Hold != auditstore.HoldPending {
		t.Fatalf("manual rollback = (%+v, %v)", storedManual, err)
	}
	var roundCount, artifactCount int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM audit_rounds WHERE audit_id = $1`, manualDraft.AuditID).Scan(&roundCount); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `
SELECT count(*)
  FROM artifact_bindings
 WHERE scope_kind = 'project' AND scope_id = $1 AND namespace LIKE 'audit-%'`, project.ProjectID).Scan(&artifactCount); err != nil {
		t.Fatal(err)
	}
	if roundCount != 0 || artifactCount != 2 {
		// The successful Audit created exactly a task package and a worklist;
		// the rejected manual Audit must not add either one.
		t.Fatalf("unsupported rollback rows = rounds %d, Audit bindings %d", roundCount, artifactCount)
	}
}

func TestAuditStartUsesOwningTransactionWithSaturatedPool(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	profiles := loadAuditServiceProfiles(t)
	gateway, err := profiles.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	development, err := managedcredentials.NewStaticProvider([]managedcredentials.StaticEntry{{
		Metadata: config.CredentialMetadata{
			Ref:        contracts.LLMCredentialRef{CredentialID: "development-worker"},
			LLMGateway: gateway.Ref, Unrestricted: true,
		},
		Token: contracts.NewSecretString("development-test-token"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	transactionCredentials, err := managedcredentials.NewTransactionLookupFactory(development)
	if err != nil {
		t.Fatal(err)
	}

	limitedConfig := pool.Config()
	limitedConfig.MaxConns = 2
	limited, err := pgxpool.NewWithConfig(ctx, limitedConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer limited.Close()
	service, err := New(Options{
		Pool: limited, Profiles: profiles,
		TransactionLLMCredentials: transactionCredentials,
		CredentialGuard:           &countingCredentialGuard{},
	})
	if err != nil {
		t.Fatal(err)
	}

	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-saturated-audit", OwnerID: "owner-saturated-audit",
		Kind: projectstore.KindProject, Name: "Saturated Audit",
		IdempotencyKey: "project-saturated-audit", RequestDigest: serviceTestDigest("saturated-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := writeChecklist(t, ctx, projectArtifacts, "saturated", "automatic")
	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-saturated", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:        ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:         map[string]contracts.ArtifactRef{"checklist": checklist.Ref},
		Scope:          Scope{Objective: "Exercise transaction-bound credential validation"},
		IdempotencyKey: "create-saturated-audit", RequestDigest: serviceTestDigest("saturated-create"),
	})
	if err != nil {
		t.Fatal(err)
	}

	listener, err := limited.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Release()
	if _, err := listener.Exec(ctx, `LISTEN contractor_audit_transaction_test`); err != nil {
		t.Fatal(err)
	}
	startCtx, startCancel := context.WithTimeout(ctx, 5*time.Second)
	defer startCancel()
	started, err := service.Start(startCtx, StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-saturated-audit", RequestDigest: serviceTestDigest("saturated-start"),
	})
	if err != nil {
		t.Fatalf("start Audit with one LISTEN connection and one transaction: %v", err)
	}
	if started.Audit.State != auditstore.AuditActive || len(started.Items) != 1 {
		t.Fatalf("started saturated Audit = %+v", started)
	}
}

type switchableProfileCatalog struct {
	mu        sync.Mutex
	snapshot  *config.Snapshot
	available bool
}

func (c *switchableProfileCatalog) AuditProfile(selector string) (config.ResolvedAuditProfile, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.available {
		return config.ResolvedAuditProfile{}, config.ErrConfigurationNotFound
	}
	return c.snapshot.AuditProfile(selector)
}

func (c *switchableProfileCatalog) AuditProfiles() []config.ResolvedAuditProfile {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.available {
		return nil
	}
	return c.snapshot.AuditProfiles()
}

func (c *switchableProfileCatalog) setAvailable(value bool) {
	c.mu.Lock()
	c.available = value
	c.mu.Unlock()
}

type switchableCredentialLookup struct {
	mu        sync.Mutex
	available bool
	gateway   contracts.LLMGatewayConfigRef
}

func (l *switchableCredentialLookup) LookupLLMCredential(
	_ context.Context, id string,
) (config.CredentialMetadata, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.available || id != "development-worker" {
		return config.CredentialMetadata{}, errors.New("credential unavailable")
	}
	return config.CredentialMetadata{
		Ref:          contracts.LLMCredentialRef{CredentialID: id},
		LLMGateway:   l.gateway,
		Unrestricted: true,
	}, nil
}

func (l *switchableCredentialLookup) setAvailable(value bool) {
	l.mu.Lock()
	l.available = value
	l.mu.Unlock()
}

type countingCredentialGuard struct {
	mu    sync.Mutex
	calls int
}

func (g *countingCredentialGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	g.mu.Lock()
	g.calls++
	g.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

func (g *countingCredentialGuard) callsCount() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.calls
}

func writeChecklist(
	t *testing.T, ctx context.Context, store artifacts.ScopedStore, name, reviewPolicy string,
) artifacts.WriteResult {
	t.Helper()
	payload := []byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check-` + name + `","version":"1","statement":"Verify ` + name + `.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"` + reviewPolicy + `"}]}`)
	result, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: name}, artifacts.Payload{
		MediaType: "application/json", Data: payload,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func loadAuditServiceProfiles(t *testing.T) *config.Snapshot {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"instructions/planner.md": "Execute the selected checklist item.",
		"instructions/worker.md":  "Read the task package and write a result package.",
		"llm-gateways/test.yaml": `apiVersion: contractor/v1alpha1
kind: LLMGatewayConfig
metadata: {name: test-gateway, version: "1"}
spec:
  protocol: openai-compatible@1
  url: http://127.0.0.1:4000/v1
`,
		"model-policies/worker.yaml": `apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata: {name: worker, version: "1"}
spec:
  model: worker-model
  maxOutputTokens: 1024
  maxModelCalls: 2
  maxToolCalls: 4
  maxTotalTokens: 4096
  temperature: 0
`,
		"agent-templates/worker.yaml": `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: audit-worker, version: "1"}
spec:
  description: Produces one deterministic test result
  runtime: adk@1
  instructions: {ref: instructions/worker.md}
  modelPolicy: worker@1
  toolsets:
    - ref: run-artifacts@1
      tools: [read_artifact, write_artifact]
  sandboxProfile: local-workdir@1
`,
		"workflows/check.yaml": `apiVersion: contractor/v1alpha1
kind: Workflow
metadata: {name: audit-check, version: "1"}
spec:
  parameters: {}
  inputs:
    task: {required: true, mediaTypes: [application/zip]}
  outputs:
    result: {required: true, mediaTypes: [application/zip]}
  executionConfig:
    workers:
      llmGateway: test-gateway@1
      credential: development-worker
  entryStage: check
  stages:
    check:
      objective: Evaluate one checklist item
      instructions: {ref: instructions/planner.md}
      planner: passthrough@1
      agents:
        worker: {template: audit-worker@1}
      context:
        artifacts:
          task: {namespace: inputs, name: task, required: true}
      result:
        artifacts:
          result: {required: true, mediaTypes: [application/zip], from: {namespace: worker, name: result}}
      workflowOutputs: {result: result}
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`,
		"audit-profiles/checklist.yaml": `apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata: {name: test-checklist, version: "1"}
spec:
  mode: custom-checklist
  standards: []
  inputs:
    checklist: {required: true, mediaTypes: [application/json]}
  inventory:
    implementation: checklist@1
    sourceInput: checklist
    itemWorkflowRole: check
  workflows:
    check:
      ref: audit-check@1
      inputs:
        task: {source: item-package}
      parameters: {}
      outputs: {result: result}
  execution:
    roundMode: fixed-barrier
    maxRounds: 1
    batchSize: 1
    maxItemsPerRound: 10
    maxItemsTotal: 10
    maxSubmittedRuns: 20
    maxItemRunAttempts: 2
    deadlineSeconds: 3600
    maxEvidenceBytes: 1048576
    incompleteRound: assess-with-gaps
  interaction:
    activeChecks: prohibited
    findingConfirmation: disabled
    notApplicable: profile-rule
    reportAcceptance: automatic
`,
	}
	for _, directory := range []string{
		"instructions", "llm-gateways", "model-policies", "execution-configs",
		"agent-templates", "workflows", "audit-profiles", "skills",
	} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for name, contents := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(contents), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatalf("load Audit service config: %v", err)
	}
	return snapshot
}

func isolatedAuditServicePool(
	t *testing.T, ctx context.Context, databaseURL string,
) *pgxpool.Pool {
	t.Helper()
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_audit_service_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	configuration, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	configuration.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	if _, err := postgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop test schema: %v", err)
		}
		admin.Close()
	})
	return pool
}

func serviceTestDigest(value string) string {
	return "sha256:" + hex.EncodeToString([]byte(strings.Repeat(value, 64))[:32])
}
