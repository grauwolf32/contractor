//go:build integration

package auditcontroller

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresControllerDispatchesOrdinaryRunsThroughDerivedWindow(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 3)
	settings := settingsstore.NewPostgresStore(harness.pool)
	if _, err := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 2, ExpectedRevision: 1,
	}); err != nil {
		t.Fatal(err)
	}

	controller := harness.controller(t)
	for operation, wantWorked := range []bool{true, true, true, false} {
		worked, err := controller.RunOnce(ctx)
		if err != nil || worked != wantWorked {
			t.Fatalf("controller operation %d = (%t, %v), want worked=%t", operation, worked, err, wantWorked)
		}
	}
	executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) != 2 {
		t.Fatalf("initial executions = (%+v, %v)", executions, err)
	}
	for _, execution := range executions {
		if execution.RunID == nil || execution.State != auditstore.ExecutionSubmitted {
			t.Fatalf("initial execution is not submitted: %+v", execution)
		}
		run, getErr := harness.runs.GetRun(ctx, *execution.RunID)
		if getErr != nil || run.PublicationMode != runstore.PublicationAuditManaged ||
			run.AuditExecutionID == nil || *run.AuditExecutionID != execution.ExecutionID {
			t.Fatalf("trusted child Run = (%+v, %v)", run, getErr)
		}
	}
	if _, err := harness.runs.TransitionRun(
		ctx, *executions[0].RunID, runstore.RunRunning, runstore.RunSucceeded,
		runstore.Reason{Code: "test_succeeded"},
	); err != nil {
		t.Fatal(err)
	}
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("observe terminal child = (%t, %v)", worked, err)
	}
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("refill derived dispatch window = (%t, %v)", worked, err)
	}
	executions, err = harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) != 3 {
		t.Fatalf("refilled executions = (%+v, %v)", executions, err)
	}
	items, err := harness.audits.ListItems(ctx, harness.started.Audit.AuditID)
	if err != nil || items[0].State != auditstore.ItemCollecting || items[0].FinalDisposition != nil {
		t.Fatalf("terminal observation prematurely settled item = (%+v, %v)", items, err)
	}
}

func TestPostgresDispatchReservationOrdersWithSettingsDecrease(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	harness := newPostgresControllerHarness(t, ctx, 3)
	settings := settingsstore.NewPostgresStore(harness.pool)
	current, err := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
		MaxConcurrentRuns: 2, ExpectedRevision: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := harness.audits.Claim(ctx, auditstore.ClaimParams{
		HolderID: "postgres-race-controller", Lease: time.Minute, Limit: 1,
	})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	if _, err := harness.audits.TransitionRound(ctx, auditstore.RoundTransitionParams{
		Claim: claim, RoundID: harness.started.Round.RoundID,
		ExpectedRevision: harness.started.Round.Revision,
		ExpectedState:    auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
	}); err != nil {
		t.Fatal(err)
	}
	snapshot, err := harness.audits.GetReconcileSnapshot(ctx, claim)
	if err != nil {
		t.Fatal(err)
	}
	builder := harness.builder(t)
	first, err := builder.Prepare(ctx, snapshot, snapshot.Items[0], 1)
	if err != nil {
		t.Fatal(err)
	}
	first.Intent.Claim = claim
	if _, inserted, err := harness.audits.CreateExecutionIntent(ctx, first.Intent); err != nil || !inserted {
		t.Fatalf("first reservation = (%t, %v)", inserted, err)
	}
	second, err := builder.Prepare(ctx, snapshot, snapshot.Items[1], 1)
	if err != nil {
		t.Fatal(err)
	}
	second.Intent.Claim = claim

	start := make(chan struct{})
	type reserveResult struct {
		inserted bool
		err      error
	}
	reserved := make(chan reserveResult, 1)
	updated := make(chan error, 1)
	go func() {
		<-start
		_, inserted, reserveErr := harness.audits.CreateExecutionIntent(ctx, second.Intent)
		reserved <- reserveResult{inserted: inserted, err: reserveErr}
	}()
	go func() {
		<-start
		_, updateErr := settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
			MaxConcurrentRuns: 1, ExpectedRevision: current.Revision,
		})
		updated <- updateErr
	}()
	close(start)
	reservation := <-reserved
	if reservation.err != nil && !errors.Is(reservation.err, auditstore.ErrPrecondition) {
		t.Fatalf("racing reservation error = %v", reservation.err)
	}
	if err := <-updated; err != nil {
		t.Fatalf("lower Scheduler setting: %v", err)
	}
	stored, err := harness.audits.GetReconcileSnapshot(ctx, claim)
	if err != nil {
		t.Fatal(err)
	}
	if reservation.inserted {
		if stored.Audit.OutstandingRunCount != 2 {
			t.Fatalf("old reservation won but outstanding = %d, want 2", stored.Audit.OutstandingRunCount)
		}
	} else if stored.Audit.OutstandingRunCount != 1 {
		t.Fatalf("lower setting won but outstanding = %d, want 1", stored.Audit.OutstandingRunCount)
	}
	third, err := builder.Prepare(ctx, stored, stored.Items[2], 1)
	if err != nil {
		t.Fatal(err)
	}
	third.Intent.Claim = claim
	if _, inserted, err := harness.audits.CreateExecutionIntent(ctx, third.Intent); !errors.Is(err, auditstore.ErrPrecondition) || inserted {
		t.Fatalf("reservation above lowered window = (%t, %v)", inserted, err)
	}
}

type postgresControllerHarness struct {
	pool       *pgxpool.Pool
	snapshot   *config.Snapshot
	artifacts  *artifacts.Service
	audits     *auditstore.PostgresStore
	runs       *runstore.PostgresStore
	runService *runservice.Service
	started    auditservice.StartedAudit
}

func newPostgresControllerHarness(
	t *testing.T, ctx context.Context, itemCount int,
) *postgresControllerHarness {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	pool := isolatedControllerPool(t, ctx, databaseURL)
	snapshot := loadControllerConfig(t)
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-audit-controller", OwnerID: "owner-audit-controller",
		Kind: projectstore.KindProject, Name: "Audit Controller test",
		IdempotencyKey: "create-controller-project", RequestDigest: postgresDigest("project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifactService.Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := `{"schema":"contractor.audit.checklist.v1","items":[`
	for index := range itemCount {
		if index > 0 {
			checklist += ","
		}
		checklist += fmt.Sprintf(
			`{"key":"check-%d","version":"1","statement":"Verify check %d.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}`,
			index, index,
		)
	}
	checklist += `]}`
	input, err := projectArtifacts.Write(
		ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "checklist"},
		artifacts.Payload{MediaType: "application/json", Data: []byte(checklist)}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	guard := controllerCredentialGuard{}
	auditService, err := auditservice.New(auditservice.Options{
		Pool: pool, Profiles: snapshot, LLMCredentials: controllerCredentialLookup{},
		CredentialGuard: guard, RuntimeCredentials: controllerRuntimeCredentials{},
	})
	if err != nil {
		t.Fatal(err)
	}
	draft, _, err := auditService.CreateDraft(ctx, auditservice.CreateDraftParams{
		AuditID: "audit-controller", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:       auditservice.ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:        map[string]contracts.ArtifactRef{"checklist": input.Ref},
		RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: "Verify the test checklist"},
		IdempotencyKey: "create-controller-audit", RequestDigest: postgresDigest("audit-create"),
	})
	if err != nil {
		t.Fatal(err)
	}
	started, err := auditService.Start(ctx, auditservice.StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-controller-audit", RequestDigest: postgresDigest("audit-start"),
	})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	audits := auditstore.NewPostgresStore(pool)
	runCreation, err := runservice.New(runservice.Options{
		Runs: runs, Workflows: snapshot, LLMCredentials: controllerCredentialLookup{},
		CredentialGuard: guard, RuntimeCredentials: controllerRuntimeCredentials{}, Projects: projects,
		SkillInitializationAvailable: true,
		PublicTransaction: func(ctx context.Context, fn func(runservice.PublicRunWriter, *artifacts.Service) error) error {
			return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
				return fn(runstore.NewPostgresStore(tx), artifacts.NewService(artifacts.NewPostgresRepository(tx)))
			})
		},
		AuditTransaction: func(ctx context.Context, fn func(runservice.AuditRunWriter, *artifacts.Service, runservice.AuditExecutionWriter) error) error {
			return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				return fn(
					runstore.NewPostgresStore(tx), artifacts.NewService(artifacts.NewPostgresRepository(tx)),
					auditstore.NewPostgresStore(tx),
				)
			})
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return &postgresControllerHarness{
		pool: pool, snapshot: snapshot, artifacts: artifactService,
		audits: audits, runs: runs, runService: runCreation, started: started,
	}
}

func (h *postgresControllerHarness) builder(t *testing.T) *PinnedSubmissionBuilder {
	t.Helper()
	access, err := NewProjectArtifactAccess(h.artifacts)
	if err != nil {
		t.Fatal(err)
	}
	builder, err := NewPinnedSubmissionBuilder(access)
	if err != nil {
		t.Fatal(err)
	}
	return builder
}

func (h *postgresControllerHarness) controller(t *testing.T) *Controller {
	t.Helper()
	var ids int
	controller, err := New(
		h.audits, h.runs, h.runService, h.builder(t), &postgresNotifier{},
		Options{
			HolderID: "postgres-audit-controller", ClaimLease: 5 * time.Second,
			OperationTimeout: time.Second, ClaimBatch: 1,
			NewID: func(prefix string) (string, error) {
				ids++
				return fmt.Sprintf("%s%d", prefix, ids), nil
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return controller
}

type postgresNotifier struct{}

func (*postgresNotifier) Wake()         {}
func (*postgresNotifier) Cancel(string) {}

type controllerCredentialLookup struct{}

func (controllerCredentialLookup) LookupLLMCredential(context.Context, string) (config.CredentialMetadata, error) {
	return config.CredentialMetadata{}, fmt.Errorf("credential is unavailable")
}

type controllerCredentialGuard struct{}

func (controllerCredentialGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

type controllerRuntimeCredentials struct{}

func (controllerRuntimeCredentials) ValidateRuntimeCredential(context.Context, string, ...string) error {
	return nil
}

func loadControllerConfig(t *testing.T) *config.Snapshot {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"instructions/planner.md": "Execute the selected checklist item.",
		"instructions/worker.md":  "Read the task package and write a result package.",
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
		t.Fatalf("load Audit Controller test configuration: %v", err)
	}
	return snapshot
}

func isolatedControllerPool(
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
		t.Skipf("PostgreSQL is unavailable: %v", err)
	}
	random := make([]byte, 8)
	if _, err := cryptorand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_audit_controller_test_" + hex.EncodeToString(random)
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
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupContext, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		_, _ = admin.Exec(cleanupContext, `DROP SCHEMA `+identifier+` CASCADE`)
		admin.Close()
	})
	return pool
}

func postgresDigest(value string) string {
	result := ""
	for len(result) < 64 {
		result += hex.EncodeToString([]byte(value))
		if value == "" {
			result += "0"
		}
	}
	return "sha256:" + result[:64]
}
