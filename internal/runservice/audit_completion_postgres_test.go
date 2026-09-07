package runservice

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestAuditCompletionPostgresAtomicCreationRestartAndAllocation(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunServicePool(t, ctx, databaseURL)
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: "project", OwnerID: "owner", Kind: projectstore.KindProject, Name: "Audit completion", IdempotencyKey: "project", RequestDigest: testDigest("1")})
	if err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	store, err := artifactService.Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	source := writeExact(t, ctx, store, "sources", "source", "application/zip", []byte("source"))
	inventory, err := auditdomain.BuildChecklistInventory([]byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check-1","version":"1","statement":"Check source","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`), "application/json", auditdomain.InventoryOptions{Round: 1, WorkflowRole: "check", SourceInputName: "source", SourceRef: source.Ref, ApprovalRequirement: auditdomain.ApprovalNone})
	if err != nil {
		t.Fatal(err)
	}
	task := writeExact(t, ctx, store, "audit-test", "task", "application/zip", inventory.Tasks[0].Package)
	manifestDoc := inventory.ExecutionManifest
	manifestDoc.Items[0].TaskRef = &task.Ref
	manifestBytes, err := auditdomain.EncodeExecutionManifest(manifestDoc)
	if err != nil {
		t.Fatal(err)
	}
	manifest := writeExact(t, ctx, store, "audit-test", "manifest", "application/json", manifestBytes)
	catalog, err := config.Load("../../testdata/configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := catalog.AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	binding := profile.Workflows["check"]
	binding.WorkerCompletion = &config.AuditWorkerCompletion{Kind: contracts.AuditCheckResultsV1, Stage: "check", Agent: "checker"}
	stage := binding.Workflow.Stages["check"]
	agent := stage.Agents["checker"]
	for i := range agent.Template.Toolsets {
		if agent.Template.Toolsets[i].Ref.ToolsetID == "audit-results" {
			agent.Template.Toolsets[i].Ref.Version = "2"
		}
	}
	agent.Template.Skills = nil
	stage.Agents["checker"] = agent
	binding.Workflow.Stages["check"] = stage
	if err := config.ValidateAuditWorkerCompletion(binding); err != nil {
		t.Fatal(err)
	}
	profileBytes, _ := json.Marshal(map[string]any{"workflows": map[string]any{"check": binding}})
	audits := auditstore.NewPostgresStore(pool)
	draft, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: "audit", OwnerID: project.OwnerID, ProjectID: project.ProjectID, Profile: auditstore.ProfileIdentity{Name: "test", Version: "1", Digest: testDigest("2")}, ProfileSnapshot: profileBytes, InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 10, MaxItemsTotal: 10, MaxSubmittedRuns: 10, MaxItemRunAttempts: 2, MaxEvidenceBytes: 1 << 20}, IdempotencyKey: "audit", RequestDigest: testDigest("3")})
	if err != nil {
		t.Fatal(err)
	}
	roundID := "round"
	_, _, err = audits.MaterializeRound(ctx, auditstore.MaterializeRoundParams{OwnerID: draft.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision, RoundID: roundID, RoundOrdinal: 1, Manifest: manifest, BaselineSnapshot: json.RawMessage(`{}`), DeadlineAt: time.Now().Add(time.Hour), Items: []auditstore.MaterializedItem{{ItemID: "item", ItemKey: "check-1", Ordinal: 0, Kind: "test", SubjectKey: manifestDoc.Items[0].SubjectKey, Task: task, Origin: runServiceAuditOrigin("check-1", task), WorkflowRole: "check", InitialState: auditstore.ItemReady, Coverage: auditstore.Coverage{Status: auditstore.CoverageNotTested, Requested: []string{}, Completed: []string{}, Gaps: []string{}}}}, IdempotencyKey: "round", RequestDigest: testDigest("4")})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := audits.Claim(ctx, auditstore.ClaimParams{HolderID: "controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatal("claim", err)
	}
	claim := claims[0]
	if _, err := audits.TransitionRound(ctx, auditstore.RoundTransitionParams{Claim: claim, RoundID: roundID, ExpectedRevision: 1, ExpectedState: auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting}); err != nil {
		t.Fatal(err)
	}
	execution, _, err := audits.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{Claim: claim, ExecutionID: "execution", RoundID: &roundID, Role: auditstore.ExecutionCheck, WorkflowRole: "check", Manifest: manifest, SubmissionKey: "submission", RequestDigest: testDigest("5"), Members: []auditstore.ExecutionMemberIntent{{ExecutionItemID: "member", ItemID: "item", BatchOrdinal: 0, ItemAttempt: 1, Task: task, Inputs: []auditstore.ExactArtifact{source}}}})
	if err != nil {
		t.Fatal(err)
	}
	manager, err := config.NewManager(config.ManagerOptions{OperatorRoot: filepath.Join("..", "config", "testdata", "valid"), ManagedRoot: filepath.Join(t.TempDir(), "managed"), Descriptors: config.MVPDescriptors()})
	if err != nil {
		t.Fatal(err)
	}
	options := Options{Runs: runstore.NewPostgresStore(pool), Workflows: manager, LLMCredentials: emptyCredentialLookup{}, CredentialGuard: openCredentialGuard{}, RuntimeCredentials: acceptRuntimeCredentials{}, Projects: projects, PublicTransaction: postgresPublicTransaction(pool), AuditTransaction: postgresAuditTransaction(pool)}
	service, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	params := AuditCreateParams{Claim: claim, ExecutionID: execution.ExecutionID, Workflow: binding.Workflow, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(), Parameters: map[string]string{}, Inputs: map[string]auditstore.ExactArtifact{"source": source, "task": task, "execution_manifest": manifest}, ExecutionManifest: manifest, RequestDigest: execution.RequestDigest, NewRunID: func() (string, error) { return "run", nil }}
	wrongWorkflow := params
	encodedWorkflow, _ := json.Marshal(params.Workflow)
	wrongWorkflow.Workflow, _ = config.DecodeResolvedWorkflowSnapshot(encodedWorkflow)
	wrongWorkflow.Workflow.Ref.Version = "different-version"
	if _, err := service.CreateAudit(ctx, wrongWorkflow); !errors.Is(err, ErrInvalid) {
		t.Fatal("different Workflow accepted", err)
	}
	wrongTask := params
	wrongTask.Inputs = cloneExactArtifacts(params.Inputs)
	wrongTask.Inputs["task"] = source
	if _, err := service.CreateAudit(ctx, wrongTask); !errors.Is(err, ErrInvalid) {
		t.Fatal("foreign task accepted", err)
	}
	// Fail after all Run/input/contract mutations, then verify the transaction
	// retained the durable intent but no Run or authority.
	rollback := errors.New("injected transaction failure")
	normalTransaction := options.AuditTransaction
	omissionOptions := options
	omissionOptions.AuditTransaction = func(ctx context.Context, fn func(AuditRunWriter, *artifacts.Service, AuditExecutionWriter) error) error {
		return normalTransaction(ctx, func(r AuditRunWriter, a *artifacts.Service, e AuditExecutionWriter) error {
			return fn(omitCompletionWriter{r}, a, e)
		})
	}
	omissionService, err := New(omissionOptions)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := omissionService.CreateAudit(ctx, params); persistencepostgres.SQLState(err) != "23514" {
		t.Fatal("opted-in Run committed without its completion authority", err)
	}

	options.AuditTransaction = func(ctx context.Context, fn func(AuditRunWriter, *artifacts.Service, AuditExecutionWriter) error) error {
		return normalTransaction(ctx, func(r AuditRunWriter, a *artifacts.Service, e AuditExecutionWriter) error {
			if err := fn(r, a, e); err != nil {
				return err
			}
			return rollback
		})
	}
	failing, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := failing.CreateAudit(ctx, params); !errors.Is(err, rollback) {
		t.Fatalf("rollback: %v", err)
	}
	runs := runstore.NewPostgresStore(pool)
	if _, err := runs.GetRun(ctx, "run"); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatal("orphan Run survived", err)
	}
	created, err := service.CreateAudit(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	completion := created.Run.AuditCompletion
	if completion == nil || completion.Contract.Task.Namespace != "inputs" || completion.Contract.Task.ValidateExact() != nil {
		t.Fatal("completion not pinned")
	}
	runArtifacts, _ := artifactService.Run("run")
	if _, err := runArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "task"}, artifacts.Payload{MediaType: "application/zip", Data: []byte("changed alias")}, completion.Contract.Task.Revision); err != nil {
		t.Fatal(err)
	}
	// A new service uses the database snapshot even though this manager only
	// contains unrelated fixture catalog versions.
	options.AuditTransaction = normalTransaction
	restarted, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := restarted.CreateAudit(ctx, params)
	if err != nil || !replay.Replayed || !reflect.DeepEqual(replay.Run.AuditCompletion, completion) {
		t.Fatalf("restart: %v %v", replay, err)
	}
	exact, err := runArtifacts.Read(ctx, completion.Contract.Task)
	if err != nil || string(exact.Payload.Data) != string(inventory.Tasks[0].Package) {
		t.Fatal("exact task changed", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE workflow_runs SET audit_completion = NULL WHERE run_id = 'run'`); persistencepostgres.SQLState(err) != "23514" {
		t.Fatal("completion rewrite accepted", err)
	}
	stageBytes, _ := json.Marshal(stage)
	executionStage, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{StageExecutionID: "stage", RunID: "run", StageName: "check", Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageBytes, StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{"task": {Required: true, Artifact: &completion.Contract.Task}, "manifest": {Required: true, Artifact: &completion.Contract.ExecutionManifest}}}})
	if err != nil {
		t.Fatal(err)
	}
	gateway := stage.ExecutionConfig.Agents["checker"].LLMGateway.Ref
	allocation := runstore.StageAllocation{AllocationID: "allocation", StageExecutionID: executionStage.StageExecutionID, LogicalAgentName: "checker", Namespace: "audit-check", AgentTemplateRef: agent.Template.Ref, WorkerRuntimeRef: agent.Template.Runtime, RuntimeAgentID: strings.Repeat("1", 64), RuntimeAgentInstanceID: "runtime", RuntimeAgentLabelRevision: 1, RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion, PerformanceCollectionPolicy: contracts.PerformanceCollectionDisabled, RuntimeConfiguration: &runstore.AllocationRuntimeConfiguration{ModelPolicy: stage.ExecutionConfig.Agents["checker"].ModelPolicy.Ref, Origins: runtimeconfig.ResolvedRuntimeConfigOrigins{}, Provenance: contracts.ResolvedRuntimeConfigProvenance{Default: contracts.RuntimeLabelBindingProvenance{Label: "default", BindingRevision: 1, Config: contracts.RuntimeConfigRef{Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion, Digest: runtimeconfig.BuiltInDigest}}, RunLabels: []contracts.RuntimeLabelBindingProvenance{}, AgentLabels: []contracts.RuntimeLabelBindingProvenance{}, RuntimeAdapters: []contracts.RuntimeAdapterRef{}, RuntimeCredentialRefs: []contracts.RuntimeCredentialRef{}, LLMGatewayConfig: &gateway}}}
	if err := runs.RecordStageAllocation(ctx, allocation); persistencepostgres.SQLState(err) != "23514" {
		t.Fatal("allocation downgrade accepted", err)
	}
	allocation.CompletionContract = &completion.Contract
	if err := runs.RecordStageAllocation(ctx, allocation); err != nil {
		t.Fatal(err)
	}
	recovered, err := runstore.NewPostgresStore(pool).ListStageAllocations(ctx, executionStage.StageExecutionID)
	if err != nil || len(recovered) != 1 || !reflect.DeepEqual(recovered[0].CompletionContract, allocation.CompletionContract) {
		t.Fatal("allocation lost contract", err)
	}
	if err := runs.RecordStageAllocation(ctx, allocation); err != nil {
		t.Fatal("allocation replay", err)
	}
	// A second Stage attempt belongs to the same Run and keeps the same output
	// binding; it must record the exact original contract again.
	abort := func(id string) {
		t.Helper()
		if err := runs.EnterAborting(ctx, runstore.EnterAbortingParams{StageExecutionID: id, ExpectedState: runstore.StagePreparing, TerminationSchemaVersion: contracts.APIVersion,
			Termination: runstore.StageTermination{Outcome: runstore.TerminationInterrupted, Code: "test_interruption", Message: "test interruption", Retryable: true, Phase: runstore.TerminationPreparing, OccurredAt: time.Now()}, AbortID: "abort-" + id, Deadline: time.Now().Add(time.Minute), Reason: runstore.Reason{Code: "test_interruption"}}); err != nil {
			t.Fatal(err)
		}
		if err := runs.CompleteStageTermination(ctx, id); err != nil {
			t.Fatal(err)
		}
	}
	abort("stage")
	previous := "stage"
	nextStage, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{StageExecutionID: "stage-retry", RunID: "run", StageName: "check", Attempt: 2, PreviousExecutionID: &previous, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageBytes, StageContextSchemaVersion: contracts.APIVersion, StageContext: executionStage.StageContext})
	if err != nil {
		t.Fatal(err)
	}
	allocation.AllocationID = "allocation-retry"
	allocation.StageExecutionID = nextStage.StageExecutionID
	if err := runs.RecordStageAllocation(ctx, allocation); err != nil {
		t.Fatal("same-Run retry", err)
	}
	abort("stage-retry")
	if _, err := runArtifacts.Write(ctx, completion.Contract.ResultArtifact, artifacts.Payload{MediaType: "application/zip", Data: []byte("prior failed attempt output")}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, "run", runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "test_failure"}); err != nil {
		t.Fatal(err)
	}
	cursor, err := runs.GetRunEventCursor(ctx, "run")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := audits.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: claim, ExecutionID: execution.ExecutionID, RunID: "run", Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)}); err != nil {
		t.Fatal(err)
	}
	failureCode := "run_failed"
	if _, _, err := audits.Collect(ctx, auditstore.CollectParams{Claim: claim, ReceiptID: "receipt", ExecutionID: execution.ExecutionID, Disposition: auditstore.CollectionExecutionFailed, ErrorCode: &failureCode, RequestDigest: testDigest("6"), Items: []auditstore.CollectionItem{{ExecutionItemID: "member", Disposition: auditstore.CollectionExecutionFailed, Retryable: true, FinalDisposition: auditstore.FinalExecutionFailed, Coverage: auditstore.Coverage{Status: auditstore.CoverageBlocked, Requested: []string{}, Completed: []string{}, Gaps: []string{"failed"}}}}}); err != nil {
		t.Fatal(err)
	}
	retriedExecution, _, err := audits.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{Claim: claim, ExecutionID: "execution-retry", RoundID: &roundID, Role: auditstore.ExecutionCheck, WorkflowRole: "check", Manifest: manifest, SubmissionKey: "submission-retry", RequestDigest: testDigest("7"), Members: []auditstore.ExecutionMemberIntent{{ExecutionItemID: "member-retry", ItemID: "item", BatchOrdinal: 0, ItemAttempt: 2, Task: task, Inputs: []auditstore.ExactArtifact{source}}}})
	if err != nil {
		t.Fatal(err)
	}
	params.ExecutionID = retriedExecution.ExecutionID
	params.RequestDigest = retriedExecution.RequestDigest
	params.NewRunID = func() (string, error) { return "run-retry", nil }
	retried, err := restarted.CreateAudit(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if retried.Run.RunID == created.Run.RunID || retried.Run.AuditCompletion == nil || !reflect.DeepEqual(retried.Run.AuditCompletion.Contract.ResultArtifact, completion.Contract.ResultArtifact) {
		t.Fatal("Audit retry lost its Run-scoped output contract")
	}
	retriedArtifacts, _ := artifactService.Run(retried.Run.RunID)
	if _, err := retriedArtifacts.Read(ctx, retried.Run.AuditCompletion.Contract.ResultArtifact); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatal("new child Run inherited previous output", err)
	}

}

func TestAuditCompletionPostgresTaskSetMembership(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedRunServicePool(t, ctx, databaseURL)
	_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "project", OwnerID: "owner", Kind: projectstore.KindProject, Name: "Batch", IdempotencyKey: "project", RequestDigest: testDigest("1")})
	if err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	store, _ := service.Project("project")
	source := writeExact(t, ctx, store, "sources", "source", "application/zip", []byte("source"))
	checklist := `{"schema":"contractor.audit.checklist.v1","items":[{"key":"first","version":"1","statement":"First check","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"},{"key":"second","version":"1","statement":"Second check","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`
	inventory, err := auditdomain.BuildChecklistInventory([]byte(checklist), "application/json", auditdomain.InventoryOptions{Round: 1, WorkflowRole: "check", SourceInputName: "source", SourceRef: source.Ref, ApprovalRequirement: auditdomain.ApprovalNone})
	if err != nil {
		t.Fatal(err)
	}
	taskA := writeExact(t, ctx, store, "audit", "first", "application/zip", inventory.Tasks[0].Package)
	taskB := writeExact(t, ctx, store, "audit", "second", "application/zip", inventory.Tasks[1].Package)
	manifestDoc := inventory.ExecutionManifest
	manifestDoc.Items[0].TaskRef = &taskA.Ref
	manifestDoc.Items[1].TaskRef = &taskB.Ref
	encoded, err := auditdomain.EncodeExecutionManifest(manifestDoc)
	if err != nil {
		t.Fatal(err)
	}
	manifest := writeExact(t, ctx, store, "audit", "manifest", "application/json", encoded)
	members := []auditdomain.PackageInput{
		{ID: "task-000", Path: "tasks/000.zip", MediaType: "application/zip", Data: inventory.Tasks[0].Package},
		{ID: "task-001", Path: "tasks/001.zip", MediaType: "application/zip", Data: inventory.Tasks[1].Package},
	}
	data, _, err := auditdomain.BuildPackage("task-set", auditdomain.PackageKindTaskSet, "", members)
	if err != nil {
		t.Fatal(err)
	}
	taskSet := writeExact(t, ctx, store, "audit", "task-set", "application/zip", data)
	intent := auditstore.RunCreationIntent{ProjectID: "project", Execution: auditstore.Execution{Manifest: manifest, WorkflowRole: "check"}, Items: []auditstore.ExecutionItem{{Task: taskA}, {Task: taskB}}}
	if err := validateCompletionMembership(ctx, service, intent, taskSet, manifest); err != nil {
		t.Fatal("valid batch", err)
	}
	if err := validateCompletionMembership(ctx, service, intent, taskA, manifest); !errors.Is(err, ErrInvalid) {
		t.Fatal("partial task set accepted", err)
	}
	members[0].Data, members[1].Data = members[1].Data, members[0].Data
	data, _, err = auditdomain.BuildPackage("reordered", auditdomain.PackageKindTaskSet, "", members)
	if err != nil {
		t.Fatal(err)
	}
	reordered := writeExact(t, ctx, store, "audit", "reordered", "application/zip", data)
	if err := validateCompletionMembership(ctx, service, intent, reordered, manifest); !errors.Is(err, ErrInvalid) {
		t.Fatal("wrong task membership accepted", err)
	}
	manifestDoc.Items[0].SubjectKey = "another-subject"
	encoded, err = auditdomain.EncodeExecutionManifest(manifestDoc)
	if err != nil {
		t.Fatal(err)
	}
	wrongManifest := writeExact(t, ctx, store, "audit", "wrong-manifest", "application/json", encoded)
	intent.Execution.Manifest = wrongManifest
	if err := validateCompletionMembership(ctx, service, intent, taskSet, wrongManifest); !errors.Is(err, ErrInvalid) {
		t.Fatal("changed task subject accepted", err)
	}
}

// Simulates a crashed/older writer that skips the new atomic pinning step.
type omitCompletionWriter struct{ AuditRunWriter }

func (omitCompletionWriter) SetAuditCompletion(context.Context, string, runstore.AuditCompletionSnapshot) error {
	return nil
}
