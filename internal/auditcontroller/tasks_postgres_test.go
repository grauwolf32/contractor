//go:build integration

package auditcontroller

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func newCommonControllerHarness(t *testing.T, configure ...func(*auditcontract.Program)) *postgresControllerHarness {
	t.Helper()
	ctx := t.Context()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{Configure: func(c *pgxpool.Config) { c.MaxConns = 1 }})
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	root := configtest.CopyWithPolicies(t, "../config/testdata/valid")
	if err := os.CopyFS(root, os.DirFS("../config/testdata/audit-common-catalog")); err != nil {
		t.Fatal(err)
	}
	if err := os.CopyFS(root, os.DirFS("../config/testdata/openapi-prepared-work")); err != nil {
		t.Fatal(err)
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	program := auditcontract.Program{
		Schema:      "contractor.audit.program.v1",
		Inputs:      map[string]auditcontract.ProgramInput{"source": {Required: true, MediaTypes: []string{"text/plain"}}},
		Workflows:   map[string]auditcontract.ProgramWorkflow{"inspect": {Ref: "artifact-copy@1", Inputs: map[string]auditcontract.InputMapping{"source": {Source: "audit-input", Name: "source"}}, Parameters: map[string]string{}, Outputs: map[string]string{"material": "result"}}},
		Hypotheses:  map[string]auditcontract.HypothesisSeed{},
		InitialWork: []auditcontract.InitialWork{{Key: "inspect-1", Workflow: "inspect", Objective: "Inspect selected source"}},
		Limits:      auditcontract.ProgramLimits{MaxTasks: 16, MaxIterations: 1, MaxTasksPerIteration: 16, BatchSize: 2, MaxSubmittedRuns: 8, MaxAttempts: 1, DeadlineSeconds: 86400, MaxEvidenceBytes: 64 << 20, MaxHypotheses: 1, MaxHypothesisRevisions: 1, MaxHypothesisBytes: 32768},
		Policy:      auditcontract.OperationPolicy{ActiveOperations: "prohibited"},
	}
	program.Workflows["check"] = auditcontract.ProgramWorkflow{Ref: "audit-contract-check@1", Inputs: map[string]auditcontract.InputMapping{"source": {Source: "task-input"}, "context": {Source: "task-context"}}, Parameters: map[string]string{}, Outputs: map[string]string{"result": "result"}}
	program.Inputs["openapi"] = auditcontract.ProgramInput{Required: true, MediaTypes: []string{"application/json"}}
	program.Workflows["prepare"] = auditcontract.ProgramWorkflow{Ref: "openapi-prepare-work@1", Inputs: map[string]auditcontract.InputMapping{"openapi": {Source: "audit-input", Name: "openapi"}}, Parameters: map[string]string{"checking_workflow": "check", "selection": "operations", "max_items": "1"}, Outputs: map[string]string{"material": "work"}}
	for _, change := range configure {
		change(&program)
	}
	data, _ := json.Marshal(program)
	pinned, err := auditprofile.PinProgram(resolved.Selector{ID: "common-controller", Version: "1"}, data, catalog.Workflow)
	if err != nil {
		t.Fatal(err)
	}
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: "common-project", OwnerID: "owner", Kind: projectstore.KindProject, Name: "Common controller", IdempotencyKey: "project", RequestDigest: contentdigest.Bytes([]byte("project"))})
	if err != nil {
		t.Fatal(err)
	}
	art := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	projectArtifacts, _ := art.Project(project.ProjectID)
	written, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("selected source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := projectArtifacts.Metadata(ctx, written.Ref)
	if err != nil {
		t.Fatal(err)
	}
	apiWrite, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "openapi"}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"openapi":"3.1.0","paths":{}}`)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	apiMetadata, err := projectArtifacts.Metadata(ctx, apiWrite.Ref)
	if err != nil {
		t.Fatal(err)
	}
	audits := auditstoretest.NewStore(pool).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	snapshot, _ := json.Marshal(pinned)
	limits := program.Limits
	audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: "common-audit", OwnerID: "owner", ProjectID: project.ProjectID, Profile: auditstore.ProfileIdentity{Name: pinned.Ref.Name, Version: pinned.Ref.Version, Digest: pinned.Ref.Digest}, ProfileSnapshot: snapshot, InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: limits.MaxIterations, BatchSize: limits.BatchSize, MaxItemsPerRound: limits.MaxTasksPerIteration, MaxItemsTotal: limits.MaxTasks, MaxSubmittedRuns: limits.MaxSubmittedRuns, MaxItemRunAttempts: limits.MaxAttempts, MaxEvidenceBytes: limits.MaxEvidenceBytes}, IdempotencyKey: "audit", RequestDigest: contentdigest.Bytes([]byte("audit"))})
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := auditservice.EncodeBaseline(auditservice.BaselineSnapshot{Schema: auditservice.BaselineSchema, Labels: map[string]string{}, RequiredPlacementLabels: []string{}, Inputs: map[string]auditstore.ExactArtifact{"source": metadata.Exact(), "openapi": apiMetadata.Exact()}, ActiveChecks: contracts.ActiveChecksProhibited, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), RuntimeLabels: []string{}, Skills: []agentskills.RunSkillSnapshot{}, LLMCredentialIDs: []string{}, RuntimeCredentialIDs: []string{}, Standards: []auditstandards.PinnedPackage{}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audits SET state='active',phase='rounds',started_at=clock_timestamp(),hold_state='held',baseline_snapshot=$2 WHERE audit_id=$1`, audit.AuditID, baseline); err != nil {
		t.Fatal(err)
	}
	service, err := auditservice.New(auditservice.Options{Pool: pool, Profiles: catalog, TransactionLLMCredentials: controllerTransactionCredentialLookup(), CredentialGuard: controllerCredentialGuard{}})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	runService, err := runservice.New(runservice.Options{Runs: runs, Workflows: catalog, LLMCredentials: controllerCredentialLookup{}, CredentialGuard: controllerCredentialGuard{}, RuntimeCredentials: controllerRuntimeCredentials{}, Projects: projects, SkillInitializationAvailable: true,
		PublicTransaction: func(context.Context, func(runservice.PublicRunWriter, *artifacts.Service) error) error {
			return errors.New("public Run creation is outside this component fixture")
		},
		AuditTransaction: func(ctx context.Context, fn func(runservice.AuditRunWriter, *artifacts.Service, runservice.AuditExecutionWriter) error) error {
			return postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				return fn(runstore.NewPostgresTxStore(tx), artifacts.NewService(artifacts.NewPostgresRepository(tx)), auditstoretest.NewStore(tx))
			})
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return &postgresControllerHarness{pool: pool, snapshot: catalog, artifacts: art, audits: audits, runs: runs, runService: runService, service: service, started: auditservice.StartedAudit{Audit: audit}}
}

func (h *postgresControllerHarness) commonController(t *testing.T) *Controller {
	t.Helper()
	access, err := auditimport.NewArtifactAccess(h.artifacts, hookstore.NewClaimedArtifacts(h.pool))
	if err != nil {
		t.Fatal(err)
	}
	collector, err := auditimport.New(h.audits, h.audits, h.runs, access)
	if err != nil {
		t.Fatal(err)
	}
	controller, err := New(h.audits, h.audits, h.runs, rundeletion.New(h.pool), h.runService, h.builder(t), &postgresNotifier{}, Options{OperationTimeout: 5 * time.Second, ClaimLease: 15 * time.Second, Collector: collector.WithTaskIntake(h.service), TaskIntake: h.service})
	if err != nil {
		t.Fatal(err)
	}
	return controller
}

func TestPostgresCommonControllerPauseHandoffResume(t *testing.T) {
	h := newCommonControllerHarness(t)
	ctx := t.Context()
	c := h.commonController(t)
	step := func() {
		t.Helper()
		if changed, err := c.RunOnce(ctx); err != nil || !changed {
			t.Fatalf("step: %v %v", changed, err)
		}
	}
	step() // accept initial ordinary work
	step() // select and submit through actual RunService
	executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("first Run: %+v %v", executions, err)
	}
	first := executions[0]
	audit, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.service.Pause(ctx, auditservice.MutationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "pause", RequestDigest: contentdigest.Bytes([]byte("pause"))}); err != nil {
		t.Fatal(err)
	}
	if changed, err := c.RunOnce(ctx); err != nil || changed {
		t.Fatalf("paused active Run: %v %v", changed, err)
	}
	firstOutput := finishCommonChild(t, h, *first.RunID, false)
	step()                    // observe terminal while paused
	step()                    // collect ordinary material while paused
	c = h.commonController(t) // process-local controller state is discarded
	if changed, err := c.RunOnce(ctx); err != nil || changed {
		t.Fatalf("empty paused Audit completed: %v %v", changed, err)
	}
	if err := rundeletion.New(h.pool).DeleteReleasedTerminalRun(ctx, "owner", *first.RunID); err != nil {
		t.Fatal(err)
	}
	audit, err = h.service.Get(ctx, "owner", audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	params := auditservice.TaskCreationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "second", RequestDigest: contentdigest.Bytes([]byte("second")), Work: []auditcontract.PreparedSelection{{Key: "second", Workflow: "check", Objective: "Inspect the selected first result", Inputs: map[string]auditcontract.Material{"source": firstOutput}}}}
	accepted, err := h.service.CreateTasks(ctx, params)
	if err != nil || accepted.Acceptance.CreatedCount != 1 {
		t.Fatalf("exact handoff after source purge: %+v %v", accepted, err)
	}
	if changed, err := c.RunOnce(ctx); err != nil || changed {
		t.Fatalf("paused second dispatch: %v %v", changed, err)
	}
	audit, err = h.service.Get(ctx, "owner", audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := h.audits.Resume(ctx, auditstore.ResumeParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "resume", RequestDigest: contentdigest.Bytes([]byte("resume"))}); err != nil {
		t.Fatal(err)
	}
	step()
	executions, err = h.audits.ListExecutions(ctx, audit.AuditID)
	if err != nil || len(executions) != 2 || executions[1].RunID == nil {
		t.Fatalf("second Run: %+v %v", executions, err)
	}
	secondRun, _ := h.artifacts.Run(*executions[1].RunID)
	input, err := secondRun.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil || string(input.Payload.Data) != "ordinary first output" {
		t.Fatalf("inexact handoff: %+v %v", input, err)
	}
	finishCommonChild(t, h, *executions[1].RunID, true)
	for range 6 {
		step()
		audit, err = h.service.Get(ctx, "owner", audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if audit.State == auditstore.AuditCompleted {
			if audit.SubmittedRunCount != 2 || audit.TaskCount != 2 || audit.OutstandingRunCount != 0 {
				t.Fatalf("bad completion: %+v", audit)
			}
			var rounds, items int
			if err := h.pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM audit_rounds WHERE audit_id=$1),(SELECT count(*) FROM audit_items WHERE audit_id=$1)`, audit.AuditID).Scan(&rounds, &items); err != nil || rounds != 0 || items != 0 {
				t.Fatalf("retired pipeline used: %d/%d %v", rounds, items, err)
			}
			return
		}
	}
	t.Fatal("ordinary Audit did not complete without findings or report approval")
}

func finishCommonChild(t *testing.T, h *postgresControllerHarness, runID string, semantic bool) auditcontract.Material {
	t.Helper()
	ctx := t.Context()
	run, _ := h.artifacts.Run(runID)
	payload := artifacts.Payload{MediaType: "text/plain", Data: []byte("ordinary first output")}
	if semantic {
		trustedContext, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "context"})
		if err != nil {
			t.Fatal(err)
		}
		var value struct {
			Items []struct {
				Evidence []auditcontract.Material `json:"evidence"`
			}
		}
		if err := json.Unmarshal(trustedContext.Payload.Data, &value); err != nil {
			t.Fatal(err)
		}
		data, _ := json.Marshal(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": []any{map[string]any{"item_key": "item-00", "rationale": "Selected material inspected", "tested_conditions": []string{"exact input bytes"}, "limitations": []string{}, "evidence": value.Items[0].Evidence}}})
		payload = artifacts.Payload{MediaType: "application/json", Data: data}
	}
	written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "result"}, payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactstest.BindOutput(ctx, h.artifacts, runID, "result", written.Ref); err != nil {
		t.Fatal(err)
	}
	if err := artifactstest.FreezeOutputs(ctx, h.artifacts, runID); err != nil {
		t.Fatal(err)
	}
	for _, transition := range [][2]runstore.WorkflowRunState{{runstore.RunPending, runstore.RunRunning}, {runstore.RunRunning, runstore.RunSucceeded}} {
		if _, err := h.runs.TransitionRun(ctx, runID, transition[0], transition[1], runstore.Reason{Code: "fixture"}); err != nil {
			t.Fatal(err)
		}
	}
	metadata, err := run.Metadata(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "result"})
	if err != nil {
		t.Fatal(err)
	}
	return auditcontract.Material{ScopeKind: "run", ScopeID: runID, Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}
}

type acknowledgeLostCreator struct {
	RunCreator
	lost bool
}

func (c *acknowledgeLostCreator) CreateAudit(ctx context.Context, p runservice.AuditCreateParams) (runservice.CreateResult, error) {
	result, err := c.RunCreator.CreateAudit(ctx, p)
	if err == nil && !c.lost {
		c.lost = true
		return runservice.CreateResult{}, io.ErrUnexpectedEOF
	}
	return result, err
}

func TestPostgresCommonControllerReconcilesLostCreationAcknowledgement(t *testing.T) {
	h := newCommonControllerHarness(t)
	c := h.commonController(t)
	c.creator = &acknowledgeLostCreator{RunCreator: h.runService}
	if changed, err := c.RunOnce(t.Context()); err != nil || !changed {
		t.Fatalf("initial intake: %v %v", changed, err)
	}
	if _, err := c.RunOnce(t.Context()); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("lost acknowledgement: %v", err)
	}
	c = h.commonController(t)
	if changed, err := c.RunOnce(t.Context()); err != nil || changed {
		t.Fatalf("recovery repeated creation: %v %v", changed, err)
	}
	executions, err := h.audits.ListExecutions(t.Context(), h.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil || executions[0].State != auditstore.ExecutionSubmitted {
		t.Fatalf("ambiguous create lost binding: %+v %v", executions, err)
	}
}

func TestPostgresCommonControllerModelQueueDoesNotHideToolWork(t *testing.T) {
	h := newCommonControllerHarness(t)
	ctx := t.Context()
	toolRuns := 1
	if _, err := settingsstore.NewPostgresStore(h.pool).UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{ExpectedRevision: 1, MaxConcurrentRuns: 1, MaxConcurrentToolRuns: &toolRuns}); err != nil {
		t.Fatal(err)
	}
	c := h.commonController(t)
	for range 2 {
		if changed, err := c.RunOnce(ctx); err != nil || !changed {
			t.Fatalf("initial: %v %v", changed, err)
		}
	}
	audit, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.service.CreateTasks(ctx, auditservice.TaskCreationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "mixed", RequestDigest: contentdigest.Bytes([]byte("mixed")), Work: []auditcontract.PreparedSelection{{Key: "model", Workflow: "inspect", Objective: "Queued model work", Inputs: map[string]auditcontract.Material{}}, {Key: "tool", Workflow: "prepare", Objective: "Independent tool work", Inputs: map[string]auditcontract.Material{}}}}); err != nil {
		t.Fatal(err)
	}
	if changed, err := c.RunOnce(ctx); err != nil || !changed {
		t.Fatalf("tool lane: %v %v", changed, err)
	}
	executions, err := h.audits.ListExecutions(ctx, audit.AuditID)
	if err != nil || len(executions) != 2 || executions[1].WorkflowRole != "prepare" || executions[1].RunID == nil {
		t.Fatalf("tool work starved: %+v %v", executions, err)
	}
	run, err := h.runs.GetRun(ctx, *executions[1].RunID)
	if err != nil || run.SchedulerLaneClass != resolved.LaneClassTool {
		t.Fatalf("wrong authoritative lane: %+v %v", run, err)
	}
}

func TestPostgresCommonControllerResumesReservedIntentAtRunBudget(t *testing.T) {
	h := newCommonControllerHarness(t, func(p *auditcontract.Program) {
		p.Limits.MaxTasks, p.Limits.MaxTasksPerIteration, p.Limits.BatchSize, p.Limits.MaxSubmittedRuns = 1, 1, 1, 1
	})
	ctx := t.Context()
	c := h.commonController(t)
	if changed, err := c.RunOnce(ctx); err != nil || !changed {
		t.Fatalf("initial: %v %v", changed, err)
	}
	page, err := h.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: h.started.Audit.AuditID, Limit: 10})
	if err != nil || len(page.Tasks) != 1 {
		t.Fatalf("task: %+v %v", page, err)
	}
	claims, err := h.audits.Claim(ctx, auditstore.ClaimParams{HolderID: "interrupted-selection", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim: %+v %v", claims, err)
	}
	selected, _, err := h.audits.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claims[0], TaskIDs: []string{page.Tasks[0].Document.TaskID}})
	if err != nil {
		t.Fatal(err)
	}
	if err := h.audits.ReleaseClaim(ctx, claims[0]); err != nil {
		t.Fatal(err)
	}
	audit, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.service.Pause(ctx, auditservice.MutationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "pause", RequestDigest: contentdigest.Bytes([]byte("pause"))}); err != nil {
		t.Fatal(err)
	}
	if changed, err := c.RunOnce(ctx); err != nil || changed {
		t.Fatalf("paused reservation was discarded: %v %v", changed, err)
	}
	audit, err = h.service.Get(ctx, "owner", audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := h.audits.Resume(ctx, auditstore.ResumeParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "resume", RequestDigest: contentdigest.Bytes([]byte("resume"))}); err != nil {
		t.Fatal(err)
	}
	c = h.commonController(t)
	if changed, err := c.RunOnce(ctx); err != nil || !changed {
		t.Fatalf("reserved Run at exhausted budget: %v %v", changed, err)
	}
	executions, err := h.audits.ListExecutions(ctx, audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].ExecutionID != selected.ExecutionID || executions[0].RunID == nil {
		t.Fatalf("reservation replaced: %+v %v", executions, err)
	}
	task, err := h.service.Task(ctx, "owner", audit.AuditID, page.Tasks[0].Document.TaskID)
	if err != nil || task.Attempts != 1 {
		t.Fatalf("pause spent another attempt: %+v %v", task, err)
	}
}

func TestPostgresCommonControllerDeadlineAndCancellation(t *testing.T) {
	for _, deadline := range []bool{true, false} {
		t.Run(map[bool]string{true: "deadline", false: "cancel"}[deadline], func(t *testing.T) {
			h := newCommonControllerHarness(t)
			ctx := t.Context()
			c := h.commonController(t)
			if deadline {
				if _, err := h.pool.Exec(ctx, `UPDATE audits SET deadline_at='2000-01-01T00:00:00Z' WHERE audit_id=$1`, h.started.Audit.AuditID); err != nil {
					t.Fatal(err)
				}
				if changed, err := c.RunOnce(ctx); err != nil || !changed {
					t.Fatalf("deadline: %v %v", changed, err)
				}
				audit, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
				if err != nil || audit.State != auditstore.AuditPaused || audit.TaskCount != 0 || audit.SubmittedRunCount != 0 {
					t.Fatalf("post-deadline work: %+v %v", audit, err)
				}
				return
			}
			if changed, err := c.RunOnce(ctx); err != nil || !changed {
				t.Fatalf("initial: %v %v", changed, err)
			}
			audit, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := h.service.Cancel(ctx, auditservice.MutationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "cancel", RequestDigest: contentdigest.Bytes([]byte("cancel"))}); err != nil {
				t.Fatal(err)
			}
			for range 4 {
				if changed, err := c.RunOnce(ctx); err != nil || !changed {
					t.Fatalf("cancel: %v %v", changed, err)
				}
				audit, err = h.service.Get(ctx, "owner", audit.AuditID)
				if err != nil {
					t.Fatal(err)
				}
				if audit.State == auditstore.AuditCancelled {
					page, err := h.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: audit.AuditID, Limit: 10})
					if err != nil || len(page.Tasks) != 1 || page.Tasks[0].State != auditstore.TaskSettled || page.Tasks[0].GapCode == nil || audit.SubmittedRunCount != 0 {
						t.Fatalf("cancelled work: %+v %v", page, err)
					}
					return
				}
			}
			t.Fatal("cancellation did not settle")
		})
	}
}
