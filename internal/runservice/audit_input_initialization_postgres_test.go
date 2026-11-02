package runservice

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5/pgxpool"
)

func auditInputFixture(t *testing.T, ctx context.Context) (Options, AuditCreateParams, *pgxpool.Pool) {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	pool := isolatedRunServicePool(t, ctx, databaseURL)
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: "project", OwnerID: "owner", Kind: projectstore.KindProject, Name: "Audit completion", IdempotencyKey: "project", RequestDigest: testDigest("1")})
	if err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifactpostgres.NewRepository(pool))
	store, err := artifactService.Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	source := writeExact(t, ctx, store, "sources", "source", "application/zip", []byte("source"))
	inventory, err := auditdomain.BuildChecklistInventory([]byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check-1","version":"1","statement":"Check source","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`), "application/json", auditdomain.InventoryOptions{Round: 1, WorkflowRole: "check", SourceInputName: "source", SourceRef: source.Ref, ApprovalRequirement: auditdomain.ApprovalNone})
	if err != nil {
		t.Fatal(err)
	}
	task := writeAuditExact(t, ctx, artifactService, project.ProjectID, "task", "application/zip", inventory.Tasks[0].Package)
	manifestDoc := inventory.ExecutionManifest
	manifestDoc.Items[0].TaskRef = &task.Ref
	manifestBytes, err := auditdomain.EncodeExecutionManifest(manifestDoc)
	if err != nil {
		t.Fatal(err)
	}
	manifest := writeAuditExact(t, ctx, artifactService, project.ProjectID, "manifest", "application/json", manifestBytes)
	catalog, _, err := configload.LoadUnionReadOnly("../../testdata/configs", filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	profile, err := catalog.AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	binding := profile.Workflows["check"]
	binding.WorkerCompletion = &auditprofile.AuditWorkerCompletion{Kind: contracts.AuditCheckResultsV1, Stage: "check", Agent: "checker"}
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
	if err := auditprofile.ValidateAuditWorkerCompletion(binding); err != nil {
		t.Fatal(err)
	}
	profileBytes, _ := json.Marshal(map[string]any{"workflows": map[string]any{"check": binding}})
	audits := auditstoretest.NewStore(pool)
	draft, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: "audit", OwnerID: project.OwnerID, ProjectID: project.ProjectID, Profile: auditstore.ProfileIdentity{Name: "test", Version: "1", Digest: testDigest("2")}, ProfileSnapshot: profileBytes, InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 10, MaxItemsTotal: 10, MaxSubmittedRuns: 10, MaxItemRunAttempts: 2, MaxEvidenceBytes: 1 << 20}, IdempotencyKey: "audit", RequestDigest: testDigest("3")})
	if err != nil {
		t.Fatal(err)
	}
	roundID := "round"
	_, _, err = audits.MaterializeRound(ctx, auditstore.MaterializeRoundParams{OwnerID: draft.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision, RoundID: roundID, RoundOrdinal: 1, Manifest: manifest, BaselineSnapshot: json.RawMessage(`{"labels":{},"requiredPlacementLabels":[]}`), DeadlineAt: time.Now().Add(time.Hour), Items: []auditstore.MaterializedItem{{ItemID: "item", ItemKey: "check-1", Ordinal: 0, Kind: "test", SubjectKey: manifestDoc.Items[0].SubjectKey, Task: task, Origin: runServiceAuditOrigin("check-1", task), WorkflowRole: "check", InitialState: auditstore.ItemReady, Coverage: auditstore.Coverage{Status: auditstore.CoverageNotTested, Requested: []string{}, Completed: []string{}, Gaps: []string{}}}}, IdempotencyKey: "round", RequestDigest: testDigest("4")})
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
	execution, _, err := audits.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{Claim: claim, ExecutionID: "execution", RoundID: &roundID, Role: auditstore.ExecutionCheck, WorkflowRole: "check", LaneClass: resolved.LaneClassModel, Manifest: manifest, SubmissionKey: "submission", RequestDigest: testDigest("5"), Members: []auditstore.ExecutionMemberIntent{{ExecutionItemID: "member", ItemID: "item", BatchOrdinal: 0, ItemAttempt: 1, Task: task, Inputs: []auditstore.ExactArtifact{source}}}})
	if err != nil {
		t.Fatal(err)
	}
	manager, _, err := configload.NewManager(config.ManagerOptions{OperatorRoot: filepath.Join("..", "config", "testdata", "valid"), ManagedRoot: filepath.Join(t.TempDir(), "managed")})
	if err != nil {
		t.Fatal(err)
	}
	options := Options{Runs: runstore.NewPostgresStore(pool), Workflows: manager, LLMCredentials: emptyCredentialLookup{}, CredentialGuard: openCredentialGuard{}, RuntimeCredentials: acceptRuntimeCredentials{}, Projects: projects, PublicTransaction: postgresPublicTransaction(pool), AuditTransaction: postgresAuditTransaction(pool)}
	params := AuditCreateParams{Claim: claim, ExecutionID: execution.ExecutionID, Workflow: binding.Workflow, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), Parameters: map[string]string{}, Inputs: map[string]auditstore.ExactArtifact{"source": source, "task": task, "execution_manifest": manifest}, ExecutionManifest: manifest, RequestDigest: execution.RequestDigest, NewRunID: func() (string, error) { return "run", nil }}
	return options, params, pool
}

func TestAuditInputPostgresResumesOneChildAndClassifiesFailures(t *testing.T) {
	for _, outcome := range []string{"complete", "first-transient", "transient", "permanent", "deadline"} {
		t.Run(outcome, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			options, params, pool := auditInputFixture(t, ctx)
			params.InputMaterializationID = testDigest("9")
			transient := errors.New("injected unavailable storage")
			calls := 0
			var originalDeadline time.Time
			params.MaterializeInputs = func(ctx context.Context, service *artifacts.Service, intent auditstore.RunCreationIntent, run runstore.WorkflowRun, state runstore.AuditInputMaterialization) (runstore.AuditInputMaterialization, error) {
				calls++
				if !state.DeadlineAt.Equal(intent.Execution.CreatedAt.Add(runstore.AuditInputMaterializationTimeout)) {
					t.Fatal("input deadline did not use the original durable intent")
				}
				if calls == 1 {
					originalDeadline = state.DeadlineAt
				} else if !originalDeadline.Equal(state.DeadlineAt) {
					t.Fatal("first slice rollback reset the input deadline")
				}
				if outcome == "first-transient" && calls == 1 {
					return state, transient
				}
				if state.Revision != 0 {
					if outcome == "permanent" {
						return state, blob.ErrIntegrity
					}
					if outcome == "transient" && calls == 2 {
						return state, transient
					}
					state.Complete = true
					state.Checkpoint = []byte(`{"slice":2}`)
					return state, nil
				}
				scope, _ := artifacts.RunScope(run.RunID)
				written, err := service.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: artifactpolicy.AuditContextNamespace, Name: "page"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("exact context")})
				if err != nil {
					return state, err
				}
				state.Refs = []contracts.ArtifactRef{written.Ref}
				state.Checkpoint = []byte(`{"slice":1}`)
				return state, nil
			}
			service, err := New(options)
			if err != nil {
				t.Fatal(err)
			}
			first, err := service.CreateAudit(ctx, params)
			if outcome == "first-transient" {
				if !errors.Is(err, transient) {
					t.Fatal("first transient outcome", err)
				}
				if _, err := runstore.NewPostgresStore(pool).GetRun(ctx, "run"); !errors.Is(err, runstore.ErrNotFound) {
					t.Fatal("rolled back first slice retained a child", err)
				}
				first, err = service.CreateAudit(ctx, params)
			}
			if err != nil || first.Run.State != runstore.RunInitializing || first.Run.StateReason.Code != runstore.AuditInputInitializationPendingReason {
				t.Fatalf("first slice: %+v %v", first, err)
			}
			var scheduled, stages, holds int
			if err := pool.QueryRow(ctx, `SELECT COALESCE(scheduler_claim_tier,0), (SELECT count(*) FROM stage_executions WHERE run_id=$1), (SELECT COALESCE(sum(hold_count),0) FROM artifact_binding_revisions WHERE scope_kind='run' AND scope_id=$1) FROM workflow_runs WHERE run_id=$1`, first.Run.RunID).Scan(&scheduled, &stages, &holds); err != nil {
				t.Fatal(err)
			}
			if scheduled != 0 || stages != 0 || holds == 0 {
				t.Fatalf("partial input allocated or unprotected: tier=%d stages=%d holds=%d", scheduled, stages, holds)
			}
			service, err = New(options)
			if err != nil {
				t.Fatal(err)
			}
			var last CreateResult
			if outcome == "deadline" {
				// Advance an injected clock; no wall-clock timing assertion.
				err = options.AuditTransaction(ctx, func(r AuditRunWriter, a *artifacts.Service, e AuditExecutionWriter) error {
					intent, err := e.GetRunCreationIntent(ctx, params.Claim, params.ExecutionID)
					if err != nil {
						return err
					}
					run, err := r.GetRun(ctx, first.Run.RunID)
					if err != nil {
						return err
					}
					run, err = completeAuditInputSlice(ctx, r, a, intent, params, run, run.CreatedAt.Add(runstore.AuditInputMaterializationTimeout))
					last = CreateResult{Run: run, Replayed: true}
					return err
				})
			} else {
				last, err = service.CreateAudit(ctx, params)
				if outcome == "transient" {
					if !errors.Is(err, transient) {
						t.Fatalf("transient slice: %v", err)
					}
					last, err = service.CreateAudit(ctx, params)
				}
			}
			want := runstore.RunPending
			if outcome == "permanent" || outcome == "deadline" {
				want = runstore.RunFailed
			}
			if err != nil || !last.Replayed || last.Run.RunID != first.Run.RunID || last.Run.State != want {
				t.Fatalf("recovery: %+v %v, want %s", last, err, want)
			}
			if want == runstore.RunFailed && last.Run.StateReason.Code != runstore.AuditInputInitializationFailedReason {
				t.Fatal("failure classification lost", last.Run.StateReason)
			}
			var children, debits int
			if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM workflow_runs WHERE audit_execution_id=$1), submitted_run_count FROM audits WHERE audit_id=$2`, params.ExecutionID, params.Claim.AuditID).Scan(&children, &debits); err != nil {
				t.Fatal(err)
			}
			if children != 1 || debits != 1 {
				t.Fatalf("recovery duplicated child/debit: %d/%d", children, debits)
			}
		})
	}
}

func TestAuditInputPostgresCanonicalByteBoundaryAndImmutability(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	options, params, pool := auditInputFixture(t, ctx)
	params.InputMaterializationID = testDigest("9")
	params.MaterializeInputs = func(_ context.Context, _ *artifacts.Service, _ auditstore.RunCreationIntent, _ runstore.WorkflowRun, state runstore.AuditInputMaterialization) (runstore.AuditInputMaterialization, error) {
		state.Checkpoint = []byte(`{"slice":1}`)
		return state, nil
	}
	service, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	created, err := service.CreateAudit(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(context.WithoutCancel(ctx))
	runs := runstore.NewPostgresTxStore(tx)
	state, err := runs.LockAuditInputMaterialization(ctx, created.Run.RunID)
	if err != nil {
		t.Fatal(err)
	}
	state.Revision++
	state.Checkpoint = json.RawMessage(`{"padding":""}`)
	data, _ := json.Marshal(state)
	state.Checkpoint = json.RawMessage(`{"padding":"` + strings.Repeat("x", runstore.MaxAuditInputMaterializationBytes-len(data)) + `"}`)
	if err := runs.SetAuditInputMaterialization(ctx, created.Run.RunID, 1, *state); err != nil {
		data, _ := json.Marshal(state)
		var length int
		var valid bool
		diagnostic := pool.QueryRow(ctx, `SELECT octet_length($1::json::text), contractor_valid_audit_input_materialization($1::json)`, data).Scan(&length, &valid)
		t.Fatalf("exact byte boundary rejected: %v len=%d valid=%v diagnostic=%v", err, length, valid, diagnostic)
	}
	readback, err := runs.LockAuditInputMaterialization(ctx, created.Run.RunID)
	if err != nil {
		t.Fatal("stored boundary unreadable", err)
	}
	encoded, _ := json.Marshal(readback)
	if len(encoded) != runstore.MaxAuditInputMaterializationBytes {
		t.Fatal("boundary fixture", len(encoded))
	}
	readback.Revision++
	readback.Checkpoint = append(readback.Checkpoint[:len(readback.Checkpoint)-2], 'x', '"', '}')
	if err := runs.SetAuditInputMaterialization(ctx, created.Run.RunID, 2, *readback); !errors.Is(err, runstore.ErrInvalid) {
		t.Fatal("oversize state accepted", err)
	}
	if err := runs.SetAuditInputMaterialization(ctx, created.Run.RunID, 1, *state); !errors.Is(err, runstore.ErrConflict) {
		t.Fatal("stale CAS accepted", err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE workflow_runs SET audit_input_materialization=jsonb_set(audit_input_materialization::jsonb,'{identity}',to_jsonb($2::text))::json WHERE run_id=$1`, created.Run.RunID, testDigest("a")); persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation {
		t.Fatal("authority rewrite accepted", err)
	}
}
