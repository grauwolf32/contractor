package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/operations"
	"github.com/grauwolf32/contractor/internal/operations/sitemapjoin"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Embedded nil interfaces panic on any call. These collaborators are mandatory
// for the common Scheduler, but a native Stage must never use them.
type unusedOperationAllocator struct{ Allocator }

func (unusedOperationAllocator) PollAllocationLosses() []controlplane.AllocationLoss { return nil }

type unusedOperationWorkers struct{ WorkerController }
type unusedOperationPlanners struct{ PlannerRegistry }
type unusedOperationSessions struct{ sessionRecovery }

type operationCallFunc func(context.Context, string, operations.Call) (stageresult.StageContentResult, error)

func (f operationCallFunc) Run(ctx context.Context, ref string, call operations.Call) (stageresult.StageContentResult, error) {
	return f(ctx, ref, call)
}

type finalizationWriteFault struct {
	*PostgresPersistence
	failed bool
}

var errInjectedOperationWrite = errors.New("injected finalization write failure")

func (p *finalizationWriteFault) EnterFinalizingWithResult(ctx context.Context, params runstore.EnterFinalizingParams) error {
	if !p.failed {
		p.failed = true
		return errInjectedOperationWrite
	}
	return p.PostgresPersistence.EnterFinalizingWithResult(ctx, params)
}

type operationHarness struct {
	pool         *pgxpool.Pool
	store        *runstore.PostgresStore
	artifacts    *artifacts.Service
	sitemap      *projectsitemap.Service
	workflow     resolved.ResolvedWorkflow
	seed         contracts.ArtifactRef
	registry     *operations.Registry
	transactions *PostgresPersistence
}

func newOperationHarness(t *testing.T, apply bool, customize ...func(*resolved.ResolvedWorkflow)) *operationHarness {
	t.Helper()
	ctx := t.Context()
	pool := isolatedSchedulerPool(t, ctx)
	root := configtest.CopyWithPolicies(t, "../../testdata/configs")
	data, err := os.ReadFile("../config/testdata/server-operations/workflows/join.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if apply {
		data = []byte(strings.Replace(string(data), contracts.SitemapOpenAPIJoinOperation, contracts.SitemapApplyOpenAPIOperation, 1))
	}
	if err := os.WriteFile(filepath.Join(root, "workflows", "sitemap_join.yaml"), data, 0600); err != nil {
		t.Fatal(err)
	}
	catalog, err := configload.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := catalog.ResolveRunWorkflow(ctx, "sitemap-join@1", resolved.ExecutionConfigPatch{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, change := range customize {
		change(&workflow)
	}
	if err := resolved.ValidateWorkflowGraph(workflow); err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	domain := projectsitemap.NewService(pool)
	store := runstore.NewPostgresStore(pool)
	projectID := "operation-project"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Operation", IdempotencyKey: "create", RequestDigest: "sha256:" + strings.Repeat("a", 64)}); err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(workflow)
	if _, err := runstoretest.CreateRun(ctx, store, runstore.CreateRunParams{RunID: "operation-run", OwnerID: "user-1", ProjectID: &projectID, LaneClass: resolved.EntryLaneClass(workflow), WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: encoded, Parameters: map[string]string{"api_origin": "https://example.test"}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	seed, err := domain.Mutate(ctx, projectsitemap.Authority{OwnerID: "user-1", ProjectID: projectID, RunID: "operation-run", SourceID: "capture", Kind: "http", CanObserve: true}, "seed", projectsitemap.Mutation{Operation: "observe", Observations: []projectsitemap.Observation{{URL: "https://example.test/users/42", Method: "GET", Outcome: "response", Status: 200, At: "2026-10-29T00:00:00Z"}}})
	if err != nil {
		t.Fatal(err)
	}
	project, _ := service.Project(projectID)
	schema, err := project.Write(ctx, contracts.ArtifactRef{Namespace: "contracts", Name: "api"}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"openapi":"3.1.0","paths":{"/users/{id}":{"get":{"responses":{"200":{}}}}}}`)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for name, ref := range map[string]contracts.ArtifactRef{"sitemap": seed.Ref, "openapi": schema.Ref} {
		if _, err := artifactstest.ForkProjectInput(ctx, service, projectID, ref, "operation-run", name); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := store.TransitionRun(ctx, "operation-run", runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	registry, err := operations.NewRegistry(sitemapjoin.NewJoin(domain), sitemapjoin.NewApply(domain))
	if err != nil {
		t.Fatal(err)
	}
	transactions, _ := NewPostgresPersistence(pool)
	return &operationHarness{pool: pool, store: store, artifacts: service, sitemap: domain, workflow: workflow, seed: seed.Ref, registry: registry, transactions: transactions}
}

func TestPostgresServerOperationMaterializesOnlyDeclaredOutputs(t *testing.T) {
	for _, mode := range []string{"compare-only", "apply-only", "report-only"} {
		t.Run(mode, func(t *testing.T) {
			apply := mode == "apply-only"
			h := newOperationHarness(t, apply, func(workflow *resolved.ResolvedWorkflow) {
				workflow.Outputs = map[string]resolved.ArtifactSlot{}
				stage := workflow.Stages[workflow.EntryStage]
				stage.WorkflowOutputs = map[string]string{}
				stage.Result.Artifacts = map[string]resolved.ArtifactSlot{}
				if mode == "report-only" {
					stage.Result.Artifacts["report"] = resolved.ArtifactSlot{Required: true, MediaTypes: []string{"application/json"}, From: &resolved.ArtifactBinding{Namespace: "join-api", Name: "report"}}
				}
				workflow.Stages[workflow.EntryStage] = stage
			})
			s := h.scheduler(t, h.transactions, h.registry)
			if _, err := s.RunOnce(t.Context()); err != nil {
				t.Fatal(err)
			}
			run, err := h.store.GetRun(t.Context(), "operation-run")
			if err != nil || run.State != runstore.RunSucceeded {
				t.Fatalf("Run: %+v %v", run, err)
			}
			scope, _ := h.artifacts.Run(run.RunID)
			if _, err := scope.Metadata(t.Context(), contracts.ArtifactRef{Namespace: "join-api", Name: "sitemap"}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
				t.Fatalf("unrequested copy exists: %v", err)
			}
			_, err = scope.Metadata(t.Context(), contracts.ArtifactRef{Namespace: "join-api", Name: "report"})
			if (err == nil) != (mode == "report-only") {
				t.Fatalf("report: %v", err)
			}
			project, _ := h.artifacts.Project("operation-project")
			live, err := project.Metadata(t.Context(), contracts.ArtifactRef{Namespace: "sitemap", Name: "map"})
			if err != nil {
				t.Fatal(err)
			}
			if live.Ref.SameExact(h.seed) == apply {
				t.Fatal("unexpected Project publication")
			}
			if apply {
				stages, _ := h.store.ListStageExecutions(t.Context(), run.RunID)
				if !strings.Contains(stages[0].AcceptedResult.Summary, *live.Ref.Revision) {
					t.Fatal("summary lost actual publication ref")
				}
			}
		})
	}
}

func (h *operationHarness) scheduler(t *testing.T, persistence AtomicPersistence, executor ServerOperationExecutor) *Scheduler {
	t.Helper()
	resolver, _ := NewArtifactServiceResolver(h.artifacts)
	s, err := New(h.store, persistence, resolver, unusedOperationAllocator{}, unusedOperationWorkers{}, unusedOperationPlanners{}, Options{ServerOperations: executor, PlannerSessions: unusedOperationSessions{}, PollInterval: time.Second, LeaseScanInterval: time.Second, ClaimDuration: time.Minute, OperationTimeout: 5 * time.Second, PlannerTimeout: 20 * time.Second, FinalizationTimeout: 5 * time.Second, AbortTimeout: 5 * time.Second, RuntimeTransport: testSchedulerRuntimeTransport(), Settings: settingsstore.NewPostgresStore(h.pool), Reports: telemetry.NewRepository(h.pool), Logger: slog.New(slog.NewTextHandler(io.Discard, nil))})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestPostgresServerOperationStagePublishesWithoutPlannerOrRuntime(t *testing.T) {
	for _, apply := range []bool{false, true} {
		t.Run(map[bool]string{false: "pure", true: "apply"}[apply], func(t *testing.T) {
			h := newOperationHarness(t, apply)
			s := h.scheduler(t, h.transactions, h.registry)
			if worked, err := s.RunOnce(t.Context()); err != nil || !worked {
				t.Fatalf("RunOnce: %v %v", worked, err)
			}
			run, err := h.store.GetRun(t.Context(), "operation-run")
			if err != nil || run.State != runstore.RunSucceeded {
				t.Fatalf("Run: %+v %v", run, err)
			}
			stages, err := h.store.ListStageExecutions(t.Context(), run.RunID)
			if err != nil || len(stages) != 1 {
				t.Fatalf("Stages: %+v %v", stages, err)
			}
			stage := stages[0]
			if stage.State != runstore.StageSucceeded || stage.PlannerSessionID != nil || stage.PlannerInvocationID != nil || stage.PlannerStartedAt != nil || stage.AdmittedAt == nil {
				t.Fatalf("synthetic Planner state: %+v", stage)
			}
			for _, table := range []string{"planner_sessions", "stage_allocations"} {
				var count int
				if err := h.pool.QueryRow(t.Context(), "SELECT count(*) FROM "+table).Scan(&count); err != nil || count != 0 {
					t.Fatalf("%s count=%d: %v", table, count, err)
				}
			}
			scope, _ := h.artifacts.Run(run.RunID)
			output, err := scope.Read(t.Context(), contracts.ArtifactRef{Namespace: "outputs", Name: "sitemap"})
			if err != nil {
				t.Fatal(err)
			}
			document, err := projectsitemap.Decode(output.Payload.Data)
			if err != nil || len(document.Rules) != 1 {
				t.Fatalf("ordinary output: %+v %v", document, err)
			}
			project, _ := h.artifacts.Project("operation-project")
			live, err := project.Metadata(t.Context(), contracts.ArtifactRef{Namespace: "sitemap", Name: "map"})
			if err != nil {
				t.Fatal(err)
			}
			if live.Ref.SameExact(h.seed) == apply {
				t.Fatal("pure/apply publication boundary is wrong")
			}
		})
	}
}

func TestPostgresServerOperationRecoveryReplaysReceipt(t *testing.T) {
	h := newOperationHarness(t, true)
	fault := &finalizationWriteFault{PostgresPersistence: h.transactions}
	calls := 0
	executor := operationCallFunc(func(ctx context.Context, ref string, call operations.Call) (stageresult.StageContentResult, error) {
		calls++
		return h.registry.Run(ctx, ref, call)
	})
	s := h.scheduler(t, fault, executor)
	if _, err := s.RunOnce(t.Context()); !errors.Is(err, errInjectedOperationWrite) {
		t.Fatalf("fault did not leave running operation: %v", err)
	}
	project, _ := h.artifacts.Project("operation-project")
	first, err := project.Metadata(t.Context(), contracts.ArtifactRef{Namespace: "sitemap", Name: "map"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.RunOnce(t.Context()); err != nil {
		t.Fatal(err)
	}
	second, err := project.Metadata(t.Context(), contracts.ArtifactRef{Namespace: "sitemap", Name: "map"})
	if err != nil || !first.Ref.SameExact(second.Ref) || calls != 2 {
		t.Fatalf("recovery republished: calls=%d %v", calls, err)
	}
	run, _ := h.store.GetRun(t.Context(), "operation-run")
	if run.State != runstore.RunSucceeded {
		t.Fatalf("recovered state: %s", run.State)
	}
}

func TestPostgresServerOperationClaimFenceRejectsOldOwner(t *testing.T) {
	h := newOperationHarness(t, true)
	ctx := t.Context()
	claimed, err := h.store.ClaimRunnableRun(ctx, "first", time.Minute, nil, resolved.LaneClassTool)
	if err != nil {
		t.Fatal(err)
	}
	w, err := decodeExecutableWorkflow(claimed)
	if err != nil {
		t.Fatal(err)
	}
	s := h.scheduler(t, h.transactions, h.registry)
	stage, err := s.createStageExecution(ctx, claimed, w)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.transactions.AdmitStage(ctx, claimed.RunID, stage.StageExecutionID); err != nil {
		t.Fatal(err)
	}
	if _, err := h.transactions.StartStageOperation(ctx, claimed.RunID, stage.StageExecutionID, "first"); err != nil {
		t.Fatal(err)
	}
	if err := h.store.ReleaseRunClaim(ctx, claimed.RunID, "first"); err != nil {
		t.Fatal(err)
	}
	if _, err := h.store.ClaimRunnableRun(ctx, "second", time.Minute, nil, resolved.LaneClassTool); err != nil {
		t.Fatal(err)
	}
	err = postgres.InTx(ctx, h.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := runstore.NewPostgresTxStore(tx).LockStageOperation(ctx, claimed.RunID, stage.StageExecutionID, "first")
		return err
	})
	if !errors.Is(err, runstore.ErrConflict) {
		t.Fatalf("old owner passed: %v", err)
	}
	if _, err := h.transactions.StartStageOperation(ctx, claimed.RunID, stage.StageExecutionID, "second"); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresServerOperationPreservesAppendsAndRejectsChangedIntent(t *testing.T) {
	for _, scenario := range []string{"append", "edit", "output-conflict", "cancel"} {
		t.Run(scenario, func(t *testing.T) {
			h := newOperationHarness(t, true)
			ctx := t.Context()
			project, _ := h.artifacts.Project("operation-project")
			executor := operationCallFunc(func(ctx context.Context, ref string, call operations.Call) (stageresult.StageContentResult, error) {
				switch scenario {
				case "append":
					_, err := h.sitemap.Mutate(ctx, projectsitemap.Authority{OwnerID: "user-1", ProjectID: "operation-project", RunID: call.RunID, SourceID: "capture", Kind: "http", CanObserve: true}, "append", projectsitemap.Mutation{Operation: "observe", Observations: []projectsitemap.Observation{{URL: "https://example.test/concurrent", Method: "GET", Outcome: "response", Status: 200, At: "2026-10-29T00:00:00Z"}}})
					if err != nil {
						t.Fatal(err)
					}
				case "edit":
					selected, err := project.Read(ctx, h.seed)
					if err != nil {
						t.Fatal(err)
					}
					document, err := projectsitemap.Decode(selected.Payload.Data)
					if err != nil {
						t.Fatal(err)
					}
					_, err = h.sitemap.Mutate(ctx, projectsitemap.Authority{OwnerID: "user-1", ProjectID: "operation-project", SourceID: "owner", Kind: "manual"}, "edit", projectsitemap.Mutation{Operation: "annotate", ExpectedRevision: *h.seed.Revision, MemberIDs: []string{document.Groups[0].Members[0].ID}, Note: "keep this edit", Tags: []string{}})
					if err != nil {
						t.Fatal(err)
					}
				case "output-conflict":
					run, _ := h.artifacts.Run(call.RunID)
					if _, err := run.Write(ctx, call.Outputs["sitemap"], artifacts.Payload{MediaType: "text/plain", Data: []byte("already exists")}, nil); err != nil {
						t.Fatal(err)
					}
				case "cancel":
					if _, err := h.store.RequestRunCancellation(ctx, call.RunID, runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: time.Now()}); err != nil {
						t.Fatal(err)
					}
				}
				return h.registry.Run(ctx, ref, call)
			})
			s := h.scheduler(t, h.transactions, executor)
			_, err := s.RunOnce(ctx)
			if scenario == "cancel" {
				if err != nil && !errors.Is(err, runstore.ErrConflict) {
					t.Fatal(err)
				}
				if _, err := s.RunOnce(ctx); err != nil {
					t.Fatal(err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			run, err := h.store.GetRun(ctx, "operation-run")
			if err != nil {
				t.Fatal(err)
			}
			want := runstore.RunFailed
			if scenario == "append" {
				want = runstore.RunSucceeded
			}
			if scenario == "cancel" {
				want = runstore.RunCancelled
			}
			if run.State != want {
				t.Fatalf("state=%s, want %s", run.State, want)
			}
			current, err := project.Read(ctx, contracts.ArtifactRef{Namespace: "sitemap", Name: "map"})
			if err != nil {
				t.Fatal(err)
			}
			document, err := projectsitemap.Decode(current.Payload.Data)
			if err != nil {
				t.Fatal(err)
			}
			if scenario == "append" {
				found := false
				for _, group := range document.Groups {
					for _, member := range group.Members {
						found = found || member.Path == "/concurrent"
					}
				}
				if !found {
					t.Fatal("append lost")
				}
			} else if len(document.Rules) != 0 {
				t.Fatal("failed operation committed source changes")
			}
			if scenario == "edit" {
				found := false
				for _, group := range document.Groups {
					for _, member := range group.Members {
						for _, annotation := range member.Annotations {
							found = found || annotation.Note == "keep this edit"
						}
					}
				}
				if !found {
					t.Fatal("owner edit lost")
				}
			}
			if scenario == "edit" || scenario == "output-conflict" {
				stages, _ := h.store.ListStageExecutions(ctx, run.RunID)
				if stages[0].Termination == nil || stages[0].Termination.Retryable {
					t.Fatal("permanent conflict became an automatic retry")
				}
			}
		})
	}
}
