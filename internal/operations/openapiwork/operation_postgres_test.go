package openapiwork

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/operations"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func ownedOperation(t *testing.T) (*Module, operations.Call, *pgxpool.Pool) {
	t.Helper()
	ctx := t.Context()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{Configure: func(cfg *pgxpool.Config) { cfg.MaxConns = 1 }})
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	root := configtest.CopyWithPolicies(t, "../../config/testdata/valid")
	data, err := os.ReadFile("../../config/testdata/openapi-prepared-work/workflows/prepare.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "workflows", "prepare.yaml"), data, 0600); err != nil {
		t.Fatal(err)
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := catalog.ResolveRunWorkflow(ctx, "openapi-prepare-work@1", nil)
	if err != nil {
		t.Fatal(err)
	}
	project := "prepared-project"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: "owner", Kind: projectstore.KindProject, Name: "Prepared work", IdempotencyKey: "create", RequestDigest: "sha256:" + strings.Repeat("a", 64)}); err != nil {
		t.Fatal(err)
	}
	parameters := map[string]string{"checking_workflow": "check", "selection": "operations", "max_items": "256"}
	encoded, _ := json.Marshal(workflow)
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	if _, err := runstoretest.CreateRun(ctx, store, runstore.CreateRunParams{RunID: "producer", OwnerID: "owner", ProjectID: &project, LaneClass: resolved.LaneClassTool, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: encoded, Parameters: parameters, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	scope, _ := artifacts.RunScope("producer")
	source, err := service.Create(ctx, scope, contracts.ArtifactRef{Namespace: "inputs", Name: "openapi"}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"openapi":"3.1.0","paths":{"/items":{"get":{}}}}`)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(ctx, "producer", runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ClaimRunnableRun(ctx, "claim", time.Minute, nil, resolved.LaneClassTool); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(ctx, "producer", runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	stage, _ := json.Marshal(workflow.Stages[workflow.EntryStage])
	if _, err := store.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{StageExecutionID: "prepare-stage", RunID: "producer", StageName: workflow.EntryStage, Attempt: 1, LaneClass: resolved.LaneClassTool, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stage, StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{Parameters: parameters, Artifacts: map[string]runstore.PinnedContextArtifact{"openapi": {Required: true, Artifact: &source.Ref}, "request_values": {Required: false}}}}); err != nil {
		t.Fatal(err)
	}
	if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		s := runstore.NewPostgresTxStore(tx, artifactpgtest.NewStorage())
		if _, err := s.LockRunState(ctx, "producer", runstore.RunRunning); err != nil {
			return err
		}
		if _, err := s.LockStageState(ctx, "producer", "prepare-stage", runstore.StagePreparing); err != nil {
			return err
		}
		if _, err := s.MarkStageAdmitted(ctx, "prepare-stage"); err != nil {
			return err
		}
		_, err := s.StartStageOperation(ctx, "producer", "prepare-stage", "claim")
		return err
	}); err != nil {
		t.Fatal(err)
	}
	return New(pool, artifactpgtest.NewStorage()), operations.Call{RunID: "producer", StageExecutionID: "prepare-stage", ClaimID: "claim", Arguments: map[string]any{"workflow": "check", "selection": "operations", "max_items": "256", "scanner": "", "server": "", "operations": "", "test_parameters": "", "credential_id": ""}, Inputs: map[string]contracts.ArtifactRef{"openapi": source.Ref}, Outputs: map[string]contracts.ArtifactRef{"work": {Namespace: "prepared-work", Name: "work"}}}, pool
}

func TestPostgresPreparedWorkPublicationReplayAndCollision(t *testing.T) {
	for _, scenario := range []string{"lost-ack", "collision"} {
		t.Run(scenario, func(t *testing.T) {
			m, call, pool := ownedOperation(t)
			ctx := t.Context()
			service := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
			scope, _ := artifacts.RunScope(call.RunID)
			store, _ := service.Run(call.RunID)
			if scenario == "collision" {
				if _, err := service.Create(ctx, scope, call.Outputs["work"], artifacts.Payload{MediaType: "application/json", Data: []byte(`{"other":true}`)}); err != nil {
					t.Fatal(err)
				}
				if _, err := m.Run(ctx, call); err == nil {
					t.Fatal("overwrote a conflicting output")
				}
				return
			}
			first, err := m.Run(ctx, call)
			if err != nil {
				t.Fatal(err)
			}
			before, err := store.Read(ctx, first.Artifacts["work"])
			if err != nil {
				t.Fatal(err)
			}
			second, err := m.Run(ctx, call)
			if err != nil {
				t.Fatal(err)
			}
			if !first.Artifacts["work"].SameExact(second.Artifacts["work"]) {
				t.Fatal("replay created a different result")
			}
			after, err := store.Read(ctx, second.Artifacts["work"])
			if err != nil || string(before.Payload.Data) != string(after.Payload.Data) {
				t.Fatal("replay changed exact materials")
			}
		})
	}
}

func TestPostgresPreparedWorkRejectsClosedAuthority(t *testing.T) {
	for _, scenario := range []string{"claim-lost", "cancelled", "project-deleting", "changed-arguments", "changed-input"} {
		t.Run(scenario, func(t *testing.T) {
			m, call, pool := ownedOperation(t)
			ctx := t.Context()
			runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
			switch scenario {
			case "claim-lost":
				if err := runs.ReleaseRunClaim(ctx, call.RunID, call.ClaimID); err != nil {
					t.Fatal(err)
				}
			case "cancelled":
				if _, err := runs.RequestRunCancellation(ctx, call.RunID, runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: time.Now()}); err != nil {
					t.Fatal(err)
				}
			case "project-deleting":
				if _, _, err := projectstore.NewPostgresStore(pool).BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: "owner", ProjectID: "prepared-project", ExpectedRevision: 1}); err != nil {
					t.Fatal(err)
				}
			case "changed-arguments":
				call.Arguments["max_items"] = "1"
			case "changed-input":
				ref := call.Inputs["openapi"]
				ref.Name = "foreign"
				call.Inputs["openapi"] = ref
			}
			if _, err := m.Run(ctx, call); err == nil {
				t.Fatal("closed or changed authority published")
			}
			store, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(call.RunID)
			if _, err := store.Metadata(ctx, call.Outputs["work"]); !errors.Is(err, artifacts.ErrArtifactNotFound) {
				t.Fatalf("published root after fence rejection: %v", err)
			}
		})
	}
}

func TestPreparedWorkRejectsBadBindingsBeforeStorage(t *testing.T) {
	for _, args := range []map[string]any{{"workflow": "check", "selection": "operations", "max_items": "257"}, {"workflow": "check", "selection": "operations", "max_items": "+1"}, {"workflow": "check", "selection": "unknown", "max_items": "1"}} {
		revision := "exact"
		_, err := New(nil, nil).Run(t.Context(), operations.Call{Arguments: args, Inputs: map[string]contracts.ArtifactRef{"openapi": {Namespace: "inputs", Name: "openapi", Revision: &revision}}, Outputs: map[string]contracts.ArtifactRef{"work": {Namespace: "prepared-work", Name: "work"}}})
		var failure *operations.Failure
		if !errors.As(err, &failure) || failure.Code != "invalid_request" || failure.Retryable {
			t.Fatalf("invalid binding reached storage: %v", err)
		}
	}
}
