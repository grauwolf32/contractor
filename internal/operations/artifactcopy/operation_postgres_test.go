package artifactcopy

import (
	"bytes"
	"context"
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
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

var sourceBytes = append([]byte("é😀\r\n \t\n"), 0, 255)

func ownedCopy(t *testing.T, project bool) (*Module, operations.Call, *pgxpool.Pool, artifacts.ScopedStore) {
	t.Helper()
	ctx := t.Context()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{Configure: func(c *pgxpool.Config) { c.MaxConns = 1 }})
	storage := artifactpgtest.NewStorage()
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	root := configtest.CopyWithPolicies(t, "../../config/testdata/valid")
	data, err := os.ReadFile("testdata/copy.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "workflows", "copy-operation.yaml"), data, 0600); err != nil {
		t.Fatal(err)
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := catalog.ResolveRunWorkflow(ctx, "copy-operation-test@1", nil)
	if err != nil {
		t.Fatal(err)
	}
	var projectID *string
	if project {
		id := "copy-project"
		if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: id, OwnerID: "owner", Kind: projectstore.KindProject, Name: "Copy fixture", IdempotencyKey: "create", RequestDigest: "sha256:" + strings.Repeat("a", 64)}); err != nil {
			t.Fatal(err)
		}
		projectID = &id
	}
	store := runstore.NewPostgresStore(pool, storage)
	encoded, _ := json.Marshal(workflow)
	if _, err := runstoretest.CreateRun(ctx, store, runstore.CreateRunParams{RunID: "copy-run", OwnerID: "owner", ProjectID: projectID, LaneClass: resolved.LaneClassTool, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: encoded, Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(storage.Repository(pool))
	scope, _ := artifacts.RunScope("copy-run")
	source, err := service.Create(ctx, scope, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "application/octet-stream", Data: sourceBytes})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(ctx, "copy-run", runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ClaimRunnableRun(ctx, "claim", time.Minute, nil, resolved.LaneClassTool); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(ctx, "copy-run", runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	stage, _ := json.Marshal(workflow.Stages[workflow.EntryStage])
	if _, err := store.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{StageExecutionID: "copy-stage", RunID: "copy-run", StageName: workflow.EntryStage, Attempt: 1, LaneClass: resolved.LaneClassTool, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stage, StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{"source": {Required: true, Artifact: &source.Ref}}}}); err != nil {
		t.Fatal(err)
	}
	if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		s := runstore.NewPostgresTxStore(tx, storage)
		if _, err := s.LockRunState(ctx, "copy-run", runstore.RunRunning); err != nil {
			return err
		}
		if _, err := s.LockStageState(ctx, "copy-run", "copy-stage", runstore.StagePreparing); err != nil {
			return err
		}
		if _, err := s.MarkStageAdmitted(ctx, "copy-stage"); err != nil {
			return err
		}
		_, err := s.StartStageOperation(ctx, "copy-run", "copy-stage", "claim")
		return err
	}); err != nil {
		t.Fatal(err)
	}
	scoped, _ := service.Run("copy-run")
	return New(pool, storage), operations.Call{RunID: "copy-run", StageExecutionID: "copy-stage", ClaimID: "claim", Arguments: map[string]any{}, Inputs: map[string]contracts.ArtifactRef{"source": source.Ref}, Outputs: map[string]contracts.ArtifactRef{"copied": {Namespace: "builder", Name: "copied"}}}, pool, scoped
}

func dispatch(t *testing.T, m *Module, ctx context.Context, call operations.Call) (stageresult.StageContentResult, error) {
	t.Helper()
	registry, err := operations.NewRegistry(m)
	if err != nil {
		t.Fatal(err)
	}
	return registry.Run(ctx, m.Ref(), call)
}

func TestPostgresCopyPreservesPinnedBytesAndReplaysOneRevision(t *testing.T) {
	for _, project := range []bool{false, true} {
		t.Run(map[bool]string{false: "ordinary-run", true: "project-run"}[project], func(t *testing.T) {
			m, call, pool, store := ownedCopy(t, project)
			ctx := t.Context()
			first, err := dispatch(t, m, ctx, call)
			if err != nil {
				t.Fatal(err)
			}
			// A lost acknowledgement must replay the original source, even after its head advances.
			if _, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("newer source")}, call.Inputs["source"].Revision); err != nil {
				t.Fatal(err)
			}
			second, err := dispatch(t, m, ctx, call)
			if err != nil || !first.Artifacts["copied"].SameExact(second.Artifacts["copied"]) {
				t.Fatalf("copy replay changed its exact receipt: %+v %v", second, err)
			}
			metadata, err := store.Metadata(ctx, second.Artifacts["copied"])
			if err != nil {
				t.Fatal(err)
			}
			read, err := store.ReadExact(ctx, metadata.Exact())
			if err != nil || !bytes.Equal(read.Payload.Data, sourceBytes) || metadata.MediaType != "application/octet-stream" {
				t.Fatalf("copy changed source bytes or media type: %v", err)
			}
			var revisions int
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind='run' AND scope_id='copy-run' AND namespace='builder' AND name='copied'`).Scan(&revisions); err != nil || revisions != 1 {
				t.Fatalf("copy replay appended revisions: %d %v", revisions, err)
			}
		})
	}
}

func TestPostgresCopyRejectsChangedAuthorityAndLifecycleBeforePublication(t *testing.T) {
	for _, scenario := range []string{"foreign-source", "foreign-output", "claim-lost", "project-deleting", "cancelled", "collision"} {
		t.Run(scenario, func(t *testing.T) {
			m, call, pool, store := ownedCopy(t, true)
			ctx := t.Context()
			switch scenario {
			case "foreign-source":
				ref := call.Inputs["source"]
				ref.Name = "other"
				call.Inputs["source"] = ref
			case "foreign-output":
				call.Outputs["copied"] = contracts.ArtifactRef{Namespace: "foreign", Name: "copied"}
			case "claim-lost":
				if err := runstore.NewPostgresStore(pool, m.storage).ReleaseRunClaim(ctx, call.RunID, call.ClaimID); err != nil {
					t.Fatal(err)
				}
			case "project-deleting":
				if _, _, err := projectstore.NewPostgresStore(pool).BeginDeletion(ctx, projectstore.BeginDeletionParams{ProjectID: "copy-project", OwnerID: "owner", ExpectedRevision: 1}); err != nil {
					t.Fatal(err)
				}
			case "cancelled":
				cancelled, cancel := context.WithCancel(ctx)
				cancel()
				if _, err := dispatch(t, m, cancelled, call); !errors.Is(err, context.Canceled) {
					t.Fatalf("cancelled copy returned %v", err)
				}
			case "collision":
				if _, err := store.Write(ctx, call.Outputs["copied"], artifacts.Payload{MediaType: "text/plain", Data: []byte("existing")}, nil); err != nil {
					t.Fatal(err)
				}
			}
			if scenario != "cancelled" {
				_, err := dispatch(t, m, ctx, call)
				if err == nil {
					t.Fatal("copy accepted changed authority or lifecycle")
				}
				if failure, ok := err.(*operations.Failure); ok && failure.Retryable {
					t.Fatalf("permanent rejection became retryable: %+v", failure)
				}
			}
			var revisions int
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind='run' AND scope_id='copy-run' AND namespace='builder' AND name='copied'`).Scan(&revisions); err != nil {
				t.Fatal(err)
			}
			want := 0
			if scenario == "collision" {
				want = 1
			}
			if revisions != want {
				t.Fatalf("rejected copy wrote a revision: %d", revisions)
			}
		})
	}
}
