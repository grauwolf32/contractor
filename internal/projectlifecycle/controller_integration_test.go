package projectlifecycle

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestProjectDeletionCancelsDrainsPurgesAndRetainsSharedResources(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	runs := runstore.NewPostgresStore(pool)
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))

	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-delete", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Disposable workspace", Description: "lifecycle integration",
		IdempotencyKey: "create-project-delete",
		RequestDigest:  "sha256:" + strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO runtime_credentials (
    credential_id, credential_kind, encryption_schema_version, key_id,
    nonce, ciphertext, created_by, created_at
) VALUES (
    'project-origin', 'http-origin-bearer@1', 'contractor.runtime-credentials/v1', $1,
    decode(repeat('00', 12), 'hex'), decode(repeat('00', 17), 'hex'),
    'user-1', clock_timestamp()
)`, "sha256:"+strings.Repeat("b", 64)); err != nil {
		t.Fatal(err)
	}
	project, err = projects.Update(ctx, projectstore.UpdateParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
		Name: project.Name, Description: project.Description,
		HTTPTarget: &contracts.HTTPOriginTargetRef{
			URL: "https://app.example.test/api",
			Credential: &contracts.RuntimeCredentialRefV2{
				CredentialID: "project-origin", Kind: contracts.RuntimeCredentialOriginBearer,
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	userStore, err := artifactService.User(project.OwnerID)
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifactService.Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	sharedPayload := artifacts.Payload{MediaType: "text/plain", Data: []byte("shared content")}
	userSource, err := userStore.Write(ctx, artifacts.ArtifactRef{
		Namespace: "sources", Name: "shared",
	}, sharedPayload, nil)
	if err != nil {
		t.Fatal(err)
	}
	userSkill, err := userStore.Write(ctx, artifacts.ArtifactRef{
		Namespace: "skills", Name: "review",
	}, artifacts.Payload{MediaType: "application/zip", Data: []byte("skill archive")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	projectSource, err := projectArtifacts.Write(ctx, artifacts.ArtifactRef{
		Namespace: "sources", Name: "shared",
	}, sharedPayload, nil)
	if err != nil {
		t.Fatal(err)
	}

	projectID := project.ProjectID
	active := createProjectRun(t, ctx, runs, "run-active", projectID)
	active, err = runs.TransitionRun(
		ctx, active.RunID, runstore.RunInitializing, runstore.RunRunning,
		runstore.Reason{Code: "initialized"},
	)
	if err != nil {
		t.Fatal(err)
	}
	execution, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: "stage-active", RunID: active.RunID, StageName: "build", Attempt: 1,
		StageSpecSchemaVersion:    contracts.APIVersion,
		StageSpecSnapshot:         json.RawMessage(`{"objective":"build"}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO stage_allocations (
    allocation_id, stage_execution_id, logical_agent_name, namespace,
    agent_template_ref, worker_runtime_ref, runtime_agent_instance_id
) VALUES ('allocation-active', $1, 'builder', 'builder', '{}'::jsonb, '{}'::jsonb, 'agent-instance')`,
		execution.StageExecutionID,
	); err != nil {
		t.Fatal(err)
	}

	completed := createProjectRun(t, ctx, runs, "run-completed", projectID)
	if err := artifactService.PinExact(
		ctx, completed.RunID, mustUserScope(t, project.OwnerID), userSkill.Ref,
		artifacts.PinRunInput, "run-completed-skill",
	); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(
		ctx, completed.RunID, runstore.RunInitializing, runstore.RunFailed,
		runstore.Reason{Code: "fixture_failed"},
	); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.UpdateOwnerQueueControl(ctx, runstore.UpdateOwnerQueueControlParams{
		OwnerID: project.OwnerID, ExpectedRevision: 0, Paused: true,
	}); err != nil {
		t.Fatal(err)
	}

	deleting, accepted, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
	})
	if err != nil || !accepted || deleting.Lifecycle != projectstore.LifecycleDeleting {
		t.Fatalf("begin deletion = (%+v, %t, %v)", deleting, accepted, err)
	}
	if _, err := projects.Update(ctx, projectstore.UpdateParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: deleting.Revision,
		Name: "fenced", Description: "fenced",
	}); !errors.Is(err, projectstore.ErrDeleting) {
		t.Fatalf("metadata fence error = %v", err)
	}
	if _, err := projectArtifacts.Write(ctx, artifacts.ArtifactRef{
		Namespace: "sources", Name: "late",
	}, sharedPayload, nil); !errors.Is(err, artifacts.ErrScopeDeleting) {
		t.Fatalf("Artifact fence error = %v", err)
	}
	if _, err := runs.CreateRun(ctx, projectRunParams("run-late", projectID)); !errors.Is(err, runstore.ErrProjectDeleting) {
		t.Fatalf("Run fence error = %v", err)
	}
	if _, _, err := runs.CreateRunIdempotent(ctx, runstore.CreateRunIdempotentParams{
		CreateRunParams: projectRunParams("run-late-idempotent", projectID),
		IdempotencyKey:  "late-project-run",
		RequestDigest:   "sha256:" + strings.Repeat("c", 64),
	}); !errors.Is(err, runstore.ErrProjectDeleting) {
		t.Fatalf("idempotent Run fence error = %v", err)
	}

	notifier := &recordingNotifier{}
	first := newTestController(t, pool, runs, notifier, "first")
	worked, err := first.RunOnce(ctx)
	if err != nil || !worked || len(notifier.runIDs) != 1 || notifier.runIDs[0] != active.RunID {
		t.Fatalf("cancellation iteration = (%t, %v), notifications=%v", worked, err, notifier.runIDs)
	}
	cancelling, err := runs.GetRun(ctx, active.RunID)
	if err != nil || cancelling.State != runstore.RunCancelling {
		t.Fatalf("cancelling Run = (%+v, %v)", cancelling, err)
	}
	if _, err := runs.TransitionRun(
		ctx, active.RunID, runstore.RunCancelling, runstore.RunCancelled,
		runstore.Reason{Code: runstore.CancellationUserRequested},
	); err != nil {
		t.Fatal(err)
	}

	// A fresh controller resumes only from the durable Project phase.
	restarted := newTestController(t, pool, runs, notifier, "restarted")
	worked, err = restarted.RunOnce(ctx)
	if err != nil || !worked {
		t.Fatalf("restart phase advance = (%t, %v)", worked, err)
	}
	current, err := projects.Get(ctx, project.OwnerID, project.ProjectID)
	if err != nil || current.Deletion == nil || current.Deletion.Phase != projectstore.DeletionDraining {
		t.Fatalf("draining Project = (%+v, %v)", current, err)
	}
	worked, err = restarted.RunOnce(ctx)
	if err != nil || worked {
		t.Fatalf("pending allocation drain = (%t, %v)", worked, err)
	}
	if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); err != nil {
		t.Fatalf("Project disappeared before release: %v", err)
	}
	if err := runs.MarkStageAllocationReleased(ctx, "allocation-active"); err != nil {
		t.Fatal(err)
	}
	expireDeletionClaim(t, ctx, pool, project.ProjectID)

	for iteration := 0; iteration < 12; iteration++ {
		_, err := restarted.RunOnce(ctx)
		if err != nil {
			t.Fatalf("cleanup iteration %d: %v", iteration, err)
		}
		if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); errors.Is(err, projectstore.ErrNotFound) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
	}
	if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); !errors.Is(err, projectstore.ErrNotFound) {
		t.Fatalf("deleted Project lookup error = %v", err)
	}

	for label, query := range map[string]string{
		"Project Runs":      `SELECT count(*) FROM workflow_runs WHERE project_id = 'project-delete'`,
		"Project scope":     `SELECT count(*) FROM artifact_scopes WHERE scope_kind = 'project' AND scope_id = 'project-delete'`,
		"Project revisions": `SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind = 'project' AND scope_id = 'project-delete'`,
		"Run pins":          `SELECT count(*) FROM artifact_pins WHERE run_id IN ('run-active', 'run-completed')`,
	} {
		var count int
		if err := pool.QueryRow(ctx, query).Scan(&count); err != nil || count != 0 {
			t.Fatalf("%s count = %d, error = %v", label, count, err)
		}
	}
	if _, err := userStore.Read(ctx, userSource.Ref); err != nil {
		t.Fatalf("shared User Artifact was removed: %v", err)
	}
	if _, err := userStore.Read(ctx, userSkill.Ref); err != nil {
		t.Fatalf("User Skill was removed: %v", err)
	}
	var credentialCount int
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM runtime_credentials WHERE credential_id = 'project-origin'`,
	).Scan(&credentialCount); err != nil || credentialCount != 1 {
		t.Fatalf("Runtime credential count = %d, error = %v", credentialCount, err)
	}
	queue, err := runs.GetOwnerQueueControl(ctx, project.OwnerID)
	if err != nil || !queue.Paused {
		t.Fatalf("paused Queue after cleanup = (%+v, %v)", queue, err)
	}
	var projectRevisionCount int
	if err := pool.QueryRow(ctx, `
SELECT count(*)
FROM artifact_binding_revisions
WHERE scope_kind = 'project' AND scope_id = 'project-delete'
  AND revision = $1`, *projectSource.Ref.Revision).Scan(&projectRevisionCount); err != nil || projectRevisionCount != 0 {
		t.Fatalf("Project Artifact revision count = %d, error = %v", projectRevisionCount, err)
	}
}

func createProjectRun(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	runID string,
	projectID string,
) runstore.WorkflowRun {
	t.Helper()
	run, err := store.CreateRun(ctx, projectRunParams(runID, projectID))
	if err != nil {
		t.Fatalf("create Project Run %q: %v", runID, err)
	}
	return run
}

func projectRunParams(runID, projectID string) runstore.CreateRunParams {
	return runstore.CreateRunParams{
		RunID: runID, OwnerID: "user-1", ProjectID: &projectID,
		WorkflowName: "artifact-copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot:      json.RawMessage(`{"ref":{"name":"artifact-copy","version":"1"}}`),
		Parameters:            map[string]string{},
		RuntimeConfig:         runtimeconfig.BuiltInRunSnapshot(),
	}
}

type recordingNotifier struct{ runIDs []string }

func (n *recordingNotifier) Cancel(runID string) { n.runIDs = append(n.runIDs, runID) }

func newTestController(
	t *testing.T,
	pool *pgxpool.Pool,
	runs *runstore.PostgresStore,
	notifier *recordingNotifier,
	id string,
) *Controller {
	t.Helper()
	sequence := 0
	controller, err := New(pool, runs, notifier, Options{
		PollInterval: time.Millisecond, ClaimDuration: time.Minute,
		OperationTimeout: 5 * time.Second,
		NewID: func(prefix string) (string, error) {
			sequence++
			return prefix + id + "-" + string(rune('a'+sequence)), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return controller
}

func expireDeletionClaim(t *testing.T, ctx context.Context, pool *pgxpool.Pool, projectID string) {
	t.Helper()
	if _, err := pool.Exec(ctx, `
UPDATE projects
SET deletion_claimed_at = clock_timestamp() - interval '2 seconds',
    deletion_claim_expires_at = clock_timestamp() - interval '1 second'
WHERE project_id = $1 AND deletion_claim_id IS NOT NULL`, projectID); err != nil {
		t.Fatal(err)
	}
}

func mustUserScope(t *testing.T, userID string) artifacts.Scope {
	t.Helper()
	scope, err := artifacts.UserScope(userID)
	if err != nil {
		t.Fatal(err)
	}
	return scope
}

func isolatedPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
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
	schema := "contractor_project_lifecycle_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
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
