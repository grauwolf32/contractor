package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5/pgxpool"
)

// postgresVerifiers are the production Planner inspector and Scheduler
// resolver over one PostgreSQL ArtifactStore.
type postgresVerifiers struct {
	inspector planner.ArtifactInspector
	scheduler *Scheduler
}

func newPostgresVerifiers(t *testing.T, pool *pgxpool.Pool) postgresVerifiers {
	t.Helper()
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	inspector, err := planner.NewArtifactServiceInspector(service)
	if err != nil {
		t.Fatal(err)
	}
	resolver, err := NewArtifactServiceResolver(service)
	if err != nil {
		t.Fatal(err)
	}
	return postgresVerifiers{inspector: inspector, scheduler: &Scheduler{artifacts: resolver}}
}

// Against PostgreSQL, the Planner and the Scheduler verify the same exact
// result reference with the same class: a revision the RunScope does not hold
// is a non-retryable contract violation, and a store that cannot answer, here
// because a lock outlasts lock_timeout, leaves the candidate retryable
// (PLN-R118, SCH-R177).
func TestPostgresResultArtifactVerificationClassifiesMissingAndUnavailable(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	workflowSnapshot := loadSchedulerWorkflow(t)
	artifactService := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	runStore := createSchedulerRun(t, ctx, runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()), artifactService, workflowSnapshot)
	written, err := runStore.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "copied"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("copied\n")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := executableWorkflow{workflow: workflowSnapshot}.selectStage(workflowSnapshot.EntryStage)
	if err != nil {
		t.Fatal(err)
	}
	candidate := func(ref contracts.ArtifactRef) stageresult.StageContentResult {
		return stageresult.StageContentResult{
			APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: "copied",
			Artifacts: map[string]contracts.ArtifactRef{"copied": ref},
		}
	}
	verify := func(t *testing.T, verifiers postgresVerifiers, ref contracts.ArtifactRef) (*planner.Error, error) {
		t.Helper()
		plannerErr := planner.ValidateCandidate(
			ctx, "run-1", workflow.stage.Result.Artifacts, candidate(ref), verifiers.inspector,
		)
		return plannerErr, verifiers.scheduler.validateCandidate(ctx, "run-1", workflow, candidate(ref))
	}
	verifiers := newPostgresVerifiers(t, pool)

	t.Run("stored exact revision", func(t *testing.T) {
		plannerErr, schedulerErr := verify(t, verifiers, written.Ref)
		if plannerErr != nil || schedulerErr != nil {
			t.Fatalf("stored revision = (%v, %v)", plannerErr, schedulerErr)
		}
	})
	missingRevision := written.Ref.Clone()
	*missingRevision.Revision = "missing-revision"
	for name, ref := range map[string]contracts.ArtifactRef{
		"missing revision": missingRevision,
		"missing binding":  exactRef("builder", "absent", *written.Ref.Revision),
	} {
		t.Run(name, func(t *testing.T) {
			plannerErr, schedulerErr := verify(t, verifiers, ref)
			if plannerErr == nil || plannerErr.Code != "result_contract_violation" || plannerErr.Retryable ||
				!errors.Is(plannerErr, artifacts.ErrArtifactNotFound) {
				t.Fatalf("Planner verdict = %+v", plannerErr)
			}
			var unavailable *candidateVerificationError
			if !errors.Is(schedulerErr, artifacts.ErrArtifactNotFound) || errors.As(schedulerErr, &unavailable) {
				t.Fatalf("Scheduler verdict = %v, want a rejected reference", schedulerErr)
			}
		})
	}

	t.Run("lock timeout", func(t *testing.T) {
		config := pool.Config()
		config.ConnConfig.RuntimeParams["lock_timeout"] = "50ms"
		bounded, err := pgxpool.NewWithConfig(ctx, config)
		if err != nil {
			t.Fatal(err)
		}
		defer bounded.Close()
		holder, err := pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = holder.Rollback(context.WithoutCancel(ctx)) }()
		if _, err := holder.Exec(ctx, `LOCK TABLE artifact_bindings IN ACCESS EXCLUSIVE MODE`); err != nil {
			t.Fatal(err)
		}
		plannerErr, schedulerErr := verify(t, newPostgresVerifiers(t, bounded), written.Ref)
		if plannerErr == nil || plannerErr.Code != "result_artifact_unavailable" || !plannerErr.Retryable ||
			persistencepostgres.SQLState(plannerErr) != persistencepostgres.SQLStateLockNotAvailable {
			t.Fatalf("Planner verdict = %+v (%v)", plannerErr, errors.Unwrap(plannerErr))
		}
		var unavailable *candidateVerificationError
		if !errors.As(schedulerErr, &unavailable) ||
			persistencepostgres.SQLState(schedulerErr) != persistencepostgres.SQLStateLockNotAvailable {
			t.Fatalf("Scheduler verdict = %v, want a retryable verification failure", schedulerErr)
		}
	})
}
