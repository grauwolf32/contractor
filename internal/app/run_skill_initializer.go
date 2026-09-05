package app

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// runSkillInitializer is shared by the public create path and Scheduler
// recovery. It never resolves a logical owner binding: only the exact source
// outcome already committed on WorkflowRun is accepted.
type runSkillInitializer struct {
	pool *pgxpool.Pool
}

func (i *runSkillInitializer) InitializeRunSkills(
	ctx context.Context,
	runID string,
) (runstore.WorkflowRun, error) {
	if i == nil || i.pool == nil {
		return runstore.WorkflowRun{}, errors.New("Run Skill initializer is not configured")
	}
	var result runstore.WorkflowRun
	err := persistencepostgres.InTx(ctx, i.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		store := runstore.NewPostgresStore(tx)
		run, err := store.LockRunSkillInitialization(ctx, runID)
		if err != nil {
			return err
		}
		result = run
		if run.State != runstore.RunInitializing ||
			run.StateReason.Code != runstore.SkillInitializationPendingReason {
			return nil
		}

		workflow, err := config.DecodeResolvedWorkflowSnapshot(run.WorkflowSnapshot)
		if err != nil {
			return invalidPinnedSkillSnapshot()
		}
		refs, err := config.WorkflowSkillRefs(workflow)
		if err != nil || !sameSelectedSkillNames(refs, run.SkillSnapshot) {
			return invalidPinnedSkillSnapshot()
		}
		service := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		catalog, err := agentskills.NewCatalog(service)
		if err != nil {
			return err
		}
		initialized, err := catalog.InitializeRun(
			ctx, run.OwnerID, run.RunID, run.SkillSnapshot, config.WorkflowSkillSets(workflow),
		)
		if err != nil {
			return err
		}
		if err := store.CompleteRunSkillInitialization(ctx, run.RunID, initialized); err != nil {
			return err
		}
		result, err = store.TransitionRun(
			ctx, run.RunID, runstore.RunInitializing, runstore.RunRunning,
			runstore.Reason{Code: "initialized"},
		)
		return err
	})
	if err == nil {
		return result, nil
	}

	var skillErr *agentskills.RunSkillError
	if !errors.As(err, &skillErr) || skillErr.Retryable {
		current, loadErr := runstore.NewPostgresStore(i.pool).GetRun(ctx, runID)
		if loadErr == nil {
			result = current
		}
		return result, err
	}
	failed, failErr := i.failPermanent(ctx, runID, skillErr)
	if failErr != nil {
		return result, errors.Join(err, failErr)
	}
	return failed, nil
}

func (i *runSkillInitializer) failPermanent(
	ctx context.Context,
	runID string,
	failure *agentskills.RunSkillError,
) (runstore.WorkflowRun, error) {
	var result runstore.WorkflowRun
	err := persistencepostgres.InTx(ctx, i.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		store := runstore.NewPostgresStore(tx)
		current, err := store.LockRunSkillInitialization(ctx, runID)
		if err != nil {
			return err
		}
		result = current
		if current.State != runstore.RunInitializing ||
			current.StateReason.Code != runstore.SkillInitializationPendingReason {
			return nil
		}
		message := ""
		if failure.Name != "" {
			message = "skills/" + failure.Name
		}
		result, err = store.TransitionRun(
			ctx, runID, runstore.RunInitializing, runstore.RunFailed,
			runstore.Reason{Code: failure.Code, Message: message},
		)
		return err
	})
	if err != nil {
		return runstore.WorkflowRun{}, fmt.Errorf("terminate invalid Run Skill initialization: %w", err)
	}
	return result, nil
}

func invalidPinnedSkillSnapshot() error {
	return &agentskills.RunSkillError{
		Code: agentskills.CodeArchiveInvalid, Retryable: false,
	}
}

func sameSelectedSkillNames(
	refs []contracts.ArtifactRef,
	selected []contracts.RunSkillSnapshot,
) bool {
	if len(refs) != len(selected) {
		return false
	}
	for index := range refs {
		if refs[index].Namespace != contracts.AgentSkillNamespace ||
			refs[index].Name != selected[index].Name {
			return false
		}
	}
	return true
}
