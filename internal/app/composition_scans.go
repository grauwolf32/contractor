package app

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/randomid"
	"github.com/grauwolf32/contractor/internal/runrepeat"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/scancontroller"
	"github.com/grauwolf32/contractor/internal/scanservice"
	"github.com/grauwolf32/contractor/internal/scanstore"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func configureSweeps(pool *pgxpool.Pool, manager *workflowconfig.Manager, credentialSet runCreationCredentials, runs *runservice.Service, notify interface {
	Wake()
	Cancel(string)
}, logger *slog.Logger) (*scanservice.Service, *scancontroller.Controller, error) {
	holder, err := randomid.New("sweep_controller_")
	if err != nil {
		return nil, nil, err
	}
	u := postgresPublicUnitOfWork{pool: pool, transactionLLMCredentials: credentialSet.transactionLookup}
	admit := u.sweepAdmitter(runs, notify)
	controller, err := scancontroller.New(scancontroller.Options{Pool: pool, HolderID: holder, Admit: admit, Now: time.Now, Logger: logger, CancelRun: func(ctx context.Context, id, owner string) (runstore.WorkflowRun, error) {
		run, err := runstore.NewPostgresStore(pool).RequestRunCancellation(ctx, id, runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: time.Now(), RequestedBy: &owner})
		if err == nil {
			if run.State == runstore.RunCancelling {
				notify.Cancel(run.RunID)
			}
			notify.Wake()
		}
		return run, err
	}})
	if err != nil {
		return nil, nil, err
	}
	service, err := scanservice.New(scanservice.Options{Pool: pool, Workflows: manager, Credentials: credentialSet.provider, Drainer: controller, NewID: randomid.New, Logger: logger,
		PinRuntime: func(ctx context.Context, tx pgx.Tx) (runtimeconfig.RunSnapshot, error) {
			lookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, credentialSet.transactionLookup)
			if err != nil {
				return runtimeconfig.RunSnapshot{}, err
			}
			return runstore.NewRunCreationPostgresStore(tx, lookup, credentials.NewRuntimeCredentialRepository(tx)).PinRuntimeLabels(ctx, nil, true)
		},
	})
	if err != nil {
		return nil, nil, err
	}
	return service, controller, nil
}

func (u postgresPublicUnitOfWork) sweepAdmitter(runs *runservice.Service, notify interface{ Wake() }) func(context.Context, persistencepostgres.ControllerClaim, scanstore.Snapshot, scanstore.Member) error {
	return func(ctx context.Context, claim persistencepostgres.ControllerClaim, snapshot scanstore.Snapshot, m scanstore.Member) error {
		var child scanstore.Child
		if strictjson.Decode(m.Child, &child) != nil {
			return scanstore.ErrInvalid
		}
		params := runservice.PublicCreateParams{
			OwnerID: snapshot.OwnerID, ProjectID: &m.ProjectID, Workflow: child.Workflow,
			PinnedWorkflow: &snapshot.Workflow, ExpectedWorkflowSHA256: snapshot.WorkflowDigest,
			CapturedTargetURL: m.Target, PinnedRuntimeConfig: &snapshot.Runtime, RuntimeLabels: snapshot.Runtime.ExplicitLabels(),
			Parameters: child.Parameters, Inputs: child.Inputs, MetadataLabels: child.Labels,
			IdempotencyKey: m.Key, RequestDigest: m.Digest, NewRunID: func() (string, error) { return randomid.New("run_") },
		}
		digest, err := runservice.PublicRequestDigest(params)
		if err != nil || digest != m.Digest {
			return scanstore.ErrInvalid
		}
		if snapshot.Rulepack != nil {
			original, ok := params.Inputs["rulepack"]
			if !ok || !original.SameExact(snapshot.Rulepack.Source.Ref) {
				return scanstore.ErrInvalid
			}
			params.Inputs = map[string]contracts.ArtifactRef{"rulepack": snapshot.Rulepack.Retained}
			params.OriginalInputSources = map[string]runrepeat.InputSource{"rulepack": {SourceScope: artifacts.ScopeUser, Artifact: snapshot.Rulepack.Source.Ref}}
		}
		transaction := func(ctx context.Context, fn func(runservice.PublicRunWriter, *artifacts.Service) error) error {
			return u.do(ctx, func(tx pgx.Tx, writer runservice.PublicRunWriter, a *artifacts.Service) error {
				store := scanstore.New(tx)
				if err := store.Lock(ctx, claim); err != nil {
					return err
				}
				parent, err := store.Get(ctx, snapshot.OwnerID, claim.SubjectID)
				if err != nil {
					return err
				}
				if parent.State != "active" {
					return scanstore.ErrConflict
				}
				if err := fn(writer, a); err != nil {
					return err
				}
				run, found, err := runstore.NewPostgresStore(tx).LookupRunIdempotency(ctx, snapshot.OwnerID, m.Key, m.Digest)
				if err != nil {
					return err
				}
				if !found {
					return fmt.Errorf("child Run transaction returned without a Run")
				}
				return store.Record(ctx, claim, m.Ordinal, "admitted", "", &run.RunID, runstore.RunLifecycleTerminal.Includes(run.State), 2)
			})
		}
		result, err := runs.CreatePublicIn(ctx, params, transaction)
		if errors.Is(err, contentdigest.ErrPinnedSelectionChanged) {
			return apierror.New(apierror.ConfigurationDrift, "Sweep pinned configuration changed")
		}
		if err != nil {
			return err
		}
		if result.Replayed {
			// Ordinary replay is deliberately before current catalog/source reads.
			// Its Run already exists; joining the parent lock only binds that fact.
			if err := persistencepostgres.InTxWithRetry(ctx, u.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				return scanstore.New(tx).Record(ctx, claim, m.Ordinal, "admitted", "", &result.Run.RunID, runstore.RunLifecycleTerminal.Includes(result.Run.State), 2)
			}); err != nil {
				return err
			}
		}
		notify.Wake()
		return nil
	}
}
