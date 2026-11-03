package app

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditcontroller"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore/claimstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type auditServices struct {
	service    *auditservice.Service
	runs       *runservice.Service
	controller *auditcontroller.Controller
}

func configureAudits(
	cfg Config,
	pool *pgxpool.Pool,
	configurationManager *workflowconfig.Manager,
	credentialSet credentialServices,
	catalogs catalogServices,
	workflows workflowServices,
	logger *slog.Logger,
	storage *artifactpostgres.Storage,
) (auditServices, error) {
	auditService, err := auditservice.New(auditservice.Options{Storage: storage,
		Pool: pool, Profiles: configurationManager,
		TransactionLLMCredentials: credentialSet.transactionLookup,
		CredentialGuard:           credentialSet.lifecycle,
	})
	if err != nil {
		return auditServices{}, fmt.Errorf("configure Audit service: %w", err)
	}
	runCreationService, err := configureRunCreation(pool, configurationManager, runCreationCredentials{
		provider: credentialSet.provider, guard: credentialSet.lifecycle,
		runtime: credentialSet.runtime, transactionLookup: credentialSet.transactionLookup,
	}, storage)
	if err != nil {
		return auditServices{}, err
	}
	// The Controller side holds the claim-bound commands and the reads of the
	// Audit store; the owner-facing service holds neither seam's handle.
	auditClaims, auditReads := claimstore.New(pool, storage).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch), querystore.New(pool, storage)
	auditImportArtifacts, err := auditimport.NewArtifactAccess(catalogs.artifacts, hookstore.NewClaimedArtifacts(pool, storage))
	if err != nil {
		return auditServices{}, fmt.Errorf("configure Audit import artifacts: %w", err)
	}
	auditImporter, err := auditimport.New(
		auditClaims, auditReads, runstore.NewPostgresStore(pool, storage), auditImportArtifacts,
		catalogs.findings,
	)
	if err != nil {
		return auditServices{}, fmt.Errorf("configure Audit importer: %w", err)
	}
	auditImporter = auditImporter.WithTaskIntake(auditService)
	auditController, err := auditcontroller.New(
		auditClaims, auditReads, runstore.NewPostgresStore(pool, storage), rundeletion.New(pool, storage),
		runCreationService, workflows.scheduler,
		auditcontroller.Options{
			PollInterval:     cfg.Operations.AuditController.PollInterval,
			ClaimLease:       cfg.Operations.AuditController.ClaimLease,
			OperationTimeout: cfg.Operations.AuditController.OperationTimeout,
			ClaimBatch:       cfg.Operations.AuditController.ClaimBatch,
			Logger:           logger, Collector: auditImporter, TaskIntake: auditService,
		},
	)
	if err != nil {
		return auditServices{}, fmt.Errorf("configure Audit Controller: %w", err)
	}
	return auditServices{service: auditService, runs: runCreationService, controller: auditController}, nil
}

type runCreationCredentials struct {
	provider          resolved.CredentialLookup
	guard             runservice.CredentialGuard
	runtime           runservice.RuntimeCredentialValidator
	transactionLookup runtimeconfig.TransactionLLMCredentialLookupFactory
}

func configureRunCreation(pool *pgxpool.Pool, configurationManager *workflowconfig.Manager, credentialSet runCreationCredentials, storage *artifactpostgres.Storage) (*runservice.Service, error) {
	runCreationService, err := runservice.New(runservice.Options{
		Runs: runstore.NewPostgresStore(pool, storage), Workflows: configurationManager,
		LLMCredentials: credentialSet.provider, CredentialGuard: credentialSet.guard,
		RuntimeCredentials: credentialSet.runtime, Projects: projectstore.NewPostgresStore(pool),
		SkillInitializationAvailable: true,
		PublicTransaction: postgresPublicUnitOfWork{storage: storage,
			pool: pool, transactionLLMCredentials: credentialSet.transactionLookup,
		}.Do,
		AuditTransaction: func(
			transactionContext context.Context,
			fn func(runservice.AuditRunWriter, *artifacts.Service, runservice.AuditExecutionWriter) error,
		) error {
			return persistencepostgres.InTx(
				transactionContext, pool, pgx.TxOptions{},
				func(tx pgx.Tx) error {
					return fn(
						runstore.NewPostgresTxStore(tx, storage),
						artifacts.NewService(storage.Repository(tx)),
						claimstore.New(tx, storage),
					)
				},
			)
		},
	})
	if err != nil {
		return nil, fmt.Errorf("configure trusted Run service: %w", err)
	}
	return runCreationService, nil
}

type postgresPublicUnitOfWork struct {
	storage *artifactpostgres.Storage

	pool                      *pgxpool.Pool
	transactionLLMCredentials runtimeconfig.TransactionLLMCredentialLookupFactory
}

func (u postgresPublicUnitOfWork) Do(
	ctx context.Context,
	fn func(runservice.PublicRunWriter, *artifacts.Service) error,
) error {
	return u.do(ctx, func(_ pgx.Tx, writer runservice.PublicRunWriter, service *artifacts.Service) error {
		return fn(writer, service)
	})
}

func (u postgresPublicUnitOfWork) do(ctx context.Context, fn func(pgx.Tx, runservice.PublicRunWriter, *artifacts.Service) error) error {
	return persistencepostgres.InTxWithRetry(ctx, u.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
		txLookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(
			tx, u.transactionLLMCredentials,
		)
		if err != nil {
			return err
		}
		return fn(tx,
			runstore.NewRunCreationPostgresStore(tx, txLookup, credentials.NewRuntimeCredentialRepository(tx), u.storage),
			artifacts.NewService(u.storage.Repository(tx)),
		)
	})
}
