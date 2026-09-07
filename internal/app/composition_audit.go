package app

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditcontroller"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	publicapi "github.com/grauwolf32/contractor/internal/httpapi/public"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
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
) (auditServices, error) {
	auditService, err := auditservice.New(auditservice.Options{
		Pool: pool, Profiles: configurationManager,
		TransactionLLMCredentials: credentialSet.transactionLookup,
		CredentialGuard:           credentialSet.lifecycle,
	})
	if err != nil {
		return auditServices{}, fmt.Errorf("configure Audit service: %w", err)
	}
	runCreationService, err := configureRunCreation(pool, configurationManager, credentialSet)
	if err != nil {
		return auditServices{}, err
	}
	auditArtifactAccess, err := auditcontroller.NewProjectArtifactAccess(catalogs.artifacts)
	if err != nil {
		return auditServices{}, fmt.Errorf("configure Audit Controller artifacts: %w", err)
	}
	auditSubmissionStore := auditstore.NewPostgresStore(pool)
	auditSubmissionBuilder, err := auditcontroller.NewPinnedSubmissionBuilder(
		auditArtifactAccess, auditSubmissionStore,
	)
	if err != nil {
		return auditServices{}, fmt.Errorf("configure Audit Controller submissions: %w", err)
	}
	auditImportArtifacts, err := auditimport.NewArtifactAccess(catalogs.artifacts)
	if err != nil {
		return auditServices{}, fmt.Errorf("configure Audit import artifacts: %w", err)
	}
	auditImporter, err := auditimport.New(
		auditstore.NewPostgresStore(pool), runstore.NewPostgresStore(pool), auditImportArtifacts,
		catalogs.findings,
	)
	if err != nil {
		return auditServices{}, fmt.Errorf("configure Audit importer: %w", err)
	}
	auditController, err := auditcontroller.New(
		auditstore.NewPostgresStore(pool), runstore.NewPostgresStore(pool),
		runCreationService, auditSubmissionBuilder, workflows.scheduler,
		auditcontroller.Options{
			PollInterval:     cfg.Operations.AuditController.PollInterval,
			ClaimLease:       cfg.Operations.AuditController.ClaimLease,
			OperationTimeout: cfg.Operations.AuditController.OperationTimeout,
			ClaimBatch:       cfg.Operations.AuditController.ClaimBatch,
			Logger:           logger, Collector: auditImporter, RoundBuilder: auditService,
		},
	)
	if err != nil {
		return auditServices{}, fmt.Errorf("configure Audit Controller: %w", err)
	}
	return auditServices{service: auditService, runs: runCreationService, controller: auditController}, nil
}

func configureRunCreation(pool *pgxpool.Pool, configurationManager *workflowconfig.Manager, credentialSet credentialServices) (*runservice.Service, error) {
	runCreationService, err := runservice.New(runservice.Options{
		Runs: runstore.NewPostgresStore(pool), Workflows: configurationManager,
		LLMCredentials: credentialSet.provider, CredentialGuard: credentialSet.lifecycle,
		RuntimeCredentials: credentialSet.runtime, Projects: projectstore.NewPostgresStore(pool),
		SkillInitializationAvailable: true,
		PublicTransaction: func(
			transactionContext context.Context,
			fn func(runservice.PublicRunWriter, *artifacts.Service) error,
		) error {
			return persistencepostgres.InTx(
				transactionContext, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead},
				func(tx pgx.Tx) error {
					txLookup, bindErr := runtimeconfig.BindTransactionLLMCredentialLookup(
						tx, credentialSet.transactionLookup,
					)
					if bindErr != nil {
						return bindErr
					}
					return fn(
						runstore.NewRunCreationPostgresStore(tx, txLookup),
						artifacts.NewService(artifacts.NewPostgresRepository(tx)),
					)
				},
			)
		},
		AuditTransaction: func(
			transactionContext context.Context,
			fn func(runservice.AuditRunWriter, *artifacts.Service, runservice.AuditExecutionWriter) error,
		) error {
			return persistencepostgres.InTx(
				transactionContext, pool, pgx.TxOptions{},
				func(tx pgx.Tx) error {
					return fn(
						runstore.NewPostgresStore(tx),
						artifacts.NewService(artifacts.NewPostgresRepository(tx)),
						auditstore.NewPostgresStore(tx),
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
	pool                      *pgxpool.Pool
	transactionLLMCredentials runtimeconfig.TransactionLLMCredentialLookupFactory
}

func (u postgresPublicUnitOfWork) Do(
	ctx context.Context,
	fn func(publicapi.RunWriter, *artifacts.Service) error,
) error {
	return persistencepostgres.InTxWithRetry(ctx, u.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
		txLookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(
			tx, u.transactionLLMCredentials,
		)
		if err != nil {
			return err
		}
		return fn(
			runstore.NewRunCreationPostgresStore(tx, txLookup),
			artifacts.NewService(artifacts.NewPostgresRepository(tx)),
		)
	})
}
