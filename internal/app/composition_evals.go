package app

import (
	"log/slog"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/evalcoordinator"
	"github.com/grauwolf32/contractor/internal/evalservice"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

type evalServices struct {
	service     *evalservice.Service
	coordinator *evalcoordinator.Coordinator
}

func configureEvals(pool *pgxpool.Pool, catalog *workflowconfig.Manager, credentials credentialServices, audits auditServices, workflows workflowServices, logger *slog.Logger, storage *artifactpostgres.Storage) (evalServices, error) {
	resolver := &evalservice.Resolver{Storage: storage, Pool: pool, Catalog: catalog, Credentials: credentials.transactionLookup, Barrier: credentials.lifecycle}
	driver := &evalservice.Driver{Storage: storage, Pool: pool, Runs: audits.runs, Audits: audits.service, Notifier: workflows.scheduler}
	service, err := evalservice.New(evalservice.Options{Storage: storage, Pool: pool, Resolver: resolver, Driver: driver, Logger: logger})
	if err != nil {
		return evalServices{}, err
	}
	coordinator, err := evalcoordinator.New(evalstore.NewPostgresStore(pool, storage), service, evalcoordinator.Options{Logger: logger})
	if err != nil {
		return evalServices{}, err
	}
	return evalServices{service: service, coordinator: coordinator}, nil
}
