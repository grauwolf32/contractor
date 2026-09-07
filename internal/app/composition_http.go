package app

import (
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/auth"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/gitimport"
	privateartifacts "github.com/grauwolf32/contractor/internal/httpapi/privateartifacts"
	publicapi "github.com/grauwolf32/contractor/internal/httpapi/public"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
	"github.com/grauwolf32/contractor/internal/performance"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

type serverHandlers struct {
	public  http.Handler
	private http.Handler
	runners backgroundRunnerGroup
}

func configureHTTP(
	pool *pgxpool.Pool,
	configurationManager *workflowconfig.Manager,
	cfg Config,
	authentication *auth.Service,
	browserOrigins auth.OriginPolicy,
	eventHub *publicevents.Hub,
	credentialSet credentialServices,
	control controlServices,
	catalogs catalogServices,
	workflows workflowServices,
	audits auditServices,
	logger *slog.Logger,
) (serverHandlers, error) {
	gitClient, err := gitimport.NewClient(cfg.GitImport)
	if err != nil {
		return serverHandlers{}, fmt.Errorf("configure Git reader: %w", err)
	}
	gitImporter, err := gitimport.NewImporter(pool, gitClient, credentialSet.gitKeys)
	if err != nil {
		return serverHandlers{}, fmt.Errorf("configure Git importer: %w", err)
	}
	performanceDiagnostics := newPerformanceDiagnostics(cfg.PerformanceMetrics, cfg.DatabaseURL)
	var performanceCollector *performance.Collector
	if cfg.PerformanceMetrics {
		performanceCollector = performance.New(performance.Options{
			ReadPool: performance.WorkingPoolReader(pool), Diagnostics: performanceDiagnostics,
		})
	}
	performanceReader := performance.NewReadService(
		cfg.PerformanceMetrics, performanceCollector, performanceDiagnostics,
		performance.NewHistoryRepository(pool, time.Now), time.Now,
	)
	collectionPublisher, err := findingintake.NewCollectionPublisher(pool, audits.service)
	if err != nil {
		return serverHandlers{}, fmt.Errorf("configure finding collection publisher: %w", err)
	}
	publicHandler, err := publicapi.NewHandler(publicapi.Dependencies{
		Authentication: authentication, BrowserOrigins: browserOrigins,
		InsecureLoopbackCookie: cfg.InsecureLoopbackCookie,
		Config:                 configurationManager, ConfigurationPublisher: configurationManager,
		Runs: runstore.NewPostgresStore(pool), RunCreator: audits.runs, Artifacts: catalogs.artifacts,
		Credentials: credentialSet.provider, ManagedCredentials: credentialSet.lifecycle,
		RuntimeConfigs: control.runtimeConfigurations, RuntimeCredentials: credentialSet.runtime,
		GitKeys:                credentialSet.gitKeys,
		GitImports:             gitImporter,
		RuntimeAgentPrincipals: control.principalOperations,
		Projects:               projectstore.NewPostgresStore(pool),
		Audits:                 audits.service,
		FindingProposals:       catalogs.findings,
		FindingCollections:     collectionPublisher,
		Metrics:                telemetry.NewRepository(pool),
		PlannerPlans:           workflows.planners.sessions,
		Operations:             control.registry,
		Performance:            performanceReader,
		AllocationResources:    telemetry.NewRepository(pool),
		OperationsInvalidator:  control.registry,
		SchedulerSettings:      workflows.settings,
		Events:                 eventHub,
		Transactions: postgresPublicUnitOfWork{
			pool: pool, transactionLLMCredentials: credentialSet.transactionLookup,
		},
		BearerToken: cfg.PublicBearerToken,
		RunNotifier: workflows.scheduler, Logger: logger,
		ProjectDeletionNotifier: workflows.projectDeletion,
		RunSkills:               &runSkillInitializer{pool: pool},
	})
	if err != nil {
		return serverHandlers{}, fmt.Errorf("configure public API: %w", err)
	}

	controlHandler, err := controlplane.NewHTTPHandler(
		control.registry, controlplane.HTTPOptions{Logger: logger, Principals: control.principals},
	)
	if err != nil {
		return serverHandlers{}, fmt.Errorf("configure private Control Plane API: %w", err)
	}
	artifactHandler, err := privateartifacts.NewHandler(privateartifacts.Dependencies{
		Registry: control.registry, Artifacts: catalogs.artifacts, Findings: catalogs.findings, Logger: logger,
	})
	if err != nil {
		return serverHandlers{}, fmt.Errorf("configure private Artifact API: %w", err)
	}
	privateHandler := newPrivateHandler(controlHandler, artifactHandler)
	processHandler, instrumentedPrivate, _ := instrumentPerformance(
		cfg.PerformanceMetrics, NewReadyHandler(pool.Ping, publicHandler), privateHandler,
		func() *performance.Collector { return performanceCollector },
	)
	runners := backgroundRunnerGroup{workflows.scheduler, audits.controller, workflows.projectDeletion}
	if performanceCollector != nil {
		runners = append(runners, performanceCollector)
	}
	if performanceDiagnostics != nil {
		runners = append(runners, performanceDiagnostics)
	}
	return serverHandlers{public: processHandler, private: instrumentedPrivate, runners: runners}, nil
}
