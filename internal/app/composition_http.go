package app

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auth"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/gitimport"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
	privateartifacts "github.com/grauwolf32/contractor/internal/httpapi/privateartifacts"
	publicapi "github.com/grauwolf32/contractor/internal/httpapi/public"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
	"github.com/grauwolf32/contractor/internal/performance"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runservice"
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
	trustedPeers auth.PeerPolicy,
	eventHub *publicevents.Hub,
	credentialSet credentialServices,
	control controlServices,
	catalogs catalogServices,
	workflows workflowServices,
	audits auditServices,
	evals evalServices,
	logger *slog.Logger,
	storage *artifactpostgres.Storage,
) (serverHandlers, error) {
	gitClient, err := gitimport.NewClient(cfg.GitImport, logger)
	if err != nil {
		return serverHandlers{}, fmt.Errorf("configure Git reader: %w", err)
	}
	gitImporter, err := gitimport.NewImporter(pool, gitClient, credentialSet.gitKeys, cfg.GitImport.EffectiveLimits().Timeout, storage)
	if err != nil {
		return serverHandlers{}, fmt.Errorf("configure Git importer: %w", err)
	}
	performanceDiagnostics := newPerformanceDiagnostics(cfg.PerformanceMetrics, cfg.DatabaseURL.Reveal())
	var performanceCollector *performance.Collector
	if cfg.PerformanceMetrics {
		performanceCollector = performance.New(performance.Options{
			ReadPool: performance.WorkingPoolReader(pool), Diagnostics: performanceDiagnostics,
			ReadGPU: performance.NewNVIDIAGPUReader(),
		})
	}
	performanceReader := performance.NewReadService(
		cfg.PerformanceMetrics, performanceCollector, performanceDiagnostics,
		performance.NewHistoryRepository(pool, time.Now), time.Now,
	)
	collectionPublisher, err := findingintake.NewCollectionPublisher(pool, audits.service, storage)
	if err != nil {
		return serverHandlers{}, fmt.Errorf("configure finding collection publisher: %w", err)
	}
	runStore := runstore.NewPostgresStore(pool, storage)
	runDetails, err := runservice.NewRunDetailReader(runservice.PostgresRunDetailTransaction(pool, storage))
	if err != nil {
		return serverHandlers{}, fmt.Errorf("configure Run detail reader: %w", err)
	}
	projects := projectstore.NewPostgresStore(pool)
	projectEditor := projectstore.NewEditor(projects, credentialSet.runtime, func(ctx context.Context, ownerID string, operations bool, credentialID, kind string) error {
		return credentialSet.runtime.ValidateRuntimeCredentialUse(ctx, credentials.RuntimeCredentialUser{UserID: ownerID, Operations: operations}, credentialID, kind)
	})
	publicHandler, err := publicapi.NewHandler(publicapi.Dependencies{
		BearerToken:  cfg.PublicBearerToken,
		NewRequestID: func() (string, error) { return randomid.New("request_") },
		Logger:       logger,
		Session: publicapi.SessionDependencies{
			Authentication: authentication, BrowserOrigins: browserOrigins, TrustedPeers: trustedPeers,
			InsecureLoopbackCookie: cfg.InsecureLoopbackCookie, Events: eventHub,
		},
		Catalog: publicapi.CatalogDependencies{
			Config: configurationManager, ConfigurationPublisher: configurationManager,
			OperationsInvalidator: control.registry,
		},
		Projects: publicapi.ProjectDependencies{
			Projects: projects, Editor: projectEditor,
			Sitemap:                 projectsitemap.NewService(pool, storage),
			ProjectDeletionNotifier: workflows.projectDeletion, NewID: randomid.New,
		},
		Runs: publicapi.RunDependencies{
			Diagnostics: telemetry.NewRepository(pool),
			Runs:        runStore, RunDetails: runDetails, RunLifecycle: runStore,
			RunDeletion: rundeletion.New(pool, storage), RunCreator: audits.runs,
			RunNotifier: workflows.scheduler, GatewayRecovery: workflows.gatewayRecovery,
			Projects: projects, NewID: randomid.New, Now: time.Now,
		},
		RunRepeat: publicapi.RunRepeatDependencies{RepeatDrafts: runservice.NewRepeatReader(runservice.RepeatOptions{
			Runs: runStore, Projects: projects, Config: configurationManager,
			Credentials: credentialSet.provider, Artifacts: catalogs.artifacts,
			RuntimeConfigs: control.runtimeConfigurations,
		})},
		Queue: publicapi.QueueDependencies{RunQueue: runStore, RunNotifier: workflows.scheduler},
		Artifacts: publicapi.ArtifactDependencies{Transfers: storage,
			Artifacts: catalogs.artifacts, Projects: projects, Runs: runStore, GitImports: gitImporter,
		},
		Audits: publicapi.AuditDependencies{Transfers: storage, Audits: audits.service, Projects: projects, NewID: randomid.New},
		Findings: publicapi.FindingDependencies{Transfers: storage,
			FindingProposals: catalogs.findings, FindingCollections: collectionPublisher, FindingExports: collectionPublisher,
			Audits: audits.service, Runs: runStore,
		},
		Evals: publicapi.EvalDependencies{
			Evals: evals.service, EvalNotifier: evals.coordinator,
			CatalogSnapshot: configurationManager.Snapshot,
		},
		Operations: publicapi.OperationsDependencies{
			ModelSelection: configurationManager,
			Operations:     control.registry, Performance: performanceReader,
			AllocationResources: telemetry.NewRepository(pool), SchedulerSettings: workflows.settings,
			OperationsInvalidator: control.registry, RunNotifier: workflows.scheduler, Logger: logger,
		},
		Credentials: publicapi.CredentialDependencies{
			ManagedCredentials: credentialSet.lifecycle, OperationsInvalidator: control.registry,
		},
		RuntimeConfiguration: publicapi.RuntimeConfigurationDependencies{
			RuntimeConfigs: control.runtimeConfigurations, RuntimeCredentials: credentialSet.runtime,
			RuntimeAgentPrincipals: control.principalOperations, OperationsInvalidator: control.registry,
			Logger: logger, Now: time.Now,
		},
		GitKeys: publicapi.GitKeyDependencies{GitKeys: credentialSet.gitKeys},
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
	artifactHandler, err := privateartifacts.NewHandler(privateartifacts.Dependencies{Transfers: storage,
		Registry: control.registry, Artifacts: catalogs.artifacts, Findings: catalogs.findings, ProjectFindings: collectionPublisher, GatewayRecovery: workflows.gatewayRecovery, Logger: logger,
		Sitemap:     projectsitemap.NewService(pool, storage),
		RunProjects: runStore,
		Projects:    projects,
	})
	if err != nil {
		return serverHandlers{}, fmt.Errorf("configure private Artifact API: %w", err)
	}
	privateHandler := newPrivateHandler(controlHandler, artifactHandler)
	processHandler, instrumentedPrivate, _ := instrumentPerformance(
		cfg.PerformanceMetrics, NewReadyHandler(pool.Ping, publicHandler), privateHandler,
		func() *performance.Collector { return performanceCollector },
	)
	runners := backgroundRunnerGroup{
		workflows.scheduler, workflows.telemetryRetention, audits.controller, workflows.projectDeletion,
	}
	if performanceCollector != nil {
		runners = append(runners, performanceCollector)
	}
	if performanceDiagnostics != nil {
		runners = append(runners, performanceDiagnostics)
	}
	return serverHandlers{public: processHandler, private: instrumentedPrivate, runners: runners}, nil
}
