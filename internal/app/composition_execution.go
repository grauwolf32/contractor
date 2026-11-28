package app

import (
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/operations"
	"github.com/grauwolf32/contractor/internal/operations/artifactcopy"
	"github.com/grauwolf32/contractor/internal/operations/memorysnapshot"
	"github.com/grauwolf32/contractor/internal/operations/openapiwork"
	"github.com/grauwolf32/contractor/internal/operations/publishfindings"
	"github.com/grauwolf32/contractor/internal/operations/sitemapjoin"
	"github.com/grauwolf32/contractor/internal/planner"
	plannera2a "github.com/grauwolf32/contractor/internal/planner/a2a"
	plannerrouter "github.com/grauwolf32/contractor/internal/planner/router"
	plannerscan "github.com/grauwolf32/contractor/internal/planner/scan"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
	"github.com/grauwolf32/contractor/internal/projectlifecycle"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/runcost"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/scheduler"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

type executorServices struct {
	registry   *planner.Registry
	operations *operations.Registry
	sessions   *plannersession.Service
}

type workflowServices struct {
	plannerOperations  *planner.Registry
	gatewayRecovery    *gatewayrecovery.Service
	scheduler          *scheduler.Scheduler
	telemetryRetention *telemetry.Retention
	projectDeletion    *projectlifecycle.Controller
	settings           *settingsstore.PostgresStore
}

// Format redacts the services as a whole for every verb (SEC-R88): through the
// unexported scheduler field fmt would print its telemetry secrets.
func (workflowServices) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "app.workflowServices([REDACTED])")
}

func configureExecutors(
	pool *pgxpool.Pool,
	artifactService *artifacts.Service,
	findings *findingintake.Service,
	runtimeClient *controlplane.RuntimeControlClient,
	cfg Config,
	storage *artifactpostgres.Storage,
) (executorServices, error) {
	files := mtls.Files{Certificate: cfg.CertificateFile, PrivateKey: cfg.PrivateKeyFile, CA: cfg.CAFile}
	artifactInspector, err := planner.NewArtifactServiceInspector(artifactService)
	if err != nil {
		return executorServices{}, err
	}
	plannerMemoryStore, err := plannermemory.NewPostgresStore(pool, storage)
	if err != nil {
		return executorServices{}, fmt.Errorf("configure Planner Memory store: %w", err)
	}
	completionWriteTimeout := max(
		planner.DefaultCompletionWriteTimeout,
		cfg.Operations.Database.AcquireTimeout+cfg.Operations.Database.LockTimeout+time.Second,
	)
	plannerSessions, err := plannersession.New(runcost.NewParticipants(pool, storage), plannersession.Options{
		CompletionWriteTimeout: completionWriteTimeout,
	})
	if err != nil {
		return executorServices{}, fmt.Errorf("configure Planner sessions: %w", err)
	}
	a2aInvoker, err := plannera2a.NewMTLS(files, cfg.RuntimeRequestTimeout, plannera2a.Options{PollInterval: cfg.Operations.A2A.PollInterval})
	if err != nil {
		return executorServices{}, fmt.Errorf("configure A2A client: %w", err)
	}
	streamlineLimits := streamline.DefaultLimits()
	streamlineLimits.MaxWallTime = cfg.PlannerTimeout
	scanArtifacts, err := plannerscan.NewServiceArtifacts(artifactService)
	if err != nil {
		return executorServices{}, err
	}
	plannerRegistry, err := composePlanners(plannerAdapters{
		sessions: plannerSessions, planSessions: plannerSessions,
		scanSessions: plannerSessions, invoker: a2aInvoker, inspector: artifactInspector,
		stateReader: runtimeClient, memory: plannerMemoryStore,
		limits: streamlineLimits, scanArtifacts: scanArtifacts,
	})
	if err != nil {
		return executorServices{}, err
	}
	operationRegistry, err := operations.NewRegistry(artifactcopy.New(pool, storage), memorysnapshot.New(plannerMemoryStore), sitemapjoin.NewJoin(projectsitemap.NewService(pool, storage)), sitemapjoin.NewApply(projectsitemap.NewService(pool, storage)), openapiwork.New(pool, storage), publishfindings.New(findings))
	if err != nil {
		return executorServices{}, err
	}
	return executorServices{registry: plannerRegistry, operations: operationRegistry, sessions: plannerSessions}, nil
}

// plannerAdapters are the dependencies of the composed PlannerFactories.
type plannerAdapters struct {
	sessions      planner.SessionService
	planSessions  planner.PlanSessionService
	scanSessions  planner.ScanSessionService
	invoker       planner.WorkerInvoker
	inspector     planner.ArtifactInspector
	stateReader   planner.WorkerStateReader
	memory        plannermemory.Store
	limits        streamline.Limits
	scanArtifacts plannerscan.ArtifactStore
}

// composePlanners builds one PlannerFactory for each registered PlannerFactory
// descriptor (CAT-R79) and fails when the two sets differ: a descriptor
// without a factory would load in the catalog and fail only when a Stage
// creates its Planner.
func composePlanners(adapters plannerAdapters) (*planner.Registry, error) {
	passthrough, err := planner.NewPassthroughFactory(adapters.sessions, adapters.invoker, adapters.inspector)
	if err != nil {
		return nil, err
	}
	streamlineFactory, err := streamline.NewRuntimeFactoryWithMemory(
		adapters.planSessions, adapters.invoker, adapters.inspector,
		adapters.stateReader, adapters.memory, adapters.limits,
	)
	if err != nil {
		return nil, fmt.Errorf("configure Streamline Planner: %w", err)
	}
	routerFactory, err := plannerrouter.NewRuntimeFactory(streamlineFactory)
	if err != nil {
		return nil, fmt.Errorf("configure Router Planner: %w", err)
	}
	scanFactory, err := plannerscan.NewFactory(adapters.scanSessions, adapters.invoker, adapters.scanArtifacts, adapters.inspector)
	if err != nil {
		return nil, err
	}
	registry, err := planner.NewRegistry(passthrough, streamlineFactory, routerFactory, scanFactory)
	if err != nil {
		return nil, err
	}
	if err := matchPlannerDescriptors(registry.Refs(), resolved.PlannerSelectors()); err != nil {
		return nil, err
	}
	return registry, nil
}

// matchPlannerDescriptors reports the first registered descriptor without a
// composed PlannerFactory, or composed factory without a descriptor.
func matchPlannerDescriptors(composed, described []string) error {
	for _, ref := range described {
		if !slices.Contains(composed, ref) {
			return fmt.Errorf("PlannerFactory descriptor %q has no composed factory", ref)
		}
	}
	for _, ref := range composed {
		if !slices.Contains(described, ref) {
			return fmt.Errorf("composed PlannerFactory %q has no descriptor", ref)
		}
	}
	return nil
}

func configureWorkflows(
	pool *pgxpool.Pool,
	cfg Config,
	catalogs catalogServices,
	control controlServices,
	credentialSet credentialServices,
	plannerTelemetryRegistry *telemetry.PlannerAdapterRegistry,
	logger *slog.Logger,
	storage *artifactpostgres.Storage,
) (workflowServices, error) {
	// Workers bound each model request by workerRequestTimeout and modeled
	// planners by plannerTimeout; a recovery probe lease outlasts its holder's.
	// Recovery is process-local; Scheduler projects waits through the Run store.
	recovery, err := gatewayrecovery.New(
		cfg.Operations.LLMRecovery, cfg.WorkerRequestTimeout, cfg.PlannerTimeout,
	)
	if err != nil {
		return workflowServices{}, err
	}
	planners, err := configureExecutors(pool, catalogs.artifacts, catalogs.findings, control.runtimeClient, cfg, storage)
	if err != nil {
		return workflowServices{}, err
	}
	artifactResolver, err := scheduler.NewArtifactServiceResolver(catalogs.artifacts)
	if err != nil {
		return workflowServices{}, err
	}
	transactions, err := scheduler.NewPostgresPersistence(pool, storage)
	if err != nil {
		return workflowServices{}, err
	}
	sourceLimits := cfg.GitImport.EffectiveLimits().Source
	runtimeTransport := scheduler.RuntimeTransportSettings{
		SourceArchiveLimits:   &sourceLimits,
		ArtifactAPIURL:        strings.TrimRight(cfg.PrivateURL, "/"),
		RequestTimeoutSeconds: int(cfg.WorkerRequestTimeout / time.Second),
	}
	schedulerSettings := settingsstore.NewPostgresStore(pool)
	telemetryReports := runservice.NewCostReports(pool)
	workflowScheduler, err := scheduler.New(
		runstore.NewPostgresStore(pool, storage),
		transactions,
		artifactResolver,
		control.allocator,
		control.workers,
		planners.registry,
		scheduler.Options{
			GatewayRecovery:     recovery,
			PlannerSessions:     planners.sessions,
			ServerOperations:    planners.operations,
			PollInterval:        cfg.Operations.Scheduler.PollInterval,
			ClaimDuration:       cfg.Operations.Scheduler.ClaimDuration,
			LeaseScanInterval:   cfg.Operations.Scheduler.LeaseScanInterval,
			OperationTimeout:    cfg.Operations.Scheduler.OperationTimeout,
			FinalizationTimeout: cfg.Operations.Scheduler.FinalizationTimeout,
			AbortTimeout:        cfg.Operations.Scheduler.AbortTimeout,
			PlannerTimeout:      cfg.PlannerTimeout,
			RuntimeTransport:    runtimeTransport,
			Credentials:         credentialSet.provider,
			RuntimeCredentials:  credentialSet.runtime,
			PlannerTelemetry:    plannerTelemetryRegistry,
			Reports:             telemetryReports,
			RunSkills:           &runSkillInitializer{storage: storage, pool: pool},
			Settings:            schedulerSettings,
			TelemetrySecrets: []string{
				cfg.DatabaseURL.Reveal(), cfg.PublicBearerToken.Reveal(),
				cfg.DevelopmentWorkerToken.Reveal(), cfg.DevelopmentPlannerToken.Reveal(),
			},
			Logger: logger,
		},
	)
	if err != nil {
		return workflowServices{}, fmt.Errorf("configure Workflow Scheduler: %w", err)
	}
	// Expired telemetry is removed by its own loop outside the Scheduler lanes;
	// a cleanup pass keeps the Scheduler store-operation budget it ran under.
	telemetryRetention, err := telemetry.NewRetention(
		telemetryReports.Repository, cfg.Operations.Scheduler.OperationTimeout, logger,
	)
	if err != nil {
		return workflowServices{}, fmt.Errorf("configure telemetry retention: %w", err)
	}
	projectDeletionController, err := projectlifecycle.New(
		pool,
		runstore.NewPostgresStore(pool, storage),
		rundeletion.New(pool, storage),
		workflowScheduler,
		projectlifecycle.Options{Storage: storage,
			OperationTimeout: cfg.Operations.ProjectLifecycle.OperationTimeout,
			PurgeTimeout:     cfg.Operations.ProjectLifecycle.PurgeTimeout,
			ClaimDuration:    cfg.Operations.ProjectLifecycle.ClaimDuration, Logger: logger,
		},
	)
	if err != nil {
		return workflowServices{}, fmt.Errorf("configure Project deletion controller: %w", err)
	}
	return workflowServices{
		gatewayRecovery: recovery, scheduler: workflowScheduler, telemetryRetention: telemetryRetention, plannerOperations: planners.registry,
		projectDeletion: projectDeletionController, settings: schedulerSettings,
	}, nil
}
