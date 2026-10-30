package app

import (
	"fmt"
	"io"
	"log/slog"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/operations"
	"github.com/grauwolf32/contractor/internal/operations/sitemapjoin"
	"github.com/grauwolf32/contractor/internal/planner"
	plannera2a "github.com/grauwolf32/contractor/internal/planner/a2a"
	plannerrouter "github.com/grauwolf32/contractor/internal/planner/router"
	plannerscan "github.com/grauwolf32/contractor/internal/planner/scan"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
	"github.com/grauwolf32/contractor/internal/projectlifecycle"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/scheduler"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/adk/v2/model"
)

type executorServices struct {
	registry   *planner.Registry
	operations *operations.Registry
	sessions   *plannersession.Service
}

type workflowServices struct {
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
	runtimeClient *controlplane.RuntimeControlClient,
	cfg Config,
) (executorServices, error) {
	files := mtls.Files{Certificate: cfg.CertificateFile, PrivateKey: cfg.PrivateKeyFile, CA: cfg.CAFile}
	artifactInspector, err := planner.NewArtifactServiceInspector(artifactService)
	if err != nil {
		return executorServices{}, err
	}
	plannerMemoryStore, err := plannermemory.NewPostgresStore(pool)
	if err != nil {
		return executorServices{}, fmt.Errorf("configure Planner Memory store: %w", err)
	}
	completionWriteTimeout := max(
		planner.DefaultCompletionWriteTimeout,
		cfg.Operations.Database.AcquireTimeout+cfg.Operations.Database.LockTimeout+time.Second,
	)
	plannerSessions, err := plannersession.New(runstore.NewPostgresStore(pool), plannersession.Options{
		CompletionWriteTimeout: completionWriteTimeout,
	})
	if err != nil {
		return executorServices{}, fmt.Errorf("configure Planner sessions: %w", err)
	}
	a2aInvoker, err := plannera2a.NewMTLS(files, cfg.RuntimeRequestTimeout, plannera2a.Options{PollInterval: cfg.Operations.A2A.PollInterval})
	if err != nil {
		return executorServices{}, fmt.Errorf("configure A2A client: %w", err)
	}
	passthrough, err := planner.NewPassthroughFactory(plannerSessions, a2aInvoker, artifactInspector)
	if err != nil {
		return executorServices{}, err
	}
	plannerModelFactory := func(access planner.ModelAccess) (model.LLM, error) {
		return streamline.NewOpenAICompatibleModel(streamline.GatewaySettings{
			Recovery: access.Recovery, FailureSignatures: access.LLMGateway.EffectiveFailureSignatures(),
			URL: access.LLMGateway.URL, Token: access.Token, Model: access.ModelPolicy.Model,
			MaxOutputTokens: access.ModelPolicy.MaxOutputTokens, RequestTimeout: cfg.PlannerTimeout,
		})
	}
	streamlineLimits := streamline.DefaultLimits()
	streamlineLimits.MaxWallTime = cfg.PlannerTimeout
	streamlineFactory, err := streamline.NewConfiguredFactoryWithMemory(
		plannerSessions, plannerSessions, a2aInvoker, artifactInspector,
		runtimeClient, plannerMemoryStore, plannerModelFactory, streamlineLimits,
	)
	if err != nil {
		return executorServices{}, fmt.Errorf("configure Streamline Planner: %w", err)
	}
	routerFactory, err := plannerrouter.NewConfiguredFactoryWithMemory(
		plannerSessions, plannerSessions, a2aInvoker, artifactInspector,
		runtimeClient, plannerMemoryStore, plannerModelFactory, streamlineLimits,
	)
	if err != nil {
		return executorServices{}, fmt.Errorf("configure Router Planner: %w", err)
	}
	scanArtifacts, err := plannerscan.NewServiceArtifacts(artifactService)
	if err != nil {
		return executorServices{}, err
	}
	scanFactory, err := plannerscan.NewFactory(plannerSessions, a2aInvoker, scanArtifacts, artifactInspector)
	if err != nil {
		return executorServices{}, err
	}
	plannerRegistry, err := planner.NewRegistry(passthrough, streamlineFactory, routerFactory, scanFactory)
	if err != nil {
		return executorServices{}, err
	}
	operationRegistry, err := operations.NewRegistry(sitemapjoin.NewJoin(projectsitemap.NewService(pool)), sitemapjoin.NewApply(projectsitemap.NewService(pool)))
	if err != nil {
		return executorServices{}, err
	}
	return executorServices{registry: plannerRegistry, operations: operationRegistry, sessions: plannerSessions}, nil
}

func configureWorkflows(
	pool *pgxpool.Pool,
	cfg Config,
	catalogs catalogServices,
	control controlServices,
	credentialSet credentialServices,
	plannerTelemetryRegistry *telemetry.PlannerAdapterRegistry,
	logger *slog.Logger,
) (workflowServices, error) {
	// Workers bound each model request by workerRequestTimeout and modeled
	// planners by plannerTimeout; a recovery probe lease outlasts its holder's.
	// Recovery locks and moves Runs only through the Run store.
	recovery, err := gatewayrecovery.New(
		pool, runstore.GatewayRecoveryRuns{}, cfg.Operations.LLMRecovery, cfg.WorkerRequestTimeout, cfg.PlannerTimeout,
	)
	if err != nil {
		return workflowServices{}, err
	}
	planners, err := configureExecutors(pool, catalogs.artifacts, control.runtimeClient, cfg)
	if err != nil {
		return workflowServices{}, err
	}
	artifactResolver, err := scheduler.NewArtifactServiceResolver(catalogs.artifacts)
	if err != nil {
		return workflowServices{}, err
	}
	transactions, err := scheduler.NewPostgresPersistence(pool)
	if err != nil {
		return workflowServices{}, err
	}
	sourceLimits := cfg.GitImport.EffectiveLimits().Source
	runtimeTransport := scheduler.RuntimeTransportSettings{
		SourceArchiveLimits:   &sourceLimits,
		ArtifactAPIURL:        strings.TrimRight(cfg.PrivateURL, "/") + "/private/v1",
		RequestTimeoutSeconds: int(cfg.WorkerRequestTimeout / time.Second),
	}
	schedulerSettings := settingsstore.NewPostgresStore(pool)
	telemetryReports := telemetry.NewRepository(pool)
	workflowScheduler, err := scheduler.New(
		runstore.NewPostgresStore(pool),
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
			RunSkills:           &runSkillInitializer{pool: pool},
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
		telemetryReports, cfg.Operations.Scheduler.OperationTimeout, logger,
	)
	if err != nil {
		return workflowServices{}, fmt.Errorf("configure telemetry retention: %w", err)
	}
	projectDeletionController, err := projectlifecycle.New(
		pool,
		runstore.NewPostgresStore(pool),
		rundeletion.New(pool),
		workflowScheduler,
		projectlifecycle.Options{
			OperationTimeout: cfg.Operations.ProjectLifecycle.OperationTimeout,
			PurgeTimeout:     cfg.Operations.ProjectLifecycle.PurgeTimeout,
			ClaimDuration:    cfg.Operations.ProjectLifecycle.ClaimDuration, Logger: logger,
		},
	)
	if err != nil {
		return workflowServices{}, fmt.Errorf("configure Project deletion controller: %w", err)
	}
	return workflowServices{
		gatewayRecovery: recovery, scheduler: workflowScheduler, telemetryRetention: telemetryRetention,
		projectDeletion: projectDeletionController, settings: schedulerSettings,
	}, nil
}
