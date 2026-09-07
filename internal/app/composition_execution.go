package app

import (
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/planner"
	plannera2a "github.com/grauwolf32/contractor/internal/planner/a2a"
	plannerrouter "github.com/grauwolf32/contractor/internal/planner/router"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
	"github.com/grauwolf32/contractor/internal/projectlifecycle"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/scheduler"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/adk/model"
)

type plannerServices struct {
	sessions *plannersession.Service
	registry *planner.Registry
}

type workflowServices struct {
	scheduler       *scheduler.Scheduler
	projectDeletion *projectlifecycle.Controller
	settings        *settingsstore.PostgresStore
	planners        plannerServices
}

func configurePlanners(
	pool *pgxpool.Pool,
	artifactService *artifacts.Service,
	runtimeClient *controlplane.RuntimeControlClient,
	cfg Config,
) (plannerServices, error) {
	files := mtls.Files{Certificate: cfg.CertificateFile, PrivateKey: cfg.PrivateKeyFile, CA: cfg.CAFile}
	artifactInspector, err := planner.NewArtifactServiceInspector(artifactService)
	if err != nil {
		return plannerServices{}, err
	}
	plannerMemoryStore, err := plannermemory.NewPostgresStore(pool)
	if err != nil {
		return plannerServices{}, fmt.Errorf("configure Planner Memory store: %w", err)
	}
	plannerSessions, err := plannersession.New(runstore.NewPostgresStore(pool), plannersession.Options{})
	if err != nil {
		return plannerServices{}, fmt.Errorf("configure Planner sessions: %w", err)
	}
	a2aInvoker, err := plannera2a.NewMTLS(files, cfg.RuntimeRequestTimeout, plannera2a.Options{PollInterval: cfg.Operations.A2A.PollInterval})
	if err != nil {
		return plannerServices{}, fmt.Errorf("configure A2A client: %w", err)
	}
	passthrough, err := planner.NewPassthroughFactory(plannerSessions, a2aInvoker, artifactInspector)
	if err != nil {
		return plannerServices{}, err
	}
	plannerModelFactory := func(access planner.ModelAccess) (model.LLM, error) {
		return streamline.NewOpenAICompatibleModel(streamline.GatewaySettings{
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
		return plannerServices{}, fmt.Errorf("configure Streamline Planner: %w", err)
	}
	routerFactory, err := plannerrouter.NewConfiguredFactoryWithMemory(
		plannerSessions, plannerSessions, a2aInvoker, artifactInspector,
		runtimeClient, plannerMemoryStore, plannerModelFactory, streamlineLimits,
	)
	if err != nil {
		return plannerServices{}, fmt.Errorf("configure Router Planner: %w", err)
	}
	plannerRegistry, err := planner.NewRegistry(passthrough, streamlineFactory, routerFactory)
	if err != nil {
		return plannerServices{}, err
	}
	return plannerServices{sessions: plannerSessions, registry: plannerRegistry}, nil
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
	planners, err := configurePlanners(pool, catalogs.artifacts, control.runtimeClient, cfg)
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
	runtimeSettings := contracts.RuntimeSettings{
		ArtifactAPIURL:        strings.TrimRight(cfg.PrivateURL, "/") + "/private/v1",
		RequestTimeoutSeconds: int(cfg.WorkerRequestTimeout / time.Second),
	}
	schedulerSettings := settingsstore.NewPostgresStore(pool)
	workflowScheduler, err := scheduler.New(
		runstore.NewPostgresStore(pool),
		transactions,
		artifactResolver,
		control.allocator,
		control.workers,
		planners.registry,
		scheduler.Options{
			OperationTimeout:    cfg.Operations.Scheduler.OperationTimeout,
			FinalizationTimeout: cfg.Operations.Scheduler.FinalizationTimeout,
			AbortTimeout:        cfg.Operations.Scheduler.AbortTimeout,
			PlannerTimeout:      cfg.PlannerTimeout,
			RuntimeSettings:     runtimeSettings,
			Credentials:         credentialSet.provider,
			RuntimeCredentials:  credentialSet.runtime,
			PlannerTelemetry:    plannerTelemetryRegistry,
			RunSkills:           &runSkillInitializer{pool: pool},
			Settings:            schedulerSettings,
			TelemetrySecrets: []string{
				cfg.DatabaseURL, cfg.PublicBearerToken.Reveal(),
				cfg.DevelopmentWorkerToken.Reveal(), cfg.DevelopmentPlannerToken.Reveal(),
			},
			Logger: logger,
		},
	)
	if err != nil {
		return workflowServices{}, fmt.Errorf("configure Workflow Scheduler: %w", err)
	}
	projectDeletionController, err := projectlifecycle.New(
		pool,
		runstore.NewPostgresStore(pool),
		workflowScheduler,
		projectlifecycle.Options{OperationTimeout: cfg.Operations.ProjectLifecycle.OperationTimeout, ClaimDuration: cfg.Operations.ProjectLifecycle.ClaimDuration, Logger: logger},
	)
	if err != nil {
		return workflowServices{}, fmt.Errorf("configure Project deletion controller: %w", err)
	}
	return workflowServices{scheduler: workflowScheduler, projectDeletion: projectDeletionController, settings: schedulerSettings, planners: planners}, nil
}
