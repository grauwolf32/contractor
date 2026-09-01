// Package app contains the Contractor Server composition root.
package app

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auth"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	litellmcredentials "github.com/grauwolf32/contractor/internal/credentials/litellm"
	privateartifacts "github.com/grauwolf32/contractor/internal/httpapi/privateartifacts"
	publicapi "github.com/grauwolf32/contractor/internal/httpapi/public"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/persistence/configaudit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	plannera2a "github.com/grauwolf32/contractor/internal/planner/a2a"
	plannerrouter "github.com/grauwolf32/contractor/internal/planner/router"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/scheduler"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/adk/model"
)

// Config contains process-level settings needed by the bootstrap server.
type Config struct {
	ListenAddress         string
	PrivateListenAddress  string
	PrivateURL            string
	ShutdownTimeout       time.Duration
	RuntimeRequestTimeout time.Duration
	DatabaseURL           string
	// ConfigRoot is the deprecated alias retained for callers that inspect
	// parsed settings. Runtime composition uses the two explicit roots below.
	ConfigRoot                  string
	OperatorConfigRoot          string
	ManagedConfigRoot           string
	CredentialMasterKeyFile     string
	LLMGatewayAdminBindingsFile string
	CAFile                      string
	CertificateFile             string
	PrivateKeyFile              string
	DevelopmentWorkerToken      contracts.SecretString
	DevelopmentPlannerToken     contracts.SecretString
	PlannerTimeout              time.Duration
	PublicBearerToken           contracts.SecretString
	PublicUserID                string
	LocalAuthFile               string
	BrowserOrigins              []string
	InsecureLoopbackCookie      bool
}

const (
	developmentWorkerCredential  = "development-worker"
	developmentPlannerCredential = "development-planner"
)

func developmentCredentials(
	snapshot *workflowconfig.Snapshot,
	cfg Config,
) (*credentials.StaticProvider, error) {
	entries := make([]credentials.StaticEntry, 0, 2)
	if cfg.DevelopmentWorkerToken.Reveal() == "" && cfg.DevelopmentPlannerToken.Reveal() == "" {
		return credentials.NewStaticProvider(entries)
	}
	gateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		return nil, errors.New("development tokens require LLMGatewayConfig local-litellm@1")
	}
	appendEntry := func(id string, token contracts.SecretString) {
		if token.Reveal() == "" {
			return
		}
		entries = append(entries, credentials.StaticEntry{
			Metadata: workflowconfig.CredentialMetadata{
				Ref: contracts.LLMCredentialRef{CredentialID: id}, LLMGateway: gateway.Ref,
				Unrestricted: true,
			},
			Token: token,
		})
	}
	appendEntry(developmentWorkerCredential, cfg.DevelopmentWorkerToken)
	appendEntry(developmentPlannerCredential, cfg.DevelopmentPlannerToken)
	return credentials.NewStaticProvider(entries)
}

// RunCLI parses process configuration and runs the Server until cancellation.
func RunCLI(
	ctx context.Context,
	args []string,
	getenv func(string) string,
	logger *slog.Logger,
) error {
	if len(args) > 0 && args[0] == "config" {
		return runConfigCLI(args[1:], logger)
	}
	if len(args) > 0 && args[0] == "migrate" {
		return runMigrateCLI(ctx, args[1:], getenv, logger)
	}
	if len(args) > 0 && args[0] == "auth" {
		return runAuthCLI(args[1:])
	}

	cfg, err := ParseConfig(args, getenv)
	if err != nil {
		return err
	}
	if strings.TrimSpace(cfg.DatabaseURL) == "" {
		return errors.New("database URL is required")
	}
	if cfg.PublicBearerToken.Reveal() == "" {
		return errors.New("CONTRACTOR_PUBLIC_BEARER_TOKEN is required")
	}
	bootstrap, err := auth.LoadBootstrap(cfg.LocalAuthFile)
	if err != nil {
		return fmt.Errorf("load local authentication: %w", err)
	}
	if strings.TrimSpace(cfg.PublicUserID) != "" && cfg.PublicUserID != bootstrap.Principal.UserID {
		return errors.New("CONTRACTOR_PUBLIC_USER_ID does not match local-auth userId")
	}
	authentication, err := auth.NewService(bootstrap, auth.Options{})
	if err != nil {
		return fmt.Errorf("configure local authentication: %w", err)
	}
	browserOrigins, err := auth.NewOriginPolicy(cfg.BrowserOrigins, cfg.InsecureLoopbackCookie)
	if err != nil {
		return fmt.Errorf("configure browser origins: %w", err)
	}
	if strings.TrimSpace(cfg.CAFile) == "" || strings.TrimSpace(cfg.CertificateFile) == "" ||
		strings.TrimSpace(cfg.PrivateKeyFile) == "" {
		return errors.New("Control Plane certificate, private key, and deployment CA are required")
	}
	if cfg.PlannerTimeout <= 0 {
		return errors.New("Planner timeout is required")
	}
	pool, err := persistencepostgres.OpenPool(ctx, cfg.DatabaseURL, persistencepostgres.PoolOptions{})
	if err != nil {
		return err
	}
	defer pool.Close()
	configurationManager, err := workflowconfig.NewManager(workflowconfig.ManagerOptions{
		OperatorRoot: cfg.OperatorConfigRoot,
		ManagedRoot:  cfg.ManagedConfigRoot,
		Descriptors:  workflowconfig.MVPDescriptors(),
		Audit:        configaudit.New(pool),
		Logger:       logger,
	})
	if err != nil {
		return fmt.Errorf("load configuration: %w", err)
	}
	snapshot := configurationManager.Snapshot()
	credentialRepository := credentials.NewRepository(pool)
	activeCredentialCount, err := credentialRepository.CountCredentials(ctx)
	if err != nil {
		return fmt.Errorf("inspect encrypted LLM credentials: %w", err)
	}
	runtimeCredentialRepository := credentials.NewRuntimeCredentialRepository(pool)
	storedRuntimeCredentialCount, err := runtimeCredentialRepository.CountStored(ctx)
	if err != nil {
		return fmt.Errorf("inspect encrypted Runtime credentials: %w", err)
	}
	if storedRuntimeCredentialCount > 0 && activeCredentialCount > math.MaxInt64-storedRuntimeCredentialCount {
		return errors.New("encrypted credential count is invalid")
	}
	activeCredentialCount += storedRuntimeCredentialCount
	tokenCipher, err := credentials.RequireTokenCipher(cfg.CredentialMasterKeyFile, activeCredentialCount)
	if err != nil {
		return fmt.Errorf("configure encrypted LLM credentials: %w", err)
	}
	if tokenCipher != nil {
		if err := credentialRepository.VerifyActiveKey(ctx, tokenCipher.KeyID()); err != nil {
			return fmt.Errorf("verify encrypted LLM credential key: %w", err)
		}
		if err := runtimeCredentialRepository.VerifyStoredKey(ctx, tokenCipher.KeyID()); err != nil {
			return fmt.Errorf("verify encrypted Runtime credential key: %w", err)
		}
	}
	encryptedCredentialProvider, err := credentials.NewEncryptedProvider(credentialRepository, tokenCipher)
	if err != nil {
		return fmt.Errorf("configure encrypted LLM credentials: %w", err)
	}
	developmentCredentialProvider, err := developmentCredentials(snapshot, cfg)
	if err != nil {
		return fmt.Errorf("configure development LLM credentials: %w", err)
	}
	credentialProvider, err := credentials.NewCompositeProvider(
		developmentCredentialProvider, encryptedCredentialProvider,
	)
	if err != nil {
		return fmt.Errorf("compose LLM credential providers: %w", err)
	}
	adminBindings, err := litellmcredentials.LoadAdminBindings(
		cfg.LLMGatewayAdminBindingsFile, configurationManager,
	)
	if err != nil {
		return fmt.Errorf("load LLM Gateway admin bindings: %w", err)
	}
	if adminBindings.Len() != 0 && tokenCipher == nil {
		return errors.New("managed LLM Gateway credentials require --credential-master-key-file")
	}
	liteLLMManager, err := litellmcredentials.NewManager(
		adminBindings, configurationManager, litellmcredentials.Options{},
	)
	if err != nil {
		return fmt.Errorf("configure LiteLLM credential manager: %w", err)
	}
	credentialManagers, err := credentials.NewManagerRegistry(credentials.ManagerRegistration{
		Implementation: contracts.LiteLLMVirtualKeysManager,
		Manager:        liteLLMManager,
	})
	if err != nil {
		return fmt.Errorf("configure Gateway credential managers: %w", err)
	}
	credentialBarrier := credentials.NewLifecycleBarrier()
	credentialLifecycle, err := credentials.NewService(credentials.ServiceOptions{
		Pool: pool, Gateways: configurationManager, Managers: credentialManagers,
		Runs: runstore.NewPostgresStore(pool), Cipher: tokenCipher, Barrier: credentialBarrier,
	})
	if err != nil {
		return fmt.Errorf("configure LLM credential lifecycle: %w", err)
	}
	if err := credentialLifecycle.Recover(ctx); err != nil {
		return fmt.Errorf("recover LLM credential operations: %w", err)
	}
	runtimeCredentialLifecycle, err := credentials.NewRuntimeCredentialService(
		credentials.RuntimeCredentialServiceOptions{
			Pool: pool, Cipher: tokenCipher, Usage: runtimeCredentialRepository,
			Barrier: credentialBarrier,
		},
	)
	if err != nil {
		return fmt.Errorf("configure Runtime credential lifecycle: %w", err)
	}
	runtimeConfigPublisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool,
		GatewayResolver: runtimeconfig.GatewayResolverFunc(func(_ context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
			return configurationManager.LLMGateway(selector)
		}),
		RuntimeCredentials: runtimeCredentialLifecycle,
	})
	if err != nil {
		return fmt.Errorf("configure RuntimeConfig publisher: %w", err)
	}
	runtimeBindingService, err := runtimeconfig.NewBindingService(pool, runtimeCredentialLifecycle)
	if err != nil {
		return fmt.Errorf("configure Runtime label bindings: %w", err)
	}
	runtimeConfigManagement, err := runtimeconfig.NewManagementService(
		pool, runtimeConfigPublisher, runtimeBindingService,
	)
	if err != nil {
		return fmt.Errorf("configure RuntimeConfig management: %w", err)
	}
	files := mtls.Files{
		Certificate: cfg.CertificateFile, PrivateKey: cfg.PrivateKeyFile, CA: cfg.CAFile,
	}
	privateTLS, err := mtls.ControlPlaneServerConfig(files)
	if err != nil {
		return fmt.Errorf("configure private mTLS server: %w", err)
	}
	registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{})
	if err != nil {
		return fmt.Errorf("configure Control Plane registry: %w", err)
	}
	principalService, err := runtimeconfig.NewPrincipalService(runtimeconfig.PrincipalServiceOptions{
		Pool: pool, DeletionGuard: registry,
	})
	if err != nil {
		return fmt.Errorf("configure Runtime Agent principals: %w", err)
	}
	placementAllocator, err := controlplane.NewPlacementAllocator(controlplane.PlacementAllocatorOptions{
		Pool: pool, Registry: registry, Gateways: configurationManager,
		LLMCredentials: credentialProvider, RuntimeCredentials: runtimeCredentialLifecycle,
		CredentialGuard: credentialLifecycle,
	})
	if err != nil {
		return fmt.Errorf("configure candidate Runtime placement: %w", err)
	}
	runtimeClient, err := controlplane.NewMTLSRuntimeControlClient(files, cfg.RuntimeRequestTimeout)
	if err != nil {
		return fmt.Errorf("configure Runtime Agent client: %w", err)
	}
	workers, err := controlplane.NewRuntimeBatchController(
		runtimeClient,
		registry,
		controlplane.RuntimeBatchOptions{CleanupTimeout: cfg.RuntimeRequestTimeout},
	)
	if err != nil {
		return fmt.Errorf("configure Runtime Agent lifecycle: %w", err)
	}
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	artifactInspector, err := planner.NewArtifactServiceInspector(artifactService)
	if err != nil {
		return err
	}
	plannerSessions, err := plannersession.New(runstore.NewPostgresStore(pool), plannersession.Options{})
	if err != nil {
		return fmt.Errorf("configure Planner sessions: %w", err)
	}
	a2aInvoker, err := plannera2a.NewMTLS(files, cfg.RuntimeRequestTimeout, plannera2a.Options{})
	if err != nil {
		return fmt.Errorf("configure A2A client: %w", err)
	}
	passthrough, err := planner.NewPassthroughFactory(plannerSessions, a2aInvoker, artifactInspector)
	if err != nil {
		return err
	}
	plannerModelFactory := func(access planner.ModelAccess) (model.LLM, error) {
		return streamline.NewOpenAICompatibleModel(streamline.GatewaySettings{
			URL: access.LLMGateway.URL, Token: access.Token, Model: access.ModelPolicy.Model,
			MaxOutputTokens: access.ModelPolicy.MaxOutputTokens, RequestTimeout: cfg.PlannerTimeout,
		})
	}
	streamlineLimits := streamline.DefaultLimits()
	streamlineLimits.MaxWallTime = cfg.PlannerTimeout
	streamlineFactory, err := streamline.NewConfiguredFactory(
		plannerSessions, plannerSessions, a2aInvoker, artifactInspector, plannerModelFactory, streamlineLimits,
	)
	if err != nil {
		return fmt.Errorf("configure Streamline Planner: %w", err)
	}
	routerFactory, err := plannerrouter.NewConfiguredFactory(
		plannerSessions, plannerSessions, a2aInvoker, artifactInspector, plannerModelFactory, streamlineLimits,
	)
	if err != nil {
		return fmt.Errorf("configure Router Planner: %w", err)
	}
	plannerRegistry, err := planner.NewRegistry(passthrough, streamlineFactory, routerFactory)
	if err != nil {
		return err
	}
	artifactResolver, err := scheduler.NewArtifactServiceResolver(artifactService)
	if err != nil {
		return err
	}
	transactions, err := scheduler.NewPostgresPersistence(pool)
	if err != nil {
		return err
	}
	runtimeSettings := contracts.RuntimeSettings{
		ArtifactAPIURL:        strings.TrimRight(cfg.PrivateURL, "/") + "/private/v1",
		RequestTimeoutSeconds: int(cfg.RuntimeRequestTimeout / time.Second),
	}
	workflowScheduler, err := scheduler.New(
		runstore.NewPostgresStore(pool),
		transactions,
		artifactResolver,
		placementAllocator,
		workers,
		plannerRegistry,
		scheduler.Options{
			OperationTimeout:   cfg.RuntimeRequestTimeout,
			PlannerTimeout:     cfg.PlannerTimeout,
			RuntimeSettings:    runtimeSettings,
			Credentials:        credentialProvider,
			RuntimeCredentials: runtimeCredentialLifecycle,
			TelemetrySecrets: []string{
				cfg.DatabaseURL, cfg.PublicBearerToken.Reveal(),
				cfg.DevelopmentWorkerToken.Reveal(), cfg.DevelopmentPlannerToken.Reveal(),
			},
			Logger: logger,
		},
	)
	if err != nil {
		return fmt.Errorf("configure Workflow Scheduler: %w", err)
	}
	eventListener, err := runstore.NewPostgresRunEventListener(pool)
	if err != nil {
		return fmt.Errorf("configure WorkflowRun event listener: %w", err)
	}
	eventHub, err := publicevents.NewHub(publicevents.Options{
		Context: ctx, Authentication: authentication, Origins: browserOrigins,
		Runs: runstore.NewPostgresStore(pool), Operations: registry,
		RunNotifications: eventListener, Logger: logger,
	})
	if err != nil {
		return fmt.Errorf("configure public event WebSocket: %w", err)
	}
	defer eventHub.Close()
	publicHandler, err := publicapi.NewHandler(publicapi.Dependencies{
		Authentication: authentication, BrowserOrigins: browserOrigins,
		InsecureLoopbackCookie: cfg.InsecureLoopbackCookie,
		Config:                 configurationManager, ConfigurationPublisher: configurationManager,
		Runs: runstore.NewPostgresStore(pool), Artifacts: artifactService,
		Credentials: credentialProvider, ManagedCredentials: credentialLifecycle,
		RuntimeConfigs: runtimeConfigManagement, RuntimeCredentials: runtimeCredentialLifecycle,
		Metrics:      telemetry.NewRepository(pool),
		PlannerPlans: plannerSessions,
		Operations:   registry, OperationsInvalidator: registry, Events: eventHub,
		Transactions: postgresPublicUnitOfWork{pool: pool},
		BearerToken:  cfg.PublicBearerToken,
		RunNotifier:  workflowScheduler, Logger: logger,
	})
	if err != nil {
		return fmt.Errorf("configure public API: %w", err)
	}

	controlHandler, err := controlplane.NewHTTPHandler(
		registry, controlplane.HTTPOptions{Logger: logger, Principals: principalService},
	)
	if err != nil {
		return fmt.Errorf("configure private Control Plane API: %w", err)
	}
	artifactHandler, err := privateartifacts.NewHandler(privateartifacts.Dependencies{
		Registry: registry, Artifacts: artifactService, Logger: logger,
	})
	if err != nil {
		return fmt.Errorf("configure private Artifact API: %w", err)
	}
	privateHandler := http.NewServeMux()
	privateHandler.Handle("/private/v1/agents/", controlHandler)
	privateHandler.Handle("/private/v1/allocations/", artifactHandler)

	publicListener, err := net.Listen("tcp", cfg.ListenAddress)
	if err != nil {
		return fmt.Errorf("listen on %q: %w", cfg.ListenAddress, err)
	}
	privateTCPListener, err := net.Listen("tcp", cfg.PrivateListenAddress)
	if err != nil {
		_ = publicListener.Close()
		return fmt.Errorf("listen privately on %q: %w", cfg.PrivateListenAddress, err)
	}
	privateListener := tls.NewListener(privateTCPListener, privateTLS)
	return ServeSystem(
		ctx,
		publicListener,
		privateListener,
		cfg.ShutdownTimeout,
		logger,
		NewHandler(publicHandler),
		privateHandler,
		workflowScheduler,
	)
}

// Serve runs the bootstrap HTTP server on an already-created listener. Taking
// the listener as input keeps shutdown behavior deterministic in tests.
func Serve(
	ctx context.Context,
	listener net.Listener,
	shutdownTimeout time.Duration,
	logger *slog.Logger,
) error {
	return ServeHandler(ctx, listener, shutdownTimeout, logger, NewHandler())
}

// ServeHandler runs a composed process handler and preserves Serve as the
// small health-only harness used by bootstrap tests.
func ServeHandler(
	ctx context.Context,
	listener net.Listener,
	shutdownTimeout time.Duration,
	logger *slog.Logger,
	handler http.Handler,
) error {
	server := &http.Server{
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		errCh <- server.Serve(listener)
	}()

	logger.Info("contractor server listening", "address", listener.Addr().String())

	select {
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return fmt.Errorf("serve HTTP: %w", err)
	case <-ctx.Done():
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		_ = server.Close()
		return fmt.Errorf("shutdown HTTP server: %w", err)
	}

	if err := <-errCh; err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("serve HTTP during shutdown: %w", err)
	}
	logger.Info("contractor server stopped")
	return nil
}

// NewHandler composes unauthenticated process health routes with the optional
// authenticated public API.
func NewHandler(publicAPI ...http.Handler) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", writeHealthy)
	mux.HandleFunc("GET /readyz", writeHealthy)
	if len(publicAPI) == 1 && publicAPI[0] != nil {
		mux.Handle("/v1/", publicAPI[0])
	}
	return mux
}

type postgresPublicUnitOfWork struct{ pool *pgxpool.Pool }

func (u postgresPublicUnitOfWork) Do(
	ctx context.Context,
	fn func(publicapi.RunWriter, *artifacts.Service) error,
) error {
	return persistencepostgres.InTx(ctx, u.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		return fn(
			runstore.NewPostgresStore(tx),
			artifacts.NewService(artifacts.NewPostgresRepository(tx)),
		)
	})
}

func writeHealthy(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, "{\"status\":\"ok\"}\n")
}
