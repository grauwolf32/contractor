// Package app contains the Contractor Server composition root.
package app

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	privateartifacts "github.com/grauwolf32/contractor/internal/httpapi/privateartifacts"
	publicapi "github.com/grauwolf32/contractor/internal/httpapi/public"
	"github.com/grauwolf32/contractor/internal/mtls"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	plannera2a "github.com/grauwolf32/contractor/internal/planner/a2a"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/scheduler"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Config contains process-level settings needed by the bootstrap server.
type Config struct {
	ListenAddress         string
	PrivateListenAddress  string
	PrivateURL            string
	ShutdownTimeout       time.Duration
	RuntimeRequestTimeout time.Duration
	DatabaseURL           string
	ConfigRoot            string
	CAFile                string
	CertificateFile       string
	PrivateKeyFile        string
	LLMGatewayURL         string
	LLMGatewayToken       contracts.SecretString
	PlannerGatewayURL     string
	PlannerGatewayToken   contracts.SecretString
	PlannerModel          string
	PlannerTimeout        time.Duration
	PublicBearerToken     contracts.SecretString
	PublicUserID          string
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

	cfg, err := ParseConfig(args, getenv)
	if err != nil {
		return err
	}
	if strings.TrimSpace(cfg.DatabaseURL) == "" {
		return errors.New("database URL is required")
	}
	if strings.TrimSpace(cfg.PublicUserID) == "" || cfg.PublicBearerToken.Reveal() == "" {
		return errors.New("CONTRACTOR_PUBLIC_USER_ID and CONTRACTOR_PUBLIC_BEARER_TOKEN are required")
	}
	if strings.TrimSpace(cfg.CAFile) == "" || strings.TrimSpace(cfg.CertificateFile) == "" ||
		strings.TrimSpace(cfg.PrivateKeyFile) == "" {
		return errors.New("Control Plane certificate, private key, and deployment CA are required")
	}
	if strings.TrimSpace(cfg.LLMGatewayURL) == "" || cfg.LLMGatewayToken.Reveal() == "" {
		return errors.New("LLM Gateway URL and token are required")
	}
	if strings.TrimSpace(cfg.PlannerGatewayURL) == "" || cfg.PlannerGatewayToken.Reveal() == "" ||
		strings.TrimSpace(cfg.PlannerModel) == "" || cfg.PlannerTimeout <= 0 {
		return errors.New("Planner LLM Gateway URL, token, model, and timeout are required")
	}
	snapshot, err := workflowconfig.Load(cfg.ConfigRoot, workflowconfig.MVPDescriptors())
	if err != nil {
		return fmt.Errorf("load configuration: %w", err)
	}
	pool, err := persistencepostgres.OpenPool(ctx, cfg.DatabaseURL, persistencepostgres.PoolOptions{})
	if err != nil {
		return err
	}
	defer pool.Close()
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
	plannerModel, err := streamline.NewOpenAICompatibleModel(streamline.GatewaySettings{
		URL: cfg.PlannerGatewayURL, Token: cfg.PlannerGatewayToken, Model: cfg.PlannerModel,
		RequestTimeout: cfg.PlannerTimeout,
	})
	if err != nil {
		return fmt.Errorf("configure Planner LLM Gateway: %w", err)
	}
	streamlineLimits := streamline.DefaultLimits()
	streamlineLimits.MaxWallTime = cfg.PlannerTimeout
	streamlineFactory, err := streamline.NewFactory(
		plannerSessions, plannerSessions, a2aInvoker, artifactInspector, plannerModel, streamlineLimits,
	)
	if err != nil {
		return fmt.Errorf("configure Streamline Planner: %w", err)
	}
	plannerRegistry, err := planner.NewRegistry(passthrough, streamlineFactory)
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
		LLMGatewayURL: cfg.LLMGatewayURL, LLMGatewayToken: cfg.LLMGatewayToken,
		ArtifactAPIURL:        strings.TrimRight(cfg.PrivateURL, "/") + "/private/v1",
		RequestTimeoutSeconds: int(cfg.RuntimeRequestTimeout / time.Second),
	}
	workflowScheduler, err := scheduler.New(
		runstore.NewPostgresStore(pool),
		transactions,
		artifactResolver,
		registry,
		workers,
		plannerRegistry,
		scheduler.Options{
			OperationTimeout: cfg.RuntimeRequestTimeout,
			PlannerTimeout:   cfg.PlannerTimeout,
			RuntimeSettings:  runtimeSettings,
			TelemetrySecrets: []string{
				cfg.DatabaseURL, cfg.PublicBearerToken.Reveal(),
				cfg.LLMGatewayToken.Reveal(), cfg.PlannerGatewayToken.Reveal(),
			},
			Logger: logger,
		},
	)
	if err != nil {
		return fmt.Errorf("configure Workflow Scheduler: %w", err)
	}
	publicHandler, err := publicapi.NewHandler(publicapi.Dependencies{
		Config: snapshot, Runs: runstore.NewPostgresStore(pool), Artifacts: artifactService,
		Metrics:      telemetry.NewRepository(pool),
		Transactions: postgresPublicUnitOfWork{pool: pool},
		BearerToken:  cfg.PublicBearerToken, UserID: cfg.PublicUserID,
		RunNotifier: workflowScheduler, Logger: logger,
	})
	if err != nil {
		return fmt.Errorf("configure public API: %w", err)
	}

	controlHandler, err := controlplane.NewHTTPHandler(
		registry, controlplane.HTTPOptions{Logger: logger},
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
