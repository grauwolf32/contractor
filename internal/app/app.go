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
	"github.com/grauwolf32/contractor/internal/auth"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/gitimport"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
	"github.com/grauwolf32/contractor/internal/persistence/configaudit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// Config contains process-level settings needed by the bootstrap server.
type Config struct {
	ListenAddress         string
	PrivateListenAddress  string
	PrivateURL            string
	ShutdownTimeout       time.Duration
	RuntimeRequestTimeout time.Duration
	WorkerRequestTimeout  time.Duration
	DatabaseURL           string
	ArtifactBlobBackend   artifacts.BlobBackend
	ArtifactBlobPath      string
	GitImport             gitimport.Config
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
	PerformanceMetrics          bool
	Pprof                       bool
	PprofListen                 string
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
	if len(args) > 0 && args[0] == "blobs" {
		return runBlobCleanupCLI(ctx, args[1:], getenv, logger)
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
	profilingServer, err := configureProfiling(cfg)
	if err != nil {
		return err
	}
	if profilingServer != nil {
		defer func() { _ = profilingServer.Close() }()
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
	pool, err := persistencepostgres.OpenPool(ctx, cfg.DatabaseURL, persistencepostgres.PoolOptions{Logger: logger})
	if err != nil {
		return err
	}
	defer pool.Close()
	var blobStore artifacts.BlobStore = artifacts.PostgresBlobStore{}
	if cfg.ArtifactBlobBackend == artifacts.BlobFilesystem {
		files, err := artifacts.OpenFilesystemBlobStore(ctx, cfg.ArtifactBlobPath)
		if err != nil {
			return err
		}
		defer files.Close()
		blobStore = files
	}
	if err := artifacts.ClaimBlobBackend(ctx, pool, cfg.ArtifactBlobBackend); err != nil {
		return err
	}
	ctx = artifacts.WithBlobRuntime(ctx, artifacts.NewBlobRuntime(blobStore, logger))
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
	credentialSet, err := configureCredentials(ctx, pool, configurationManager, cfg)
	if err != nil {
		return err
	}
	plannerTelemetryRegistry, err := telemetry.NewBuiltinPlannerAdapterRegistry()
	if err != nil {
		return fmt.Errorf("configure Planner telemetry adapters: %w", err)
	}

	control, err := configureControlPlane(
		pool, configurationManager, cfg, credentialSet, plannerTelemetryRegistry,
	)
	if err != nil {
		return err
	}
	catalogs, err := configureCatalogs(ctx, pool, configurationManager, cfg, bootstrap.Principal.UserID, logger)
	if err != nil {
		return err
	}
	workflows, err := configureWorkflows(
		pool, cfg, catalogs, control, credentialSet, plannerTelemetryRegistry, logger,
	)
	if err != nil {
		return err
	}
	eventListener, err := runstore.NewPostgresRunEventListener(pool)
	if err != nil {
		return fmt.Errorf("configure WorkflowRun event listener: %w", err)
	}
	eventHub, err := publicevents.NewHub(publicevents.Options{
		Context: ctx, Authentication: authentication, Origins: browserOrigins,
		Runs: runstore.NewPostgresStore(pool), Operations: control.registry,
		RunNotifications: eventListener, Logger: logger,
	})
	if err != nil {
		return fmt.Errorf("configure public event WebSocket: %w", err)
	}
	defer eventHub.Close()
	audits, err := configureAudits(pool, configurationManager, credentialSet, catalogs, workflows, logger)
	if err != nil {
		return err
	}
	handlers, err := configureHTTP(
		pool, configurationManager, cfg, authentication, browserOrigins, eventHub,
		credentialSet, control, catalogs, workflows, audits, logger,
	)
	if err != nil {
		return err
	}
	runners := handlers.runners
	if profilingServer != nil {
		runners = append(runners, profilingServer)
	}

	publicListener, err := net.Listen("tcp", cfg.ListenAddress)
	if err != nil {
		return fmt.Errorf("listen on %q: %w", cfg.ListenAddress, err)
	}
	privateTCPListener, err := net.Listen("tcp", cfg.PrivateListenAddress)
	if err != nil {
		_ = publicListener.Close()
		return fmt.Errorf("listen privately on %q: %w", cfg.PrivateListenAddress, err)
	}
	privateListener := tls.NewListener(privateTCPListener, control.tls)
	return ServeSystem(
		ctx,
		publicListener,
		privateListener,
		cfg.ShutdownTimeout,
		logger,
		handlers.public,
		handlers.private,
		runners,
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
	return newProcessHandler(http.HandlerFunc(writeHealthy), publicAPI...)
}

func newProcessHandler(readiness http.Handler, publicAPI ...http.Handler) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", writeHealthy)
	mux.Handle("GET /readyz", readiness)
	if len(publicAPI) == 1 && publicAPI[0] != nil {
		mux.Handle("/v1/", publicAPI[0])
	}
	return mux
}

func newPrivateHandler(controlHandler http.Handler, artifactHandler http.Handler) http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/private/v1/agents/", controlHandler)
	mux.Handle("/private/v1/allocations/", artifactHandler)
	return mux
}

func writeHealthy(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, "{\"status\":\"ok\"}\n")
}
