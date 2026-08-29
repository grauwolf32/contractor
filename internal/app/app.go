// Package app contains the Contractor Server composition root.
package app

import (
	"context"
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
	publicapi "github.com/grauwolf32/contractor/internal/httpapi/public"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Config contains process-level settings needed by the bootstrap server.
type Config struct {
	ListenAddress     string
	ShutdownTimeout   time.Duration
	DatabaseURL       string
	ConfigRoot        string
	PublicBearerToken contracts.SecretString
	PublicUserID      string
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
	snapshot, err := workflowconfig.Load(cfg.ConfigRoot, workflowconfig.MVPDescriptors())
	if err != nil {
		return fmt.Errorf("load configuration: %w", err)
	}
	pool, err := persistencepostgres.OpenPool(ctx, cfg.DatabaseURL, persistencepostgres.PoolOptions{})
	if err != nil {
		return err
	}
	defer pool.Close()
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	publicHandler, err := publicapi.NewHandler(publicapi.Dependencies{
		Config: snapshot, Runs: runstore.NewPostgresStore(pool), Artifacts: artifactService,
		Transactions: postgresPublicUnitOfWork{pool: pool},
		BearerToken:  cfg.PublicBearerToken, UserID: cfg.PublicUserID,
	})
	if err != nil {
		return fmt.Errorf("configure public API: %w", err)
	}

	listener, err := net.Listen("tcp", cfg.ListenAddress)
	if err != nil {
		return fmt.Errorf("listen on %q: %w", cfg.ListenAddress, err)
	}

	return ServeHandler(ctx, listener, cfg.ShutdownTimeout, logger, NewHandler(publicHandler))
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
