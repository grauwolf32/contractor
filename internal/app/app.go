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
	"time"
)

// Config contains process-level settings needed by the bootstrap server.
type Config struct {
	ListenAddress   string
	ShutdownTimeout time.Duration
}

// RunCLI parses process configuration and runs the Server until cancellation.
func RunCLI(
	ctx context.Context,
	args []string,
	getenv func(string) string,
	logger *slog.Logger,
) error {
	cfg, err := ParseConfig(args, getenv)
	if err != nil {
		return err
	}

	listener, err := net.Listen("tcp", cfg.ListenAddress)
	if err != nil {
		return fmt.Errorf("listen on %q: %w", cfg.ListenAddress, err)
	}

	return Serve(ctx, listener, cfg.ShutdownTimeout, logger)
}

// Serve runs the bootstrap HTTP server on an already-created listener. Taking
// the listener as input keeps shutdown behavior deterministic in tests.
func Serve(
	ctx context.Context,
	listener net.Listener,
	shutdownTimeout time.Duration,
	logger *slog.Logger,
) error {
	server := &http.Server{
		Handler:           NewHandler(),
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

// NewHandler returns the process-level health routes. Domain routes are added
// by later implementation tasks.
func NewHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", writeHealthy)
	mux.HandleFunc("GET /readyz", writeHealthy)
	return mux
}

func writeHealthy(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, "{\"status\":\"ok\"}\n")
}
