package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/grauwolf32/contractor/internal/app"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if err := app.RunCLI(ctx, os.Args[1:], os.Getenv, logger); err != nil {
		logger.Error("contractor server stopped", "error", err)
		os.Exit(1)
	}
}
