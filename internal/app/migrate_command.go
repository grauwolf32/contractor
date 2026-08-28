package app

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"strings"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func runMigrateCLI(
	ctx context.Context,
	args []string,
	getenv func(string) string,
	logger *slog.Logger,
) error {
	databaseURL, err := parseMigrationDatabaseURL(args, getenv)
	if err != nil {
		return err
	}
	pool, err := persistencepostgres.OpenPool(ctx, databaseURL, persistencepostgres.PoolOptions{})
	if err != nil {
		return fmt.Errorf("open migration database: %w", err)
	}
	defer pool.Close()
	result, err := persistencepostgres.ApplyMigrations(ctx, pool)
	if err != nil {
		return fmt.Errorf("migrate database: %w", err)
	}
	logger.Info(
		"database migrations complete",
		"applied", len(result.AppliedVersions),
		"current_version", result.CurrentVersion,
	)
	return nil
}

func parseMigrationDatabaseURL(args []string, getenv func(string) string) (string, error) {
	databaseURL := getenv("CONTRACTOR_DATABASE_URL")
	flags := flag.NewFlagSet("contractor-server migrate", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.StringVar(&databaseURL, "database-url", databaseURL, "PostgreSQL connection URL")
	if err := flags.Parse(args); err != nil {
		return "", fmt.Errorf("parse migrate flags: %w", err)
	}
	if flags.NArg() != 0 {
		return "", fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}
	if strings.TrimSpace(databaseURL) == "" {
		return "", fmt.Errorf("database URL is required; set CONTRACTOR_DATABASE_URL or --database-url")
	}
	return databaseURL, nil
}
