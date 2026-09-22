package app

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func runMigrateCLI(
	ctx context.Context,
	args []string,
	getenv func(string) string,
	logger *slog.Logger,
) error {
	inputs, err := parseMigrationInputs(args, getenv)
	if err != nil {
		return err
	}
	pool, err := persistencepostgres.OpenPool(ctx, inputs.databaseURL, persistencepostgres.PoolOptions{Logger: logger})
	if err != nil {
		return fmt.Errorf("open migration database: %w", err)
	}
	defer pool.Close()
	result, err := persistencepostgres.ApplyMigrationsWithBudgets(ctx, pool, inputs.budgets)
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

type migrationInputs struct {
	databaseURL string
	budgets     persistencepostgres.MigrationBudgets
}

// Migration timeouts are deliberately separate from ServerConfig database
// budgets: schema changes may legitimately run longer than request work.
func parseMigrationInputs(args []string, getenv func(string) string) (migrationInputs, error) {
	inputs := migrationInputs{
		databaseURL: getenv("CONTRACTOR_DATABASE_URL"),
		budgets:     persistencepostgres.DefaultMigrationBudgets(),
	}
	for _, setting := range []struct {
		env    string
		target *time.Duration
	}{
		{"CONTRACTOR_MIGRATE_STATEMENT_TIMEOUT", &inputs.budgets.StatementTimeout},
		{"CONTRACTOR_MIGRATE_LOCK_TIMEOUT", &inputs.budgets.LockTimeout},
	} {
		if encoded := getenv(setting.env); encoded != "" {
			parsed, err := time.ParseDuration(encoded)
			if err != nil {
				return migrationInputs{}, fmt.Errorf("%s must be a duration", setting.env)
			}
			*setting.target = parsed
		}
	}
	flags := flag.NewFlagSet("contractor-server migrate", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.StringVar(&inputs.databaseURL, "database-url", inputs.databaseURL, "PostgreSQL connection URL")
	flags.DurationVar(&inputs.budgets.StatementTimeout, "statement-timeout", inputs.budgets.StatementTimeout, "per-statement migration timeout")
	flags.DurationVar(&inputs.budgets.LockTimeout, "lock-timeout", inputs.budgets.LockTimeout, "migration lock wait timeout")
	if err := parseCommandFlags(flags, args); err != nil {
		return migrationInputs{}, fmt.Errorf("parse migrate flags: %w", err)
	}
	if flags.NArg() != 0 {
		return migrationInputs{}, fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}
	if strings.TrimSpace(inputs.databaseURL) == "" {
		return migrationInputs{}, fmt.Errorf("database URL is required; set CONTRACTOR_DATABASE_URL or --database-url")
	}
	if inputs.budgets.StatementTimeout <= 0 || inputs.budgets.LockTimeout <= 0 {
		return migrationInputs{}, fmt.Errorf("migration statement and lock timeouts must be positive")
	}
	if err := inputs.budgets.Validate(); err != nil {
		return migrationInputs{}, err
	}
	return inputs, nil
}
