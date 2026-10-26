package app

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func runDataMigrateCLI(ctx context.Context, args []string, getenv func(string) string, logger *slog.Logger) error {
	if err := commandGroupHelp(args, "data-migrate", [2]string{"finding-checks", "derive exact finding check indexes from retained proposal bytes"}); err != nil {
		return err
	}
	if len(args) == 0 || args[0] != "finding-checks" {
		return errors.New("expected data-migrate finding-checks")
	}
	databaseURL, path := getenv("CONTRACTOR_DATABASE_URL"), getenv("CONTRACTOR_ARTIFACT_BLOB_PATH")
	batch := 128
	flags := flag.NewFlagSet("contractor-server data-migrate finding-checks", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.StringVar(&databaseURL, "database-url", databaseURL, "PostgreSQL registry URL")
	flags.StringVar(&path, "artifact-blob-path", path, "existing filesystem blob root, including when new writes use PostgreSQL")
	flags.IntVar(&batch, "batch-size", batch, "receipt headers per batch, 1 to 128")
	if err := parseCommandFlags(flags, args[1:]); err != nil {
		return fmt.Errorf("parse data migration flags: %w", err)
	}
	if flags.NArg() != 0 || databaseURL == "" || batch < 1 || batch > 128 {
		return errors.New("data migration requires a database URL and batch size between 1 and 128")
	}
	pool, err := persistencepostgres.OpenPool(ctx, databaseURL, persistencepostgres.PoolOptions{MaxConnections: 1, Logger: logger})
	if err != nil {
		return err
	}
	defer pool.Close()
	if err := persistencepostgres.VerifyMigrations(ctx, pool); err != nil {
		return err
	}
	_, store, err := artifacts.OpenBlobStore(ctx, artifacts.BlobPostgres, path, artifacts.DefaultBlobInlineMaxBytes)
	if err != nil {
		return err
	}
	if closer, ok := store.(io.Closer); ok {
		defer closer.Close()
	}
	ctx = artifacts.WithBlobRuntime(ctx, artifacts.NewBlobRuntime(store, logger))
	service, err := findingintake.New(pool)
	if err != nil {
		return err
	}
	var total findingintake.ChecksRebuildReport
	for {
		report, err := service.RebuildChecksProjectionBatch(ctx, batch)
		total.Examined += report.Examined
		total.Rebuilt += report.Rebuilt
		total.More = report.More
		if err != nil {
			return fmt.Errorf("derive finding check indexes: %w", err)
		}
		if !report.More {
			return json.NewEncoder(os.Stdout).Encode(total)
		}
		if report.Examined == 0 {
			return errors.New("finding check data migration made no progress")
		}
	}
}
