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
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

type blobCleanupConfig struct {
	databaseURL, path string
	apply             bool
}

func parseBlobCleanupConfig(args []string, getenv func(string) string) (cfg blobCleanupConfig, err error) {
	if len(args) == 0 || args[0] != "cleanup" {
		return cfg, errors.New("expected blobs cleanup")
	}
	cfg.databaseURL = getenv("CONTRACTOR_DATABASE_URL")
	cfg.path = getenv("CONTRACTOR_ARTIFACT_BLOB_PATH")
	offline := false
	flags := flag.NewFlagSet("contractor-server blobs cleanup", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.StringVar(&cfg.databaseURL, "database-url", cfg.databaseURL, "PostgreSQL registry URL")
	flags.StringVar(&cfg.path, "artifact-blob-path", cfg.path, "filesystem blob directory")
	flags.BoolVar(&cfg.apply, "apply", false, "remove unreferenced files (default: dry-run)")
	flags.BoolVar(&offline, "offline", false, "acknowledge every Server/writer using this store is stopped")
	if err = flags.Parse(args[1:]); err != nil {
		return cfg, errors.New("invalid blob cleanup flags")
	}
	if flags.NArg() != 0 {
		return cfg, errors.New("unexpected blob cleanup arguments")
	}
	if cfg.apply && !offline {
		return cfg, errors.New("cleanup --apply requires --offline and all Server/writer processes stopped")
	}
	if cfg.databaseURL == "" {
		return cfg, errors.New("database URL is required")
	}
	_, err = artifacts.ValidateBlobConfig(string(artifacts.BlobFilesystem), cfg.path)
	return
}

func runBlobCleanupCLI(ctx context.Context, args []string, getenv func(string) string, logger *slog.Logger) error {
	cfg, err := parseBlobCleanupConfig(args, getenv)
	if err != nil {
		return err
	}
	pool, err := postgres.OpenPool(ctx, cfg.databaseURL, postgres.PoolOptions{MaxConnections: 1, Logger: logger})
	if err != nil {
		return err
	}
	defer pool.Close()
	report, err := artifacts.CleanupFilesystemBlobs(ctx, pool, cfg.path, cfg.apply)
	if err != nil {
		return fmt.Errorf("offline blob cleanup: %w", err)
	}
	return json.NewEncoder(os.Stdout).Encode(report)
}
