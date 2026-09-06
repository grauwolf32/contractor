package postgres

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io/fs"
	"regexp"
	"sort"
	"strconv"

	"github.com/grauwolf32/contractor/internal/persistence/migrations"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const migrationLockKey int64 = 0x436f6e7472616374

var (
	// ErrMigrationDrift means an already-recorded version no longer matches the
	// embedded migration name or checksum.
	ErrMigrationDrift = errors.New("PostgreSQL migration drift")
	migrationName     = regexp.MustCompile(`^([0-9]{6})_([a-z0-9_]+)\.sql$`)
)

type MigrationResult struct {
	AppliedVersions []int64
	CurrentVersion  int64
}

type migration struct {
	version  int64
	name     string
	contents []byte
	checksum [sha256.Size]byte
}

// ApplyMigrations serializes migrators with a transaction advisory lock,
// verifies recorded checksums, and applies every pending embedded migration.
func ApplyMigrations(ctx context.Context, pool *pgxpool.Pool) (MigrationResult, error) {
	ctx, cancel := WithMigrationBudget(ctx)
	defer cancel()
	available, err := loadMigrations()
	if err != nil {
		return MigrationResult{}, err
	}
	var result MigrationResult
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock($1)`, migrationLockKey); err != nil {
			return fmt.Errorf("lock PostgreSQL migrations: %w", err)
		}
		if _, err := tx.Exec(ctx, `
CREATE TABLE IF NOT EXISTS contractor_schema_migrations (
    version bigint PRIMARY KEY CHECK (version > 0),
    name text NOT NULL CHECK (btrim(name) <> ''),
    checksum bytea NOT NULL CHECK (octet_length(checksum) = 32),
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
)`); err != nil {
			return fmt.Errorf("create migration ledger: %w", err)
		}

		applied, err := readAppliedMigrations(ctx, tx)
		if err != nil {
			return err
		}
		availableVersions := make(map[int64]struct{}, len(available))
		for _, item := range available {
			availableVersions[item.version] = struct{}{}
		}
		for version := range applied {
			if _, exists := availableVersions[version]; !exists {
				return fmt.Errorf("%w: database contains unknown version %06d", ErrMigrationDrift, version)
			}
		}
		for _, item := range available {
			if existing, ok := applied[item.version]; ok {
				if existing.name != item.name || existing.checksum != item.checksum {
					return fmt.Errorf("%w: version %06d differs from embedded %s", ErrMigrationDrift, item.version, item.name)
				}
				result.CurrentVersion = item.version
				continue
			}
			if _, err := tx.Exec(ctx, string(item.contents)); err != nil {
				return fmt.Errorf("apply migration %s: %w", item.name, err)
			}
			if _, err := tx.Exec(ctx,
				`INSERT INTO contractor_schema_migrations (version, name, checksum) VALUES ($1, $2, $3)`,
				item.version, item.name, item.checksum[:],
			); err != nil {
				return fmt.Errorf("record migration %s: %w", item.name, err)
			}
			result.AppliedVersions = append(result.AppliedVersions, item.version)
			result.CurrentVersion = item.version
		}
		return nil
	})
	if err != nil {
		return MigrationResult{}, err
	}
	return result, nil
}

type appliedMigration struct {
	name     string
	checksum [sha256.Size]byte
}

func readAppliedMigrations(ctx context.Context, tx pgx.Tx) (map[int64]appliedMigration, error) {
	rows, err := tx.Query(ctx, `SELECT version, name, checksum FROM contractor_schema_migrations ORDER BY version`)
	if err != nil {
		return nil, fmt.Errorf("read applied migrations: %w", err)
	}
	defer rows.Close()
	result := make(map[int64]appliedMigration)
	for rows.Next() {
		var version int64
		var name string
		var rawChecksum []byte
		if err := rows.Scan(&version, &name, &rawChecksum); err != nil {
			return nil, fmt.Errorf("scan applied migration: %w", err)
		}
		if len(rawChecksum) != sha256.Size {
			return nil, fmt.Errorf("%w: version %06d has invalid checksum length", ErrMigrationDrift, version)
		}
		var checksum [sha256.Size]byte
		copy(checksum[:], rawChecksum)
		result[version] = appliedMigration{name: name, checksum: checksum}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate applied migrations: %w", err)
	}
	return result, nil
}

func loadMigrations() ([]migration, error) {
	entries, err := fs.ReadDir(migrations.Files, ".")
	if err != nil {
		return nil, fmt.Errorf("list embedded migrations: %w", err)
	}
	result := make([]migration, 0, len(entries))
	seen := make(map[int64]string)
	for _, entry := range entries {
		if entry.IsDir() || entry.Name() == "embed.go" {
			continue
		}
		match := migrationName.FindStringSubmatch(entry.Name())
		if match == nil {
			return nil, fmt.Errorf("embedded migration has invalid name %q", entry.Name())
		}
		version, err := strconv.ParseInt(match[1], 10, 64)
		if err != nil || version <= 0 {
			return nil, fmt.Errorf("embedded migration has invalid version %q", entry.Name())
		}
		if previous, duplicate := seen[version]; duplicate {
			return nil, fmt.Errorf("embedded migrations %q and %q share version %06d", previous, entry.Name(), version)
		}
		seen[version] = entry.Name()
		contents, err := migrations.Files.ReadFile(entry.Name())
		if err != nil {
			return nil, fmt.Errorf("read embedded migration %q: %w", entry.Name(), err)
		}
		result = append(result, migration{
			version: version, name: entry.Name(), contents: contents, checksum: sha256.Sum256(contents),
		})
	}
	sort.Slice(result, func(i, j int) bool { return result[i].version < result[j].version })
	if len(result) == 0 {
		return nil, errors.New("no embedded PostgreSQL migrations")
	}
	return result, nil
}
