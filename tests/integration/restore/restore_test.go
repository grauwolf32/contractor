//go:build integration

package restore_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// This is an offline recovery rehearsal: writers are quiescent during both
// snapshots. Filesystem objects are backed up with the database, not inferred
// from its metadata. Only databases created by this test are ever dropped.
func TestBackupRestorePreservesExactArtifactsAndCAS(t *testing.T) {
	for _, backend := range []artifacts.BlobBackend{artifacts.BlobPostgres, artifacts.BlobFilesystem} {
		t.Run(string(backend), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
			defer cancel()
			admin, config := restoreAdmin(t, ctx)
			source, sourceName := restoreDatabase(t, ctx, admin, config)
			target, targetName := restoreDatabase(t, ctx, admin, config)
			initial, err := postgres.ApplyMigrations(ctx, source)
			if err != nil || len(initial.AppliedVersions) == 0 {
				t.Fatalf("fresh migration: %+v, %v", initial, err)
			}
			if err := artifacts.ClaimBlobBackend(ctx, source, backend); err != nil {
				t.Fatal(err)
			}
			sourceDir, backupDir, targetDir := t.TempDir(), filepath.Join(t.TempDir(), "blobs"), t.TempDir()
			sourceCtx := restoreBlobContext(t, ctx, backend, sourceDir)
			projects := projectstore.NewPostgresStore(source)
			params := projectstore.CreateParams{
				ProjectID: "restore-project", OwnerID: "restore-owner", Kind: projectstore.KindProject,
				Name: "Restored project", IdempotencyKey: "restore-project-create",
				RequestDigest: "sha256:" + strings.Repeat("a", 64),
			}
			project, _, err := projects.Create(sourceCtx, params)
			if err != nil {
				t.Fatal(err)
			}
			store, err := artifacts.NewService(artifacts.NewPostgresRepository(source)).Project(project.ProjectID)
			if err != nil {
				t.Fatal(err)
			}
			ref := artifacts.ArtifactRef{Namespace: "docs", Name: "recovery"}
			firstBody := []byte("retained first revision: Unicode Привет and <>&\n")
			first, err := store.Write(sourceCtx, ref, artifacts.Payload{MediaType: "text/plain", Data: firstBody}, nil)
			if err != nil {
				t.Fatal(err)
			}
			latestBody := bytes.Repeat([]byte("binary\x00payload\xff\n"), 1024)
			latest, err := store.Write(sourceCtx, ref, artifacts.Payload{MediaType: "application/octet-stream", Data: latestBody}, first.Ref.Revision)
			if err != nil {
				t.Fatal(err)
			}
			before, err := store.Metadata(sourceCtx, latest.Ref)
			if err != nil {
				t.Fatal(err)
			}
			archive := filepath.Join(t.TempDir(), "database.dump")
			runPostgresTool(t, ctx, config, sourceName, "pg_dump", "--format=custom", "--no-owner", "--file", archive)
			if backend == artifacts.BlobFilesystem {
				if err := os.CopyFS(backupDir, os.DirFS(sourceDir)); err != nil {
					t.Fatal(err)
				}
				// Recovery must not accidentally continue reading the original store.
				if err := os.RemoveAll(sourceDir); err != nil {
					t.Fatal(err)
				}
			}
			source.Close()
			if _, err := admin.Exec(ctx, "DROP DATABASE "+pgx.Identifier{sourceName}.Sanitize()); err != nil {
				t.Fatal(err)
			}
			runPostgresTool(t, ctx, config, targetName, "pg_restore", "--exit-on-error", "--no-owner", "--dbname", targetName, archive)
			restored, err := postgres.ApplyMigrations(ctx, target)
			if err != nil || len(restored.AppliedVersions) != 0 || restored.CurrentVersion != initial.CurrentVersion {
				t.Fatalf("restored migration ledger: %+v, %v", restored, err)
			}
			if err := artifacts.ClaimBlobBackend(ctx, target, backend); err != nil {
				t.Fatal(err)
			}
			targetCtx := restoreBlobContext(t, ctx, backend, targetDir)
			recoveredStore, err := artifacts.NewService(artifacts.NewPostgresRepository(target)).Project(project.ProjectID)
			if err != nil {
				t.Fatal(err)
			}
			if backend == artifacts.BlobFilesystem {
				if _, err := recoveredStore.Read(targetCtx, latest.Ref); err == nil {
					t.Fatal("database-only restore fabricated missing filesystem payload")
				}
				if err := os.CopyFS(targetDir, os.DirFS(backupDir)); err != nil {
					t.Fatal(err)
				}
			}
			for _, expected := range []struct {
				ref  artifacts.ArtifactRef
				body []byte
			}{{first.Ref, firstBody}, {latest.Ref, latestBody}, {ref, latestBody}} {
				read, err := recoveredStore.Read(targetCtx, expected.ref)
				if err != nil || !bytes.Equal(read.Payload.Data, expected.body) {
					t.Fatalf("restored exact/current content differs: %v", err)
				}
			}
			after, err := recoveredStore.Metadata(targetCtx, latest.Ref)
			if err != nil || after.Digest != before.Digest || after.Size != before.Size || after.MediaType != before.MediaType {
				t.Fatalf("restored metadata differs: %v", err)
			}
			replayed, created, err := projectstore.NewPostgresStore(target).Create(targetCtx, params)
			if err != nil || created || replayed.ProjectID != project.ProjectID || replayed.Revision != project.Revision {
				t.Fatalf("restored idempotency receipt: %+v, created=%v, %v", replayed, created, err)
			}
			if _, err := recoveredStore.Write(targetCtx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("stale")}, first.Ref.Revision); !errors.Is(err, artifacts.ErrArtifactConflict) {
				t.Fatalf("restored stale CAS: %v", err)
			}
			if _, err := recoveredStore.Write(targetCtx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("resumed")}, latest.Ref.Revision); err != nil {
				t.Fatalf("resumed write: %v", err)
			}
			t.Logf("restored schema %d, two exact revisions, idempotency and CAS; backend=%s", restored.CurrentVersion, backend)
		})
	}
}

func restoreAdmin(t *testing.T, ctx context.Context) (*pgxpool.Pool, *pgxpool.Config) {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL must point to disposable PostgreSQL")
	}
	config, err := pgxpool.ParseConfig(url)
	if err != nil {
		t.Fatal("invalid test database configuration")
	}
	config.MaxConns, config.MinConns, config.MinIdleConns = 2, 0, 0
	admin, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal("open isolated test database")
	}
	t.Cleanup(admin.Close)
	return admin, config
}

func restoreDatabase(t *testing.T, ctx context.Context, admin *pgxpool.Pool, config *pgxpool.Config) (*pgxpool.Pool, string) {
	t.Helper()
	var random [8]byte
	if _, err := rand.Read(random[:]); err != nil {
		t.Fatal(err)
	}
	name := "contractor_restore_" + hex.EncodeToString(random[:])
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+pgx.Identifier{name}.Sanitize()); err != nil {
		t.Fatal(err)
	}
	copy := config.Copy()
	copy.ConnConfig.Database = name
	pool, err := pgxpool.NewWithConfig(ctx, copy)
	if err != nil {
		t.Fatal("open fresh restore database")
	}
	t.Cleanup(func() {
		pool.Close()
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, "DROP DATABASE IF EXISTS "+pgx.Identifier{name}.Sanitize()); err != nil {
			t.Errorf("clean up owned restore database: %v", err)
		}
	})
	return pool, name
}

func restoreBlobContext(t *testing.T, ctx context.Context, backend artifacts.BlobBackend, directory string) context.Context {
	t.Helper()
	if backend != artifacts.BlobFilesystem {
		return ctx
	}
	store, err := artifacts.OpenFilesystemBlobStore(ctx, directory)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return artifacts.WithBlobRuntime(ctx, artifacts.NewBlobRuntime(store, nil))
}

func runPostgresTool(t *testing.T, ctx context.Context, config *pgxpool.Config, database, tool string, args ...string) {
	t.Helper()
	command := exec.CommandContext(ctx, tool, args...)
	command.Env = append(os.Environ(),
		"PGHOST="+config.ConnConfig.Host,
		"PGPORT="+strconv.Itoa(int(config.ConnConfig.Port)),
		"PGUSER="+config.ConnConfig.User,
		"PGPASSWORD="+config.ConnConfig.Password,
		"PGDATABASE="+database,
		"PGSSLMODE=disable",
	)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("%s failed: %v; %s", tool, err, strings.ReplaceAll(string(output), config.ConnConfig.Password, "<redacted>"))
	}
}
