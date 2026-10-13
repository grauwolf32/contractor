package postgres

import (
	"bytes"
	"context"
	"crypto/sha256"
	"os"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestPostgresPerBlobStorageMigrationPreservesLocations(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	for _, backend := range []string{"postgresql", "filesystem"} {
		t.Run(backend, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool, _ := isolatedPools(t, ctx, url)
			installMigrationPrefix(t, ctx, pool, 14)
			data := []byte("retained " + backend)
			digest := sha256.Sum256(data)
			var payload []byte
			var key *string
			if backend == "postgresql" {
				payload = data
			} else {
				value := "ab/" + strings.Repeat("1", 32)
				key = &value
			}
			if _, err := pool.Exec(ctx, "INSERT INTO artifact_blobs(sha256,payload,size_bytes,backend,object_key) VALUES($1,$2,$3,$4,$5)", digest[:], payload, len(data), backend, key); err != nil {
				t.Fatal(err)
			}
			var before []byte
			if err := pool.QueryRow(ctx, "SELECT to_jsonb(blob) FROM artifact_blobs blob WHERE sha256=$1", digest[:]).Scan(&before); err != nil {
				t.Fatal(err)
			}
			result, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
			if err != nil || !slices.Contains(result.AppliedVersions, 15) {
				t.Fatalf("remove global storage claim: %+v, %v", result, err)
			}
			var removed bool
			if err := pool.QueryRow(ctx, "SELECT to_regclass('artifact_blob_settings') IS NULL AND to_regprocedure('artifact_blob_backend_guard()') IS NULL AND NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid='artifact_blobs'::regclass AND tgname='artifact_blob_backend_guard')").Scan(&removed); err != nil || !removed {
				t.Fatalf("global guard retained: %v", err)
			}
			var after []byte
			if err := pool.QueryRow(ctx, "SELECT to_jsonb(blob) FROM artifact_blobs blob WHERE sha256=$1", digest[:]).Scan(&after); err != nil || !bytes.Equal(before, after) {
				t.Fatalf("migration rewrote the retained blob: %v", err)
			}
			newData := []byte("other backend " + backend)
			newDigest := sha256.Sum256(newData)
			if backend == "postgresql" {
				newKey := "cd/" + strings.Repeat("2", 32)
				_, err = pool.Exec(ctx, "INSERT INTO artifact_blobs(sha256,payload,size_bytes,backend,object_key) VALUES($1,NULL,$2,'filesystem',$3)", newDigest[:], len(newData), newKey)
			} else {
				_, err = pool.Exec(ctx, "INSERT INTO artifact_blobs(sha256,payload,size_bytes,backend) VALUES($1,$2,$3,'postgresql')", newDigest[:], newData, len(newData))
			}
			if err != nil {
				t.Fatalf("mixed backend insert rejected: %v", err)
			}
			_, err = pool.Exec(ctx, "INSERT INTO artifact_blobs(sha256,payload,size_bytes,backend) VALUES(sha256('invalid'::bytea),'different'::bytea,9,'postgresql')")
			if Classify(err) != ClassInvalid {
				t.Fatalf("inline checksum validation lost: %v", err)
			}
			_, err = pool.Exec(ctx, "INSERT INTO artifact_blobs(sha256,payload,size_bytes,backend,object_key) VALUES(sha256('file'::bytea),'file'::bytea,4,'filesystem','invalid')")
			if Classify(err) != ClassInvalid {
				t.Fatalf("filesystem shape validation lost: %v", err)
			}
		})
	}
}
