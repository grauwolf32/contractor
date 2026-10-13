package app

import (
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestBlobStorageConfigDefaultsDirectoryAndPrecedence(t *testing.T) {
	defaults, err := ParseConfig(nil, func(string) string { return "" })
	if err != nil || defaults.ArtifactBlobBackend != artifacts.BlobAuto ||
		defaults.ArtifactBlobPath != "" || defaults.ArtifactBlobInlineMaxBytes != 1048576 {
		t.Fatalf("default blob settings: %v, %v", defaults.ArtifactBlobBackend, err)
	}
	directory := t.TempDir()
	path := filepath.Join(directory, "server.yaml")
	writeServerConfigTestFile(t, path, `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  artifactBlobPath: blobs
  artifactBlobInlineMaxBytes: 262144
`)
	env := map[string]string{"CONTRACTOR_SERVER_CONFIG": path}
	getenv := func(key string) string { return env[key] }
	cfg, err := ParseConfig(nil, getenv)
	if err != nil || cfg.ArtifactBlobBackend != artifacts.BlobAuto ||
		cfg.ArtifactBlobPath != filepath.Join(directory, "blobs") || cfg.ArtifactBlobInlineMaxBytes != 262144 {
		t.Fatalf("YAML blob settings: %v, %v", cfg.ArtifactBlobBackend, err)
	}
	env["CONTRACTOR_ARTIFACT_BLOB_BACKEND"] = "postgresql"
	env["CONTRACTOR_ARTIFACT_BLOB_PATH"] = filepath.Join(directory, "environment")
	env["CONTRACTOR_ARTIFACT_BLOB_INLINE_MAX_BYTES"] = "65536"
	cfg, err = ParseConfig(nil, getenv)
	if err != nil || cfg.ArtifactBlobBackend != artifacts.BlobPostgres ||
		cfg.ArtifactBlobPath != env["CONTRACTOR_ARTIFACT_BLOB_PATH"] || cfg.ArtifactBlobInlineMaxBytes != 65536 {
		t.Fatalf("environment blob settings: %v, %v", cfg.ArtifactBlobBackend, err)
	}
	env["CONTRACTOR_ARTIFACT_BLOB_INLINE_MAX_BYTES"] = "invalid overridden value"
	flagPath := filepath.Join(directory, "flags")
	cfg, err = ParseConfig([]string{
		"--artifact-blob-backend=filesystem", "--artifact-blob-path=" + flagPath,
		"--artifact-blob-inline-max-bytes=0",
	}, getenv)
	if err != nil || cfg.ArtifactBlobBackend != artifacts.BlobFilesystem ||
		cfg.ArtifactBlobPath != flagPath || cfg.ArtifactBlobInlineMaxBytes != 0 {
		t.Fatalf("flag blob settings: %v, %v", cfg.ArtifactBlobBackend, err)
	}
}

func TestBlobStorageConfigRejectsInvalidThresholds(t *testing.T) {
	for _, raw := range []string{"-1", "67108865", "1.5", "1MiB", "9223372036854775808"} {
		t.Run(raw, func(t *testing.T) {
			getenv := func(key string) string {
				if key == "CONTRACTOR_ARTIFACT_BLOB_INLINE_MAX_BYTES" {
					return raw
				}
				return ""
			}
			if _, err := ParseConfig(nil, getenv); err == nil {
				t.Fatal("invalid threshold accepted")
			}
		})
	}
	for _, raw := range []string{"-1", "67108865", "1.5"} {
		path := filepath.Join(t.TempDir(), "server.yaml")
		writeServerConfigTestFile(t, path, "apiVersion: contractor/v1alpha1\nkind: ServerConfig\nspec:\n  artifactBlobInlineMaxBytes: "+raw+"\n")
		if _, err := ParseConfig([]string{"--config=" + path}, func(string) string { return "" }); err == nil {
			t.Fatalf("invalid YAML threshold %q accepted", raw)
		}
	}
	for _, raw := range []string{"0", "67108864"} {
		if _, err := ParseConfig([]string{"--artifact-blob-inline-max-bytes=" + raw}, func(string) string { return "" }); err != nil {
			t.Fatalf("valid threshold boundary %s rejected: %v", raw, err)
		}
	}
}
