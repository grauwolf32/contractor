package app

import (
	"path/filepath"
	"strconv"
	"testing"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
)

// The retained-history limits default to LIM-ART-9 and follow the Server
// configuration precedence: defaults < YAML < environment < flags.
func TestArtifactHistoryConfigDefaultsAndPrecedence(t *testing.T) {
	defaults, err := ParseConfig(nil, func(string) string { return "" })
	if err != nil || defaults.ArtifactHistory != artifactpostgres.DefaultHistoryPolicy() {
		t.Fatalf("default history limits: %+v, %v", defaults.ArtifactHistory, err)
	}
	directory := t.TempDir()
	path := filepath.Join(directory, "server.yaml")
	writeServerConfigTestFile(t, path, `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  artifactHistoryMaxRevisions: 500
  artifactHistoryMaxBytes: 268435456
`)
	env := map[string]string{"CONTRACTOR_SERVER_CONFIG": path}
	getenv := func(key string) string { return env[key] }
	assertHistory := func(name string, args []string, revisions, bytes int64) {
		t.Helper()
		cfg, err := ParseConfig(args, getenv)
		if err != nil || cfg.ArtifactHistory.MaxRevisions() != revisions || cfg.ArtifactHistory.MaxBytes() != bytes {
			t.Fatalf("%s history limits = %d revisions, %d bytes, %v; want %d, %d",
				name, cfg.ArtifactHistory.MaxRevisions(), cfg.ArtifactHistory.MaxBytes(), err, revisions, bytes)
		}
	}
	assertHistory("YAML", nil, 500, 268435456)
	env["CONTRACTOR_ARTIFACT_HISTORY_MAX_REVISIONS"] = strconv.FormatInt(artifactpostgres.HistoryRevisionsCeiling, 10)
	env["CONTRACTOR_ARTIFACT_HISTORY_MAX_BYTES"] = strconv.FormatInt(artifactpostgres.HistoryBytesCeiling, 10)
	assertHistory("environment", nil, artifactpostgres.HistoryRevisionsCeiling, artifactpostgres.HistoryBytesCeiling)
	env["CONTRACTOR_ARTIFACT_HISTORY_MAX_REVISIONS"] = "invalid overridden value"
	assertHistory("flag", []string{
		"--artifact-history-max-revisions=2", "--artifact-history-max-bytes=" + strconv.FormatInt(artifactpostgres.HistoryBytesFloor, 10),
	}, 2, artifactpostgres.HistoryBytesFloor)
}

func TestArtifactHistoryConfigRejectsLimitsOutsideTheirRange(t *testing.T) {
	for _, test := range []struct{ name, revisions, bytes string }{
		{"revisions below floor", strconv.FormatInt(artifactpostgres.HistoryRevisionsFloor-1, 10), ""},
		{"revisions above ceiling", strconv.FormatInt(artifactpostgres.HistoryRevisionsCeiling+1, 10), ""},
		{"revisions not an integer", "2k", ""},
		{"bytes below floor", "", strconv.FormatInt(artifactpostgres.HistoryBytesFloor-1, 10)},
		{"bytes above ceiling", "", strconv.FormatInt(artifactpostgres.HistoryBytesCeiling+1, 10)},
		{"bytes not an integer", "", "1GiB"},
	} {
		t.Run(test.name, func(t *testing.T) {
			getenv := func(key string) string {
				switch key {
				case "CONTRACTOR_ARTIFACT_HISTORY_MAX_REVISIONS":
					return test.revisions
				case "CONTRACTOR_ARTIFACT_HISTORY_MAX_BYTES":
					return test.bytes
				}
				return ""
			}
			if _, err := ParseConfig(nil, getenv); err == nil {
				t.Fatal("invalid history limit accepted")
			}
		})
	}
	path := filepath.Join(t.TempDir(), "server.yaml")
	writeServerConfigTestFile(t, path, "apiVersion: contractor/v1alpha1\nkind: ServerConfig\nspec:\n  artifactHistoryMaxRevisions: 0\n")
	if _, err := ParseConfig([]string{"--config=" + path}, func(string) string { return "" }); err == nil {
		t.Fatal("invalid YAML history limit accepted")
	}
}
