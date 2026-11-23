package app

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/gitimport"
)

func TestGitImportBudgetConfiguration(t *testing.T) {
	empty := func(string) string { return "" }
	cfg, err := ParseConfig(nil, empty)
	if err != nil || cfg.GitImport.EffectiveLimits() != gitimport.DefaultLimits() {
		t.Fatalf("default budgets = %+v, %v", cfg.GitImport, err)
	}
	path := filepath.Join(t.TempDir(), "server.yaml")
	write := func(body string) {
		t.Helper()
		if err := os.WriteFile(path, []byte("apiVersion: contractor/v1alpha1\nkind: ServerConfig\nspec:\n  gitImport:\n"+body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	write("    maxExpandedBytes: 134217728\n    maxFileBytes: 16777216\n    timeout: 3m\n")
	cfg, err = ParseConfig([]string{"--config", path}, empty)
	if err != nil || cfg.GitImport.EffectiveLimits().Source.MaxExpandedBytes != 128<<20 || cfg.GitImport.EffectiveLimits().Source.MaxFileBytes != 16<<20 || cfg.GitImport.EffectiveLimits().Timeout != 3*time.Minute {
		t.Fatalf("YAML budgets = %+v, %v", cfg.GitImport, err)
	}
	cfg, err = ParseConfig([]string{"--config", path, "--git-import-max-file-bytes=8388608", "--git-import-timeout=4m"}, func(key string) string {
		switch key {
		case "CONTRACTOR_GIT_IMPORT_MAX_FILE_BYTES":
			return "12582912"
		case "CONTRACTOR_GIT_IMPORT_MAX_ENTRIES":
			return "12000"
		case "CONTRACTOR_GIT_IMPORT_TIMEOUT":
			return "5m"
		}
		return ""
	})
	if err != nil || cfg.GitImport.EffectiveLimits().Source.MaxFileBytes != 8<<20 || cfg.GitImport.EffectiveLimits().Source.MaxEntries != 12000 || cfg.GitImport.EffectiveLimits().Timeout != 4*time.Minute {
		t.Fatalf("CLI precedence = %+v, %v", cfg.GitImport, err)
	}
	for _, body := range []string{
		"    maxExpandedBytes: 0\n", "    maxFileBytes: 0\n", "    maxEntries: 0\n",
		"    maxReceivedBytes: 0\n", "    maxReceivedBytes: 1.5\n", "    maxDecodedBytes: 0\n", "    timeout: 0s\n",
		"    maxExpandedBytes: 1073741825\n", "    maxFileBytes: 67108865\n", "    maxEntries: 50001\n",
		"    maxReceivedBytes: 1073741825\n", "    maxDecodedBytes: 2147483649\n", "    timeout: 31m\n",
		"    maxExpandedBytes: 10\n    maxFileBytes: 11\n", "    maxDecodedBytes: 1\n", "    unexpected: 1\n",
	} {
		write(body)
		if _, err := ParseConfig([]string{"--config", path}, empty); err == nil {
			t.Fatalf("accepted invalid budgets: %s", body)
		}
	}
	write("    maxFileBytes: 4194304\n")
	if _, err := ParseConfig([]string{"--config", path}, func(key string) string {
		if key == "CONTRACTOR_GIT_IMPORT_MAX_FILE_BYTES" {
			return "not-a-byte-count"
		}
		return ""
	}); err == nil {
		t.Fatal("accepted malformed environment integer")
	}
}
