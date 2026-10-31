package app

import (
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedAuditCommonOfflineCommand(t *testing.T) {
	root := configtest.CopyWithPolicies(t, "../config/testdata/valid")
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	for _, file := range []string{"programs.json", "results.json", "controls.json", "proposals.json"} {
		var fixtures struct {
			Cases []struct {
				Document string `json:"document"`
				Context  string `json:"context"`
				Valid    bool   `json:"valid"`
			}
		}
		if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", file), &fixtures); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), "document.json")
		contextPath := filepath.Join(t.TempDir(), "context.json")
		for _, c := range fixtures.Cases {
			if err := os.WriteFile(path, []byte(c.Document), 0600); err != nil {
				t.Fatal(err)
			}
			args := []string{"validate", "--root", root, "--audit-document", path}
			if c.Context != "" {
				if err := os.WriteFile(contextPath, []byte(c.Context), 0600); err != nil {
					t.Fatal(err)
				}
				args = append(args, "--audit-context", contextPath)
			}
			err := runConfigCLI(args, logger)
			if (err == nil) != c.Valid {
				t.Fatalf("%s valid=%v error=%v", file, c.Valid, err)
			}
		}
	}
}
