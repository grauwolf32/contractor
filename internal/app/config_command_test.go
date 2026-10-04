package app

import (
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
)

func TestConfigValidateMatchesServerForFilesystemAndUnionFailures(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(t *testing.T, operator, managed string)
		want   string
	}{
		{"symlinked root", func(t *testing.T, operator, _ string) {
			real := operator + "-real"
			if err := os.Rename(operator, real); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(real, operator); err != nil {
				t.Fatal(err)
			}
		}, "operator configuration root: root must be a real directory"},
		{"symlinked subtree", func(t *testing.T, operator, _ string) {
			original := filepath.Join(operator, "model-policies")
			if err := os.Rename(original, original+"-real"); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink("model-policies-real", original); err != nil {
				t.Fatal(err)
			}
		}, "configuration subtree model-policies must be a real directory"},
		{"symlinked manifest", func(t *testing.T, operator, _ string) {
			if err := os.Symlink("worker.yaml", filepath.Join(operator, "model-policies", "linked.yaml")); err != nil {
				t.Fatal(err)
			}
		}, "configuration path"},
		{"symlinked instruction", func(t *testing.T, operator, _ string) {
			if err := os.Symlink("artifact-builder.md", filepath.Join(operator, "instructions", "linked.md")); err != nil {
				t.Fatal(err)
			}
		}, "instruction path"},
		{"duplicate managed identity", func(t *testing.T, operator, managed string) {
			if err := os.MkdirAll(filepath.Join(managed, "model-policies"), 0o755); err != nil {
				t.Fatal(err)
			}
			payload, err := os.ReadFile(filepath.Join(operator, "model-policies", "worker.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(managed, "model-policies", "duplicate.yaml"), payload, 0o600); err != nil {
				t.Fatal(err)
			}
		}, "duplicate ModelPolicy identity"},
	} {
		t.Run(test.name, func(t *testing.T) {
			operator := copyConfigValidationTree(t)
			managed := filepath.Join(filepath.Dir(operator), "managed-configs")
			test.mutate(t, operator, managed)
			validationErr := validateConfigForTest(operator, managed)
			if validationErr == nil || !strings.Contains(validationErr.Error(), test.want) {
				t.Fatalf("config validate error = %v, want %q", validationErr, test.want)
			}
			_, serverErr := config.NewManager(config.ManagerOptions{
				OperatorRoot: operator, ManagedRoot: managed, Descriptors: config.MVPDescriptors(),
			})
			if serverErr == nil || !strings.Contains(serverErr.Error(), test.want) {
				t.Fatalf("Server load error = %v, want %q", serverErr, test.want)
			}
		})
	}
}

func TestConfigValidateNeverCreatesManagedRootOrSubtrees(t *testing.T) {
	operator := copyConfigValidationTree(t)
	managed := filepath.Join(filepath.Dir(operator), "managed-configs")
	if err := validateConfigForTest(operator, managed); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(managed); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("validation created managed root: %v", err)
	}
	if err := os.Mkdir(managed, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(managed, "model-policies"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := validateConfigForTest(operator, managed); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(managed, "workflows")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("validation created managed subtree: %v", err)
	}
	if _, err := config.NewManager(config.ManagerOptions{
		OperatorRoot: operator, ManagedRoot: managed, Descriptors: config.MVPDescriptors(),
	}); err != nil {
		t.Fatalf("Server rejected configuration accepted by config validate: %v", err)
	}
}

func TestConfigValidateDefaultsToSiblingManagedRoot(t *testing.T) {
	operator := copyConfigValidationTree(t)
	managed := filepath.Join(filepath.Dir(operator), "managed-configs")
	if err := os.MkdirAll(filepath.Join(managed, "model-policies"), 0o755); err != nil {
		t.Fatal(err)
	}
	payload, err := os.ReadFile(filepath.Join(operator, "model-policies", "worker.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(managed, "model-policies", "duplicate.yaml"), payload, 0o600); err != nil {
		t.Fatal(err)
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	err = runConfigCLI([]string{"validate", "--root", operator}, logger)
	if err == nil || !strings.Contains(err.Error(), "duplicate ModelPolicy identity") {
		t.Fatalf("default managed root was not included: %v", err)
	}
}

func TestConfigValidateTreatsUncleanRootLikeCleanPath(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	for _, suffix := range []string{"/", "//", "/."} {
		t.Run(suffix, func(t *testing.T) {
			operator := copyConfigValidationTree(t)
			root := operator + suffix
			if err := runConfigCLI([]string{"validate", "--root", root}, logger); err != nil {
				t.Fatalf("config validate --root %s: %v", root, err)
			}
			if _, err := os.Stat(filepath.Join(operator, "managed-configs")); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("validation derived a managed root inside the operator root: %v", err)
			}
			// A duplicate in the sibling root proves validation reads it.
			managed := filepath.Join(filepath.Dir(operator), "managed-configs", "model-policies")
			if err := os.MkdirAll(managed, 0o755); err != nil {
				t.Fatal(err)
			}
			payload, err := os.ReadFile(filepath.Join(operator, "model-policies", "worker.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(managed, "duplicate.yaml"), payload, 0o600); err != nil {
				t.Fatal(err)
			}
			err = runConfigCLI([]string{"validate", "--root", root}, logger)
			if err == nil || !strings.Contains(err.Error(), "duplicate ModelPolicy identity") {
				t.Fatalf("sibling managed root was not used for %s: %v", root, err)
			}
		})
	}
}

func TestConfigValidateRejectsMissingManagedRootUnderOperatorSymlink(t *testing.T) {
	operator := copyConfigValidationTree(t)
	alias := filepath.Join(filepath.Dir(operator), "operator-alias")
	if err := os.Symlink(operator, alias); err != nil {
		t.Fatal(err)
	}
	managed := filepath.Join(alias, "managed-configs")
	err := validateConfigForTest(operator, managed)
	if err == nil || !strings.Contains(err.Error(), "roots must not overlap") {
		t.Fatalf("validation accepted overlapping managed root through symlink: %v", err)
	}
	if _, err := os.Stat(filepath.Join(operator, "managed-configs")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("validation created overlapping managed root: %v", err)
	}
}

func TestConfigValidateChecksBundledStandardSelections(t *testing.T) {
	for _, test := range []struct {
		name, from, to, want string
	}{
		{"unknown entry", "- v5.0.0-1.2.4", "- v5.0.0-1.2.0", "standard selection"},
		{"database-only standard", `version: "5.0.0"`, `version: "9.9.9"`, "is not bundled"},
	} {
		t.Run(test.name, func(t *testing.T) {
			operator := copyConfigValidationTree(t)
			profilePath := filepath.Join(operator, "audit-profiles", "owasp_asvs_5_0_l1_source_pilot.yaml")
			original, err := os.ReadFile(profilePath)
			if err != nil {
				t.Fatal(err)
			}
			modified := strings.Replace(string(original), test.from, test.to, 1)
			if modified == string(original) {
				t.Fatal("standard test did not change the profile")
			}
			if err := os.WriteFile(profilePath, []byte(modified), 0o600); err != nil {
				t.Fatal(err)
			}
			err = validateConfigForTest(operator, filepath.Join(filepath.Dir(operator), "managed-configs"))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("config validate error = %v, want %q", err, test.want)
			}
		})
	}
}

// copyConfigValidationTree builds a mutable operator root from the shared
// Server test catalog plus this package's bundled ASVS pilot standard and the
// AuditProfile that selects it.
func copyConfigValidationTree(t *testing.T) string {
	t.Helper()
	root := filepath.Join(t.TempDir(), "configs")
	for _, fixture := range []string{"../../testdata/configs", "testdata/config-validation"} {
		if err := os.CopyFS(root, os.DirFS(fixture)); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func validateConfigForTest(operator, managed string) error {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return runConfigCLI([]string{"validate", "--root", operator, "--managed-root", managed}, logger)
}
