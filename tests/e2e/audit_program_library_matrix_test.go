package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

type auditProgramLibraryMatrix struct {
	SchemaVersion string `yaml:"schema_version"`
	Scope         string `yaml:"scope"`
	Programs      []struct {
		Profile string `yaml:"profile"`
		Source  string `yaml:"source"`
		Name    string `yaml:"name"`
	} `yaml:"programs"`
	Gate string `yaml:"gate"`
}

func validateAuditProgramLibraryMatrix(root string, m auditProgramLibraryMatrix) error {
	expected := []string{"openapi-operation-observe@4", "openapi-operation-trace@5", "openapi-sqlmap-scan@1", "openapi-nuclei-scan@1", "web-scan@1", "source-openapi-sqlmap-scan@3"}
	if m.SchemaVersion != "2.0" || m.Scope == "" || m.Gate != "make test-audit-common-programs-e2e" || len(m.Programs) != len(expected) {
		return fmt.Errorf("invalid common program matrix")
	}
	seen := map[string]bool{}
	for _, p := range m.Programs {
		if !slices.Contains(expected, p.Profile) || seen[p.Profile] {
			return fmt.Errorf("invalid program identity %s", p.Profile)
		}
		seen[p.Profile] = true
		if err := validateMatrixNamedOwner(root, p.Source, "go_test", p.Name); err != nil {
			return err
		}
	}
	return nil
}
func TestAuditProgramLibraryMatrixIsComplete(t *testing.T) {
	data, err := os.ReadFile("audit_program_library_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	m, err := decodeStrictMatrix[auditProgramLibraryMatrix](data, "common programs")
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join("..", "..")
	if err := validateAuditProgramLibraryMatrix(root, m); err != nil {
		t.Fatal(err)
	}
	t.Run("missing program", func(t *testing.T) {
		bad := m
		bad.Programs = bad.Programs[1:]
		if validateAuditProgramLibraryMatrix(root, bad) == nil {
			t.Fatal("missing program accepted")
		}
	})
	t.Run("retired program", func(t *testing.T) {
		bad := mustDecodeStrictMatrix[auditProgramLibraryMatrix](t, data, "common programs")
		bad.Programs[0].Profile = "source-checklist@4"
		if validateAuditProgramLibraryMatrix(root, bad) == nil {
			t.Fatal("retired program accepted")
		}
	})
	t.Run("missing process owner", func(t *testing.T) {
		bad := mustDecodeStrictMatrix[auditProgramLibraryMatrix](t, data, "common programs")
		bad.Programs[0].Name = "TestAbsent"
		if validateAuditProgramLibraryMatrix(root, bad) == nil {
			t.Fatal("missing process test accepted")
		}
	})
}
