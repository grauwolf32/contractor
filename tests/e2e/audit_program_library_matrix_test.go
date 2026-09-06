package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

type auditProgramLibraryMatrix struct {
	SchemaVersion string                       `yaml:"schema_version"`
	Policy        auditProgramLibraryPolicy    `yaml:"policy"`
	Programs      []auditProgramLibraryProgram `yaml:"programs"`
	Cases         []auditProgramLibraryCase    `yaml:"cases"`
	Gates         []matrixGate                 `yaml:"gates"`
}

type auditProgramLibraryPolicy struct {
	Identity  string `yaml:"identity"`
	Coverage  string `yaml:"coverage"`
	Reporting string `yaml:"reporting"`
	Lifecycle string `yaml:"lifecycle"`
}

type auditProgramLibraryProgram struct {
	ID                  string                      `yaml:"id"`
	Profile             string                      `yaml:"profile"`
	Mode                string                      `yaml:"mode"`
	Standard            auditProgramLibraryStandard `yaml:"standard"`
	SelectedItems       int                         `yaml:"selected_items"`
	ExpectedAssessments []string                    `yaml:"expected_assessments"`
	ProhibitedClaims    []string                    `yaml:"prohibited_claims"`
}

type auditProgramLibraryStandard struct {
	Scheme         string `yaml:"scheme"`
	Version        string `yaml:"version"`
	SourceRevision string `yaml:"source_revision"`
	License        string `yaml:"license"`
}

type auditProgramLibraryCase struct {
	ID         string         `yaml:"id"`
	Phase      string         `yaml:"phase"`
	Programs   []string       `yaml:"programs"`
	Acceptance []string       `yaml:"acceptance"`
	Expected   string         `yaml:"expected"`
	Test       auditTestOwner `yaml:"test"`
}

var expectedAuditPrograms = map[string]auditProgramLibraryProgram{
	"top10-2025": {
		ID: "top10-2025", Profile: "owasp-top10-2025-source-risk@1", Mode: "risk-assessment",
		Standard: auditProgramLibraryStandard{
			Scheme: "owasp-web-top10", Version: "2025",
			SourceRevision: "66ebc4798d2ca72973967a20264bdeb70dcf0a13", License: "CC-BY-SA-4.0",
		},
		SelectedItems:       10,
		ExpectedAssessments: []string{"inconclusive", "not-tested", "satisfied", "violated"},
		ProhibitedClaims:    []string{"application is compliant", "application is secure", "OWASP certified"},
	},
	"asvs-5.0-l1-pilot": {
		ID: "asvs-5.0-l1-pilot", Profile: "owasp-asvs-5-0-l1-source-review@1",
		Mode: "requirements-verification",
		Standard: auditProgramLibraryStandard{
			Scheme: "owasp-asvs", Version: "5.0.0",
			SourceRevision: "5cf9b032440be53ce345ab3c130fda46ba1ce7a2", License: "CC-BY-SA-4.0",
		},
		SelectedItems:       5,
		ExpectedAssessments: []string{"inconclusive", "not-applicable", "not-tested", "satisfied", "violated"},
		ProhibitedClaims:    []string{"application is compliant", "application is secure", "ASVS certified"},
	},
}

var auditProgramLibraryPhases = []string{
	"browser", "capability", "definition", "lifecycle", "matrix", "persistence", "process",
}

func TestAuditProgramLibraryMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("audit_program_library_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeStrictMatrix[auditProgramLibraryMatrix](data, "Audit program library")
	if err != nil {
		t.Fatal(err)
	}
	if err := validateAuditProgramLibraryMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("program identity drift", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditProgramLibraryMatrix](t, data, "Audit program library")
		broken.Programs[0].Standard.Version = "latest"
		requireAuditProgramLibraryMatrixError(t, repositoryRoot, broken, "program identity")
	})
	t.Run("missing program phase", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditProgramLibraryMatrix](t, data, "Audit program library")
		broken.Cases = slices.DeleteFunc(broken.Cases, func(value auditProgramLibraryCase) bool {
			return value.Phase == "browser"
		})
		requireAuditProgramLibraryMatrixError(t, repositoryRoot, broken, "browser")
	})
	t.Run("renamed owner", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditProgramLibraryMatrix](t, data, "Audit program library")
		broken.Cases[0].Test.Name += "Renamed"
		requireAuditProgramLibraryMatrixError(t, repositoryRoot, broken, "does not define")
	})
	t.Run("missing release gate", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditProgramLibraryMatrix](t, data, "Audit program library")
		broken.Gates = slices.DeleteFunc(broken.Gates, func(value matrixGate) bool {
			return value.ID == "release"
		})
		requireAuditProgramLibraryMatrixError(t, repositoryRoot, broken, "release")
	})
}

func requireAuditProgramLibraryMatrixError(
	t *testing.T,
	repositoryRoot string,
	matrix auditProgramLibraryMatrix,
	want string,
) {
	t.Helper()
	err := validateAuditProgramLibraryMatrix(repositoryRoot, matrix)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("matrix error = %v, want %q", err, want)
	}
}

func validateAuditProgramLibraryMatrix(
	repositoryRoot string,
	matrix auditProgramLibraryMatrix,
) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.Identity) == "" ||
		strings.TrimSpace(matrix.Policy.Coverage) == "" || strings.TrimSpace(matrix.Policy.Reporting) == "" ||
		strings.TrimSpace(matrix.Policy.Lifecycle) == "" {
		return fmt.Errorf("incomplete Audit program library policy: %+v", matrix.Policy)
	}
	if len(matrix.Programs) != len(expectedAuditPrograms) {
		return fmt.Errorf("Audit program library has %d programs, want %d", len(matrix.Programs), len(expectedAuditPrograms))
	}
	seenPrograms := make(map[string]bool, len(matrix.Programs))
	for _, program := range matrix.Programs {
		expected, exists := expectedAuditPrograms[program.ID]
		if !exists || seenPrograms[program.ID] || !reflect.DeepEqual(program, expected) {
			return fmt.Errorf("invalid Audit program identity %q: %+v", program.ID, program)
		}
		seenPrograms[program.ID] = true
	}

	seenCases := make(map[string]bool, len(matrix.Cases))
	seenOwners := make(map[string]string, len(matrix.Cases))
	coverage := make(map[string]map[string]bool, len(expectedAuditPrograms))
	seenAcceptance := map[string]bool{}
	for program := range expectedAuditPrograms {
		coverage[program] = make(map[string]bool, len(auditProgramLibraryPhases))
	}
	for _, item := range matrix.Cases {
		if item.ID == "" || seenCases[item.ID] || !slices.Contains(auditProgramLibraryPhases, item.Phase) ||
			len(item.Programs) == 0 || len(item.Acceptance) == 0 || strings.TrimSpace(item.Expected) == "" {
			return fmt.Errorf("invalid Audit program library case: %+v", item)
		}
		seenCases[item.ID] = true
		seenCasePrograms := map[string]bool{}
		for _, program := range item.Programs {
			if _, exists := expectedAuditPrograms[program]; !exists || seenCasePrograms[program] {
				return fmt.Errorf("case %q has invalid program %q", item.ID, program)
			}
			seenCasePrograms[program] = true
			coverage[program][item.Phase] = true
		}
		for _, acceptance := range item.Acceptance {
			if acceptance != "A1" && acceptance != "A2" {
				return fmt.Errorf("case %q has invalid acceptance %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		owner := item.Test.Source + "#" + item.Test.Kind + "#" + item.Test.Name
		if previous, duplicate := seenOwners[owner]; duplicate {
			return fmt.Errorf("duplicate test owner %q in cases %q and %q", owner, previous, item.ID)
		}
		seenOwners[owner] = item.ID
		if err := validateMatrixNamedOwner(repositoryRoot, item.Test.Source, item.Test.Kind, item.Test.Name); err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
	}
	for _, acceptance := range []string{"A1", "A2"} {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required acceptance %q is absent", acceptance)
		}
	}
	for program, phases := range coverage {
		for _, phase := range auditProgramLibraryPhases {
			if !phases[phase] {
				return fmt.Errorf("program %q has no %s coverage", program, phase)
			}
		}
	}
	return validateMatrixGates(
		"Audit program library", matrix.Gates,
		[]string{"browser", "hardening", "matrix", "process", "release"}, true,
	)
}
