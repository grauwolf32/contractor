package e2e

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

type auditsMatrix struct {
	SchemaVersion string             `yaml:"schema_version"`
	Policy        auditsMatrixPolicy `yaml:"policy"`
	Cases         []auditsMatrixCase `yaml:"cases"`
	Gates         []matrixGate       `yaml:"gates"`
}

type auditsMatrixPolicy struct {
	Orchestration string `yaml:"orchestration"`
	Settlement    string `yaml:"settlement"`
	Retention     string `yaml:"retention"`
	Reporting     string `yaml:"reporting"`
	Isolation     string `yaml:"isolation"`
}

type auditsMatrixCase struct {
	ID         string         `yaml:"id"`
	Category   string         `yaml:"category"`
	Acceptance []string       `yaml:"acceptance"`
	Faults     []string       `yaml:"faults"`
	Expected   string         `yaml:"expected"`
	Test       auditTestOwner `yaml:"test"`
}

type auditTestOwner struct {
	Source string `yaml:"source"`
	Kind   string `yaml:"kind"`
	Name   string `yaml:"name"`
	Gate   string `yaml:"gate"`
}

var (
	auditsAcceptance = []string{"A1", "A2", "A3", "A4", "A5"}
	auditsCategories = []string{
		"browser", "cancellation", "capability", "concurrency", "configuration",
		"findings", "input_security", "inventory", "isolation", "matrix", "package",
		"process", "recovery", "regression", "reporting", "retention", "retry",
		"review", "scheduler", "settlement",
	}
	auditsFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
		"F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20",
		"F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28",
	}
)

func TestAuditsHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("audits_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeStrictMatrix[auditsMatrix](data, "Audits")
	if err != nil {
		t.Fatal(err)
	}
	if err := validateAuditsMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}
	if err := validateAuditsGateCoverage(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("missing policy", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		broken.Policy.Settlement = ""
		requireAuditsMatrixError(t, repositoryRoot, broken, "policy")
	})
	t.Run("missing acceptance", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		for index := range broken.Cases {
			broken.Cases[index].Acceptance = slices.DeleteFunc(
				broken.Cases[index].Acceptance, func(value string) bool { return value == "A5" },
			)
		}
		requireAuditsMatrixError(t, repositoryRoot, broken, "A5")
	})
	t.Run("missing category", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		broken.Cases = slices.DeleteFunc(broken.Cases, func(value auditsMatrixCase) bool {
			return value.Category == "capability"
		})
		requireAuditsMatrixError(t, repositoryRoot, broken, `category "capability"`)
	})
	t.Run("missing fault", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		for index := range broken.Cases {
			broken.Cases[index].Faults = slices.DeleteFunc(
				broken.Cases[index].Faults, func(value string) bool { return value == "F28" },
			)
		}
		requireAuditsMatrixError(t, repositoryRoot, broken, "F28")
	})
	t.Run("duplicate owner", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		broken.Cases[1].Test = broken.Cases[0].Test
		requireAuditsMatrixError(t, repositoryRoot, broken, "duplicate test owner")
	})
	t.Run("renamed owner", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		broken.Cases[0].Test.Name += "Renamed"
		requireAuditsMatrixError(t, repositoryRoot, broken, "does not define")
	})
	t.Run("owner outside declared gate", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		broken.Cases[6].Test.Gate = "process"
		err := validateAuditsGateCoverage(repositoryRoot, broken)
		if err == nil || !strings.Contains(err.Error(), "does not select") {
			t.Fatalf("gate error = %v, want selection failure", err)
		}
	})
	t.Run("missing release gate", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		broken.Gates = slices.DeleteFunc(broken.Gates, func(value matrixGate) bool {
			return value.ID == "release"
		})
		requireAuditsMatrixError(t, repositoryRoot, broken, `gate "release"`)
	})
}

func requireAuditsMatrixError(t *testing.T, repositoryRoot string, matrix auditsMatrix, want string) {
	t.Helper()
	err := validateAuditsMatrix(repositoryRoot, matrix)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("matrix error = %v, want %q", err, want)
	}
}

func validateAuditsMatrix(repositoryRoot string, matrix auditsMatrix) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.Orchestration) == "" ||
		strings.TrimSpace(matrix.Policy.Settlement) == "" || strings.TrimSpace(matrix.Policy.Retention) == "" ||
		strings.TrimSpace(matrix.Policy.Reporting) == "" || strings.TrimSpace(matrix.Policy.Isolation) == "" {
		return fmt.Errorf("incomplete Audits policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenAcceptance := map[string]bool{}
	seenCategories := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || strings.TrimSpace(item.Expected) == "" ||
			seenCases[item.ID] || len(item.Acceptance) == 0 {
			return fmt.Errorf("invalid or duplicate Audits case: %+v", item)
		}
		if !slices.Contains(auditsCategories, item.Category) {
			return fmt.Errorf("case %q has unknown category %q", item.ID, item.Category)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, acceptance := range item.Acceptance {
			if !slices.Contains(auditsAcceptance, acceptance) {
				return fmt.Errorf("case %q has unknown acceptance %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(auditsFaults, fault) || seenFaults[fault] {
				return fmt.Errorf("case %q has unknown or duplicate fault %q", item.ID, fault)
			}
			seenFaults[fault] = true
		}
		owner := item.Test.Source + "#" + item.Test.Kind + "#" + item.Test.Name
		if previous, exists := seenOwners[owner]; exists {
			return fmt.Errorf("duplicate test owner %q in cases %q and %q", owner, previous, item.ID)
		}
		seenOwners[owner] = item.ID
		if err := validateMatrixNamedOwner(repositoryRoot, item.Test.Source, item.Test.Kind, item.Test.Name); err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
		if !slices.Contains([]string{"browser", "hardening", "matrix", "process", "release"}, item.Test.Gate) {
			return fmt.Errorf("case %q has invalid declared gate %q", item.ID, item.Test.Gate)
		}
	}
	for _, acceptance := range auditsAcceptance {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required Audits acceptance %q is absent", acceptance)
		}
	}
	for _, category := range auditsCategories {
		if !seenCategories[category] {
			return fmt.Errorf("required Audits category %q is absent", category)
		}
	}
	for _, fault := range auditsFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required Audits fault %q is absent", fault)
		}
	}
	return validateMatrixGates(
		"Audits", matrix.Gates,
		[]string{"browser", "hardening", "matrix", "process", "release"}, true,
	)
}

// The matrix names a concrete gate for every owner. Validate the expanded Make
// recipe, including package, build tag and -run; a defined test alone is not
// evidence that release-verify can execute it.
func validateAuditsGateCoverage(repositoryRoot string, matrix auditsMatrix) error {
	commandsByGate := make(map[string][]string, len(matrix.Gates))
	stackSource, err := os.ReadFile(filepath.Join(repositoryRoot, "tests", "ui-stack", "stack_test.go"))
	if err != nil {
		return fmt.Errorf("read UI stack selection: %w", err)
	}
	for _, gate := range matrix.Gates {
		args := append([]string{"-n"}, strings.Fields(strings.TrimPrefix(gate.Command, "make "))...)
		command := exec.Command("make", args...)
		command.Dir = repositoryRoot
		output, err := command.CombinedOutput()
		if err != nil {
			return fmt.Errorf("expand gate %q: %w: %s", gate.ID, err, output)
		}
		commandsByGate[gate.ID] = strings.Split(string(output), "\n")
	}
	for _, item := range matrix.Cases {
		owner := item.Test
		commands, ok := commandsByGate[owner.Gate]
		if !ok {
			return fmt.Errorf("case %q declares unknown gate %q", item.ID, owner.Gate)
		}
		matched := false
		for _, line := range commands {
			if owner.Kind == "test_title" {
				// The UI stack runs every spec; its report validator requires this
				// spec file and the named title is checked by validateMatrixNamedOwner.
				fields := strings.Fields(line)
				matched = strings.Contains(line, "go test ") && slices.Contains(fields, "-tags=e2e") &&
					slices.Contains(fields, "./tests/ui-stack") &&
					bytes.Contains(stackSource, []byte("\""+strings.TrimPrefix(owner.Source, "ui/")+"\""))
			} else if owner.Kind == "go_test" {
				matched = goGateSelectsOwner(repositoryRoot, line, owner)
			}
			if matched {
				break
			}
		}
		if !matched {
			return fmt.Errorf("case %q: gate %q does not select %s", item.ID, owner.Gate, owner.Name)
		}
	}
	return nil
}

func goGateSelectsOwner(repositoryRoot, line string, owner auditTestOwner) bool {
	if !strings.Contains(line, "go test ") {
		return false
	}
	fields := strings.Fields(line)
	packagePath := "./" + filepath.ToSlash(filepath.Dir(owner.Source))
	if !slices.Contains(fields, packagePath) {
		return false
	}
	data, err := os.ReadFile(filepath.Join(repositoryRoot, filepath.FromSlash(owner.Source)))
	if err != nil {
		return false
	}
	if bytes.HasPrefix(data, []byte("//go:build integration")) && !slices.Contains(fields, "-tags=integration") {
		return false
	}
	if bytes.HasPrefix(data, []byte("//go:build e2e")) && !slices.Contains(fields, "-tags=e2e") {
		return false
	}
	for index, field := range fields {
		if field != "-run" {
			continue
		}
		if index+1 == len(fields) {
			return false
		}
		pattern := strings.Trim(fields[index+1], "'\"")
		re, err := regexp.Compile(pattern)
		return err == nil && re.MatchString(owner.Name)
	}
	return true
}
