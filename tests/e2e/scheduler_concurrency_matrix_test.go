package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type schedulerConcurrencyMatrix struct {
	SchemaVersion string                           `yaml:"schema_version"`
	Policy        schedulerConcurrencyMatrixPolicy `yaml:"policy"`
	Cases         []schedulerConcurrencyMatrixCase `yaml:"cases"`
	Gates         []matrixGate                     `yaml:"gates"`
}

type schedulerConcurrencyMatrixPolicy struct {
	Limit    string `yaml:"limit"`
	Resize   string `yaml:"resize"`
	Capacity string `yaml:"capacity"`
	Recovery string `yaml:"recovery"`
}

type schedulerConcurrencyMatrixCase struct {
	ID         string                     `yaml:"id"`
	Category   string                     `yaml:"category"`
	Acceptance []string                   `yaml:"acceptance"`
	Faults     []string                   `yaml:"faults"`
	Expected   string                     `yaml:"expected"`
	Test       lifecycleControlsTestOwner `yaml:"test"`
}

var (
	schedulerConcurrencyAcceptance = []string{"A1", "A2", "A3", "A4"}
	schedulerConcurrencyCategories = []string{
		"browser", "cancellation", "claims", "default", "deferred", "invalidation",
		"lease_loss", "matrix", "pause", "persistence", "process", "project_deletion",
		"release", "resize", "runtime_capacity", "shutdown",
	}
	schedulerConcurrencyFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
		"F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19",
	}
)

func TestSchedulerConcurrencyMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("scheduler_concurrency_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeStrictMatrix[schedulerConcurrencyMatrix](data, "Scheduler concurrency")
	if err != nil {
		t.Fatal(err)
	}
	if err := validateSchedulerConcurrencyMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("missing policy", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[schedulerConcurrencyMatrix](t, data, "Scheduler concurrency")
		broken.Policy.Capacity = ""
		requireSchedulerConcurrencyMatrixError(t, repositoryRoot, broken, "policy")
	})
	t.Run("missing acceptance", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[schedulerConcurrencyMatrix](t, data, "Scheduler concurrency")
		for index := range broken.Cases {
			broken.Cases[index].Acceptance = slices.DeleteFunc(
				broken.Cases[index].Acceptance, func(value string) bool { return value == "A4" },
			)
		}
		requireSchedulerConcurrencyMatrixError(t, repositoryRoot, broken, "A4")
	})
	t.Run("missing category", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[schedulerConcurrencyMatrix](t, data, "Scheduler concurrency")
		broken.Cases = slices.DeleteFunc(
			broken.Cases, func(value schedulerConcurrencyMatrixCase) bool { return value.Category == "lease_loss" },
		)
		requireSchedulerConcurrencyMatrixError(t, repositoryRoot, broken, `category "lease_loss"`)
	})
	t.Run("missing fault", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[schedulerConcurrencyMatrix](t, data, "Scheduler concurrency")
		for index := range broken.Cases {
			broken.Cases[index].Faults = slices.DeleteFunc(
				broken.Cases[index].Faults, func(value string) bool { return value == "F19" },
			)
		}
		requireSchedulerConcurrencyMatrixError(t, repositoryRoot, broken, "F19")
	})
	t.Run("renamed owner", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[schedulerConcurrencyMatrix](t, data, "Scheduler concurrency")
		broken.Cases[0].Test.Name += "Renamed"
		requireSchedulerConcurrencyMatrixError(t, repositoryRoot, broken, "does not define")
	})
	t.Run("missing release gate", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[schedulerConcurrencyMatrix](t, data, "Scheduler concurrency")
		broken.Gates = slices.DeleteFunc(broken.Gates, func(value matrixGate) bool {
			return value.ID == "release"
		})
		requireSchedulerConcurrencyMatrixError(t, repositoryRoot, broken, `gate "release"`)
	})
}

func requireSchedulerConcurrencyMatrixError(
	t *testing.T,
	repositoryRoot string,
	matrix schedulerConcurrencyMatrix,
	want string,
) {
	t.Helper()
	err := validateSchedulerConcurrencyMatrix(repositoryRoot, matrix)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("matrix error = %v, want %q", err, want)
	}
}

func validateSchedulerConcurrencyMatrix(repositoryRoot string, matrix schedulerConcurrencyMatrix) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.Limit) == "" ||
		strings.TrimSpace(matrix.Policy.Resize) == "" || strings.TrimSpace(matrix.Policy.Capacity) == "" ||
		strings.TrimSpace(matrix.Policy.Recovery) == "" {
		return fmt.Errorf("incomplete Scheduler-concurrency policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenAcceptance := map[string]bool{}
	seenCategories := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || strings.TrimSpace(item.Expected) == "" ||
			seenCases[item.ID] || len(item.Acceptance) == 0 {
			return fmt.Errorf("invalid or duplicate Scheduler-concurrency case: %+v", item)
		}
		if !slices.Contains(schedulerConcurrencyCategories, item.Category) {
			return fmt.Errorf("case %q has unknown category %q", item.ID, item.Category)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, acceptance := range item.Acceptance {
			if !slices.Contains(schedulerConcurrencyAcceptance, acceptance) {
				return fmt.Errorf("case %q has unknown acceptance %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(schedulerConcurrencyFaults, fault) || seenFaults[fault] {
				return fmt.Errorf("case %q has unknown or duplicate fault %q", item.ID, fault)
			}
			seenFaults[fault] = true
		}
		owner := item.Test.Source + "#" + item.Test.Kind + "#" + item.Test.Name
		if previous, exists := seenOwners[owner]; exists {
			return fmt.Errorf("duplicate test owner %q in cases %q and %q", owner, previous, item.ID)
		}
		seenOwners[owner] = item.ID
		if err := validateMatrixNamedOwner(
			repositoryRoot, item.Test.Source, item.Test.Kind, item.Test.Name,
		); err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
	}
	for _, acceptance := range schedulerConcurrencyAcceptance {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required Scheduler-concurrency acceptance %q is absent", acceptance)
		}
	}
	for _, category := range schedulerConcurrencyCategories {
		if !seenCategories[category] {
			return fmt.Errorf("required Scheduler-concurrency category %q is absent", category)
		}
	}
	for _, fault := range schedulerConcurrencyFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required Scheduler-concurrency fault %q is absent", fault)
		}
	}
	return validateMatrixGates(
		"Scheduler concurrency", matrix.Gates,
		[]string{"browser", "hardening", "matrix", "process", "release"}, true,
	)
}
