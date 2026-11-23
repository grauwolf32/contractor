package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type lifecycleControlsMatrix struct {
	SchemaVersion string                        `yaml:"schema_version"`
	Policy        lifecycleControlsMatrixPolicy `yaml:"policy"`
	Cases         []lifecycleControlsMatrixCase `yaml:"cases"`
	Gates         []matrixGate                  `yaml:"gates"`
}

type lifecycleControlsMatrixPolicy struct {
	Admission string `yaml:"admission"`
	Deletion  string `yaml:"deletion"`
	Retention string `yaml:"retention"`
	Recovery  string `yaml:"recovery"`
}

type lifecycleControlsMatrixCase struct {
	ID         string                     `yaml:"id"`
	Category   string                     `yaml:"category"`
	Acceptance []string                   `yaml:"acceptance"`
	Faults     []string                   `yaml:"faults"`
	Expected   string                     `yaml:"expected"`
	Test       lifecycleControlsTestOwner `yaml:"test"`
}

type lifecycleControlsTestOwner struct {
	Source string `yaml:"source"`
	Kind   string `yaml:"kind"`
	Name   string `yaml:"name"`
}

var (
	lifecycleControlsAcceptance = []string{"A1", "A2", "A3"}
	lifecycleControlsCategories = []string{
		"artifact_exclusion", "browser", "matrix", "pause_semantics",
		"process", "project_deletion", "queue_serialization", "regression",
		"restart_recovery", "retention", "run_purge",
	}
	lifecycleControlsFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08",
		"F09", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
	}
)

func TestLifecycleControlsHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("lifecycle_controls_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeLifecycleControlsMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateLifecycleControlsMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("missing acceptance", func(t *testing.T) {
		broken := mustDecodeLifecycleControlsMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Acceptance = slices.DeleteFunc(
				broken.Cases[index].Acceptance,
				func(value string) bool { return value == "A2" },
			)
		}
		requireLifecycleControlsMatrixError(t, repositoryRoot, broken, "A2")
	})
	t.Run("missing category", func(t *testing.T) {
		broken := mustDecodeLifecycleControlsMatrix(t, data)
		broken.Cases = slices.DeleteFunc(
			broken.Cases,
			func(value lifecycleControlsMatrixCase) bool { return value.Category == "retention" },
		)
		requireLifecycleControlsMatrixError(t, repositoryRoot, broken, `category "retention"`)
	})
	t.Run("missing fault", func(t *testing.T) {
		broken := mustDecodeLifecycleControlsMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Faults = slices.DeleteFunc(
				broken.Cases[index].Faults,
				func(value string) bool { return value == "F16" },
			)
		}
		requireLifecycleControlsMatrixError(t, repositoryRoot, broken, "F16")
	})
	t.Run("duplicate owner", func(t *testing.T) {
		broken := mustDecodeLifecycleControlsMatrix(t, data)
		broken.Cases[1].Test = broken.Cases[0].Test
		requireLifecycleControlsMatrixError(t, repositoryRoot, broken, "duplicate test owner")
	})
	t.Run("renamed owner", func(t *testing.T) {
		broken := mustDecodeLifecycleControlsMatrix(t, data)
		broken.Cases[0].Test.Name += "Renamed"
		requireLifecycleControlsMatrixError(t, repositoryRoot, broken, "does not define")
	})
	t.Run("missing gate", func(t *testing.T) {
		broken := mustDecodeLifecycleControlsMatrix(t, data)
		broken.Gates = slices.DeleteFunc(
			broken.Gates,
			func(value matrixGate) bool { return value.ID == "browser" },
		)
		requireLifecycleControlsMatrixError(t, repositoryRoot, broken, `gate "browser"`)
	})
}

func decodeLifecycleControlsMatrix(data []byte) (lifecycleControlsMatrix, error) {
	return decodeStrictMatrix[lifecycleControlsMatrix](data, "lifecycle controls")
}

func mustDecodeLifecycleControlsMatrix(t *testing.T, data []byte) lifecycleControlsMatrix {
	t.Helper()
	return mustDecodeStrictMatrix[lifecycleControlsMatrix](t, data, "lifecycle controls")
}

func requireLifecycleControlsMatrixError(
	t *testing.T,
	repositoryRoot string,
	matrix lifecycleControlsMatrix,
	want string,
) {
	t.Helper()
	err := validateLifecycleControlsMatrix(repositoryRoot, matrix)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("matrix error = %v, want %q", err, want)
	}
}

func validateLifecycleControlsMatrix(repositoryRoot string, matrix lifecycleControlsMatrix) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.Admission) == "" ||
		strings.TrimSpace(matrix.Policy.Deletion) == "" ||
		strings.TrimSpace(matrix.Policy.Retention) == "" ||
		strings.TrimSpace(matrix.Policy.Recovery) == "" {
		return fmt.Errorf("incomplete lifecycle-controls policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenAcceptance := map[string]bool{}
	seenCategories := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || strings.TrimSpace(item.Expected) == "" ||
			seenCases[item.ID] || len(item.Acceptance) == 0 {
			return fmt.Errorf("invalid or duplicate lifecycle-controls case: %+v", item)
		}
		if !slices.Contains(lifecycleControlsCategories, item.Category) {
			return fmt.Errorf("case %q has unknown category %q", item.ID, item.Category)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, acceptance := range item.Acceptance {
			if !slices.Contains(lifecycleControlsAcceptance, acceptance) {
				return fmt.Errorf("case %q has unknown acceptance %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(lifecycleControlsFaults, fault) || seenFaults[fault] {
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
	for _, acceptance := range lifecycleControlsAcceptance {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required lifecycle-controls acceptance %q is absent", acceptance)
		}
	}
	for _, category := range lifecycleControlsCategories {
		if !seenCategories[category] {
			return fmt.Errorf("required lifecycle-controls category %q is absent", category)
		}
	}
	for _, fault := range lifecycleControlsFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required lifecycle-controls fault %q is absent", fault)
		}
	}
	return validateMatrixGates(
		"lifecycle controls", matrix.Gates,
		[]string{"browser", "hardening", "matrix", "process", "release"}, true,
	)
}
