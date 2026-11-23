package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type workerSummarizerMatrix struct {
	SchemaVersion string                       `yaml:"schema_version"`
	Policy        workerSummarizerMatrixPolicy `yaml:"policy"`
	Cases         []workerSummarizerMatrixCase `yaml:"cases"`
	Gates         []matrixGate                 `yaml:"gates"`
}

type workerSummarizerMatrixPolicy struct {
	Strategy   string `yaml:"strategy"`
	Accounting string `yaml:"accounting"`
	Retention  string `yaml:"retention"`
}

type workerSummarizerMatrixCase struct {
	ID         string                      `yaml:"id"`
	Category   string                      `yaml:"category"`
	Acceptance []string                    `yaml:"acceptance"`
	Faults     []string                    `yaml:"faults"`
	Expected   string                      `yaml:"expected"`
	Test       workerObservationsTestOwner `yaml:"test"`
}

var (
	workerSummarizerAcceptance = []string{"A1", "A2", "A3", "A4"}
	workerSummarizerCategories = []string{
		"configuration", "isolation", "lifecycle", "process", "result",
		"retention", "trigger", "usage", "validation",
	}
	workerSummarizerFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08",
		"F09", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
	}
)

func TestWorkerSummarizerHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("worker_summarizer_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeWorkerSummarizerMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateWorkerSummarizerMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("missing acceptance", func(t *testing.T) {
		broken := mustDecodeWorkerSummarizerMatrix(t, data)
		for index := range broken.Cases {
			for acceptance := range broken.Cases[index].Acceptance {
				if broken.Cases[index].Acceptance[acceptance] == "A3" {
					broken.Cases[index].Acceptance[acceptance] = "A2"
				}
			}
		}
		requireWorkerSummarizerMatrixError(t, repositoryRoot, broken, "A3")
	})
	t.Run("missing fault", func(t *testing.T) {
		broken := mustDecodeWorkerSummarizerMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Faults = slices.DeleteFunc(
				broken.Cases[index].Faults, func(value string) bool { return value == "F16" },
			)
		}
		requireWorkerSummarizerMatrixError(t, repositoryRoot, broken, "F16")
	})
	t.Run("missing category", func(t *testing.T) {
		broken := mustDecodeWorkerSummarizerMatrix(t, data)
		broken.Cases = slices.DeleteFunc(
			broken.Cases, func(value workerSummarizerMatrixCase) bool { return value.Category == "validation" },
		)
		requireWorkerSummarizerMatrixError(t, repositoryRoot, broken, "validation")
	})
	t.Run("duplicate owner", func(t *testing.T) {
		broken := mustDecodeWorkerSummarizerMatrix(t, data)
		broken.Cases[1].Test = broken.Cases[0].Test
		requireWorkerSummarizerMatrixError(t, repositoryRoot, broken, "duplicate test owner")
	})
	t.Run("renamed owner", func(t *testing.T) {
		broken := mustDecodeWorkerSummarizerMatrix(t, data)
		broken.Cases[0].Test.Symbol += "Renamed"
		requireWorkerSummarizerMatrixError(t, repositoryRoot, broken, "does not define")
	})
	t.Run("missing release gate", func(t *testing.T) {
		broken := mustDecodeWorkerSummarizerMatrix(t, data)
		broken.Gates = slices.DeleteFunc(
			broken.Gates, func(value matrixGate) bool { return value.ID == "release" },
		)
		requireWorkerSummarizerMatrixError(t, repositoryRoot, broken, `gate "release"`)
	})
}

func decodeWorkerSummarizerMatrix(data []byte) (workerSummarizerMatrix, error) {
	return decodeStrictMatrix[workerSummarizerMatrix](data, "Worker summarizer")
}

func mustDecodeWorkerSummarizerMatrix(t *testing.T, data []byte) workerSummarizerMatrix {
	t.Helper()
	return mustDecodeStrictMatrix[workerSummarizerMatrix](t, data, "Worker summarizer")
}

func requireWorkerSummarizerMatrixError(
	t *testing.T,
	repositoryRoot string,
	matrix workerSummarizerMatrix,
	want string,
) {
	t.Helper()
	err := validateWorkerSummarizerMatrix(repositoryRoot, matrix)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("matrix error = %v, want %q", err, want)
	}
}

func validateWorkerSummarizerMatrix(
	repositoryRoot string,
	matrix workerSummarizerMatrix,
) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.Strategy) == "" ||
		strings.TrimSpace(matrix.Policy.Accounting) == "" ||
		strings.TrimSpace(matrix.Policy.Retention) == "" {
		return fmt.Errorf("incomplete Worker summarizer policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenAcceptance := map[string]bool{}
	seenCategories := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || strings.TrimSpace(item.Expected) == "" ||
			seenCases[item.ID] || len(item.Acceptance) == 0 {
			return fmt.Errorf("invalid or duplicate Worker summarizer case: %+v", item)
		}
		if !slices.Contains(workerSummarizerCategories, item.Category) {
			return fmt.Errorf("case %q has unknown category %q", item.ID, item.Category)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, acceptance := range item.Acceptance {
			if !slices.Contains(workerSummarizerAcceptance, acceptance) {
				return fmt.Errorf("case %q has unknown acceptance %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(workerSummarizerFaults, fault) || seenFaults[fault] {
				return fmt.Errorf("case %q has unknown or duplicate fault %q", item.ID, fault)
			}
			seenFaults[fault] = true
		}
		owner := item.Test.Source + "#" + item.Test.Symbol
		if previous, exists := seenOwners[owner]; exists {
			return fmt.Errorf("duplicate test owner %q in cases %q and %q", owner, previous, item.ID)
		}
		seenOwners[owner] = item.ID
		if err := validateMatrixSymbolOwner(repositoryRoot, item.Test.Source, item.Test.Symbol); err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
	}
	for _, acceptance := range workerSummarizerAcceptance {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required Worker summarizer acceptance %q is absent", acceptance)
		}
	}
	for _, category := range workerSummarizerCategories {
		if !seenCategories[category] {
			return fmt.Errorf("required Worker summarizer category %q is absent", category)
		}
	}
	for _, fault := range workerSummarizerFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required Worker summarizer fault %q is absent", fault)
		}
	}
	return validateMatrixGates(
		"Worker summarizer", matrix.Gates,
		[]string{"e2e", "hardening", "matrix", "release", "runtime"}, true,
	)
}
