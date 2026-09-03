package e2e

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"go.yaml.in/yaml/v4"
)

type workerObservationsMatrix struct {
	SchemaVersion string                         `yaml:"schema_version"`
	Policy        workerObservationsMatrixPolicy `yaml:"policy"`
	Cases         []workerObservationsMatrixCase `yaml:"cases"`
	Gates         []workerObservationsMatrixGate `yaml:"gates"`
}

type workerObservationsMatrixPolicy struct {
	ResultBoundary string `yaml:"result_boundary"`
	LiveState      string `yaml:"live_state"`
	Retention      string `yaml:"retention"`
}

type workerObservationsMatrixCase struct {
	ID         string                      `yaml:"id"`
	Category   string                      `yaml:"category"`
	Acceptance []string                    `yaml:"acceptance"`
	Faults     []string                    `yaml:"faults"`
	Expected   string                      `yaml:"expected"`
	Test       workerObservationsTestOwner `yaml:"test"`
}

type workerObservationsTestOwner struct {
	Source string `yaml:"source"`
	Symbol string `yaml:"symbol"`
}

type workerObservationsMatrixGate struct {
	ID      string `yaml:"id"`
	Command string `yaml:"command"`
}

var (
	workerObservationsAcceptance = []string{"A1", "A2", "A3", "A4"}
	workerObservationsCategories = []string{
		"correlation", "lifecycle", "observation", "planner", "process", "result",
		"retention", "transport", "validation",
	}
	workerObservationsFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08",
		"F09", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
	}
)

func TestWorkerObservationsHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("worker_observations_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeWorkerObservationsMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateWorkerObservationsMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("missing acceptance", func(t *testing.T) {
		broken := mustDecodeWorkerObservationsMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Acceptance = slices.DeleteFunc(
				broken.Cases[index].Acceptance, func(value string) bool { return value == "A3" },
			)
		}
		requireWorkerObservationsMatrixError(t, repositoryRoot, broken, "A3")
	})
	t.Run("missing fault", func(t *testing.T) {
		broken := mustDecodeWorkerObservationsMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Faults = slices.DeleteFunc(
				broken.Cases[index].Faults, func(value string) bool { return value == "F16" },
			)
		}
		requireWorkerObservationsMatrixError(t, repositoryRoot, broken, "F16")
	})
	t.Run("duplicate owner", func(t *testing.T) {
		broken := mustDecodeWorkerObservationsMatrix(t, data)
		broken.Cases[1].Test = broken.Cases[0].Test
		requireWorkerObservationsMatrixError(t, repositoryRoot, broken, "duplicate test owner")
	})
	t.Run("renamed owner", func(t *testing.T) {
		broken := mustDecodeWorkerObservationsMatrix(t, data)
		broken.Cases[0].Test.Symbol += "Renamed"
		requireWorkerObservationsMatrixError(t, repositoryRoot, broken, "does not define")
	})
	t.Run("missing gate", func(t *testing.T) {
		broken := mustDecodeWorkerObservationsMatrix(t, data)
		broken.Gates = slices.DeleteFunc(
			broken.Gates, func(value workerObservationsMatrixGate) bool { return value.ID == "e2e" },
		)
		requireWorkerObservationsMatrixError(t, repositoryRoot, broken, `gate "e2e"`)
	})
}

func decodeWorkerObservationsMatrix(data []byte) (workerObservationsMatrix, error) {
	var matrix workerObservationsMatrix
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&matrix); err != nil {
		return matrix, fmt.Errorf("decode strict Worker observations matrix: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err == nil {
		return matrix, errors.New("Worker observations matrix contains a trailing YAML document")
	} else if !errors.Is(err, io.EOF) {
		return matrix, fmt.Errorf("decode Worker observations matrix trailer: %w", err)
	}
	return matrix, nil
}

func mustDecodeWorkerObservationsMatrix(t *testing.T, data []byte) workerObservationsMatrix {
	t.Helper()
	matrix, err := decodeWorkerObservationsMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	return matrix
}

func requireWorkerObservationsMatrixError(
	t *testing.T,
	repositoryRoot string,
	matrix workerObservationsMatrix,
	want string,
) {
	t.Helper()
	err := validateWorkerObservationsMatrix(repositoryRoot, matrix)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("matrix error = %v, want %q", err, want)
	}
}

func validateWorkerObservationsMatrix(
	repositoryRoot string,
	matrix workerObservationsMatrix,
) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.ResultBoundary) == "" ||
		strings.TrimSpace(matrix.Policy.LiveState) == "" || strings.TrimSpace(matrix.Policy.Retention) == "" {
		return fmt.Errorf("incomplete Worker observations policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenAcceptance := map[string]bool{}
	seenCategories := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || strings.TrimSpace(item.Expected) == "" ||
			seenCases[item.ID] || len(item.Acceptance) == 0 {
			return fmt.Errorf("invalid or duplicate Worker observations case: %+v", item)
		}
		if !slices.Contains(workerObservationsCategories, item.Category) {
			return fmt.Errorf("case %q has unknown category %q", item.ID, item.Category)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, acceptance := range item.Acceptance {
			if !slices.Contains(workerObservationsAcceptance, acceptance) {
				return fmt.Errorf("case %q has unknown acceptance %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(workerObservationsFaults, fault) || seenFaults[fault] {
				return fmt.Errorf("case %q has unknown or duplicate fault %q", item.ID, fault)
			}
			seenFaults[fault] = true
		}
		owner := item.Test.Source + "#" + item.Test.Symbol
		if previous, exists := seenOwners[owner]; exists {
			return fmt.Errorf("duplicate test owner %q in cases %q and %q", owner, previous, item.ID)
		}
		seenOwners[owner] = item.ID
		if err := validateWorkerObservationsOwner(repositoryRoot, item.Test); err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
	}
	for _, acceptance := range workerObservationsAcceptance {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required Worker observations acceptance %q is absent", acceptance)
		}
	}
	for _, category := range workerObservationsCategories {
		if !seenCategories[category] {
			return fmt.Errorf("required Worker observations category %q is absent", category)
		}
	}
	for _, fault := range workerObservationsFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required Worker observations fault %q is absent", fault)
		}
	}
	requiredGates := []string{"e2e", "hardening", "matrix", "release", "runtime"}
	seenGates := map[string]bool{}
	for _, gate := range matrix.Gates {
		if gate.ID == "" || gate.Command == "" || seenGates[gate.ID] ||
			!slices.Contains(requiredGates, gate.ID) || !strings.HasPrefix(gate.Command, "make ") {
			return fmt.Errorf("invalid or duplicate Worker observations gate: %+v", gate)
		}
		seenGates[gate.ID] = true
	}
	for _, gate := range requiredGates {
		if !seenGates[gate] {
			return fmt.Errorf("required Worker observations gate %q is absent", gate)
		}
	}
	return nil
}

func validateWorkerObservationsOwner(
	repositoryRoot string,
	owner workerObservationsTestOwner,
) error {
	if owner.Source == "" || owner.Symbol == "" || filepath.IsAbs(owner.Source) ||
		strings.Contains(owner.Source, "..") {
		return fmt.Errorf("invalid test owner: %+v", owner)
	}
	extension := filepath.Ext(owner.Source)
	if !slices.Contains([]string{".go", ".py"}, extension) {
		return fmt.Errorf("unsupported test owner: %+v", owner)
	}
	data, err := os.ReadFile(filepath.Join(repositoryRoot, filepath.FromSlash(owner.Source)))
	if err != nil {
		return fmt.Errorf("read test owner %q: %w", owner.Source, err)
	}
	prefix := "func "
	if extension == ".py" {
		prefix = "def "
	}
	if !bytes.Contains(data, []byte(prefix+owner.Symbol+"(")) {
		return fmt.Errorf("%s does not define %s", owner.Source, owner.Symbol)
	}
	return nil
}
