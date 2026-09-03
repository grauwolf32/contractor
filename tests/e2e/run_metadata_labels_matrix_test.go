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

type runMetadataLabelsMatrix struct {
	SchemaVersion string                        `yaml:"schema_version"`
	Policy        runMetadataLabelsMatrixPolicy `yaml:"policy"`
	Cases         []runMetadataLabelsMatrixCase `yaml:"cases"`
	Gates         []runMetadataLabelsMatrixGate `yaml:"gates"`
}

type runMetadataLabelsMatrixPolicy struct {
	ExecutionInertness string `yaml:"execution_inertness"`
	Isolation          string `yaml:"isolation"`
	RetainedSurfaces   string `yaml:"retained_surfaces"`
}

type runMetadataLabelsMatrixCase struct {
	ID         string                     `yaml:"id"`
	Category   string                     `yaml:"category"`
	Acceptance []string                   `yaml:"acceptance"`
	Faults     []string                   `yaml:"faults"`
	Expected   string                     `yaml:"expected"`
	Test       runMetadataLabelsTestOwner `yaml:"test"`
}

type runMetadataLabelsTestOwner struct {
	Source string `yaml:"source"`
	Kind   string `yaml:"kind"`
	Name   string `yaml:"name"`
}

type runMetadataLabelsMatrixGate struct {
	ID      string `yaml:"id"`
	Command string `yaml:"command"`
}

var (
	runMetadataLabelsAcceptance = []string{"A1", "A2", "A3", "A4"}
	runMetadataLabelsCategories = []string{
		"idempotency", "lifecycle", "matrix", "persistence", "placement",
		"private_wire", "process", "query", "runtime_isolation", "telemetry",
		"ui", "validation",
	}
	runMetadataLabelsFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
		"F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20",
	}
)

func TestRunMetadataLabelsHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("run_metadata_labels_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeRunMetadataLabelsMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateRunMetadataLabelsMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("missing acceptance", func(t *testing.T) {
		broken := mustDecodeRunMetadataLabelsMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Acceptance = slices.DeleteFunc(
				broken.Cases[index].Acceptance, func(value string) bool { return value == "A1" },
			)
		}
		requireRunMetadataLabelsMatrixError(t, repositoryRoot, broken, "A1")
	})
	t.Run("missing category", func(t *testing.T) {
		broken := mustDecodeRunMetadataLabelsMatrix(t, data)
		broken.Cases = slices.DeleteFunc(
			broken.Cases, func(value runMetadataLabelsMatrixCase) bool { return value.Category == "query" },
		)
		requireRunMetadataLabelsMatrixError(t, repositoryRoot, broken, `category "query"`)
	})
	t.Run("missing fault", func(t *testing.T) {
		broken := mustDecodeRunMetadataLabelsMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Faults = slices.DeleteFunc(
				broken.Cases[index].Faults, func(value string) bool { return value == "F20" },
			)
		}
		requireRunMetadataLabelsMatrixError(t, repositoryRoot, broken, "F20")
	})
	t.Run("duplicate owner", func(t *testing.T) {
		broken := mustDecodeRunMetadataLabelsMatrix(t, data)
		broken.Cases[1].Test = broken.Cases[0].Test
		requireRunMetadataLabelsMatrixError(t, repositoryRoot, broken, "duplicate test owner")
	})
	t.Run("renamed owner", func(t *testing.T) {
		broken := mustDecodeRunMetadataLabelsMatrix(t, data)
		broken.Cases[0].Test.Name += "Renamed"
		requireRunMetadataLabelsMatrixError(t, repositoryRoot, broken, "does not define")
	})
	t.Run("missing gate", func(t *testing.T) {
		broken := mustDecodeRunMetadataLabelsMatrix(t, data)
		broken.Gates = slices.DeleteFunc(
			broken.Gates, func(value runMetadataLabelsMatrixGate) bool { return value.ID == "process" },
		)
		requireRunMetadataLabelsMatrixError(t, repositoryRoot, broken, `gate "process"`)
	})
}

func decodeRunMetadataLabelsMatrix(data []byte) (runMetadataLabelsMatrix, error) {
	var matrix runMetadataLabelsMatrix
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&matrix); err != nil {
		return matrix, fmt.Errorf("decode strict Run metadata-label matrix: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err == nil {
		return matrix, errors.New("Run metadata-label matrix contains a trailing YAML document")
	} else if !errors.Is(err, io.EOF) {
		return matrix, fmt.Errorf("decode Run metadata-label matrix trailer: %w", err)
	}
	return matrix, nil
}

func mustDecodeRunMetadataLabelsMatrix(t *testing.T, data []byte) runMetadataLabelsMatrix {
	t.Helper()
	matrix, err := decodeRunMetadataLabelsMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	return matrix
}

func requireRunMetadataLabelsMatrixError(
	t *testing.T,
	repositoryRoot string,
	matrix runMetadataLabelsMatrix,
	want string,
) {
	t.Helper()
	err := validateRunMetadataLabelsMatrix(repositoryRoot, matrix)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("matrix error = %v, want %q", err, want)
	}
}

func validateRunMetadataLabelsMatrix(
	repositoryRoot string,
	matrix runMetadataLabelsMatrix,
) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.ExecutionInertness) == "" ||
		strings.TrimSpace(matrix.Policy.Isolation) == "" ||
		strings.TrimSpace(matrix.Policy.RetainedSurfaces) == "" {
		return fmt.Errorf("incomplete Run metadata-label policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenAcceptance := map[string]bool{}
	seenCategories := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || strings.TrimSpace(item.Expected) == "" ||
			seenCases[item.ID] || len(item.Acceptance) == 0 {
			return fmt.Errorf("invalid or duplicate Run metadata-label case: %+v", item)
		}
		if !slices.Contains(runMetadataLabelsCategories, item.Category) {
			return fmt.Errorf("case %q has unknown category %q", item.ID, item.Category)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, acceptance := range item.Acceptance {
			if !slices.Contains(runMetadataLabelsAcceptance, acceptance) {
				return fmt.Errorf("case %q has unknown acceptance %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(runMetadataLabelsFaults, fault) || seenFaults[fault] {
				return fmt.Errorf("case %q has unknown or duplicate fault %q", item.ID, fault)
			}
			seenFaults[fault] = true
		}
		owner := item.Test.Source + "#" + item.Test.Kind + "#" + item.Test.Name
		if previous, exists := seenOwners[owner]; exists {
			return fmt.Errorf("duplicate test owner %q in cases %q and %q", owner, previous, item.ID)
		}
		seenOwners[owner] = item.ID
		if err := validateRunMetadataLabelsOwner(repositoryRoot, item.Test); err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
	}
	for _, acceptance := range runMetadataLabelsAcceptance {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required Run metadata-label acceptance %q is absent", acceptance)
		}
	}
	for _, category := range runMetadataLabelsCategories {
		if !seenCategories[category] {
			return fmt.Errorf("required Run metadata-label category %q is absent", category)
		}
	}
	for _, fault := range runMetadataLabelsFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required Run metadata-label fault %q is absent", fault)
		}
	}
	requiredGates := []string{"hardening", "matrix", "process", "release", "ui"}
	seenGates := map[string]bool{}
	for _, gate := range matrix.Gates {
		if gate.ID == "" || gate.Command == "" || seenGates[gate.ID] ||
			!slices.Contains(requiredGates, gate.ID) || !strings.HasPrefix(gate.Command, "make ") {
			return fmt.Errorf("invalid or duplicate Run metadata-label gate: %+v", gate)
		}
		seenGates[gate.ID] = true
	}
	for _, gate := range requiredGates {
		if !seenGates[gate] {
			return fmt.Errorf("required Run metadata-label gate %q is absent", gate)
		}
	}
	return nil
}

func validateRunMetadataLabelsOwner(
	repositoryRoot string,
	owner runMetadataLabelsTestOwner,
) error {
	if owner.Source == "" || owner.Name == "" || filepath.IsAbs(owner.Source) ||
		strings.Contains(owner.Source, "..") {
		return fmt.Errorf("invalid test owner: %+v", owner)
	}
	extension := filepath.Ext(owner.Source)
	expectedExtension := map[string][]string{
		"go_test":    {".go"},
		"pytest":     {".py"},
		"test_title": {".ts", ".tsx"},
	}
	allowedExtensions, exists := expectedExtension[owner.Kind]
	if !exists || !slices.Contains(allowedExtensions, extension) {
		return fmt.Errorf("unsupported test owner: %+v", owner)
	}
	data, err := os.ReadFile(filepath.Join(repositoryRoot, filepath.FromSlash(owner.Source)))
	if err != nil {
		return fmt.Errorf("read test owner %q: %w", owner.Source, err)
	}
	defined := false
	switch owner.Kind {
	case "go_test":
		defined = bytes.Contains(data, []byte("func "+owner.Name+"("))
	case "pytest":
		defined = bytes.Contains(data, []byte("def "+owner.Name+"("))
	case "test_title":
		for _, call := range []string{"it", "test"} {
			for _, quote := range []string{"\"", "'", "`"} {
				if bytes.Contains(data, []byte(call+"("+quote+owner.Name+quote)) {
					defined = true
				}
			}
		}
	}
	if !defined {
		return fmt.Errorf("%s does not define %s owner %s", owner.Source, owner.Kind, owner.Name)
	}
	return nil
}
