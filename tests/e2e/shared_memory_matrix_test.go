package e2e

import (
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type sharedMemoryMatrix struct {
	SchemaVersion string                   `yaml:"schema_version"`
	Policy        sharedMemoryMatrixPolicy `yaml:"policy"`
	Cases         []sharedMemoryMatrixCase `yaml:"cases"`
	Gates         []matrixGate             `yaml:"gates"`
}

type sharedMemoryMatrixPolicy struct {
	RetainedSurface string `yaml:"retained_surface"`
	Isolation       string `yaml:"isolation"`
	SerializedV1    string `yaml:"serialized_v1"`
}

type sharedMemoryMatrixCase struct {
	ID         string                 `yaml:"id"`
	Category   string                 `yaml:"category"`
	Invariants []string               `yaml:"invariants"`
	Faults     []string               `yaml:"faults"`
	Expected   string                 `yaml:"expected"`
	Test       sharedMemoryMatrixTest `yaml:"test"`
}

type sharedMemoryMatrixTest struct {
	Source string `yaml:"source"`
	Symbol string `yaml:"symbol"`
}

var (
	sharedMemoryRequiredInvariants = []string{
		"I01", "I02", "I03", "I04", "I05", "I06", "I07", "I08", "I09", "I10", "I11",
	}
	sharedMemoryRequiredFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
		"F11", "F12", "F13", "F14", "F15", "F16", "F17",
	}
)

func TestSharedMemoryHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("shared_memory_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeSharedMemoryMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateSharedMemoryMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("validator_rejects_absent_coverage", func(t *testing.T) {
		broken := mustDecodeSharedMemoryMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Invariants = slices.DeleteFunc(
				broken.Cases[index].Invariants, func(value string) bool { return value == "I01" },
			)
		}
		if err := validateSharedMemoryMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "I01") {
			t.Fatalf("absent invariant validation error = %v", err)
		}
	})
	t.Run("validator_rejects_renamed_owner", func(t *testing.T) {
		broken := mustDecodeSharedMemoryMatrix(t, data)
		broken.Cases[0].Test.Symbol += "Renamed"
		if err := validateSharedMemoryMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "does not define") {
			t.Fatalf("renamed owner validation error = %v", err)
		}
	})
	t.Run("validator_rejects_duplicate_owner", func(t *testing.T) {
		broken := mustDecodeSharedMemoryMatrix(t, data)
		broken.Cases[1].Test = broken.Cases[0].Test
		if err := validateSharedMemoryMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "duplicate test owner") {
			t.Fatalf("duplicate owner validation error = %v", err)
		}
	})
}

func TestSharedMemoryUsesOnlyArtifactPlaneInventories(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	for _, relative := range []string{
		"api/openapi/contractor-public-v1.yaml",
		"internal/httpapi/public/router.go",
		"internal/httpapi/privateartifacts/handler.go",
	} {
		data, err := os.ReadFile(filepath.Join(repositoryRoot, relative))
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(bytes.ToLower(data), []byte("/memory")) {
			t.Fatalf("Memory-specific HTTP route appears in %s", relative)
		}
	}
	err := filepath.WalkDir(
		filepath.Join(repositoryRoot, "internal", "persistence", "migrations"),
		func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() || filepath.Ext(path) != ".sql" {
				return nil
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			if bytes.Contains(bytes.ToLower(data), []byte("memory")) {
				return fmt.Errorf("Memory-specific persistence appears in %s", path)
			}
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
}

func decodeSharedMemoryMatrix(data []byte) (sharedMemoryMatrix, error) {
	return decodeStrictMatrix[sharedMemoryMatrix](data, "shared-memory")
}

func mustDecodeSharedMemoryMatrix(t *testing.T, data []byte) sharedMemoryMatrix {
	t.Helper()
	return mustDecodeStrictMatrix[sharedMemoryMatrix](t, data, "shared-memory")
}

func validateSharedMemoryMatrix(repositoryRoot string, matrix sharedMemoryMatrix) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.RetainedSurface) == "" ||
		strings.TrimSpace(matrix.Policy.Isolation) == "" || strings.TrimSpace(matrix.Policy.SerializedV1) == "" {
		return fmt.Errorf("incomplete shared-memory matrix policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenInvariants := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || item.Expected == "" || seenCases[item.ID] ||
			len(item.Invariants)+len(item.Faults) == 0 {
			return fmt.Errorf("invalid or duplicate shared-memory case: %+v", item)
		}
		seenCases[item.ID] = true
		for _, invariant := range item.Invariants {
			if !slices.Contains(sharedMemoryRequiredInvariants, invariant) || seenInvariants[invariant] {
				return fmt.Errorf("case %q has unknown or duplicate invariant %q", item.ID, invariant)
			}
			seenInvariants[invariant] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(sharedMemoryRequiredFaults, fault) || seenFaults[fault] {
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
	for _, invariant := range sharedMemoryRequiredInvariants {
		if !seenInvariants[invariant] {
			return fmt.Errorf("required invariant %s has no owning case", invariant)
		}
	}
	for _, fault := range sharedMemoryRequiredFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required fault %s has no owning case", fault)
		}
	}

	return validateMatrixGates(
		"shared-memory", matrix.Gates,
		[]string{"contracts", "faults", "hardening", "matrix", "process", "release"}, false,
	)
}
