package e2e

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"go.yaml.in/yaml/v4"
)

type projectWorkspaceMatrix struct {
	SchemaVersion string                        `yaml:"schema_version"`
	Policy        projectWorkspacePolicy        `yaml:"policy"`
	Cases         []projectWorkspaceCase        `yaml:"cases"`
	Gates         []projectWorkspaceReleaseGate `yaml:"gates"`
}

type projectWorkspacePolicy struct {
	Ownership       string `yaml:"ownership"`
	ExactLineage    string `yaml:"exact_lineage"`
	FiniteRecovery  string `yaml:"finite_recovery"`
	SecretRetention string `yaml:"secret_retention"`
}

type projectWorkspaceCase struct {
	ID           string               `yaml:"id"`
	Category     string               `yaml:"category"`
	Requirements []string             `yaml:"requirements"`
	Expected     string               `yaml:"expected"`
	Test         projectWorkspaceTest `yaml:"test"`
}

type projectWorkspaceTest struct {
	Source string `yaml:"source"`
	Symbol string `yaml:"symbol"`
}

type projectWorkspaceReleaseGate struct {
	ID      string `yaml:"id"`
	Command string `yaml:"command"`
}

func TestProjectWorkspaceHardeningMatrixIsComplete(t *testing.T) {
	data, err := os.ReadFile("project_workspace_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	var matrix projectWorkspaceMatrix
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&matrix); err != nil {
		t.Fatalf("decode strict Project workspace matrix: %v", err)
	}
	if matrix.SchemaVersion != "1.0" || matrix.Policy.Ownership == "" ||
		matrix.Policy.ExactLineage == "" || matrix.Policy.FiniteRecovery == "" ||
		matrix.Policy.SecretRetention == "" {
		t.Fatalf("incomplete Project workspace policy: %+v", matrix.Policy)
	}

	requiredCategories := []string{
		"browser", "cas_replay", "lineage", "ownership", "placement",
		"process_recovery", "publication_race", "secrets", "standalone_regression",
	}
	seenCategories := map[string]bool{}
	seenRequirements := map[string]bool{}
	seenCases := map[string]bool{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || item.Expected == "" ||
			len(item.Requirements) == 0 || seenCases[item.ID] {
			t.Fatalf("invalid or duplicate Project workspace case: %+v", item)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, requirement := range item.Requirements {
			if !slices.Contains([]string{"A1", "A2", "A3"}, requirement) {
				t.Fatalf("case %q has unknown requirement %q", item.ID, requirement)
			}
			seenRequirements[requirement] = true
		}
		assertProjectWorkspaceTest(t, item.Test)
	}
	for _, category := range requiredCategories {
		if !seenCategories[category] {
			t.Errorf("Project workspace category %q has no case", category)
		}
	}
	for _, requirement := range []string{"A1", "A2", "A3"} {
		if !seenRequirements[requirement] {
			t.Errorf("acceptance requirement %q has no case", requirement)
		}
	}

	requiredGates := []string{"browser", "hardening", "matrix", "process", "release"}
	seenGates := map[string]bool{}
	for _, gate := range matrix.Gates {
		if gate.ID == "" || gate.Command == "" || seenGates[gate.ID] ||
			!strings.HasPrefix(gate.Command, "make ") {
			t.Fatalf("invalid or duplicate Project release gate: %+v", gate)
		}
		seenGates[gate.ID] = true
	}
	for _, gate := range requiredGates {
		if !seenGates[gate] {
			t.Errorf("Project release gate %q is absent", gate)
		}
	}
}

func assertProjectWorkspaceTest(t *testing.T, ref projectWorkspaceTest) {
	t.Helper()
	if ref.Source == "" || ref.Symbol == "" || filepath.IsAbs(ref.Source) ||
		strings.Contains(ref.Source, "..") {
		t.Fatalf("invalid Project workspace test reference: %+v", ref)
	}
	extension := filepath.Ext(ref.Source)
	if !slices.Contains([]string{".go", ".py"}, extension) {
		t.Fatalf("unsupported Project workspace test source: %+v", ref)
	}
	data, err := os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(ref.Source)))
	if err != nil {
		t.Fatalf("read referenced test %q: %v", ref.Source, err)
	}
	prefix := "func "
	if extension == ".py" {
		prefix = "def "
	}
	if !bytes.Contains(data, []byte(prefix+ref.Symbol+"(")) {
		t.Fatalf("%s does not define %s", ref.Source, ref.Symbol)
	}
}
