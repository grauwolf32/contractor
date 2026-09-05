package e2e

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

type projectWorkspaceMatrix struct {
	SchemaVersion string                 `yaml:"schema_version"`
	Policy        projectWorkspacePolicy `yaml:"policy"`
	Cases         []projectWorkspaceCase `yaml:"cases"`
	Gates         []matrixGate           `yaml:"gates"`
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

func TestProjectWorkspaceHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("project_workspace_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeStrictMatrix[projectWorkspaceMatrix](data, "Project workspace")
	if err != nil {
		t.Fatal(err)
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
		if err := validateMatrixSymbolOwner(repositoryRoot, item.Test.Source, item.Test.Symbol); err != nil {
			t.Fatalf("case %q: %v", item.ID, err)
		}
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

	if err := validateMatrixGates(
		"Project release", matrix.Gates,
		[]string{"browser", "hardening", "matrix", "process", "release"}, false,
	); err != nil {
		t.Fatal(err)
	}
}
