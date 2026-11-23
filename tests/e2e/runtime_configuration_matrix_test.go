package e2e

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

type runtimeConfigurationMatrix struct {
	SchemaVersion string                         `yaml:"schema_version"`
	Policy        runtimeConfigurationPolicy     `yaml:"policy"`
	Mutations     []runtimeConfigurationMutation `yaml:"mutations"`
	Cases         []runtimeConfigurationCase     `yaml:"cases"`
	Gates         []matrixGate                   `yaml:"gates"`
}

type runtimeConfigurationPolicy struct {
	RealBoundary    string `yaml:"real_boundary"`
	FiniteFailure   string `yaml:"finite_failure"`
	SecretRetention string `yaml:"secret_retention"`
}

type runtimeConfigurationMutation struct {
	ID           string                   `yaml:"id"`
	ResponseLoss string                   `yaml:"response_loss"`
	Test         runtimeConfigurationTest `yaml:"test"`
}

type runtimeConfigurationCase struct {
	ID           string                   `yaml:"id"`
	Category     string                   `yaml:"category"`
	Requirements []string                 `yaml:"requirements"`
	Expected     string                   `yaml:"expected"`
	Test         runtimeConfigurationTest `yaml:"test"`
}

type runtimeConfigurationTest struct {
	Source string `yaml:"source"`
	Symbol string `yaml:"symbol"`
}

func TestRuntimeConfigurationHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("runtime_configuration_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeStrictMatrix[runtimeConfigurationMatrix](data, "Runtime configuration")
	if err != nil {
		t.Fatal(err)
	}
	if matrix.SchemaVersion != "1.0" || matrix.Policy.RealBoundary == "" ||
		matrix.Policy.FiniteFailure == "" || matrix.Policy.SecretRetention == "" {
		t.Fatalf("incomplete Runtime configuration matrix policy: %+v", matrix.Policy)
	}

	requiredMutations := []string{
		"run.create",
		"runtime_agent_labels.put_delete",
		"runtime_config.publish",
		"runtime_credential.create_delete",
		"runtime_label.put_delete",
	}
	seenMutations := map[string]bool{}
	for _, mutation := range matrix.Mutations {
		if mutation.ID == "" || mutation.ResponseLoss == "" || seenMutations[mutation.ID] {
			t.Fatalf("invalid or duplicate mutation: %+v", mutation)
		}
		seenMutations[mutation.ID] = true
		if err := validateMatrixSymbolOwner(
			repositoryRoot, mutation.Test.Source, mutation.Test.Symbol,
		); err != nil {
			t.Fatalf("mutation %q: %v", mutation.ID, err)
		}
	}
	for _, mutation := range requiredMutations {
		if !seenMutations[mutation] {
			t.Errorf("mutation %q has no response-loss contract", mutation)
		}
	}

	requiredCategories := []string{
		"adapter_failure", "browser", "identity", "lease_release", "process", "race", "secrets",
	}
	seenCategories := map[string]bool{}
	seenRequirements := map[string]bool{}
	seenCases := map[string]bool{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Expected == "" || item.Category == "" ||
			len(item.Requirements) == 0 || seenCases[item.ID] {
			t.Fatalf("invalid or duplicate hardening case: %+v", item)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, requirement := range item.Requirements {
			if !slices.Contains([]string{"A1", "A2", "A3", "A4"}, requirement) {
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
			t.Errorf("hardening category %q has no case", category)
		}
	}
	for _, requirement := range []string{"A1", "A2", "A3", "A4"} {
		if !seenRequirements[requirement] {
			t.Errorf("acceptance requirement %q has no case", requirement)
		}
	}

	if err := validateMatrixGates(
		"Runtime configuration", matrix.Gates,
		[]string{"browser", "hardening", "matrix", "process", "release"}, false,
	); err != nil {
		t.Fatal(err)
	}
}
