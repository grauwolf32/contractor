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

type runtimeConfigurationMatrix struct {
	SchemaVersion string                         `yaml:"schema_version"`
	Policy        runtimeConfigurationPolicy     `yaml:"policy"`
	Mutations     []runtimeConfigurationMutation `yaml:"mutations"`
	Cases         []runtimeConfigurationCase     `yaml:"cases"`
	Gates         []runtimeConfigurationGate     `yaml:"gates"`
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

type runtimeConfigurationGate struct {
	ID      string `yaml:"id"`
	Command string `yaml:"command"`
}

func TestRuntimeConfigurationHardeningMatrixIsComplete(t *testing.T) {
	data, err := os.ReadFile("runtime_configuration_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	var matrix runtimeConfigurationMatrix
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&matrix); err != nil {
		t.Fatalf("decode strict Runtime configuration matrix: %v", err)
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
		assertRuntimeConfigurationTest(t, mutation.Test)
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
		assertRuntimeConfigurationTest(t, item.Test)
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

	requiredGates := []string{"browser", "hardening", "matrix", "process", "release"}
	seenGates := map[string]bool{}
	for _, gate := range matrix.Gates {
		if gate.ID == "" || gate.Command == "" || seenGates[gate.ID] ||
			!strings.HasPrefix(gate.Command, "make ") {
			t.Fatalf("invalid or duplicate gate: %+v", gate)
		}
		seenGates[gate.ID] = true
	}
	for _, gate := range requiredGates {
		if !seenGates[gate] {
			t.Errorf("release gate %q is absent", gate)
		}
	}
}

func assertRuntimeConfigurationTest(t *testing.T, ref runtimeConfigurationTest) {
	t.Helper()
	if ref.Source == "" || ref.Symbol == "" || filepath.IsAbs(ref.Source) ||
		strings.Contains(ref.Source, "..") {
		t.Fatalf("invalid test reference: %+v", ref)
	}
	extension := filepath.Ext(ref.Source)
	if !slices.Contains([]string{".go", ".py"}, extension) {
		t.Fatalf("unsupported test source: %+v", ref)
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
