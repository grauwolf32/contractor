package faults

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"go.yaml.in/yaml/v4"
)

type matrix struct {
	SchemaVersion string      `yaml:"schema_version"`
	Policy        policy      `yaml:"policy"`
	Operations    []operation `yaml:"operations"`
	Cases         []faultCase `yaml:"cases"`
}

type policy struct {
	RetryRule  string `yaml:"retry_rule"`
	CrashRule  string `yaml:"crash_rule"`
	SecretRule string `yaml:"secret_rule"`
}

type operation struct {
	ID           string  `yaml:"id"`
	Semantics    string  `yaml:"semantics"`
	Identity     string  `yaml:"identity"`
	ResponseLoss string  `yaml:"response_loss"`
	Test         testRef `yaml:"test"`
}

type faultCase struct {
	ID           string   `yaml:"id"`
	Category     string   `yaml:"category"`
	Phase        string   `yaml:"phase"`
	Boundary     string   `yaml:"boundary"`
	Fault        string   `yaml:"fault"`
	Expected     string   `yaml:"expected"`
	Requirements []string `yaml:"requirements"`
	Test         testRef  `yaml:"test"`
}

type testRef struct {
	Source string `yaml:"source"`
	Symbol string `yaml:"symbol"`
}

func TestFaultMatrixIsCompleteAndReferencesRealTests(t *testing.T) {
	data, err := os.ReadFile("matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	var value matrix
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&value); err != nil {
		t.Fatalf("decode strict fault matrix: %v", err)
	}
	if value.SchemaVersion != "1.0" || value.Policy.RetryRule == "" ||
		value.Policy.CrashRule == "" || value.Policy.SecretRule == "" {
		t.Fatalf("fault matrix policy/version is incomplete: %+v", value.Policy)
	}

	requiredOperations := []string{
		"control.agent.heartbeat",
		"control.agent.register",
		"planner.worker.invoke",
		"private.artifact.put",
		"public.artifact.put",
		"public.run.cancel",
		"public.run.create",
		"runtime.allocation.abort",
		"runtime.allocation.finalize",
		"runtime.allocation.prepare",
		"runtime.allocation.release",
		"scheduler.result.commit",
		"telemetry.report.persist",
	}
	seenOperations := make(map[string]bool, len(value.Operations))
	for _, item := range value.Operations {
		if item.ID == "" || seenOperations[item.ID] || item.Semantics == "" ||
			item.Identity == "" || item.ResponseLoss == "" {
			t.Fatalf("invalid or duplicate operation: %+v", item)
		}
		seenOperations[item.ID] = true
		assertTestRef(t, item.Test)
	}
	for _, operationID := range requiredOperations {
		if !seenOperations[operationID] {
			t.Errorf("mutating operation %q has no response-loss contract", operationID)
		}
	}

	requiredPhases := []string{"preparing", "running", "finalizing", "aborting", "terminal"}
	requiredCategories := []string{"crash", "retry", "race", "attack", "resource"}
	requiredRequirements := []string{"R1", "R2", "R3", "R4", "R5", "R6"}
	seenCases := make(map[string]bool, len(value.Cases))
	seenPhases := map[string]bool{}
	seenCategories := map[string]bool{}
	seenRequirements := map[string]bool{}
	for _, item := range value.Cases {
		if item.ID == "" || seenCases[item.ID] || item.Boundary == "" || item.Fault == "" ||
			item.Expected == "" || len(item.Requirements) == 0 {
			t.Fatalf("invalid or duplicate fault case: %+v", item)
		}
		seenCases[item.ID] = true
		seenPhases[item.Phase] = true
		seenCategories[item.Category] = true
		for _, requirement := range item.Requirements {
			seenRequirements[requirement] = true
		}
		assertTestRef(t, item.Test)
	}
	for _, phase := range requiredPhases {
		if !seenPhases[phase] {
			t.Errorf("durable phase %q has no fault case", phase)
		}
	}
	for _, category := range requiredCategories {
		if !seenCategories[category] {
			t.Errorf("fault category %q has no case", category)
		}
	}
	for _, requirement := range requiredRequirements {
		if !seenRequirements[requirement] {
			t.Errorf("hardening requirement %q has no fault case", requirement)
		}
	}
}

func assertTestRef(t *testing.T, ref testRef) {
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
