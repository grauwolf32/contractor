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

type httpCaidoMatrix struct {
	SchemaVersion string                `yaml:"schema_version"`
	Policy        httpCaidoMatrixPolicy `yaml:"policy"`
	Cases         []httpCaidoMatrixCase `yaml:"cases"`
	Gates         []httpCaidoMatrixGate `yaml:"gates"`
}

type httpCaidoMatrixPolicy struct {
	EgressBoundary       string `yaml:"egress_boundary"`
	GraphQLCompatibility string `yaml:"graphql_compatibility"`
	Retention            string `yaml:"retention"`
}

type httpCaidoMatrixCase struct {
	ID           string           `yaml:"id"`
	Category     string           `yaml:"category"`
	Requirements []string         `yaml:"requirements"`
	Faults       []string         `yaml:"faults"`
	Expected     string           `yaml:"expected"`
	Test         httpCaidoTestRef `yaml:"test"`
}

type httpCaidoTestRef struct {
	Source string `yaml:"source"`
	Symbol string `yaml:"symbol"`
}

type httpCaidoMatrixGate struct {
	ID      string `yaml:"id"`
	Command string `yaml:"command"`
}

var (
	httpCaidoCategories = []string{
		"cancellation", "compatibility", "concurrency", "graphql_bounds", "input_bounds",
		"lifecycle", "process", "proxy_egress", "redaction", "redirect_egress", "response_bounds",
	}
	httpCaidoFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10", "F11", "F12", "F13",
	}
)

func TestHTTPCaidoHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("http_caido_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeHTTPCaidoMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateHTTPCaidoMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("validator_rejects_missing_category", func(t *testing.T) {
		broken := mustDecodeHTTPCaidoMatrix(t, data)
		broken.Cases = slices.DeleteFunc(
			broken.Cases,
			func(item httpCaidoMatrixCase) bool { return item.Category == "compatibility" },
		)
		if err := validateHTTPCaidoMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "compatibility") {
			t.Fatalf("missing category validation error = %v", err)
		}
	})
	t.Run("validator_rejects_missing_fault", func(t *testing.T) {
		broken := mustDecodeHTTPCaidoMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Faults = slices.DeleteFunc(
				broken.Cases[index].Faults,
				func(value string) bool { return value == "F01" },
			)
		}
		if err := validateHTTPCaidoMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "F01") {
			t.Fatalf("missing fault validation error = %v", err)
		}
	})
	t.Run("validator_rejects_renamed_owner", func(t *testing.T) {
		broken := mustDecodeHTTPCaidoMatrix(t, data)
		broken.Cases[0].Test.Symbol += "Renamed"
		if err := validateHTTPCaidoMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "does not define") {
			t.Fatalf("renamed owner validation error = %v", err)
		}
	})
}

func decodeHTTPCaidoMatrix(data []byte) (httpCaidoMatrix, error) {
	var matrix httpCaidoMatrix
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&matrix); err != nil {
		return matrix, fmt.Errorf("decode strict HTTP/Caido matrix: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err == nil {
		return matrix, errors.New("HTTP/Caido matrix contains a trailing YAML document")
	} else if !errors.Is(err, io.EOF) {
		return matrix, fmt.Errorf("decode HTTP/Caido matrix trailer: %w", err)
	}
	return matrix, nil
}

func mustDecodeHTTPCaidoMatrix(t *testing.T, data []byte) httpCaidoMatrix {
	t.Helper()
	matrix, err := decodeHTTPCaidoMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	return matrix
}

func validateHTTPCaidoMatrix(repositoryRoot string, matrix httpCaidoMatrix) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.EgressBoundary) == "" ||
		strings.TrimSpace(matrix.Policy.GraphQLCompatibility) == "" ||
		strings.TrimSpace(matrix.Policy.Retention) == "" {
		return fmt.Errorf("incomplete HTTP/Caido matrix policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenCategories := map[string]bool{}
	seenRequirements := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || item.Expected == "" || seenCases[item.ID] ||
			len(item.Requirements) == 0 {
			return fmt.Errorf("invalid or duplicate HTTP/Caido case: %+v", item)
		}
		if !slices.Contains(httpCaidoCategories, item.Category) {
			return fmt.Errorf("case %q has unknown category %q", item.ID, item.Category)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, requirement := range item.Requirements {
			if !slices.Contains([]string{"A1", "A2", "A3"}, requirement) {
				return fmt.Errorf("case %q has unknown requirement %q", item.ID, requirement)
			}
			seenRequirements[requirement] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(httpCaidoFaults, fault) || seenFaults[fault] {
				return fmt.Errorf("case %q has unknown or duplicate fault %q", item.ID, fault)
			}
			seenFaults[fault] = true
		}
		owner := item.Test.Source + "#" + item.Test.Symbol
		if previous, exists := seenOwners[owner]; exists {
			return fmt.Errorf("duplicate test owner %q in cases %q and %q", owner, previous, item.ID)
		}
		seenOwners[owner] = item.ID
		if err := validateHTTPCaidoTestOwner(repositoryRoot, item.Test); err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
	}
	for _, category := range httpCaidoCategories {
		if !seenCategories[category] {
			return fmt.Errorf("required HTTP/Caido category %q is absent", category)
		}
	}
	for _, requirement := range []string{"A1", "A2", "A3"} {
		if !seenRequirements[requirement] {
			return fmt.Errorf("required HTTP/Caido requirement %q is absent", requirement)
		}
	}
	for _, fault := range httpCaidoFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required HTTP/Caido fault %q is absent", fault)
		}
	}

	requiredGates := []string{"architecture", "hardening", "matrix", "process", "release", "runtime"}
	seenGates := map[string]bool{}
	for _, gate := range matrix.Gates {
		if gate.ID == "" || gate.Command == "" || seenGates[gate.ID] ||
			!slices.Contains(requiredGates, gate.ID) || !strings.HasPrefix(gate.Command, "make ") {
			return fmt.Errorf("invalid or duplicate HTTP/Caido gate: %+v", gate)
		}
		seenGates[gate.ID] = true
	}
	for _, gate := range requiredGates {
		if !seenGates[gate] {
			return fmt.Errorf("required HTTP/Caido gate %q is absent", gate)
		}
	}
	return nil
}

func validateHTTPCaidoTestOwner(repositoryRoot string, ref httpCaidoTestRef) error {
	if ref.Source == "" || ref.Symbol == "" || filepath.IsAbs(ref.Source) ||
		strings.Contains(ref.Source, "..") {
		return fmt.Errorf("invalid test reference: %+v", ref)
	}
	extension := filepath.Ext(ref.Source)
	if !slices.Contains([]string{".go", ".py"}, extension) {
		return fmt.Errorf("unsupported test source: %+v", ref)
	}
	data, err := os.ReadFile(filepath.Join(repositoryRoot, filepath.FromSlash(ref.Source)))
	if err != nil {
		return fmt.Errorf("read referenced test %q: %w", ref.Source, err)
	}
	prefix := "func "
	if extension == ".py" {
		prefix = "def "
	}
	if !bytes.Contains(data, []byte(prefix+ref.Symbol+"(")) {
		return fmt.Errorf("%s does not define %s", ref.Source, ref.Symbol)
	}
	return nil
}
