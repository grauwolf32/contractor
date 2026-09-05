package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type httpCaidoMatrix struct {
	SchemaVersion string                `yaml:"schema_version"`
	Policy        httpCaidoMatrixPolicy `yaml:"policy"`
	Cases         []httpCaidoMatrixCase `yaml:"cases"`
	Gates         []matrixGate          `yaml:"gates"`
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
	return decodeStrictMatrix[httpCaidoMatrix](data, "HTTP/Caido")
}

func mustDecodeHTTPCaidoMatrix(t *testing.T, data []byte) httpCaidoMatrix {
	t.Helper()
	return mustDecodeStrictMatrix[httpCaidoMatrix](t, data, "HTTP/Caido")
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
		if err := validateMatrixSymbolOwner(repositoryRoot, item.Test.Source, item.Test.Symbol); err != nil {
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

	return validateMatrixGates(
		"HTTP/Caido", matrix.Gates,
		[]string{"architecture", "hardening", "matrix", "process", "release", "runtime"}, true,
	)
}
