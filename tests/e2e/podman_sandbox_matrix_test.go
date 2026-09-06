package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

type podmanMatrix struct {
	SchemaVersion string       `yaml:"schema_version"`
	Cases         []podmanCase `yaml:"cases"`
	Gates         []matrixGate `yaml:"gates"`
}

type podmanCase struct {
	ID       string                 `yaml:"id"`
	Spec     int                    `yaml:"spec"`
	Expected string                 `yaml:"expected"`
	Tests    []projectWorkspaceTest `yaml:"tests"`
	Real     []projectWorkspaceTest `yaml:"real"`
}

func TestPodmanSandboxMatrixCoversEveryAcceptanceCase(t *testing.T) {
	data, err := os.ReadFile("podman_sandbox_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix := mustDecodeStrictMatrix[podmanMatrix](t, data, "Podman")
	if matrix.SchemaVersion != "1.0" {
		t.Fatal("unsupported Podman matrix version")
	}
	seen, ids := map[int]bool{}, map[string]bool{}
	for _, item := range matrix.Cases {
		if item.Spec < 1 || item.Spec > 11 || seen[item.Spec] || item.ID == "" || ids[item.ID] || item.Expected == "" || len(item.Tests) == 0 || len(item.Real) == 0 {
			t.Fatalf("invalid or duplicate acceptance case: %+v", item)
		}
		seen[item.Spec], ids[item.ID] = true, true
		for _, reference := range append(item.Tests, item.Real...) {
			if err := validateMatrixSymbolOwner(filepath.Join("..", ".."), reference.Source, reference.Symbol); err != nil {
				t.Fatal(err)
			}
		}
	}
	for number := 1; number <= 11; number++ {
		if !seen[number] {
			t.Error(fmt.Sprintf("Podman acceptance case %d is unmapped", number))
		}
	}
	if err := validateMatrixGates("Podman", matrix.Gates, []string{"matrix", "unit", "local_direct", "supervisor", "workflow", "e2e", "release", "aggregate"}, true); err != nil {
		t.Fatal(err)
	}
}
