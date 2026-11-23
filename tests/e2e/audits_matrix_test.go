package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type auditTestOwner struct {
	Source string `yaml:"source"`
	Kind   string `yaml:"kind"`
	Name   string `yaml:"name"`
	Gate   string `yaml:"gate"`
}
type auditsMatrix struct {
	SchemaVersion string            `yaml:"schema_version"`
	Policy        map[string]string `yaml:"policy"`
	Cases         []struct {
		ID         string         `yaml:"id"`
		Category   string         `yaml:"category"`
		Acceptance []string       `yaml:"acceptance"`
		Expected   string         `yaml:"expected"`
		Test       auditTestOwner `yaml:"test"`
	} `yaml:"cases"`
	Gates []matrixGate `yaml:"gates"`
}

func validateAuditsMatrix(root string, m auditsMatrix) error {
	if m.SchemaVersion != "2.0" {
		return fmt.Errorf("unsupported common matrix version")
	}
	for _, name := range []string{"orchestration", "settlement", "retention", "reporting", "isolation"} {
		if strings.TrimSpace(m.Policy[name]) == "" {
			return fmt.Errorf("missing policy %s", name)
		}
	}
	categories := map[string]bool{}
	acceptance := map[string]bool{}
	ids := map[string]bool{}
	owners := map[string]bool{}
	for _, c := range m.Cases {
		if c.ID == "" || ids[c.ID] || strings.TrimSpace(c.Expected) == "" || len(c.Acceptance) == 0 {
			return fmt.Errorf("invalid case %q", c.ID)
		}
		ids[c.ID] = true
		categories[c.Category] = true
		for _, a := range c.Acceptance {
			if !slices.Contains([]string{"A1", "A2", "A3", "A4", "A5", "A6"}, a) {
				return fmt.Errorf("unknown acceptance %s", a)
			}
			acceptance[a] = true
		}
		key := c.Test.Source + "#" + c.Test.Name
		if owners[key] {
			return fmt.Errorf("duplicate test owner")
		}
		owners[key] = true
		if err := validateMatrixNamedOwner(root, c.Test.Source, c.Test.Kind, c.Test.Name); err != nil {
			return err
		}
		if !slices.Contains([]string{"hardening", "process", "browser"}, c.Test.Gate) {
			return fmt.Errorf("invalid gate")
		}
	}
	for _, c := range []string{"configuration", "exact-inputs", "controls", "replay", "collection", "findings", "limits", "process", "prepared", "browser"} {
		if !categories[c] {
			return fmt.Errorf("missing category %s", c)
		}
	}
	for _, a := range []string{"A1", "A2", "A3", "A4", "A6"} {
		if !acceptance[a] {
			return fmt.Errorf("missing acceptance %s", a)
		}
	}
	return validateMatrixGates("common Audit", m.Gates, []string{"matrix", "hardening", "process", "browser", "release"}, true)
}
func TestAuditsHardeningMatrixIsComplete(t *testing.T) {
	data, err := os.ReadFile("audits_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	m, err := decodeStrictMatrix[auditsMatrix](data, "common Audit")
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join("..", "..")
	if err := validateAuditsMatrix(root, m); err != nil {
		t.Fatal(err)
	}
	t.Run("missing case cannot pass", func(t *testing.T) {
		bad := m
		bad.Cases = bad.Cases[1:]
		if validateAuditsMatrix(root, bad) == nil {
			t.Fatal("missing required category accepted")
		}
	})
	t.Run("missing owner cannot pass", func(t *testing.T) {
		bad := mustDecodeStrictMatrix[auditsMatrix](t, data, "common Audit")
		bad.Cases[0].Test.Name = "TestAbsent"
		if validateAuditsMatrix(root, bad) == nil {
			t.Fatal("absent test accepted")
		}
	})
}
