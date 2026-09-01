package e2e

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"go.yaml.in/yaml/v4"
)

type agentSkillsMatrix struct {
	SchemaVersion string                  `yaml:"schema_version"`
	Policy        agentSkillsMatrixPolicy `yaml:"policy"`
	Cases         []agentSkillsMatrixCase `yaml:"cases"`
	Gates         []agentSkillsMatrixGate `yaml:"gates"`
}

type agentSkillsMatrixPolicy struct {
	WorkerContent string `yaml:"worker_content"`
	Authority     string `yaml:"authority"`
	Retention     string `yaml:"retention"`
}

type agentSkillsMatrixCase struct {
	ID       string                 `yaml:"id"`
	Covers   []string               `yaml:"covers"`
	Faults   []string               `yaml:"faults"`
	Expected string                 `yaml:"expected"`
	Test     sharedMemoryMatrixTest `yaml:"test"`
}

type agentSkillsMatrixGate struct {
	ID      string `yaml:"id"`
	Command string `yaml:"command"`
}

func TestAgentSkillsHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("agent_skills_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeAgentSkillsMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateAgentSkillsMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("missing acceptance coverage", func(t *testing.T) {
		broken, err := decodeAgentSkillsMatrix(data)
		if err != nil {
			t.Fatal(err)
		}
		for index := range broken.Cases {
			broken.Cases[index].Covers = slices.DeleteFunc(
				broken.Cases[index].Covers, func(value string) bool { return value == "A1" },
			)
			if len(broken.Cases[index].Covers) == 0 {
				broken.Cases[index].Covers = []string{"A2"}
			}
		}
		if err := validateAgentSkillsMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "A1") {
			t.Fatalf("missing acceptance validation error = %v", err)
		}
	})
	t.Run("renamed test owner", func(t *testing.T) {
		broken, err := decodeAgentSkillsMatrix(data)
		if err != nil {
			t.Fatal(err)
		}
		broken.Cases[0].Test.Symbol += "Renamed"
		if err := validateAgentSkillsMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "does not define") {
			t.Fatalf("renamed owner validation error = %v", err)
		}
	})
}

func TestAgentSkillsUseOnlyArtifactPlaneInventories(t *testing.T) {
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
		lower := bytes.ToLower(data)
		for _, route := range [][]byte{[]byte("/v1/skills"), []byte("/private/v1/skills")} {
			if bytes.Contains(lower, route) {
				t.Fatalf("Skill-specific HTTP route %q appears in %s", route, relative)
			}
		}
	}

	createTable := regexp.MustCompile(`(?i)create\s+table(?:\s+if\s+not\s+exists)?\s+([a-z_][a-z0-9_]*)`)
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
			lower := bytes.ToLower(data)
			if bytes.Contains(lower, []byte("systemscope")) ||
				bytes.Contains(lower, []byte("system_scope")) {
				return fmt.Errorf("SystemScope appears in %s", path)
			}
			for _, match := range createTable.FindAllSubmatch(data, -1) {
				if bytes.Contains(bytes.ToLower(match[1]), []byte("skill")) {
					return fmt.Errorf("Skill-specific table %q appears in %s", match[1], path)
				}
			}
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
}

func decodeAgentSkillsMatrix(data []byte) (agentSkillsMatrix, error) {
	var matrix agentSkillsMatrix
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&matrix); err != nil {
		return matrix, fmt.Errorf("decode strict Agent Skills matrix: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err == nil {
		return matrix, errors.New("Agent Skills matrix contains a trailing YAML document")
	} else if !errors.Is(err, io.EOF) {
		return matrix, fmt.Errorf("decode Agent Skills matrix trailer: %w", err)
	}
	return matrix, nil
}

func validateAgentSkillsMatrix(repositoryRoot string, matrix agentSkillsMatrix) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.WorkerContent) == "" ||
		strings.TrimSpace(matrix.Policy.Authority) == "" ||
		strings.TrimSpace(matrix.Policy.Retention) == "" {
		return fmt.Errorf("incomplete Agent Skills matrix policy: %+v", matrix.Policy)
	}
	wantAcceptance := []string{"A1", "A2", "A3", "A4", "A5"}
	wantFaults := []string{
		"F01", "F02", "F03", "F04", "F05", "F06",
		"F07", "F08", "F09", "F10", "F11", "F12",
	}
	seenCases := map[string]bool{}
	seenAcceptance := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Expected == "" || seenCases[item.ID] || len(item.Covers) == 0 {
			return fmt.Errorf("invalid or duplicate Agent Skills case: %+v", item)
		}
		seenCases[item.ID] = true
		for _, acceptance := range item.Covers {
			if !slices.Contains(wantAcceptance, acceptance) {
				return fmt.Errorf("case %q has unknown acceptance criterion %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(wantFaults, fault) || seenFaults[fault] {
				return fmt.Errorf("case %q has unknown or duplicate fault %q", item.ID, fault)
			}
			seenFaults[fault] = true
		}
		owner := item.Test.Source + "#" + item.Test.Symbol
		if previous, exists := seenOwners[owner]; exists {
			return fmt.Errorf("duplicate test owner %q in cases %q and %q", owner, previous, item.ID)
		}
		seenOwners[owner] = item.ID
		if err := validateSharedMemoryTestOwner(repositoryRoot, item.Test); err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
	}
	for _, acceptance := range wantAcceptance {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required acceptance criterion %s has no owning case", acceptance)
		}
	}
	for _, fault := range wantFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required fault %s has no owning case", fault)
		}
	}

	wantGates := []string{"corpus", "hardening", "matrix", "packages", "process", "races", "release", "runtime"}
	seenGates := map[string]bool{}
	for _, gate := range matrix.Gates {
		if gate.ID == "" || gate.Command == "" || seenGates[gate.ID] ||
			!strings.HasPrefix(gate.Command, "make ") {
			return fmt.Errorf("invalid or duplicate Agent Skills gate: %+v", gate)
		}
		seenGates[gate.ID] = true
	}
	for _, gate := range wantGates {
		if !seenGates[gate] {
			return fmt.Errorf("required Agent Skills gate %q is absent", gate)
		}
	}
	return nil
}
