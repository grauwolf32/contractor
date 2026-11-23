package agentinstructions

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

type variantPair struct {
	Baseline          string `json:"baseline"`
	Candidate         string `json:"candidate"`
	BaselineSHA256    string `json:"baseline_sha256"`
	CandidateSHA256   string `json:"candidate_sha256"`
	BaselineSelector  string `json:"baseline_selector"`
	CandidateSelector string `json:"candidate_selector"`
}

// This offline archive gate compares pinned authoring data. It does not load
// obsolete contracts through the current Server or measure LLM quality.
func TestInstructionVariantsPreserveExecutionContracts(t *testing.T) {
	var manifest struct {
		Instructions []variantPair `json:"instructions"`
		Templates    []variantPair `json:"templates"`
		Workflows    []variantPair `json:"workflows"`
	}
	readJSON(t, "variants.json", &manifest)
	root := filepath.Join("..", "..", "..")
	var frozen struct {
		Files map[string]string `json:"files"`
	}
	readJSON(t, "catalog-baseline.json", &frozen)
	baselineBytes := func(path string) []byte {
		data, ok := frozen.Files[path]
		if !ok {
			t.Fatalf("frozen catalog lacks %s", path)
		}
		return []byte(data)
	}
	for _, pairs := range [][]variantPair{manifest.Instructions, manifest.Templates, manifest.Workflows} {
		for _, pair := range pairs {
			for path, want := range map[string]string{
				pair.Baseline: pair.BaselineSHA256, pair.Candidate: pair.CandidateSHA256,
			} {
				var data []byte
				if path == pair.Baseline {
					data = baselineBytes(path)
				} else {
					var err error
					data, err = os.ReadFile(filepath.Join(root, path))
					if err != nil {
						t.Fatal(err)
					}
				}
				if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != want {
					t.Fatalf("%s changed since the experiment was pinned", path)
				}
			}
		}
	}
	catalog := archivedCatalog(t, frozen.Files, "candidate/configs")
	for _, pair := range manifest.Templates {
		before := archivedSelection(t, catalog, "AgentTemplate", pair.BaselineSelector)
		after := archivedSelection(t, catalog, "AgentTemplate", pair.CandidateSelector)
		assertArchivedTemplateContract(t, before, after)
	}
	for _, pair := range manifest.Workflows {
		before := archivedSelection(t, catalog, "Workflow", pair.BaselineSelector)
		after := archivedSelection(t, catalog, "Workflow", pair.CandidateSelector)
		assertArchivedWorkflowContract(t, catalog, before, after)
	}
}

func readJSON(t *testing.T, path string, result any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, result); err != nil {
		t.Fatal(err)
	}
}
