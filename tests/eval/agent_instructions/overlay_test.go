package agentinstructions

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type variantPair struct {
	Baseline          string `json:"baseline"`
	Candidate         string `json:"candidate"`
	BaselineSHA256    string `json:"baseline_sha256"`
	CandidateSHA256   string `json:"candidate_sha256"`
	BaselineSelector  string `json:"baseline_selector"`
	CandidateSelector string `json:"candidate_selector"`
}

// This is a compatibility gate, not an LLM quality evaluation. It prevents an
// instruction experiment from silently changing tools, budgets or output contracts.
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
	for _, variant := range []string{"baseline", "candidate"} {
		t.Run(variant, func(t *testing.T) {
			directory := t.TempDir()
			for path, data := range frozen.Files {
				if strings.HasPrefix(path, "configs/audit-profiles/") {
					continue
				}
				relative, err := filepath.Rel("configs", path)
				if err != nil {
					t.Fatal(err)
				}
				destination := filepath.Join(directory, relative)
				if err := os.MkdirAll(filepath.Dir(destination), 0o700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(destination, []byte(data), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			copyArchivedExecutionConfigs(t, directory, "candidate/configs")
			if variant == "baseline" {
				// Keep the same test catalog identities in A and B; vary only text.
				for _, pair := range manifest.Instructions {
					data := baselineBytes(pair.Baseline)
					if err := os.WriteFile(filepath.Join(directory, "instructions", filepath.Base(pair.Candidate)), data, 0o600); err != nil {
						t.Fatal(err)
					}
				}
			}
			registerArchivedToolsets(t, directory)
			snapshot, err := configload.Load(directory)
			if err != nil {
				t.Fatal(err)
			}
			for _, pair := range manifest.Templates {
				before, err := snapshot.AgentTemplate(pair.BaselineSelector)
				if err != nil {
					t.Fatal(err)
				}
				after, err := snapshot.AgentTemplate(pair.CandidateSelector)
				if err != nil {
					t.Fatal(err)
				}
				assertTemplateContract(t, before, after)
			}
			for _, pair := range manifest.Workflows {
				before, err := snapshot.Workflow(pair.BaselineSelector)
				if err != nil {
					t.Fatal(err)
				}
				after, err := snapshot.Workflow(pair.CandidateSelector)
				if err != nil {
					t.Fatal(err)
				}
				after.Ref.Version = before.Ref.Version
				for name, stage := range after.Stages {
					old, ok := before.Stages[name]
					if !ok {
						t.Fatalf("unexpected stage %s in %s", name, pair.CandidateSelector)
					}
					stage.Instructions = old.Instructions
					for role, binding := range stage.Agents {
						prior, ok := old.Agents[role]
						if !ok {
							t.Fatalf("unexpected agent role %s", role)
						}
						assertTemplateContract(t, prior.Template, binding.Template)
						binding.Template = prior.Template
						stage.Agents[role] = binding
					}
					after.Stages[name] = stage
				}
				if !reflect.DeepEqual(before, after) {
					t.Errorf("%s changes workflow behavior beyond instructions and versioned templates", pair.CandidateSelector)
				}
			}
		})
	}
}

func assertTemplateContract(t *testing.T, before, after contracts.ResolvedAgentTemplate) {
	t.Helper()
	after.Ref.Version = before.Ref.Version
	after.Ref.Digest = before.Ref.Digest
	after.Instructions = before.Instructions
	if !reflect.DeepEqual(before, after) {
		t.Errorf("%s changes execution capabilities or policy beyond instructions", after.Ref.TemplateID)
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

// archivedToolsets maps each Toolset version the frozen V40 experiment selects
// but the Server no longer registers to the registered version that exports
// the same tools with the same finding classification.
var archivedToolsets = map[string]string{
	"audit-results@1":     "audit-results@2",
	"security-findings@2": "security-findings@1",
}

// registerArchivedToolsets moves the AgentTemplates of a materialized
// experiment catalog onto registered Toolset versions, since the loader checks
// every selection against the Server's one descriptor table. It rewrites only
// the temporary copy, and both arms alike, so the A/B contract comparison is
// unchanged; nothing here builds or registers a tool.
func registerArchivedToolsets(t *testing.T, directory string) {
	t.Helper()
	err := filepath.WalkDir(filepath.Join(directory, "agent-templates"), func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		text := string(data)
		for archived, registered := range archivedToolsets {
			text = strings.ReplaceAll(text, "- ref: "+archived+"\n", "- ref: "+registered+"\n")
		}
		return os.WriteFile(path, []byte(text), 0o600)
	})
	if err != nil {
		t.Fatal(err)
	}
}
