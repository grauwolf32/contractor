package projectfacts_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestFrozenMemoryExtractionRecipeUsesNativeContracts(t *testing.T) {
	var lock struct {
		Files map[string]string `json:"files"`
	}
	data, err := os.ReadFile("corpus-v5.lock.json")
	if err != nil {
		t.Fatal(err)
	}
	if json.Unmarshal(data, &lock) != nil || len(lock.Files) != 8 {
		t.Fatal("incomplete frozen corpus")
	}
	for name, digest := range lock.Files {
		payload, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(payload)
		if digest != "sha256:"+hex.EncodeToString(sum[:]) {
			t.Fatalf("corpus drift: %s", name)
		}
	}
	dir := t.TempDir()
	revisions := filepath.Join(dir, "revisions.json")
	if err := os.WriteFile(revisions, []byte(`{"technology":"r1"}`), 0600); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"dataset", "experiment"} {
		output := filepath.Join(dir, kind+".json")
		args := []string{"prepare_recipe.py", kind, "--case", "technology", "--output", output}
		schema := "CreateExperiment"
		if kind == "dataset" {
			args = append(args, "--project", "fixture-project", "--revisions", revisions)
			schema = "DatasetInput"
		} else {
			args = append(args, "--dataset-revision", "r1", "--repetitions", "1")
		}
		if data, err := exec.Command("python3", args...).CombinedOutput(); err != nil {
			t.Fatalf("prepare %s: %s: %v", kind, data, err)
		}
		payload, err := os.ReadFile(output)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := evaldomain.Freeze(schema, payload); err != nil {
			t.Fatalf("%s contract: %v", schema, err)
		}
		if kind == "experiment" {
			var experiment struct {
				Draft struct {
					Variants []struct {
						Selector string `json:"selector"`
					} `json:"variants"`
				} `json:"draft"`
			}
			if err := json.Unmarshal(payload, &experiment); err != nil {
				t.Fatal(err)
			}
			if len(experiment.Draft.Variants) != 2 {
				t.Fatal("expected two experiment variants")
			}
			for _, variant := range experiment.Draft.Variants {
				if variant.Selector != "project-facts-eval@1" {
					t.Fatalf("recipe selects retired catalog: %s", variant.Selector)
				}
			}
		}
	}
}

func TestOptionalMemoryEvalConfigurationUsesOrdinaryAuditExecution(t *testing.T) {
	repository := filepath.Join("..", "..", "..")
	root := filepath.Join(t.TempDir(), "configs")
	if err := os.CopyFS(root, os.DirFS(filepath.Join(repository, "configs"))); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(root, "catalog.lock.json")); err != nil {
		t.Fatal(err)
	}
	optional := filepath.Join(repository, "configs", "evals", "project-facts")
	for _, directory := range []string{"agent-templates", "workflows", "audit-profiles", "instructions"} {
		entries, err := os.ReadDir(filepath.Join(optional, directory))
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			data, err := os.ReadFile(filepath.Join(optional, directory, entry.Name()))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, directory, entry.Name()), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := catalog.AuditProfile("project-facts-eval@1")
	if err != nil {
		t.Fatal(err)
	}
	baselineWorkflow, err := catalog.Workflow("fact-extraction-audit-eval@1")
	if err != nil {
		t.Fatal(err)
	}
	for _, version := range []string{"1", "2"} {
		profile, err := catalog.AuditProfile("project-facts-eval@" + version)
		if err != nil {
			t.Fatal(err)
		}
		if len(profile.Program.InitialWork) != 1 {
			t.Fatal("Eval must use one ordinary Audit task")
		}
		if !reflect.DeepEqual(profile.Program.Limits, baseline.Program.Limits) || !reflect.DeepEqual(profile.Program.Policy, baseline.Program.Policy) {
			t.Fatal("candidate changed extraction budgets or authority")
		}
		workflow, err := catalog.Workflow("fact-extraction-audit-eval@" + version)
		if err != nil {
			t.Fatal(err)
		}
		stage := workflow.Stages["extract"]
		if !reflect.DeepEqual(stage.ExecutionConfig, baselineWorkflow.Stages["extract"].ExecutionConfig) || !reflect.DeepEqual(stage.Agents["extractor"].Template.Toolsets, baselineWorkflow.Stages["extract"].Agents["extractor"].Template.Toolsets) {
			t.Fatal("candidate changed configured model or tools")
		}
		if stage.Planner.PlannerID != "passthrough" || stage.ExecutionConfig.Planner != nil || len(stage.Agents) != 1 || stage.ExecutionConfig.Agents["extractor"].ModelPolicy.Ref.PolicyID != "worker" {
			t.Fatal("Eval changed model/execution owners")
		}
	}
}

func TestFactsCandidateRecipePreservesBaselineAndPinsNewSelection(t *testing.T) {
	output := filepath.Join(t.TempDir(), "experiment.json")
	data, err := exec.Command("python3", "prepare_recipe.py", "experiment", "--candidate", "--dataset-revision", "r1", "--output", output).CombinedOutput()
	if err != nil {
		t.Fatalf("candidate recipe: %s: %v", data, err)
	}
	payload, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := evaldomain.Freeze("CreateExperiment", payload); err != nil {
		t.Fatal(err)
	}
	var experiment struct {
		Draft evaldomain.Draft `json:"draft"`
	}
	if err := json.Unmarshal(payload, &experiment); err != nil {
		t.Fatal(err)
	}
	d := experiment.Draft
	if len(d.Variants) != 2 || d.Variants[0].Selector != "project-facts-eval@1" || d.Variants[1].Selector != "project-facts-eval@2" || d.Repetitions != 3 || d.Budgets.MaxMembers != 36 {
		t.Fatal("candidate selection or repeated matrix drift")
	}
	if len(d.Comparison.AllowedDifferences) != 2 || d.Comparison.AllowedDifferences[0] != "instructions" || d.Comparison.AllowedDifferences[1] != "audit-execution" {
		t.Fatal("unexpected candidate differences")
	}
	// Validate the shipped production selector as well as the opt-in Audit profiles.
	repository := filepath.Join("..", "..", "..")
	catalog, _, err := configload.LoadUnionReadOnly(filepath.Join(repository, "configs"), filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, version := range []string{"1", "2"} {
		workflow, err := catalog.Workflow("project-facts-extract@" + version)
		if err != nil {
			t.Fatal(err)
		}
		stage := workflow.Stages["extract"]
		if stage.ExecutionConfig.Agents["extractor"].ModelPolicy.Ref.PolicyID != "worker" || len(stage.Agents) != 1 || stage.Planner.PlannerID != "passthrough" {
			t.Fatal("production extraction execution drift")
		}
	}
}
