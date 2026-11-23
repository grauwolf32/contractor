package agentinstructions

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestEvaluationReleasePreservesPairedContracts(t *testing.T) {
	var frozen struct {
		Files map[string]string `json:"files"`
	}
	readJSON(t, "catalog-baseline.json", &frozen)
	var manifest struct {
		Baseline string            `json:"baseline_catalog_sha256"`
		Variants string            `json:"variants_sha256"`
		Files    map[string]string `json:"files"`
		Cases    map[string]map[string]struct {
			Workflow string `json:"workflow"`
			Source   string `json:"source_workflow"`
			Profile  string `json:"audit_profile"`
		} `json:"cases"`
	}
	readJSON(t, "evaluation-configs/manifest.json", &manifest)
	for path, want := range map[string]string{"catalog-baseline.json": manifest.Baseline, "variants.json": manifest.Variants} {
		checkEvaluationDigest(t, path, want)
	}
	for path, want := range manifest.Files {
		checkEvaluationDigest(t, filepath.Join("evaluation-configs", path), want)
	}
	catalog := archivedCatalog(t, frozen.Files, "candidate/configs", "evaluation-configs/configs")
	if len(manifest.Cases) != 6 {
		t.Fatalf("expected six pilot cases, got %d", len(manifest.Cases))
	}
	for _, id := range []string{"d1", "o1", "l1", "t1", "t2", "a1"} {
		t.Run(id, func(t *testing.T) {
			arms := manifest.Cases[id]
			var pair []map[string]any
			for _, arm := range []string{"baseline", "candidate"} {
				entry := arms[arm]
				workflow := archivedSelection(t, catalog, "Workflow", entry.Workflow)
				source := archivedSelection(t, catalog, "Workflow", entry.Source)
				if id != "d1" {
					copy := archiveDocument(t, catalog["Workflow/"+entry.Workflow])
					copy["metadata"] = source["metadata"]
					if !reflect.DeepEqual(source, copy) {
						t.Fatal("wrapper changed source workflow")
					}
				} else {
					spec := workflow["spec"].(map[string]any)
					stages := spec["stages"].(map[string]any)
					if len(stages) != 1 || spec["entryStage"] != "dependency_discovery" {
						t.Fatal("D1 must execute only dependency discovery")
					}
					stage := stages["dependency_discovery"].(map[string]any)
					prior := source["spec"].(map[string]any)["stages"].(map[string]any)["dependency_discovery"].(map[string]any)
					stage["on"] = prior["on"]
					if value, ok := prior["workflowOutputs"]; ok {
						stage["workflowOutputs"] = value
					} else {
						delete(stage, "workflowOutputs")
					}
					if !reflect.DeepEqual(prior, stage) {
						t.Fatal("D1 changed discovery execution contract")
					}
				}
				pair = append(pair, workflow)
				if id == "a1" {
					assertArchivedProfileWrapper(t, frozen.Files["configs/audit-profiles/source-checklist.yaml"],
						filepath.Join("evaluation-configs/configs/audit-profiles", "eval-v40-a1-"+arm+".yaml"),
						entry.Profile, entry.Workflow)
				}
			}
			assertArchivedWorkflowContract(t, catalog, pair[0], pair[1])
		})
	}
}

func checkEvaluationDigest(t *testing.T, path, want string) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(raw)) != want {
		t.Fatalf("release pin changed: %s", path)
	}
}
