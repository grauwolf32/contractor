package agentinstructions

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
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
	directory := t.TempDir()
	for path, raw := range frozen.Files {
		rel, err := filepath.Rel("configs", path)
		if err != nil {
			t.Fatal(err)
		}
		dest := filepath.Join(directory, rel)
		if err := os.MkdirAll(filepath.Dir(dest), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dest, []byte(raw), 0600); err != nil {
			t.Fatal(err)
		}
	}
	for _, overlay := range []string{"candidate/configs", "evaluation-configs/configs"} {
		if err := os.CopyFS(directory, os.DirFS(overlay)); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := config.Load(directory, archivedExperimentDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Cases) != 6 {
		t.Fatalf("expected six pilot cases, got %d", len(manifest.Cases))
	}
	for _, id := range []string{"d1", "o1", "l1", "t1", "t2", "a1"} {
		t.Run(id, func(t *testing.T) {
			arms := manifest.Cases[id]
			var pair []config.ResolvedWorkflow
			for _, arm := range []string{"baseline", "candidate"} {
				entry := arms[arm]
				workflow, err := snapshot.Workflow(entry.Workflow)
				if err != nil {
					t.Fatal(err)
				}
				source, err := snapshot.Workflow(entry.Source)
				if err != nil {
					t.Fatal(err)
				}
				if id != "d1" {
					copy := workflow
					copy.Ref = source.Ref
					if !reflect.DeepEqual(source, copy) {
						t.Fatal("wrapper changed source workflow")
					}
				} else {
					if len(workflow.Stages) != 1 || workflow.EntryStage != "dependency_discovery" {
						t.Fatal("D1 must execute only dependency discovery")
					}
					stage := workflow.Stages["dependency_discovery"]
					prior := source.Stages["dependency_discovery"]
					stage.On = prior.On
					stage.WorkflowOutputs = prior.WorkflowOutputs
					if !reflect.DeepEqual(prior, stage) {
						t.Fatal("D1 changed discovery execution contract")
					}
				}
				pair = append(pair, workflow)
				if id == "a1" {
					profile, err := snapshot.AuditProfile(entry.Profile)
					if err != nil {
						t.Fatal(err)
					}
					original, err := snapshot.AuditProfile("source-checklist@1")
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(profile.Workflows["check"].Workflow, workflow) {
						t.Fatal("profile must pin its arm's exact child workflow")
					}
					profile.Ref = original.Ref
					child := profile.Workflows["check"]
					old := original.Workflows["check"]
					child.Workflow = old.Workflow
					profile.Workflows["check"] = child
					if !reflect.DeepEqual(original, profile) {
						t.Fatal("audit policy, inventory or evidence mapping changed")
					}
				}
			}
			before, after := pair[0], pair[1]
			after.Ref = before.Ref
			for key, stage := range after.Stages {
				prior := before.Stages[key]
				stage.Instructions = prior.Instructions
				for role, agent := range stage.Agents {
					old := prior.Agents[role]
					assertTemplateContract(t, old.Template, agent.Template)
					agent.Template = old.Template
					stage.Agents[role] = agent
				}
				after.Stages[key] = stage
			}
			if !reflect.DeepEqual(before, after) {
				t.Fatal("A/B differs beyond instructions and dependent refs")
			}
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
