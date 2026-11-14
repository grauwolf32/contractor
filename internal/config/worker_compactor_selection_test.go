package config

import (
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

func TestWorkerCompactorSelectionLayersAndPinnedReplay(t *testing.T) {
	root := copyCoreFixture(t, "context-compaction")
	path := filepath.Join(root, compactWorkflowPath)
	policyPath := filepath.Join(root, "model-policies/test-compactor.yaml")
	writeFile(t, filepath.Join(root, "model-policies/test-compactor-two.yaml"), []byte(strings.ReplaceAll(
		string(readFile(t, policyPath)), "test-compactor", "test-compactor-two")))
	base := mustLoad(t, root)
	workflow, err := base.Workflow("compact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	selection := workflow.Stages["copy"].ExecutionConfig.Agents["builder"]
	if selection.CompactorModelPolicy == nil || selection.CompactorModelPolicy.Ref.PolicyID != "test-compactor" || selection.Origins.CompactorModelPolicy != "workflow.executionConfig.workers" {
		t.Fatalf("default selection = %+v", selection)
	}
	replaceFile(t, path, "  entryStage: copy", `    stages:
      copy:
        agents:
          builder: {compactorModelPolicy: test-compactor-two@1}

  entryStage: copy`)
	replaceFile(t, path, "        failed:\n          fail: {}", `        failed:
          escalate:
            maxAttempts: 1
            executionConfig:
              agents:
                builder: {compactorModelPolicy: test-compactor@1}
            then: {fail: {}}`)
	snapshot := mustLoad(t, root)
	workflow, err = snapshot.Workflow("compact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["copy"]
	selected := stage.ExecutionConfig.Agents["builder"]
	escalated := stage.On.Failed.Escalate.ExecutionConfig.Effective.Agents["builder"]
	if selected.CompactorModelPolicy.Ref.PolicyID != "test-compactor-two" || selected.Origins.CompactorModelPolicy != "workflow.executionConfig.stages.copy.agents.builder" ||
		escalated.CompactorModelPolicy.Ref.PolicyID != "test-compactor" || escalated.Origins.CompactorModelPolicy != "workflow.stages.copy.on.failed.escalate.executionConfig.agents.builder" {
		t.Fatalf("specific=%+v escalated=%+v", selected, escalated)
	}
	if !reflect.DeepEqual(stage.Agents["builder"].ContextManagement, base.workflows["compact-copy@1"].Stages["copy"].Agents["builder"].ContextManagement) {
		t.Fatal("model selection changed compaction behavior")
	}
	// Captured Eval selections use the same owner as ordinary Run snapshots.
	lookup := developmentCredentialLookup(t, snapshot)
	if _, err := snapshot.ResolvePinnedRunWorkflow(t.Context(), workflow, lookup); err != nil {
		t.Fatal(err)
	}
	replaceFile(t, policyPath, "model: compactor-model", "model: changed-provider-model")
	if _, err := mustLoad(t, root).ResolvePinnedRunWorkflow(t.Context(), workflow, lookup); !errors.Is(err, contentdigest.ErrPinnedSelectionChanged) {
		t.Fatalf("drifted pinned Eval selection = %v", err)
	}
}

func TestWorkerCompactorRejectsRemovedPolicyAndMissingSelection(t *testing.T) {
	for name, change := range map[string]func(string) string{
		"missing selection": func(s string) string {
			return strings.Replace(s, "      compactorModelPolicy: test-compactor@1\n", "", 1)
		},
		"null selection": func(s string) string {
			return strings.Replace(s, "compactorModelPolicy: test-compactor@1", "compactorModelPolicy: null", 1)
		},
		"removed selector": func(s string) string {
			return strings.Replace(s, "            strategy: compact@1", "            strategy: compact@1\n            modelPolicy: test-compactor@1", 1)
		},
		"removed null": func(s string) string {
			return strings.Replace(s, "            strategy: compact@1", "            strategy: compact@1\n            modelPolicy: null", 1)
		},
		"removed merged selector": func(s string) string {
			return strings.Replace(s, "            strategy: compact@1", "            strategy: compact@1\n            <<: {modelPolicy: test-compactor@1}", 1)
		},
		"removed merged null": func(s string) string {
			return strings.Replace(s, "            strategy: compact@1", "            strategy: compact@1\n            <<: {modelPolicy: null}", 1)
		},
	} {
		t.Run(name, func(t *testing.T) {
			root := copyCoreFixture(t, "context-compaction")
			path := filepath.Join(root, compactWorkflowPath)
			writeFile(t, path, []byte(change(string(readFile(t, path)))))
			if _, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed")); err == nil {
				t.Fatal("invalid compactor authoring accepted")
			}
		})
	}
}

func TestWorkerCompactorMixedRouterRequiresSpecificSelection(t *testing.T) {
	root := copyCoreFixture(t, "context-compaction")
	path := filepath.Join(root, compactWorkflowPath)
	replaceFile(t, path, "      planner: passthrough@1", "      planner: router@1")
	replaceFile(t, path, "      agents:\n        builder:", "      agents:\n        observer: {template: artifact_builder@1}\n        builder:")
	replaceFile(t, path, "  executionConfig:\n", "  executionConfig:\n    planner: {modelPolicy: test-planner@1, llmGateway: local-litellm@1}\n")
	if _, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed")); err == nil || !strings.Contains(err.Error(), "compactorModelPolicy requires compact@1") {
		t.Fatalf("inapplicable broad selection = %v", err)
	}
	replaceFile(t, path, "      compactorModelPolicy: test-compactor@1\n", "")
	replaceFile(t, path, "  entryStage: copy", `    stages:
      copy:
        agents:
          builder: {compactorModelPolicy: test-compactor@1}

  entryStage: copy`)
	workflow, err := mustLoad(t, root).Workflow("compact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["copy"]
	if stage.ExecutionConfig.Agents["observer"].CompactorModelPolicy != nil || stage.ExecutionConfig.Agents["builder"].CompactorModelPolicy == nil || stage.ExecutionConfig.Planner.CompactorModelPolicy != nil {
		t.Fatal("specific Worker selection leaked to another consumer")
	}
}

func TestWorkerCompactorSnapshotSurvivesPostgresAndCatalogDrift(t *testing.T) {
	pool := modelAuditPool(t)
	root := copyCoreFixture(t, "context-compaction")
	snapshot := mustLoad(t, root)
	workflow, err := snapshot.Workflow("compact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	params := runstore.CreateRunIdempotentParams{
		CreateRunParams: runstore.CreateRunParams{
			RunID: "worker-compactor-pinned", OwnerID: "worker-compactor-owner",
			WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
			WorkflowSchemaVersion: "contractor/v1alpha1", WorkflowSnapshot: encoded,
			LaneClass: resolved.LaneClassModel, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
		},
		IdempotencyKey: "worker-compactor-key", RequestDigest: contentdigest.Bytes(encoded),
	}
	if _, created, err := store.CreateRunIdempotent(t.Context(), params); err != nil || !created {
		t.Fatalf("create pinned Run: created=%t err=%v", created, err)
	}
	replaceFile(t, filepath.Join(root, "model-policies/test-compactor.yaml"), "model: compactor-model", "model: changed-provider-model")
	current, err := mustLoad(t, root).Workflow("compact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	want := workflow.Stages["copy"].ExecutionConfig.Agents["builder"].CompactorModelPolicy
	if reflect.DeepEqual(want, current.Stages["copy"].ExecutionConfig.Agents["builder"].CompactorModelPolicy) {
		t.Fatal("fixture did not change the catalog policy")
	}
	for _, replay := range []bool{false, true} {
		stored, err := store.GetRun(t.Context(), params.RunID)
		if replay {
			var created bool
			stored, created, err = store.CreateRunIdempotent(t.Context(), params)
			if created {
				t.Fatal("replay created another Run")
			}
		}
		if err != nil {
			t.Fatal(err)
		}
		retained, err := resolved.DecodeResolvedWorkflowSnapshot(stored.WorkflowSnapshot)
		if err != nil || !reflect.DeepEqual(want, retained.Stages["copy"].ExecutionConfig.Agents["builder"].CompactorModelPolicy) {
			t.Fatalf("stored compactor selection changed: %v", err)
		}
	}
}
