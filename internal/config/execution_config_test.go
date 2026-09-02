package config

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestResolveRunWorkflowAppliesAllWorkerLayersAndPinsBodies(t *testing.T) {
	root := copyConfigTree(t)
	writeFile(t, filepath.Join(root, "llm-gateways/secondary.yaml"), []byte(secondaryGatewayYAML))
	workflowPath := filepath.Join(root, "workflows/artifact_copy.yaml")
	replaceFile(t, workflowPath, `  executionConfig:
    workers:
      llmGateway: local-litellm@1
      credential: development-worker
`, `  executionConfig:
    workers:
      modelPolicy: domain_worker@1
      llmGateway: local-litellm@1
      credential: workflow-worker
    stages:
      copy:
        agents:
          builder:
            modelPolicy: worker@1
`)

	snapshot := mustLoad(t, root, MVPDescriptors())
	localGateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	credentials := metadataLookup{
		"workflow-worker": credentialMetadata("workflow-worker", localGateway.Ref),
		"run-worker":      credentialMetadata("run-worker", localGateway.Ref),
	}

	base, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", ExecutionConfigPatch{}, credentials)
	if err != nil {
		t.Fatal(err)
	}
	baseSelection := base.Stages["copy"].ExecutionConfig.Agents["builder"]
	if baseSelection.ModelPolicy.Ref.PolicyID != "worker" ||
		baseSelection.Origins.ModelPolicy != "workflow.executionConfig.stages.copy.agents.builder" ||
		baseSelection.LLMGateway.Ref != localGateway.Ref ||
		baseSelection.Origins.LLMGateway != "workflow.executionConfig.workers" ||
		baseSelection.Credential == nil || baseSelection.Credential.CredentialID != "workflow-worker" ||
		baseSelection.Origins.Credential != "workflow.executionConfig.workers" {
		t.Fatalf("resolved Workflow defaults = %+v", baseSelection)
	}

	globalPatch := decodeExecutionConfigPatch(t, `{
  "workers": {"modelPolicy": "domain_worker@1", "credential": "run-worker"}
}`)
	global, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", globalPatch, credentials)
	if err != nil {
		t.Fatal(err)
	}
	globalSelection := global.Stages["copy"].ExecutionConfig.Agents["builder"]
	if globalSelection.ModelPolicy.Ref.PolicyID != "domain_worker" ||
		globalSelection.Origins.ModelPolicy != "run.executionConfig.workers" ||
		globalSelection.Credential == nil || globalSelection.Credential.CredentialID != "run-worker" ||
		globalSelection.Origins.Credential != "run.executionConfig.workers" {
		t.Fatalf("resolved Run-wide override = %+v", globalSelection)
	}

	bindingPatch := decodeExecutionConfigPatch(t, `{
  "workers": {"modelPolicy": "domain_worker@1", "credential": "run-worker"},
  "stages": {"copy": {"agents": {"builder": {
    "modelPolicy": "worker@1",
    "llmGateway": "secondary@1",
    "credential": null
  }}}}
}`)
	resolved, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", bindingPatch, credentials)
	if err != nil {
		t.Fatal(err)
	}
	selection := resolved.Stages["copy"].ExecutionConfig.Agents["builder"]
	if selection.ModelPolicy.Ref.PolicyID != "worker" || selection.LLMGateway.Ref.GatewayID != "secondary" ||
		selection.Credential != nil ||
		selection.Origins.ModelPolicy != "run.executionConfig.stages.copy.agents.builder" ||
		selection.Origins.LLMGateway != "run.executionConfig.stages.copy.agents.builder" ||
		selection.Origins.Credential != "run.executionConfig.stages.copy.agents.builder" {
		t.Fatalf("resolved binding override = %+v", selection)
	}

	pinned, err := json.Marshal(resolved)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(pinned), "token") {
		t.Fatalf("Run snapshot contains secret-shaped data: %s", pinned)
	}
	replaceFile(
		t, filepath.Join(root, "model-policies/worker.yaml"),
		"model: worker-model", "model: worker-model-v2",
	)
	reloaded := mustLoad(t, root, MVPDescriptors())
	newResolution, err := reloaded.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", bindingPatch, credentials,
	)
	if err != nil {
		t.Fatal(err)
	}
	var stored ResolvedWorkflow
	if err := json.Unmarshal(pinned, &stored); err != nil {
		t.Fatal(err)
	}
	oldModel := stored.Stages["copy"].ExecutionConfig.Agents["builder"].ModelPolicy.Model
	newModel := newResolution.Stages["copy"].ExecutionConfig.Agents["builder"].ModelPolicy.Model
	if oldModel != "worker-model" || newModel != "worker-model-v2" {
		t.Fatalf("pinned/new models = %q/%q", oldModel, newModel)
	}
}

func TestResolveRunWorkflowAllowsOnlyWorkerPhysicalGatewayToRemainAbsent(t *testing.T) {
	root := copyConfigTree(t)
	workflowPath := filepath.Join(root, "workflows/artifact_copy.yaml")
	replaceFile(t, workflowPath, `  executionConfig:
    workers:
      llmGateway: local-litellm@1
      credential: development-worker

`, "")
	snapshot, err := Load(root, MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", ExecutionConfigPatch{}, metadataLookup{},
	)
	if err != nil {
		t.Fatalf("resolve label-routed Worker: %v", err)
	}
	selection := workflow.Stages[workflow.EntryStage].ExecutionConfig.Agents["builder"]
	if selection.LLMGateway != nil || selection.ModelPolicy.Ref.PolicyID != "worker" ||
		selection.Origins.ModelPolicy != originAgentTemplate || selection.Origins.LLMGateway != "" {
		t.Fatalf("partially routed Worker selection = %+v", selection)
	}

	planner := selection
	planner.ModelPolicy, err = snapshot.ModelPolicy("planner@1")
	if err != nil {
		t.Fatal(err)
	}
	if err := validateConsumerExecutionConfig(planner, true, false); err == nil ||
		!strings.Contains(err.Error(), "Planner llmGateway is required") {
		t.Fatalf("incomplete Planner validation error = %v", err)
	}
}

func TestResolveRunWorkflowKeepsPlannerAndWorkerSelectionsIndependent(t *testing.T) {
	root := copyConfigTree(t)
	writeFile(t, filepath.Join(root, "llm-gateways/secondary.yaml"), []byte(secondaryGatewayYAML))
	workflowPath := filepath.Join(root, "workflows/artifact_copy.yaml")
	replaceFile(t, workflowPath, "      planner: passthrough@1", "      planner: streamline@1")
	replaceFile(t, workflowPath, `  executionConfig:
    workers:
      llmGateway: local-litellm@1
      credential: development-worker
`, `  executionConfig:
    planner:
      modelPolicy: planner@1
      llmGateway: local-litellm@1
      credential: planner-credential
    workers:
      modelPolicy: domain_worker@1
      llmGateway: secondary@1
      credential: worker-credential
`)

	snapshot := mustLoad(t, root, MVPDescriptors())
	plannerGateway, _ := snapshot.LLMGateway("local-litellm@1")
	workerGateway, _ := snapshot.LLMGateway("secondary@1")
	lookup := metadataLookup{
		"planner-credential": credentialMetadata("planner-credential", plannerGateway.Ref),
		"worker-credential":  credentialMetadata("worker-credential", workerGateway.Ref),
	}
	workflow, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", ExecutionConfigPatch{}, lookup,
	)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["copy"]
	if stage.ExecutionConfig.Planner == nil {
		t.Fatal("modeled Planner has no resolved execution config")
	}
	plannerSelection := *stage.ExecutionConfig.Planner
	workerSelection := stage.ExecutionConfig.Agents["builder"]
	if plannerSelection.ModelPolicy.Ref.PolicyID != "planner" ||
		plannerSelection.LLMGateway.Ref != plannerGateway.Ref ||
		plannerSelection.Credential == nil || plannerSelection.Credential.CredentialID != "planner-credential" ||
		workerSelection.ModelPolicy.Ref.PolicyID != "domain_worker" ||
		workerSelection.LLMGateway.Ref != workerGateway.Ref ||
		workerSelection.Credential == nil || workerSelection.Credential.CredentialID != "worker-credential" {
		t.Fatalf("Planner/Worker selections = planner:%+v worker:%+v", plannerSelection, workerSelection)
	}
}

func TestWorkflowWidePlannerDefaultSkipsPassthroughStages(t *testing.T) {
	root := copyConfigTree(t)
	workflow := strings.Replace(
		multiStageWorkflowYAML,
		"  executionConfig:\n    workers: {llmGateway: local-litellm@1}\n",
		"  executionConfig:\n    planner: {modelPolicy: planner@1, llmGateway: local-litellm@1}\n    workers: {llmGateway: local-litellm@1}\n",
		1,
	)
	workflow = strings.Replace(workflow, "      planner: passthrough@1", "      planner: streamline@1", 2)
	// Restore the entry Stage to passthrough; only review is model-backed.
	workflow = strings.Replace(workflow, "      planner: streamline@1", "      planner: passthrough@1", 1)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(workflow))

	snapshot := mustLoad(t, root, MVPDescriptors())
	resolved, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", ExecutionConfigPatch{}, metadataLookup{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Stages["build"].ExecutionConfig.Planner != nil ||
		resolved.Stages["review"].ExecutionConfig.Planner == nil {
		t.Fatalf(
			"mixed Planner defaults = build:%+v review:%+v",
			resolved.Stages["build"].ExecutionConfig.Planner,
			resolved.Stages["review"].ExecutionConfig.Planner,
		)
	}
}

func TestExecutionConfigRejectsConsumerIncompatibilityAndInvalidWorkflowPatches(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*testing.T, string)
		want   string
	}{
		{
			name: "Planner policy missing maxWorkerCalls",
			mutate: func(t *testing.T, root string) {
				path := filepath.Join(root, "workflows/artifact_copy.yaml")
				replaceFile(t, path, "      planner: passthrough@1", "      planner: streamline@1")
				replaceFile(t, path, "  executionConfig:\n", "  executionConfig:\n    planner: {modelPolicy: worker@1, llmGateway: local-litellm@1}\n")
			},
			want: "Planner modelPolicy requires",
		},
		{
			name: "tool Worker policy missing maxToolCalls",
			mutate: func(t *testing.T, root string) {
				path := filepath.Join(root, "workflows/artifact_copy.yaml")
				replaceFile(t, path, "    workers:\n", "    workers:\n      modelPolicy: planner@1\n")
			},
			want: "tool-using Worker modelPolicy requires maxToolCalls",
		},
		{
			name: "Workflow credential null",
			mutate: func(t *testing.T, root string) {
				path := filepath.Join(root, "workflows/artifact_copy.yaml")
				replaceFile(t, path, "      credential: development-worker\n", "      credential: null\n")
			},
			want: "workers.credential must not be null",
		},
		{
			name: "unknown Stage",
			mutate: func(t *testing.T, root string) {
				path := filepath.Join(root, "workflows/artifact_copy.yaml")
				replaceFile(
					t, path,
					"    workers:\n      llmGateway: local-litellm@1\n      credential: development-worker\n",
					"    workers:\n      llmGateway: local-litellm@1\n      credential: development-worker\n    stages:\n      absent:\n        agents: {}\n",
				)
			},
			want: "unknown Stage",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := copyConfigTree(t)
			test.mutate(t, root)
			snapshot, err := Load(root, MVPDescriptors())
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() = (%v, %v), want error containing %q", snapshot, err, test.want)
			}
		})
	}
}

func TestResolveRunWorkflowRejectsMissingOrMismatchedCredentialSafely(t *testing.T) {
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	localGateway, _ := snapshot.LLMGateway("local-litellm@1")
	patch := decodeExecutionConfigPatch(t, `{"workers":{"credential":"selected-credential"}}`)
	tests := []struct {
		name   string
		lookup CredentialLookup
	}{
		{"missing", metadataLookup{}},
		{"wrong Gateway", metadataLookup{
			"selected-credential": credentialMetadata(
				"selected-credential",
				contracts.LLMGatewayConfigRef{
					GatewayID: "other", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
				},
			),
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", patch, test.lookup)
			if err == nil || strings.Contains(err.Error(), "secret-value") ||
				(!strings.Contains(err.Error(), "unavailable") && !strings.Contains(err.Error(), "another LLMGatewayConfig")) {
				t.Fatalf("unsafe or missing credential error = %v", err)
			}
		})
	}
	if _, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", patch,
		metadataLookup{"selected-credential": credentialMetadata("selected-credential", localGateway.Ref)},
	); err != nil {
		t.Fatalf("matching credential was rejected: %v", err)
	}
}

func TestExecutionConfigPatchJSONIsStrictAndPreservesCredentialNull(t *testing.T) {
	invalid := []string{
		`null`,
		`{"planner":null}`,
		`{"workers":{}}`,
		`{"workers":{"modelPolicy":null}}`,
		`{"workers":{"credential":42}}`,
		`{"workers":{"unknown":"value"}}`,
		`{"unknown":{}}`,
		`{"stages":{"copy":{"agents":{"builder":null}}}}`,
	}
	for _, raw := range invalid {
		var patch ExecutionConfigPatch
		if err := json.Unmarshal([]byte(raw), &patch); err == nil {
			t.Fatalf("json.Unmarshal(%s) unexpectedly succeeded", raw)
		}
	}

	raw := `{"stages":{},"workers":{"credential":null}}`
	patch := decodeExecutionConfigPatch(t, raw)
	encoded, err := json.Marshal(patch)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"workers":{"credential":null}}`
	if string(encoded) != want {
		t.Fatalf("tri-state patch canonical form = %s, want %s", encoded, want)
	}
}

type metadataLookup map[string]CredentialMetadata

func (l metadataLookup) LookupLLMCredential(_ context.Context, id string) (CredentialMetadata, error) {
	metadata, ok := l[id]
	if !ok {
		return CredentialMetadata{}, errors.New("secret-value must not escape")
	}
	return metadata, nil
}

func credentialMetadata(id string, gateway contracts.LLMGatewayConfigRef) CredentialMetadata {
	return CredentialMetadata{
		Ref: contracts.LLMCredentialRef{CredentialID: id}, LLMGateway: gateway,
	}
}

func developmentCredentialLookup(t *testing.T, snapshot *Snapshot) metadataLookup {
	t.Helper()
	gateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	return metadataLookup{
		"development-worker":  credentialMetadata("development-worker", gateway.Ref),
		"development-planner": credentialMetadata("development-planner", gateway.Ref),
	}
}

func decodeExecutionConfigPatch(t *testing.T, raw string) ExecutionConfigPatch {
	t.Helper()
	var patch ExecutionConfigPatch
	if err := json.Unmarshal([]byte(raw), &patch); err != nil {
		t.Fatalf("decode executionConfig patch: %v", err)
	}
	return patch
}

const secondaryGatewayYAML = `apiVersion: contractor/v1alpha1
kind: LLMGatewayConfig
metadata:
  name: secondary
  version: "1"
spec:
  protocol: openai-compatible@1
  url: https://secondary.example/v1
`
