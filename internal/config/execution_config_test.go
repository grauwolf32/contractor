package config

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"go.yaml.in/yaml/v4"
	"os"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/credentialerrors"
)

func TestResolveRunWorkflowPinsAuthoredWorkerLayers(t *testing.T) {
	root := copyCoreFixture(t)
	writeFile(t, filepath.Join(root, "llm-gateways/secondary.yaml"), []byte(secondaryGatewayYAML))
	workflowPath := filepath.Join(root, "workflows/artifact_copy.yaml")
	replaceFile(t, workflowPath, `  executionConfig:
    workers:
      llmGateway: local-litellm@1
      credential: development-worker
      modelPolicy: test-worker@1
`, `  executionConfig:
    workers:
      modelPolicy: test-domain-worker@1
      llmGateway: local-litellm@1
      credential: workflow-worker
    stages:
      copy:
        agents:
          builder:
            modelPolicy: test-worker@1
`)

	snapshot := mustLoad(t, root)
	localGateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	credentials := metadataLookup{
		"workflow-worker":  credentialMetadata("workflow-worker", localGateway.Ref),
		"secondary-worker": credentialMetadata("secondary-worker", mustGateway(t, snapshot, "secondary@1")),
	}

	base, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", credentials)
	if err != nil {
		t.Fatal(err)
	}
	baseSelection := base.Stages["copy"].ExecutionConfig.Agents["builder"]
	if baseSelection.ModelPolicy.Ref.PolicyID != "test-worker" ||
		baseSelection.Origins.ModelPolicy != "workflow.executionConfig.stages.copy.agents.builder" ||
		baseSelection.LLMGateway.Ref != localGateway.Ref ||
		baseSelection.Origins.LLMGateway != "workflow.executionConfig.workers" ||
		baseSelection.Credential == nil || baseSelection.Credential.CredentialID != "workflow-worker" ||
		baseSelection.Origins.Credential != "workflow.executionConfig.workers" {
		t.Fatalf("resolved Workflow defaults = %+v", baseSelection)
	}

	writeWorkflowExecutionConfig(t, root, "artifact_copy.yaml", `{"workers":{"modelPolicy":"test-domain-worker@1","credential":"workflow-worker","llmGateway":"local-litellm@1"},"stages":{"copy":{"agents":{"builder":{"modelPolicy":"test-worker@1","llmGateway":"secondary@1","credential":"secondary-worker"}}}}}`)
	snapshot = mustLoad(t, root)
	runWorkflow, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", credentials)
	if err != nil {
		t.Fatal(err)
	}
	selection := runWorkflow.Stages["copy"].ExecutionConfig.Agents["builder"]
	if selection.ModelPolicy.Ref.PolicyID != "test-worker" || selection.LLMGateway.Ref.GatewayID != "secondary" || selection.Credential == nil || selection.Credential.CredentialID != "secondary-worker" || selection.Origins.Credential != "workflow.executionConfig.stages.copy.agents.builder" {
		t.Fatalf("binding selection = %+v", selection)
	}
	pinned, err := json.Marshal(runWorkflow)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(pinned), "token") {
		t.Fatalf("Run snapshot contains secret-shaped data: %s", pinned)
	}
	replaceFile(
		t, filepath.Join(root, "model-policies/test-worker.yaml"),
		"model: worker-model", "model: worker-model-v2",
	)
	reloaded := mustLoad(t, root)
	newResolution, err := reloaded.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", credentials,
	)
	if err != nil {
		t.Fatal(err)
	}
	var stored resolved.ResolvedWorkflow
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
	root := copyCoreFixture(t)
	workflowPath := filepath.Join(root, "workflows/artifact_copy.yaml")
	replaceFile(t, workflowPath, `  executionConfig:
    workers:
      llmGateway: local-litellm@1
      credential: development-worker
      modelPolicy: test-worker@1

`, "  executionConfig:\n    workers: {modelPolicy: test-worker@1}\n\n")
	snapshot, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", metadataLookup{},
	)
	if err != nil {
		t.Fatalf("resolve label-routed Worker: %v", err)
	}
	selection := workflow.Stages[workflow.EntryStage].ExecutionConfig.Agents["builder"]
	if selection.LLMGateway != nil || selection.ModelPolicy.Ref.PolicyID != "test-worker" ||
		selection.Origins.ModelPolicy != originWorkflowPrefix+".workers" || selection.Origins.LLMGateway != "" {
		t.Fatalf("partially routed Worker selection = %+v", selection)
	}

	planner := selection
	planner.ModelPolicy, err = snapshot.ModelPolicy("test-planner@1")
	if err != nil {
		t.Fatal(err)
	}
	modeled := workflow.Stages[workflow.EntryStage]
	modeled.Planner = resolved.PlannerRef{PlannerID: "streamline", Version: "1"}
	modeled.ExecutionConfig.Planner = &planner
	if err := resolved.ValidateStageExecutionConfig(workflow.EntryStage, modeled); err == nil ||
		!strings.Contains(err.Error(), "Planner llmGateway is required") {
		t.Fatalf("incomplete Planner validation error = %v", err)
	}
}

func TestResolveRunWorkflowKeepsPlannerAndWorkerSelectionsIndependent(t *testing.T) {
	root := copyCoreFixture(t)
	writeFile(t, filepath.Join(root, "llm-gateways/secondary.yaml"), []byte(secondaryGatewayYAML))
	workflowPath := filepath.Join(root, "workflows/artifact_copy.yaml")
	replaceFile(t, workflowPath, "      planner: passthrough@1", "      planner: streamline@1")
	replaceFile(t, workflowPath, `  executionConfig:
    workers:
      llmGateway: local-litellm@1
      credential: development-worker
      modelPolicy: test-worker@1
`, `  executionConfig:
    planner:
      modelPolicy: test-planner@1
      llmGateway: local-litellm@1
      credential: planner-credential
    workers:
      modelPolicy: test-domain-worker@1
      llmGateway: secondary@1
      credential: worker-credential
`)

	snapshot := mustLoad(t, root)
	plannerGateway, _ := snapshot.LLMGateway("local-litellm@1")
	workerGateway, _ := snapshot.LLMGateway("secondary@1")
	lookup := metadataLookup{
		"planner-credential": credentialMetadata("planner-credential", plannerGateway.Ref),
		"worker-credential":  credentialMetadata("worker-credential", workerGateway.Ref),
	}
	workflow, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", lookup,
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
	if plannerSelection.ModelPolicy.Ref.PolicyID != "test-planner" ||
		plannerSelection.LLMGateway.Ref != plannerGateway.Ref ||
		plannerSelection.Credential == nil || plannerSelection.Credential.CredentialID != "planner-credential" ||
		workerSelection.ModelPolicy.Ref.PolicyID != "test-domain-worker" ||
		workerSelection.LLMGateway.Ref != workerGateway.Ref ||
		workerSelection.Credential == nil || workerSelection.Credential.CredentialID != "worker-credential" {
		t.Fatalf("Planner/Worker selections = planner:%+v worker:%+v", plannerSelection, workerSelection)
	}
}

func TestWorkflowWidePlannerDefaultSkipsPassthroughStages(t *testing.T) {
	root := copyCoreFixture(t)
	workflow := strings.Replace(
		multiStageWorkflowYAML,
		"  executionConfig:\n    workers: {modelPolicy: test-worker@1, llmGateway: local-litellm@1}\n",
		"  executionConfig:\n    planner: {modelPolicy: test-planner@1, llmGateway: local-litellm@1}\n    workers: {modelPolicy: test-worker@1, llmGateway: local-litellm@1}\n",
		1,
	)
	workflow = strings.Replace(workflow, "      planner: passthrough@1", "      planner: streamline@1", 2)
	// Restore the entry Stage to passthrough; only review is model-backed.
	workflow = strings.Replace(workflow, "      planner: streamline@1", "      planner: passthrough@1", 1)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(workflow))

	snapshot := mustLoad(t, root)
	runWorkflow, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", metadataLookup{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if runWorkflow.Stages["build"].ExecutionConfig.Planner != nil ||
		runWorkflow.Stages["review"].ExecutionConfig.Planner == nil {
		t.Fatalf(
			"mixed Planner defaults = build:%+v review:%+v",
			runWorkflow.Stages["build"].ExecutionConfig.Planner,
			runWorkflow.Stages["review"].ExecutionConfig.Planner,
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
				replaceFile(t, path, "  executionConfig:\n", "  executionConfig:\n    planner: {modelPolicy: test-worker@1, llmGateway: local-litellm@1}\n")
			},
			want: "Planner modelPolicy requires",
		},
		{
			name: "tool Worker policy missing maxToolCalls",
			mutate: func(t *testing.T, root string) {
				path := filepath.Join(root, "workflows/artifact_copy.yaml")
				replaceFile(t, path, "modelPolicy: test-worker@1", "modelPolicy: test-planner@1")
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
					"    workers:\n      llmGateway: local-litellm@1\n      credential: development-worker\n      modelPolicy: test-worker@1\n",
					"    workers:\n      llmGateway: local-litellm@1\n      credential: development-worker\n      modelPolicy: test-worker@1\n    stages:\n      absent:\n        agents: {}\n",
				)
			},
			want: "unknown Stage",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := copyCoreFixture(t)
			test.mutate(t, root)
			snapshot, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() = (%v, %v), want error containing %q", snapshot, err, test.want)
			}
		})
	}
}

func TestResolveRunWorkflowRejectsMissingOrMismatchedCredentialSafely(t *testing.T) {
	root := copyCoreFixture(t)
	replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), "credential: development-worker", "credential: selected-credential")
	snapshot := mustLoad(t, root)
	localGateway, _ := snapshot.LLMGateway("local-litellm@1")
	tests := []struct {
		name   string
		lookup resolved.CredentialLookup
	}{
		{"missing", metadataLookup{}},
		{"wrong Gateway", metadataLookup{
			"selected-credential": credentialMetadata(
				"selected-credential",
				llmgateway.LLMGatewayConfigRef{
					GatewayID: "other", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
				},
			),
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", test.lookup)
			if err == nil || strings.Contains(err.Error(), "secret-value") ||
				(!strings.Contains(err.Error(), "unavailable") && !strings.Contains(err.Error(), "another LLMGatewayConfig")) {
				t.Fatalf("unsafe or missing credential error = %v", err)
			}
		})
	}
	if _, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1",
		metadataLookup{"selected-credential": credentialMetadata("selected-credential", localGateway.Ref)},
	); err != nil {
		t.Fatalf("matching credential was rejected: %v", err)
	}
}

func TestCredentialLookupFailuresKeepTheirClassAndCause(t *testing.T) {
	root := copyCoreFixture(t)
	replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), "credential: development-worker", "credential: selected-credential")
	snapshot := mustLoad(t, root)
	localGateway, _ := snapshot.LLMGateway("local-litellm@1")
	runWorkflow, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1",
		metadataLookup{"selected-credential": credentialMetadata("selected-credential", localGateway.Ref)},
	)
	if err != nil {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	recovery := errors.New("fixture: credential recovery in progress, secret-value must not escape")
	outage := fmt.Errorf("fixture: storage outage: %w", errors.New("secret-value must not escape"))
	for _, test := range []struct {
		name   string
		ctx    context.Context
		lookup resolved.CredentialLookup
		cause  error
	}{
		{"recovery", t.Context(), failingCredentialLookup{recovery}, recovery},
		{"storage outage", t.Context(), failingCredentialLookup{outage}, outage},
		{"cancelled while absent", cancelled, metadataLookup{}, context.Canceled},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, resolveErr := snapshot.ResolveRunWorkflow(test.ctx, "artifact-copy@1", test.lookup)
			validateErr := resolved.ValidateResolvedWorkflowCredentials(test.ctx, runWorkflow, test.lookup)
			for name, err := range map[string]error{"resolve": resolveErr, "revalidate": validateErr} {
				if !errors.Is(err, resolved.ErrCredentialLookup) || !errors.Is(err, test.cause) ||
					errors.Is(err, credentialerrors.LLMNotFound) || strings.Contains(err.Error(), "secret-value") {
					t.Fatalf("%s lookup failure = %v; classified=%t cause=%t", name, err,
						errors.Is(err, resolved.ErrCredentialLookup), errors.Is(err, test.cause))
				}
			}
		})
	}
	t.Run("confirmed absence", func(t *testing.T) {
		_, resolveErr := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", metadataLookup{})
		validateErr := resolved.ValidateResolvedWorkflowCredentials(t.Context(), runWorkflow, metadataLookup{})
		for name, err := range map[string]error{"resolve": resolveErr, "revalidate": validateErr} {
			if err == nil || errors.Is(err, resolved.ErrCredentialLookup) || errors.Is(err, credentialerrors.LLMNotFound) ||
				!strings.Contains(err.Error(), "credential is unavailable") || strings.Contains(err.Error(), "secret-value") {
				t.Fatalf("%s missing credential = %v", name, err)
			}
		}
	})
}

type failingCredentialLookup struct{ err error }

func (l failingCredentialLookup) LookupLLMCredential(context.Context, string) (resolved.CredentialMetadata, error) {
	return resolved.CredentialMetadata{}, l.err
}

type metadataLookup map[string]resolved.CredentialMetadata

func (l metadataLookup) LookupLLMCredential(_ context.Context, id string) (resolved.CredentialMetadata, error) {
	metadata, ok := l[id]
	if !ok {
		return resolved.CredentialMetadata{}, fmt.Errorf("%w: secret-value must not escape", credentialerrors.LLMNotFound)
	}
	return metadata, nil
}

func credentialMetadata(id string, gateway llmgateway.LLMGatewayConfigRef) resolved.CredentialMetadata {
	return resolved.CredentialMetadata{
		Ref: llmgateway.LLMCredentialRef{CredentialID: id}, LLMGateway: gateway,
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

const secondaryGatewayYAML = `apiVersion: contractor/v1alpha1
kind: LLMGatewayConfig
metadata:
  name: secondary
  version: "1"
spec:
  protocol: openai-compatible@1
  url: https://secondary.example/v1
`

func writeWorkflowExecutionConfig(t *testing.T, root, file, value string) {
	t.Helper()
	path := filepath.Join(root, "workflows", file)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var document yaml.Node
	if err := yaml.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	var selection yaml.Node
	if err := yaml.Unmarshal([]byte(value), &selection); err != nil {
		t.Fatal(err)
	}
	mapping := document.Content[0]
	for i := 0; i < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value != "spec" {
			continue
		}
		spec := mapping.Content[i+1]
		for j := 0; j < len(spec.Content); j += 2 {
			if spec.Content[j].Value == "executionConfig" {
				spec.Content[j+1] = selection.Content[0]
				data, err = yaml.Marshal(&document)
				if err != nil {
					t.Fatal(err)
				}
				writeFile(t, path, data)
				return
			}
		}
		spec.Content = append(spec.Content, &yaml.Node{Kind: yaml.ScalarNode, Value: "executionConfig"}, selection.Content[0])
		data, err = yaml.Marshal(&document)
		if err != nil {
			t.Fatal(err)
		}
		writeFile(t, path, data)
		return
	}
	t.Fatal("Workflow spec missing")
}

func mustGateway(t *testing.T, snapshot *Snapshot, ref string) llmgateway.LLMGatewayConfigRef {
	t.Helper()
	gateway, err := snapshot.LLMGateway(ref)
	if err != nil {
		t.Fatal(err)
	}
	return gateway.Ref
}
