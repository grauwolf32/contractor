package config

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configtest"
)

const openAPIValidationEscalationYAML = `        failed:
          escalate:
            maxAttempts: 1
            executionConfig:
              agents:
                validator:
                  modelPolicy: test-strong-worker@1
            then:
              fail: {}
`

func TestInlineEscalationPinsResolvedStageVariant(t *testing.T) {
	snapshot := mustLoad(t, copyEscalationConfigTree(t))
	patch := decodeExecutionConfigPatch(t, `{"stages":{"openapi_validate":{"agents":{"validator":{"modelPolicy":"test-worker@1"}}}}}`)
	workflow, err := snapshot.ResolveRunWorkflow(t.Context(), "openapi-from-workspace@5", patch, developmentCredentialLookup(t, snapshot))
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["openapi_validate"]
	if stage.ExecutionConfig.Agents["validator"].ModelPolicy.Ref.PolicyID != "test-worker" {
		t.Fatal("Run override was lost")
	}
	variant := stage.On.Failed.Escalate.ExecutionConfig
	if variant.Effective.Agents["validator"].ModelPolicy.Ref.PolicyID != "test-strong-worker" ||
		variant.Effective.Agents["validator"].Origins.ModelPolicy != "workflow.stages.openapi_validate.on.failed.escalate.executionConfig.agents.validator" {
		t.Fatalf("escalation = %+v", variant)
	}
}

func TestInlineEscalationSnapshotSurvivesWorkflowChange(t *testing.T) {
	root := copyEscalationConfigTree(t)
	first := mustLoad(t, root)
	stored, err := first.ResolveRunWorkflow(t.Context(), "openapi-from-workspace@5", resolved.ExecutionConfigPatch{}, developmentCredentialLookup(t, first))
	if err != nil {
		t.Fatal(err)
	}
	replaceFile(t, filepath.Join(root, "workflows/openapi_from_workspace_v5.yaml"), "test-strong-worker@1", "test-domain-worker@1")
	second := mustLoad(t, root)
	current, err := second.Workflow("openapi-from-workspace@5")
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(current, stored) {
		t.Fatal("inline override change did not change the Workflow snapshot")
	}
	if current.Stages["openapi_validate"].On.Failed.Escalate.ExecutionConfig.Effective.Agents["validator"].ModelPolicy.Ref.PolicyID != "test-domain-worker" {
		t.Fatal("new inline override was not resolved")
	}
	encoded, err := json.Marshal(stored)
	if err != nil {
		t.Fatal(err)
	}
	durable, err := resolved.DecodeResolvedWorkflowSnapshot(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(stored, durable) {
		t.Fatal("stored inline escalation changed after reload")
	}
}

func TestInlineEscalationYAMLAnchorsAndMerges(t *testing.T) {
	for _, merge := range []bool{false, true} {
		t.Run(map[bool]string{false: "alias", true: "merge"}[merge], func(t *testing.T) {
			root := copyEscalationConfigTree(t)
			path := filepath.Join(root, "workflows/openapi_from_workspace_v5.yaml")
			failed := strings.Replace(openAPIValidationEscalationYAML, "executionConfig:", "executionConfig: &strong_execution", 1)
			interrupted := `        interrupted:
          escalate:
            maxAttempts: 1
            executionConfig: *strong_execution
            then:
              fail: {}`
			if merge {
				interrupted = strings.Replace(interrupted, "executionConfig: *strong_execution", "executionConfig:\n              <<: *strong_execution", 1)
			}
			replaceEscalation(t, root, failed)
			replaceFile(t, path, "        interrupted: {fail: {}}", interrupted)
			aliased := mustLoad(t, root)
			workflow, err := aliased.Workflow("openapi-from-workspace@5")
			if err != nil {
				t.Fatal(err)
			}
			// Expanded and aliased authoring have the same canonical Workflow snapshot.
			replaceFile(t, path, failed, openAPIValidationEscalationYAML)
			replaceFile(t, path, interrupted, strings.TrimSuffix(strings.Replace(openAPIValidationEscalationYAML, "failed:", "interrupted:", 1), "\n"))
			expanded, err := mustLoad(t, root).Workflow("openapi-from-workspace@5")
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(workflow, expanded) {
				t.Fatal("YAML reuse changed the resolved Workflow")
			}
			stage := workflow.Stages["openapi_validate"]
			left := stage.On.Failed.Escalate.ExecutionConfig
			right := stage.On.Interrupted.Escalate.ExecutionConfig
			left.Override.Agents["validator"] = resolved.ResolvedExecutionSelectionOverride{}
			delete(left.Effective.Agents, "validator")
			if right.Override.Agents["validator"].ModelPolicy == nil || right.Effective.Agents["validator"].ModelPolicy.Ref.PolicyID != "test-strong-worker" {
				t.Fatal("alias variants share mutable state")
			}
			again, _ := aliased.Workflow("openapi-from-workspace@5")
			if !reflect.DeepEqual(again, expanded) {
				t.Fatal("mutating a variant changed the catalog snapshot")
			}
		})
	}
}

func TestInlineEscalationSupportsCredentialClear(t *testing.T) {
	root := copyEscalationConfigTree(t)
	replaceEscalation(t, root, strings.Replace(openAPIValidationEscalationYAML, "modelPolicy: test-strong-worker@1", "modelPolicy: test-strong-worker@1\n                  credential: null", 1))
	workflow, err := mustLoad(t, root).Workflow("openapi-from-workspace@5")
	if err != nil {
		t.Fatal(err)
	}
	variant := workflow.Stages["openapi_validate"].On.Failed.Escalate.ExecutionConfig
	credential := variant.Override.Agents["validator"].Credential
	if credential == nil || !credential.Clear || credential.Ref != nil || variant.Effective.Agents["validator"].Credential != nil {
		t.Fatalf("credential clear = %+v", variant)
	}
}

func TestRunCreationValidatesCredentialsInEveryEscalationVariant(t *testing.T) {
	root := copyEscalationConfigTree(t)
	replaceEscalation(t, root, strings.Replace(openAPIValidationEscalationYAML, "modelPolicy: test-strong-worker@1", "modelPolicy: test-strong-worker@1\n                  credential: escalation-credential", 1))
	snapshot := mustLoad(t, root)
	lookup := developmentCredentialLookup(t, snapshot)
	_, err := snapshot.ResolveRunWorkflow(t.Context(), "openapi-from-workspace@5", resolved.ExecutionConfigPatch{}, lookup)
	if err == nil || !strings.Contains(err.Error(), "failed escalation") || !strings.Contains(err.Error(), "credential is unavailable") {
		t.Fatalf("missing escalation credential error = %v", err)
	}
	gateway, _ := snapshot.LLMGateway("local-litellm@1")
	lookup["escalation-credential"] = credentialMetadata("escalation-credential", gateway.Ref)
	if _, err := snapshot.ResolveRunWorkflow(t.Context(), "openapi-from-workspace@5", resolved.ExecutionConfigPatch{}, lookup); err != nil {
		t.Fatal(err)
	}
}

func TestInlineEscalationRejectsInvalidShapesAndConsumers(t *testing.T) {
	for _, test := range []struct{ name, body, want string }{
		{"removed ref", "{ref: strong-oas-review@1}", "field ref not found"},
		{"ref plus inline", "{ref: strong-oas-review@1, agents: {validator: {modelPolicy: test-worker@1}}}", "field ref not found"},
		{"empty", "{}", "must contain planner and/or agents"},
		{"null planner", "{planner: null}", "planner must not be null"},
		{"null agents", "{agents: null}", "agents must not be null"},
		{"empty agents", "{agents: {}}", "agents must be a non-empty mapping"},
		{"empty leaf", "{agents: {validator: {}}}", "must select at least one field"},
		{"unknown binding", "{agents: {ghost: {modelPolicy: test-worker@1}}}", "unknown logical Agent"},
		{"invalid name", "{agents: {'bad name': {modelPolicy: test-worker@1}}}", "logical Agent name"},
		{"unknown policy", "{agents: {validator: {modelPolicy: absent@1}}}", "unknown ModelPolicy"},
		{"incompatible policy", "{agents: {validator: {modelPolicy: test-planner@1}}}", "tool-using Worker modelPolicy requires maxToolCalls"},
		{"passthrough planner", "{planner: {modelPolicy: test-planner@1}}", "passthrough@1 does not accept Planner"},
		{"duplicate member", "{agents: {validator: {modelPolicy: test-worker@1, modelPolicy: test-strong-worker@1}}}", "already defined"},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := copyEscalationConfigTree(t)
			replaceEscalation(t, root, "        failed:\n          escalate:\n            maxAttempts: 1\n            executionConfig: "+test.body+"\n            then:\n              fail: {}\n")
			snapshot, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load = (%v, %v), want %q", snapshot, err, test.want)
			}
		})
	}
	if _, err := ParseConfigurationKind("execution-configs"); err == nil {
		t.Fatal("removed catalog kind accepted")
	}
}

func TestInlineEscalationRejectsInvalidTransition(t *testing.T) {
	for _, test := range []struct{ name, old, replacement, want string }{
		{"zero attempts", "maxAttempts: 1", "maxAttempts: 0", "maxAttempts must be at least 1"},
		{"invalid then action", "fail: {}", "succeed: {}", `action "succeed" is not allowed`},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := copyEscalationConfigTree(t)
			replaceEscalation(t, root, strings.Replace(openAPIValidationEscalationYAML, test.old, test.replacement, 1))
			if snapshot, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed")); err == nil || snapshot != nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load = (%v, %v), want %q", snapshot, err, test.want)
			}
		})
	}
}

func TestValidateWorkflowGraphRejectsTamperedEscalationVariant(t *testing.T) {
	snapshot := mustLoad(t, copyEscalationConfigTree(t))
	workflow, err := snapshot.Workflow("openapi-from-workspace@5")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["openapi_validate"]
	stage.On.Failed.Escalate.ExecutionConfig.Effective.Agents["validator"] = stage.ExecutionConfig.Agents["validator"]
	workflow.Stages["openapi_validate"] = stage
	if err := resolved.ValidateWorkflowGraph(workflow); err == nil || !strings.Contains(err.Error(), "does not match its pinned override") {
		t.Fatalf("ValidateWorkflowGraph = %v", err)
	}
}

func TestInlineEscalationSnapshotRejectsRemovedProfileRef(t *testing.T) {
	workflow, err := mustLoad(t, copyEscalationConfigTree(t)).Workflow("openapi-from-workspace@5")
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	encoded = []byte(strings.Replace(string(encoded), `"executionConfig":{"override":`, `"executionConfig":{"ref":{"configId":"old","version":"1","digest":"sha256:`+strings.Repeat("0", 64)+`"},"override":`, 1))
	if _, err := resolved.DecodeResolvedWorkflowSnapshot(encoded); err == nil {
		t.Fatal("removed snapshot profile ref accepted")
	}
}

func replaceEscalation(t *testing.T, root, replacement string) {
	t.Helper()
	replaceFile(t, filepath.Join(root, "workflows/openapi_from_workspace_v5.yaml"), openAPIValidationEscalationYAML, replacement)
}
func copyEscalationConfigTree(t *testing.T) string {
	t.Helper()
	return configtest.CopyWithEscalation(t, "../../testdata/configs")
}
