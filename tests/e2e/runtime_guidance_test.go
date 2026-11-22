//go:build e2e

package e2e

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const executionFactsPrefix = "Current execution facts (authoritative for this request):\n"

func validateExecutionFacts(request map[string]any, field string) error {
	messages, _ := request["messages"].([]any)
	count := 0
	for _, raw := range messages {
		message, _ := raw.(map[string]any)
		text, _ := message["content"].(string)
		if !strings.HasPrefix(text, executionFactsPrefix) {
			continue
		}
		count++
		var facts map[string]any
		if len(text) > 2048 {
			return fmt.Errorf("execution facts exceed byte bound")
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(text, executionFactsPrefix)), &facts); err != nil {
			return err
		}
		if _, ok := facts[field]; !ok {
			return fmt.Errorf("missing current execution field %s", field)
		}
	}
	if count != 1 {
		return fmt.Errorf("execution facts blocks = %d", count)
	}
	return nil
}

func TestRuntimeGuidanceAcrossProductionProcesses(t *testing.T) {
	t.Run("http-and-audit-argument-contracts", runtimeArgumentContractsScenario)
	t.Run("streamline", func(t *testing.T) { workerCompactionSelectionScenario(t, "streamline") })
	t.Run("exact-reads-and-checkpoint-images", func(t *testing.T) {
		var data bytes.Buffer
		if err := png.Encode(&data, image.NewNRGBA(image.Rect(0, 0, 2, 1))); err != nil {
			t.Fatal(err)
		}
		stages := []domainGatewayStage{}
		for _, media := range []string{"text/plain", "image/png"} {
			read := func(extra map[string]any) domainGatewayStep {
				return toolGatewayStep("read_artifact", stageRefArguments("source", extra))
			}
			check := func(request map[string]any) error { return validateExecutionFacts(request, "remainingToolCalls") }
			third := read(nil)
			afterRefusal := read(map[string]any{"max_lines": 1})
			if media == "image/png" {
				afterRefusal = read(nil)
			}
			afterRefusal.validate = func(request map[string]any) error {
				if err := check(request); err != nil {
					return err
				}
				responses := findingsToolResponses(request, "read_artifact")
				if len(responses) != 1 {
					return fmt.Errorf("checkpoint lost the retained refusal: %v", responses)
				}
				detail, _ := responses[0]["error"].(map[string]any)
				if detail["code"] != "tool_repeat_limit" {
					return fmt.Errorf("third read was not refused: %v", responses)
				}
				return nil
			}
			steps := []domainGatewayStep{read(nil), read(nil), third, afterRefusal}
			if media == "image/png" {
				steps = append(steps, toolGatewayStep("list_artifacts", fixedArguments(map[string]any{"namespace": "inputs"})))
			}
			steps = append(steps, read(nil))
			final := finalGatewayStep("Exact reads and checkpoints checked", nil)
			final.validate = func(request map[string]any) error {
				if err := check(request); err != nil {
					return err
				}
				responses := findingsToolResponses(request, "read_artifact")
				if len(responses) == 0 {
					return fmt.Errorf("no post-checkpoint read result")
				}
				last := responses[len(responses)-1]
				if media == "text/plain" {
					if last["text"] != "first\nsecond\n" {
						return fmt.Errorf("text reread lost exact bytes")
					}
					return nil
				}
				messages, _ := request["messages"].([]any)
				found := false
				for _, raw := range messages {
					message, _ := raw.(map[string]any)
					parts, _ := message["content"].([]any)
					for _, rawPart := range parts {
						part, _ := rawPart.(map[string]any)
						if part["type"] != "image_url" {
							continue
						}
						url, _ := part["image_url"].(map[string]any)
						value, _ := url["url"].(string)
						decoded, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(value, "data:image/png;base64,"))
						if err != nil || !bytes.Equal(decoded, data.Bytes()) {
							return fmt.Errorf("post-checkpoint native image changed")
						}
						found = true
					}
				}
				if !found || last["imageDelivery"] != "inline" {
					return fmt.Errorf("no post-checkpoint native image")
				}
				return nil
			}
			steps = append(steps, final)
			for i := range steps {
				if steps[i].validate == nil {
					steps[i].validate = check
				}
			}
			stages = append(stages, domainGatewayStage{name: media, tools: []string{"read_artifact", "list_artifacts", "read_context_history"}, steps: steps})
		}
		h := startConfiguredFindingsHarness(t, stages, func(root string) {
			workflow := strings.ReplaceAll(imageFixtureWorkflow, "mediaTypes: [image/png]", "mediaTypes: [text/plain, image/png]")
			workflow = strings.Replace(workflow, "    workers:\n", "    workers:\n      compactorModelPolicy: worker_compactor@1\n", 1)
			workflow = strings.Replace(workflow, "builder: {template: artifact_image_fixture@1}", `builder:
          template: artifact_image_fixture@1
          contextManagement:
            strategy: compact@1
            includeReasoning: false
            instructions: {ref: instructions/worker-compaction.md}
            contextWindowRatio: 0.5
            keepRecentGroups: 1
            maxCompactions: 4`, 1)
			compactorPolicy, err := os.ReadFile(filepath.Join(repoRoot(t), "configs/e2e/model-policies/worker_compactor.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			files := map[string]string{
				"model-policies/worker_compactor.yaml":           string(compactorPolicy),
				"instructions/worker-compaction.md":              "Keep exact input references, completed reads and open work.",
				"agent-templates/artifact_image_fixture.yaml":    strings.Replace(imageFixtureTemplate, "[read_artifact]", "[read_artifact, list_artifacts]", 1),
				"model-policies/image_fixture.yaml":              strings.Replace(imageFixturePolicy, "maxToolCalls: 16", "maxToolCalls: 3", 1),
				"workflows/artifact_image_fixture.yaml":          workflow,
				"instructions/artifact-image-fixture.md":         "Read the exact assigned input; retain evidence across checkpoints.",
				"instructions/artifact-image-planner-fixture.md": "Check exact reads.",
			}
			for name, content := range files {
				if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0600); err != nil {
					t.Fatal(err)
				}
			}
		}, func(h *findingsHarness) {
			h.gateway.mu.Lock()
			h.gateway.compactSummary = "The exact input was already read; continue from retained results."
			h.gateway.mu.Unlock()
		})
		for i, media := range []string{"text/plain", "image/png"} {
			payload := []byte("first\nsecond\n")
			if media == "image/png" {
				payload = data.Bytes()
			}
			input := uploadProjectArtifact(t, h.client, h.baseURL, fmt.Sprint("guidance-source-", i), media, payload)
			run := createWorkflowRun(t, h.client, h.baseURL, "artifact-image-fixture@1", fmt.Sprint("guidance-", i), input)
			status := waitForFindingsRun(t, h, run, "succeeded")
			if len(status.Attempts) != 1 {
				t.Fatal("unexpected retry")
			}
			metrics := status.Attempts[0].Metrics
			// Stage totals include the deterministic passthrough Worker dispatch.
			expectedCalls := int64(6 + i)
			if metrics == nil || !metrics.ReportsComplete || metrics.ToolCalls != expectedCalls || metrics.ToolFailures != 1 {
				t.Fatalf("repeat accounting = %+v, want %d calls / one refusal", metrics, expectedCalls)
			}
		}
		if h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) != 0 {
			t.Fatal("scripts incomplete", h.gateway.Failures())
		}
		h.gateway.mu.Lock()
		compactions := h.gateway.compactions
		h.gateway.mu.Unlock()
		if compactions != 3 {
			t.Fatalf("compactions = %d, want 3", compactions)
		}
	})
}
