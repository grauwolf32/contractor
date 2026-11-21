//go:build e2e

package e2e

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

func TestArtifactImagesAcrossProcesses(t *testing.T) {
	fixture := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	fixture.SetNRGBA(0, 0, color.NRGBA{R: 255, A: 255})
	fixture.SetNRGBA(1, 0, color.NRGBA{G: 128, B: 255, A: 255})
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, fixture); err != nil {
		t.Fatal(err)
	}
	var payloadReads atomic.Int32
	stages := make([]domainGatewayStage, 0, 2)
	for _, vision := range []bool{true, false} {
		final := finalGatewayStep("Image delivery checked", nil)
		final.validate = func(request map[string]any) error {
			expected, err := stageRefArguments("source", nil)(request)
			if err != nil {
				return err
			}
			responses := findingsToolResponses(request, "read_artifact")
			if len(responses) != 1 || !reflect.DeepEqual(responses[0]["artifact"], expected) {
				return fmt.Errorf("image source lost its exact assigned ref")
			}
			messages, _ := request["messages"].([]any)
			var images []string
			for _, raw := range messages {
				message, _ := raw.(map[string]any)
				if message["role"] == "tool" {
					text, _ := message["content"].(string)
					if strings.Contains(text, "data:image") || strings.Contains(text, "contentCacheKey") || strings.Contains(text, "dataBase64") {
						return fmt.Errorf("image bytes or private identity entered tool text")
					}
				}
				if parts, ok := message["content"].([]any); ok {
					for _, rawPart := range parts {
						part, _ := rawPart.(map[string]any)
						if part["type"] == "image_url" {
							url, _ := part["image_url"].(map[string]any)
							value, _ := url["url"].(string)
							images = append(images, value)
						}
					}
				}
			}
			if !vision {
				if len(images) != 0 || responses[0]["imageDelivery"] != "unavailable" || responses[0]["reason"] != "image_support_unconfirmed" {
					return fmt.Errorf("unconfirmed route received image content or lacked its unavailable outcome")
				}
				return nil
			}
			if len(images) != 1 || responses[0]["imageDelivery"] != "inline" {
				return fmt.Errorf("native image missing")
			}
			prefix := "data:image/png;base64,"
			if !strings.HasPrefix(images[0], prefix) {
				return fmt.Errorf("wrong native image media")
			}
			data, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(images[0], prefix))
			if err != nil {
				return err
			}
			if !bytes.Equal(data, encoded.Bytes()) {
				return fmt.Errorf("native image changed the exact source bytes")
			}
			decoded, err := png.Decode(bytes.NewReader(data))
			if err != nil {
				return err
			}
			if decoded.Bounds() != fixture.Bounds() {
				return fmt.Errorf("native image changed dimensions")
			}
			for x := 0; x < 2; x++ {
				r, g, b, a := decoded.At(x, 0).RGBA()
				er, eg, eb, ea := fixture.At(x, 0).RGBA()
				if r != er || g != eg || b != eb || a != ea {
					return fmt.Errorf("native image changed pixels")
				}
			}
			return nil
		}
		stages = append(stages, domainGatewayStage{name: fmt.Sprint("image-route-", vision), tools: []string{"read_artifact"}, steps: []domainGatewayStep{
			toolGatewayStep("read_artifact", stageRefArguments("source", nil)), final,
		}})
	}
	h := startConfiguredFindingsHarness(t, stages, func(root string) {
		for name, data := range map[string]string{
			"agent-templates/artifact_image_fixture.yaml":    imageFixtureTemplate,
			"model-policies/image_fixture.yaml":              imageFixturePolicy,
			"workflows/artifact_image_fixture.yaml":          imageFixtureWorkflow,
			"workflows/artifact_image_unknown_fixture.yaml":  strings.ReplaceAll(strings.ReplaceAll(imageFixtureWorkflow, "artifact-image-fixture", "artifact-image-unknown-fixture"), "modelPolicy: image-fixture@1", "modelPolicy: worker@2"),
			"instructions/artifact-image-fixture.md":         "Read the assigned source with read_artifact. Report whether the configured route received an image or an explicit unavailable outcome. Do not reconstruct source bytes.",
			"instructions/artifact-image-planner-fixture.md": "Check image delivery through the prepared Worker.",
		} {
			if err := os.WriteFile(filepath.Join(root, name), []byte(data), 0600); err != nil {
				t.Fatal(err)
			}
		}
	}, func(h *findingsHarness) {
		h.proxyPrivateAPI(func(response *http.Response) error {
			if response.Request.Method == http.MethodGet && strings.Contains(response.Request.URL.Path, "/artifacts/inputs/source") && response.Request.Header.Get("Accept") == "*/*" && response.StatusCode == http.StatusOK {
				payloadReads.Add(1)
			}
			return nil
		})
	})
	input := uploadProjectArtifact(t, h.client, h.baseURL, "image-source", "image/png", encoded.Bytes())
	for i, name := range []string{"artifact-image-fixture@1", "artifact-image-unknown-fixture@1"} {
		run := createWorkflowRun(t, h.client, h.baseURL, name, fmt.Sprint("image-", i), input)
		waitForFindingsRun(t, h, run, "succeeded")
	}
	if h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) != 0 {
		t.Fatal("native image process scripts did not complete", h.gateway.Failures())
	}
	if payloadReads.Load() != 1 {
		t.Fatalf("image content downloads=%d, want one confirmed-route read", payloadReads.Load())
	}
}

const imageFixtureTemplate = `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: artifact_image_fixture, version: "1"}
spec:
  description: Reads one image through the configured model route
  runtime: adk@1
  instructions: {ref: instructions/artifact-image-fixture.md}
  toolsets:
    - {ref: run-artifacts@1, tools: [read_artifact]}
  sandboxProfile: local-workdir@1
`
const imageFixturePolicy = `apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata: {name: image-fixture, version: "1"}
spec:
  model: worker-model
  supportsImages: true
  contextWindowTokens: 131072
  maxOutputTokens: 4096
  maxModelCalls: 8
  maxToolCalls: 16
  maxTotalTokens: 32768
`
const imageFixtureWorkflow = `apiVersion: contractor/v1alpha1
kind: Workflow
metadata: {name: artifact-image-fixture, version: "1"}
spec:
  parameters: {}
  inputs:
    source: {required: true, mediaTypes: [image/png]}
  outputs: {}
  executionConfig:
    workers:
      llmGateway: local-litellm@1
      credential: development-worker
      modelPolicy: image-fixture@1
  entryStage: inspect
  stages:
    inspect:
      objective: Check native image delivery from the exact assigned source
      instructions: {ref: instructions/artifact-image-planner-fixture.md}
      planner: passthrough@1
      agents:
        builder: {template: artifact_image_fixture@1}
      context:
        artifacts:
          source: {namespace: inputs, name: source, required: true}
      result: {artifacts: {}}
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`
