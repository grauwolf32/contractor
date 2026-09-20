package projectworkflows

import (
	"bytes"
	"context"
	"encoding/json"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestProjectWorkflowModelSelection(t *testing.T) {
	repository := liveRepositoryRoot(t)
	original := liveCatalogBytes(t, filepath.Join(repository, "configs"))
	fixture := t.TempDir()
	if err := os.CopyFS(filepath.Join(fixture, "configs"), os.DirFS(filepath.Join(repository, "configs"))); err != nil {
		t.Fatal(err)
	}
	// Neither a legacy/current filename nor the old model alias is a selection API.
	for _, pair := range [][2]string{
		{"model-policies/worker.yaml", "model-policies/renamed_worker.yaml"},
		{"llm-gateways/local_litellm.yaml", "llm-gateways/renamed_gateway.yaml"},
	} {
		if err := os.Rename(filepath.Join(fixture, "configs", pair[0]), filepath.Join(fixture, "configs", pair[1])); err != nil {
			t.Fatal(err)
		}
	}
	policyPath := filepath.Join(fixture, "configs/model-policies/renamed_worker.yaml")
	data, err := os.ReadFile(policyPath)
	if err != nil {
		t.Fatal(err)
	}
	changed := bytes.Replace(data, []byte("model: worker-model"), []byte("model: previous-alias"), 1)
	if bytes.Equal(changed, data) {
		t.Fatal("fixture model marker was absent")
	}
	if err := os.WriteFile(policyPath, changed, 0o600); err != nil {
		t.Fatal(err)
	}
	sourceBefore := liveCatalogBytes(t, filepath.Join(fixture, "configs"))
	settings := liveSettings{
		model: `offline/custom: "選択"`, gatewayURL: "https://gateway.invalid/isolated/v1",
		gatewayToken: "never-retain-gateway-credential",
		workflows:    []string{"openapi-from-workspace@7", "likec4-from-workspace@7"},
	}
	target := filepath.Join(t.TempDir(), "configs")
	selections, err := copyLiveConfiguration(fixture, target, settings)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := config.Load(target, config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := resolveLiveWorkers(snapshot, settings.workflows)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 8 || len(selections) != len(resolved) {
		t.Fatalf("expected both four-Stage workflows, got %d Workers", len(resolved))
	}
	var probes []modelProbeCase
	for i, worker := range resolved {
		if worker.evidence != selections[i] {
			t.Fatal("evidence does not identify the actual effective configuration")
		}
		probes = append(probes, modelProbeCase{
			Label:  worker.evidence.Workflow + "/" + worker.evidence.Stage + "/" + worker.evidence.Agent,
			Policy: worker.config.ModelPolicy, GatewayURL: worker.config.LLMGateway.URL,
		})
	}
	runModelProbe(t, repository, probes)

	// Only the selected policy and route may change in the temporary catalog.
	expected := liveCatalogBytes(t, target)
	for path, data := range sourceBefore {
		if path != "model-policies/renamed_worker.yaml" && path != "llm-gateways/renamed_gateway.yaml" && !bytes.Equal(data, expected[path]) {
			t.Fatalf("unrelated catalog bytes changed: %s", path)
		}
	}
	base, err := config.Load(filepath.Join(fixture, "configs"), config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	beforePolicy, _ := base.ModelPolicy("worker@2")
	afterPolicy, _ := snapshot.ModelPolicy("worker@2")
	if beforePolicy.Ref.Digest == afterPolicy.Ref.Digest {
		t.Fatal("changed policy retained its old digest")
	}
	afterPolicy.Ref, afterPolicy.Model = beforePolicy.Ref, beforePolicy.Model
	if !reflect.DeepEqual(beforePolicy, afterPolicy) {
		t.Fatal("model override changed budget or sampling policy")
	}
	for _, name := range settings.workflows {
		before, _ := base.Workflow(name)
		after, _ := snapshot.Workflow(name)
		for stageName, stage := range before.Stages {
			for agentName, agent := range stage.Agents {
				if !reflect.DeepEqual(agent.Template.Summarizer, after.Stages[stageName].Agents[agentName].Template.Summarizer) {
					t.Fatal("summarizer policy changed")
				}
			}
		}
	}
	if !reflect.DeepEqual(original, liveCatalogBytes(t, filepath.Join(repository, "configs"))) ||
		!reflect.DeepEqual(sourceBefore, liveCatalogBytes(t, filepath.Join(fixture, "configs"))) {
		t.Fatal("source catalog was mutated")
	}

	evidence := newLiveEvaluationEvidence(settings)
	evidence.ModelSelections = selections
	evidence.FinishedAt = time.Now().UTC()
	evidenceRoot := t.TempDir()
	location, err := persistLiveEvidence(evidenceRoot, evidence)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(evidenceRoot, location, "summary.json"))
	if err != nil {
		t.Fatal(err)
	}
	var saved map[string]json.RawMessage
	if err := json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	if _, legacy := saved["modelSha256"]; legacy || string(saved["upstreamModelRevision"]) != "null" ||
		evidence.RequestedModelAliasSHA256 != liveAliasSHA256(settings.model) ||
		evidence.ModelSelectionBasis != "resolved_configuration" {
		t.Fatal("ambiguous or unsupported model provenance")
	}
	for _, secret := range []string{settings.model, settings.gatewayURL, settings.gatewayToken, "contractor-offline-provider-body-canary"} {
		if bytes.Contains(raw, []byte(secret)) {
			t.Fatal("failure evidence retained raw model or sensitive route/provider data")
		}
	}
	t.Log("8 resolved Workers across both workflows: actual Gateway requests and rejection without fallback verified")
}

func TestProjectWorkflowModelSelectionRejectsUnresolvedRoutes(t *testing.T) {
	for _, broken := range []string{"model-policies/worker.yaml", "llm-gateways/local_litellm.yaml", "selected-workflow"} {
		t.Run(broken, func(t *testing.T) {
			fixture := t.TempDir()
			if err := os.CopyFS(filepath.Join(fixture, "configs"), os.DirFS(filepath.Join(liveRepositoryRoot(t), "configs"))); err != nil {
				t.Fatal(err)
			}
			settings := liveSettings{
				model: "selected-offline-model", gatewayURL: "https://gateway.invalid/v1",
				workflows: []string{"openapi-from-workspace@7", "likec4-from-workspace@7"},
			}
			if broken == "selected-workflow" {
				settings.workflows = []string{"unavailable-workflow@1"}
			} else if err := os.Remove(filepath.Join(fixture, "configs", broken)); err != nil {
				t.Fatal(err)
			}
			selections, err := copyLiveConfiguration(fixture, filepath.Join(t.TempDir(), "configs"), settings)
			if err == nil || selections != nil {
				t.Fatal("unresolved selection must fail before execution, without default-model evidence")
			}
			if strings.Contains(err.Error(), settings.model) || strings.Contains(err.Error(), settings.gatewayURL) {
				t.Fatal("configuration error reflected the supplied model or URL")
			}
		})
	}
}

type modelProbeCase struct {
	Label      string                        `json:"label"`
	Policy     contracts.ResolvedModelPolicy `json:"policy"`
	GatewayURL string                        `json:"gatewayUrl"`
}

func runModelProbe(t *testing.T, repository string, cases []modelProbeCase) {
	t.Helper()
	input, err := json.Marshal(cases)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	command := exec.CommandContext(ctx, filepath.Join(repository, "runtime/.venv/bin/python"),
		filepath.Join(repository, "tests/eval/project_workflows/testdata/gateway_model_probe.py"))
	command.Dir = repository
	command.Stdin = bytes.NewReader(input)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	output, err := command.Output()
	if err != nil {
		t.Fatalf("offline Runtime probe failed (locked runtime/.venv required): %v; %s", err, stderr.String())
	}
	var results []struct {
		Label        string   `json:"label"`
		Models       []string `json:"models"`
		RejectedType string   `json:"rejectedType"`
		Retryable    bool     `json:"retryable"`
	}
	if err := json.Unmarshal(output, &results); err != nil {
		t.Fatal("invalid offline probe response")
	}
	if len(results) != len(cases) || stderr.Len() != 0 {
		t.Fatal("offline probe did not execute every selected Worker cleanly")
	}
	for i, result := range results {
		expected := cases[i]
		if result.Label != expected.Label ||
			!reflect.DeepEqual(result.Models, []string{expected.Policy.Model, expected.Policy.Model}) ||
			result.RejectedType != "GatewayModelError" || result.Retryable {
			t.Fatalf("effective model or no-fallback assertion failed for %s", expected.Label)
		}
	}
}

func liveCatalogBytes(t *testing.T, root string) map[string][]byte {
	t.Helper()
	files := map[string][]byte{}
	if err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		files[relative] = data
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return files
}
