//go:build e2e

package e2e

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
)

func TestHTTPCaidoE2EConfigurationClosure(t *testing.T) {
	repositoryRoot := repoRoot(t)
	fixture, _, err := configload.LoadUnionReadOnly(filepath.Join(repositoryRoot, "configs", "e2e"), filepath.Join(t.TempDir(), "fixture-managed"))
	if err != nil {
		t.Fatal(err)
	}
	worker, err := fixture.ModelPolicy("worker@1")
	if err != nil {
		t.Fatal(err)
	}
	production, _, err := configload.LoadUnionReadOnly(filepath.Join(repositoryRoot, "configs"), filepath.Join(t.TempDir(), "production-managed"))
	if err != nil {
		t.Fatal(err)
	}
	root := stageE2EConfiguration(t, filepath.Join(repositoryRoot, "configs"), filepath.Join(t.TempDir(), "configs"), "http://127.0.0.1:1/v1")
	writeHTTPAnalysisE2EWorkflow(t, root)
	for _, files := range [][2]string{
		{"e2e/model-policies/worker.yaml", "model-policies/http_e2e_worker.yaml"},
		{"model-policies/worker.yaml", "model-policies/worker.yaml"},
		{"model-policies/summarizer.yaml", "model-policies/summarizer.yaml"},
	} {
		original, err := os.ReadFile(filepath.Join(repositoryRoot, "configs", files[0]))
		if err != nil {
			t.Fatal(err)
		}
		staged, err := os.ReadFile(filepath.Join(root, files[1]))
		if err != nil || !bytes.Equal(staged, original) {
			t.Fatalf("HTTP fixture changed original policy bytes for %s: %v", files[0], err)
		}
	}
	snapshot, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatalf("shared HTTP/Caido process configuration must load: %v", err)
	}
	stagedWorker, err := snapshot.ModelPolicy("worker@1")
	if err != nil || !reflect.DeepEqual(stagedWorker, worker) {
		t.Fatalf("HTTP fixture changed the original worker@1 policy: %v", err)
	}
	for _, selector := range []string{"worker@2", "summarizer@1"} {
		expected, err := production.ModelPolicy(selector)
		if err != nil {
			t.Fatal(err)
		}
		actual, err := snapshot.ModelPolicy(selector)
		if err != nil || !reflect.DeepEqual(actual, expected) {
			t.Fatalf("HTTP fixture staging changed production %s model or budgets: %v", selector, err)
		}
	}
	workflow, err := snapshot.Workflow("http-analysis-e2e@1")
	if err != nil {
		t.Fatal(err)
	}
	selection := workflow.Stages["analyze"].ExecutionConfig.Agents["explorer"]
	policy := selection.ModelPolicy
	expected := worker
	expected.Ref = policy.Ref
	expected.MaxModelCalls = httpCaidoFixtureModelCalls()
	if policy.Ref.PolicyID != "http-caido-worker" || policy.Ref.Version != "1" || expected.MaxModelCalls != 15 || !reflect.DeepEqual(policy, expected) {
		t.Fatalf("HTTP fixture must preserve its model and other budgets with exactly 15 calls for the longest unchanged script and finalizer: %+v", policy)
	}
	summarizer, err := production.ModelPolicy("summarizer@1")
	if err != nil || selection.SummarizerModelPolicy == nil || !reflect.DeepEqual(*selection.SummarizerModelPolicy, summarizer) {
		t.Fatalf("HTTP fixture must retain its template's original summarizer@1 model and budgets: %v", err)
	}
}
