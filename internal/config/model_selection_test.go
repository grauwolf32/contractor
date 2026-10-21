package config

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

type memoryModelSettings struct {
	mu    sync.Mutex
	value settingsstore.ModelSelectionSettings
}

func (s *memoryModelSettings) InitializeModelSelection(ctx context.Context, value settingsstore.ModelSelectionValue) (settingsstore.ModelSelectionSettings, error) {
	if err := value.Validate(); err != nil {
		return settingsstore.ModelSelectionSettings{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.value.Revision == 0 {
		s.value = settingsstore.ModelSelectionSettings{Value: value.Clone(), Revision: 1, UpdatedAt: time.Date(2026, 10, 21, 0, 0, 0, 0, time.UTC)}
	}
	return s.copy(), nil
}
func (s *memoryModelSettings) copy() settingsstore.ModelSelectionSettings {
	value := s.value
	value.Value = s.value.Value.Clone()
	return value
}
func (s *memoryModelSettings) GetModelSelection(context.Context) (settingsstore.ModelSelectionSettings, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.copy(), nil
}
func (s *memoryModelSettings) UpdateModelSelection(_ context.Context, value settingsstore.ModelSelectionValue, revision uint64) (settingsstore.ModelSelectionSettings, error) {
	if err := value.Validate(); err != nil {
		return settingsstore.ModelSelectionSettings{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if revision != s.value.Revision {
		return settingsstore.ModelSelectionSettings{}, settingsstore.ErrPrecondition
	}
	if value.DefaultModel != s.value.Value.DefaultModel || !maps.Equal(value.Roles, s.value.Value.Roles) {
		s.value.Value = value.Clone()
		s.value.Revision++
		s.value.UpdatedAt = s.value.UpdatedAt.Add(time.Microsecond)
	}
	return s.copy(), nil
}

func modelSelectionFixture(t *testing.T) (string, []ModelChoice) {
	t.Helper()
	root := copyCoreFixture(t)
	installWorkerSummarizer(t, root)
	replaceFile(t, templatePath(root), "test-worker@1", "worker@1")
	replaceFile(t, templatePath(root), "terminal_summarizer@1", "summarizer@1")
	replaceFile(t, summaryPolicyPath(root), "name: terminal_summarizer", "name: summarizer")
	replaceFile(t, filepath.Join(root, "model-policies/worker.yaml"), "  model: worker-model\n", "  model: worker-model\n  contextWindowTokens: 131072\n")
	models := []ModelChoice{}
	for index, id := range []string{"fixture-a", "fixture-b"} {
		digest := strings.Repeat(string(rune('a'+index)), 64)
		window := 131072 / (index + 1)
		model := ModelChoice{ID: id, Digest: "sha256:" + digest, UpstreamModel: "fixture/" + id, ContextWindowTokens: window, MaxOutputTokens: 8192, Policies: map[string]string{}}
		for _, role := range modelRoles {
			name := role + "_" + strings.ReplaceAll(id, "-", "_")
			spec := map[string]any{"model": model.UpstreamModel + "@" + digest[:16] + "/" + role, "contextWindowTokens": window, "maxOutputTokens": 4096, "maxModelCalls": 8, "temperature": 0.1}
			switch role {
			case "planner":
				spec["maxOutputTokens"] = 2048
				spec["maxWorkerCalls"] = 8
				spec["maxTotalTokens"] = 32768
			case "summarizer":
				spec["maxOutputTokens"] = 2048
				spec["maxModelCalls"] = 1
			default:
				spec["maxToolCalls"] = 16
				spec["maxTotalTokens"] = 32768
			}
			document := map[string]any{"apiVersion": "contractor/v1alpha1", "kind": "ModelPolicy", "metadata": map[string]string{"name": name, "version": "1"}, "spec": spec}
			data, err := json.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			writeFile(t, filepath.Join(root, "model-policies", name+".yaml"), data)
			model.Policies[role] = name + "@1"
			if index == 0 && role == "planner" {
				document["metadata"] = map[string]string{"name": "planner", "version": "1"}
				data, _ = json.Marshal(document)
				writeFile(t, filepath.Join(root, "model-policies/planner.yaml"), data)
			}
		}
		models = append(models, model)
	}
	appendFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), strings.ReplaceAll(summaryTransitionYAML, "test-planner@1", "planner@1"))
	base := string(readFile(t, filepath.Join(root, "workflows/artifact_copy.yaml")))
	modeled := strings.ReplaceAll(strings.TrimSuffix(base, strings.ReplaceAll(summaryTransitionYAML, "test-planner@1", "planner@1")), "name: artifact-copy", "name: artifact-modeled")
	modeled = strings.ReplaceAll(modeled, "planner: passthrough@1", "planner: streamline@1")
	modeled = strings.Replace(modeled, "  executionConfig:\n", "  executionConfig:\n    planner:\n      modelPolicy: planner@1\n      llmGateway: local-litellm@1\n", 1)
	writeFile(t, filepath.Join(root, "workflows/artifact_modeled.yaml"), []byte(modeled))
	catalog := modelChoiceCatalog{SchemaVersion: "1", InitialSelection: settingsstore.ModelSelectionValue{DefaultModel: "fixture-a", Roles: map[string]string{}}, Models: models}
	data, err := json.Marshal(catalog)
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(root, "model-selection.json"), data)
	return root, models
}

func configuredModelManager(t *testing.T, root string, store *memoryModelSettings) *Manager {
	t.Helper()
	manager, err := NewManager(ManagerOptions{OperatorRoot: root, ManagedRoot: filepath.Join(t.TempDir(), "managed")})
	if err != nil {
		t.Fatal(err)
	}
	if err := manager.ConfigureModelSelection(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	return manager
}

func TestDeploymentModelSelectionPinsMainSummaryPlannerAndExplicitVariants(t *testing.T) {
	root, models := modelSelectionFixture(t)
	store := &memoryModelSettings{}
	manager := configuredModelManager(t, root, store)
	credentials := developmentCredentialLookup(t, manager.Snapshot())
	resolve := func(patch resolved.ExecutionConfigPatch) resolved.ResolvedWorkflow {
		t.Helper()
		workflow, err := manager.ResolveRunWorkflow(t.Context(), "artifact-modeled@1", patch, credentials)
		if err != nil {
			t.Fatal(err)
		}
		return workflow
	}
	// The deployed passthrough workflow starts a modeled Planner only after
	// a terminal summary; it must receive the same captured Planner choice.
	before := resolve(resolved.ExecutionConfigPatch{})
	original, _ := json.Marshal(before)
	stage := before.Stages["copy"]
	if stage.ExecutionConfig.Planner.ModelPolicy.Model != "fixture/fixture-a@"+strings.Repeat("a", 16)+"/planner" {
		t.Fatal("Planner did not use the selected model")
	}
	main := stage.ExecutionConfig.Agents["builder"]
	if main.ModelPolicy.ContextWindowTokens != 131072 || main.SummarizerModelPolicy == nil || main.SummarizerModelPolicy.Model != "fixture/fixture-a@"+strings.Repeat("a", 16)+"/summarizer" {
		t.Fatalf("initial selection: %+v", main)
	}
	info, err := manager.GetModelSelection(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	changed, err := manager.ReplaceModelSelection(t.Context(), settingsstore.ModelSelectionValue{DefaultModel: "fixture-b", Roles: map[string]string{"summarizer": "fixture-a"}}, 1, info.CatalogDigest)
	if err != nil {
		t.Fatal(err)
	}
	continuation, err := manager.ResolveRunWorkflow(t.Context(), "artifact-copy@1", resolved.ExecutionConfigPatch{}, credentials)
	if err != nil {
		t.Fatal(err)
	}
	if continuation.Stages["copy"].On.Summarized.Escalate.ExecutionConfig.Effective.Planner.ModelPolicy.ContextWindowTokens != 65536 {
		t.Fatal("summary continuation ignored the selected Planner model")
	}
	after := resolve(resolved.ExecutionConfigPatch{})
	selected := after.Stages["copy"].ExecutionConfig.Agents["builder"]
	if changed.Revision != "2" || selected.ModelPolicy.ContextWindowTokens != 65536 || selected.ModelPolicy.MaxTotalTokens != 32768 || selected.SummarizerModelPolicy.Model != main.SummarizerModelPolicy.Model {
		t.Fatal("model change changed spend or lost summary override")
	}
	if !reflect.DeepEqual(stage.Agents["builder"].Template, after.Stages["copy"].Agents["builder"].Template) {
		t.Fatal("selection mutated immutable template")
	}
	if fresh, _ := json.Marshal(before); string(fresh) != string(original) {
		t.Fatal("selection mutated earlier Run snapshot")
	}
	pinned, err := manager.ResolvePinnedRunWorkflow(t.Context(), before, credentials)
	if err != nil {
		t.Fatal(err)
	}
	if data, _ := json.Marshal(pinned); string(data) != string(original) {
		t.Fatal("prepared Eval was reinterpreted through current defaults")
	}
	override := decodeExecutionConfigPatch(t, `{"workers":{"modelPolicy":"`+models[0].Policies["worker"]+`","summarizerModelPolicy":"`+models[1].Policies["summarizer"]+`"}}`)
	explicit := resolve(override).Stages["copy"].ExecutionConfig.Agents["builder"]
	if explicit.ModelPolicy.Ref.PolicyID != strings.TrimSuffix(models[0].Policies["worker"], "@1") || explicit.SummarizerModelPolicy.ContextWindowTokens != 65536 {
		t.Fatal("explicit Eval choices did not win")
	}
	restarted := configuredModelManager(t, root, store)
	saved, err := restarted.GetModelSelection(t.Context())
	if err != nil || saved.Selection.DefaultModel != "fixture-b" || saved.Selection.Roles["summarizer"] != "fixture-a" {
		t.Fatal("restart replaced durable selection")
	}
	tampered := before.Clone()
	bad := tampered.Stages["copy"]
	value := bad.ExecutionConfig.Agents["builder"]
	value.ModelPolicy.Model = "unconfigured/replacement"
	bad.ExecutionConfig.Agents["builder"] = value
	tampered.Stages["copy"] = bad
	if _, err := manager.ResolvePinnedRunWorkflow(t.Context(), tampered, credentials); !errors.Is(err, contentdigest.ErrPinnedSelectionChanged) {
		t.Fatalf("tampered historical model accepted: %v", err)
	}
}

func TestDeploymentModelSelectionRejectsStaleUnknownAndIncompatibleChoices(t *testing.T) {
	root, models := modelSelectionFixture(t)
	store := &memoryModelSettings{}
	manager := configuredModelManager(t, root, store)
	info, _ := manager.GetModelSelection(t.Context())
	for _, test := range []struct {
		value    settingsstore.ModelSelectionValue
		revision uint64
		digest   string
		want     error
	}{
		{settingsstore.ModelSelectionValue{DefaultModel: "missing", Roles: map[string]string{}}, 1, info.CatalogDigest, settingsstore.ErrModelSelectionInvalid},
		{*info.Selection, 2, info.CatalogDigest, settingsstore.ErrPrecondition},
		{*info.Selection, 1, "sha256:" + strings.Repeat("0", 64), settingsstore.ErrPrecondition},
	} {
		if _, err := manager.ReplaceModelSelection(t.Context(), test.value, test.revision, test.digest); !errors.Is(err, test.want) {
			t.Fatalf("invalid replacement: %v", err)
		}
	}
	current, _ := manager.GetModelSelection(t.Context())
	if !reflect.DeepEqual(current, info) {
		t.Fatal("rejected replacement changed selection")
	}
	// A profile may support only smaller-output roles. It is offered for those
	// roles but cannot silently shrink the Worker budget to become a default.
	delete(models[1].Policies, "worker")
	delete(models[1].Policies, "audit_completion_worker")
	models[1].MaxOutputTokens = 2048
	data, _ := json.Marshal(modelChoiceCatalog{SchemaVersion: "1", InitialSelection: *info.Selection, Models: models})
	writeFile(t, filepath.Join(root, "model-selection.json"), data)
	partial := configuredModelManager(t, root, &memoryModelSettings{})
	partialInfo, _ := partial.GetModelSelection(t.Context())
	if _, err := partial.ReplaceModelSelection(t.Context(), settingsstore.ModelSelectionValue{DefaultModel: "fixture-a", Roles: map[string]string{"summarizer": "fixture-b"}}, 1, partialInfo.CatalogDigest); err != nil {
		t.Fatal(err)
	}
	if _, err := partial.ReplaceModelSelection(t.Context(), settingsstore.ModelSelectionValue{DefaultModel: "fixture-b", Roles: map[string]string{}}, 2, partialInfo.CatalogDigest); !errors.Is(err, settingsstore.ErrModelSelectionInvalid) {
		t.Fatalf("unsupported Worker default accepted: %v", err)
	}
}
