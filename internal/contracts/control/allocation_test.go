package control

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestAllocationRunMetadataLabelCasesAreStrictAndDetached(t *testing.T) {
	t.Parallel()

	type labelCase struct {
		Name  string          `json:"name"`
		Value json.RawMessage `json:"value"`
		Omit  bool            `json:"omit"`
	}
	var cases struct {
		Valid   []labelCase `json:"valid"`
		Invalid []labelCase `json:"invalid"`
	}
	path := contractstest.Path(t, "api", "testdata", "v1alpha1", "run-metadata-label-cases.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	var baseline map[string]json.RawMessage
	if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "allocation-spec.json"), &baseline); err != nil {
		t.Fatal(err)
	}
	decode := func(candidate labelCase) error {
		copy := make(map[string]json.RawMessage, len(baseline))
		for key, value := range baseline {
			copy[key] = append(json.RawMessage(nil), value...)
		}
		if candidate.Omit {
			delete(copy, "runMetadataLabels")
		} else {
			copy["runMetadataLabels"] = candidate.Value
		}
		encoded, marshalErr := json.Marshal(copy)
		if marshalErr != nil {
			return marshalErr
		}
		_, decodeErr := contracts.DecodeStrict[AllocationSpec](encoded)
		return decodeErr
	}
	for _, candidate := range cases.Valid {
		if err := decode(candidate); err != nil {
			t.Errorf("valid case %q failed: %v", candidate.Name, err)
		}
	}
	for _, candidate := range cases.Invalid {
		if err := decode(candidate); err == nil {
			t.Errorf("invalid case %q was accepted", candidate.Name)
		}
	}

	source := map[string]string{"eval.id": "eval_01"}
	normalized, err := contracts.NormalizeRunMetadataLabels(source)
	if err != nil {
		t.Fatal(err)
	}
	source["eval.id"] = "changed"
	if normalized["eval.id"] != "eval_01" || normalized.Clone() == nil {
		t.Fatalf("normalized labels alias source or lost explicit empty semantics: %v", normalized)
	}
}

func TestAllocationWorkerSessionModeCasesAreStrict(t *testing.T) {
	t.Parallel()

	type modeCase struct {
		Name  string          `json:"name"`
		Value json.RawMessage `json:"value"`
		Omit  bool            `json:"omit"`
	}
	var cases struct {
		Valid   []modeCase `json:"valid"`
		Invalid []modeCase `json:"invalid"`
	}
	path := contractstest.Path(t, "api", "testdata", "v1alpha1", "worker-session-mode-cases.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	var baseline map[string]json.RawMessage
	if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "allocation-spec.json"), &baseline); err != nil {
		t.Fatal(err)
	}
	decode := func(candidate modeCase) error {
		copy := make(map[string]json.RawMessage, len(baseline))
		for key, value := range baseline {
			copy[key] = append(json.RawMessage(nil), value...)
		}
		if candidate.Omit {
			delete(copy, "workerSessionMode")
		} else {
			copy["workerSessionMode"] = candidate.Value
		}
		encoded, marshalErr := json.Marshal(copy)
		if marshalErr != nil {
			return marshalErr
		}
		_, decodeErr := contracts.DecodeStrict[AllocationSpec](encoded)
		return decodeErr
	}
	for _, candidate := range cases.Valid {
		if err := decode(candidate); err != nil {
			t.Errorf("valid case %q failed: %v", candidate.Name, err)
		}
	}
	for _, candidate := range cases.Invalid {
		if err := decode(candidate); err == nil {
			t.Errorf("invalid case %q was accepted", candidate.Name)
		}
	}
}

func TestAllocationResolvedSkillsRejectsEveryManifestMismatch(t *testing.T) {
	t.Parallel()

	var baseline map[string]any
	if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "allocation-spec-skills.json"), &baseline); err != nil {
		t.Fatal(err)
	}
	tests := map[string]func(map[string]any){
		"missing": func(candidate map[string]any) { delete(candidate, "resolvedSkills") },
		"extra": func(candidate map[string]any) {
			skills := candidate["resolvedSkills"].([]any)
			candidate["resolvedSkills"] = append(skills, skills[1])
		},
		"duplicate": func(candidate map[string]any) {
			skills := candidate["resolvedSkills"].([]any)
			skills[1] = skills[0]
		},
		"unsorted": func(candidate map[string]any) {
			skills := candidate["resolvedSkills"].([]any)
			skills[0], skills[1] = skills[1], skills[0]
		},
		"versionless": func(candidate map[string]any) {
			skill := candidate["resolvedSkills"].([]any)[0].(map[string]any)
			delete(skill["artifact"].(map[string]any), "revision")
		},
		"wrong namespace": func(candidate map[string]any) {
			skill := candidate["resolvedSkills"].([]any)[0].(map[string]any)
			skill["artifact"].(map[string]any)["namespace"] = "other"
		},
		"name mismatch": func(candidate map[string]any) {
			skill := candidate["resolvedSkills"].([]any)[0].(map[string]any)
			skill["artifact"].(map[string]any)["name"] = "review"
		},
		"malformed digest": func(candidate map[string]any) {
			skill := candidate["resolvedSkills"].([]any)[0].(map[string]any)
			skill["packageDigest"] = "sha256:ABC"
		},
	}
	for name, mutate := range tests {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			encoded, _ := json.Marshal(baseline)
			var candidate map[string]any
			_ = json.Unmarshal(encoded, &candidate)
			mutate(candidate)
			encoded, _ = json.Marshal(candidate)
			if _, err := contracts.DecodeStrict[AllocationSpec](encoded); err == nil {
				t.Fatal("invalid resolvedSkills manifest was accepted")
			}
		})
	}
}

func TestAllocationSpecRequiresBoundedWorkerBudgets(t *testing.T) {
	t.Parallel()

	var baseline map[string]any
	if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "allocation-spec.json"), &baseline); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name, field string
		value       any
		remove      bool
	}{
		{"missing output tokens", "maxOutputTokens", nil, true},
		{"zero output tokens", "maxOutputTokens", float64(0), false},
		{"missing model calls", "maxModelCalls", nil, true},
		{"zero model calls", "maxModelCalls", float64(0), false},
		{"too many model calls", "maxModelCalls", float64(contracts.MaxWorkerModelCalls + 1), false},
		{"negative tool calls", "maxToolCalls", float64(-1), false},
		{"too many tool calls", "maxToolCalls", float64(contracts.MaxWorkerToolCalls + 1), false},
		{"missing total tokens", "maxTotalTokens", nil, true},
		{"too many total tokens", "maxTotalTokens", float64(contracts.MaxWorkerTotalTokens + 1), false},
		{"Planner-only Worker calls", "maxWorkerCalls", float64(1), false},
	} {
		for _, target := range []string{"AgentTemplate default", "effective allocation"} {
			t.Run(test.name+"/"+target, func(t *testing.T) {
				encoded, _ := json.Marshal(baseline)
				var candidate map[string]any
				_ = json.Unmarshal(encoded, &candidate)
				var policy map[string]any
				if target == "AgentTemplate default" {
					policy = candidate["agentTemplate"].(map[string]any)["modelPolicy"].(map[string]any)
				} else {
					policy = candidate["modelPolicy"].(map[string]any)
				}
				if test.remove {
					delete(policy, test.field)
				} else {
					policy[test.field] = test.value
				}
				encoded, _ = json.Marshal(candidate)
				if _, err := contracts.DecodeStrict[AllocationSpec](encoded); err == nil {
					t.Fatal("invalid Worker budget was accepted")
				}
			})
		}
	}
}

func TestAllocationSpecRequiresValidWorkerSummarizer(t *testing.T) {
	t.Parallel()

	var baseline map[string]any
	if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "allocation-spec-summarizer.json"), &baseline); err != nil {
		t.Fatal(err)
	}
	tests := map[string]func(map[string]any){
		"missing context ratio": func(summarizer map[string]any) {
			delete(summarizer, "contextWindowRatio")
		},
		"zero total threshold": func(summarizer map[string]any) {
			summarizer["cumulativeBudget"] = float64(0)
		},
		"total threshold reaches Worker hard bound": func(summarizer map[string]any) {
			summarizer["cumulativeBudget"] = float64(32768)
		},
		"zero context ratio": func(summarizer map[string]any) {
			summarizer["contextWindowRatio"] = float64(0)
		},
		"unit context ratio": func(summarizer map[string]any) {
			summarizer["contextWindowRatio"] = float64(1)
		},
		"missing output budget": func(summarizer map[string]any) {
			delete(summarizer["modelPolicy"].(map[string]any), "maxOutputTokens")
		},
		"missing summary context window": func(summarizer map[string]any) {
			delete(summarizer["modelPolicy"].(map[string]any), "contextWindowTokens")
		},
		"more than one model call": func(summarizer map[string]any) {
			summarizer["modelPolicy"].(map[string]any)["maxModelCalls"] = float64(2)
		},
		"tool budget": func(summarizer map[string]any) {
			summarizer["modelPolicy"].(map[string]any)["maxToolCalls"] = float64(1)
		},
		"Worker-call budget": func(summarizer map[string]any) {
			summarizer["modelPolicy"].(map[string]any)["maxWorkerCalls"] = float64(1)
		},
	}
	for name, mutate := range tests {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			encoded, _ := json.Marshal(baseline)
			var candidate map[string]any
			_ = json.Unmarshal(encoded, &candidate)
			summarizer := candidate["agentTemplate"].(map[string]any)["summarizer"].(map[string]any)
			mutate(summarizer)
			encoded, _ = json.Marshal(candidate)
			if _, err := contracts.DecodeStrict[AllocationSpec](encoded); err == nil {
				t.Fatal("invalid Worker summarizer was accepted")
			}
		})
	}
	encoded, _ := json.Marshal(baseline)
	var effectiveBound map[string]any
	_ = json.Unmarshal(encoded, &effectiveBound)
	effectiveBound["modelPolicy"].(map[string]any)["maxTotalTokens"] = float64(20000)
	encoded, _ = json.Marshal(effectiveBound)
	if _, err := contracts.DecodeStrict[AllocationSpec](encoded); err == nil {
		t.Fatal("summarizer cumulative budget at effective Worker hard bound was accepted")
	}

	encoded, _ = json.Marshal(baseline)
	var missingWorkerContext map[string]any
	_ = json.Unmarshal(encoded, &missingWorkerContext)
	delete(missingWorkerContext["modelPolicy"].(map[string]any), "contextWindowTokens")
	encoded, _ = json.Marshal(missingWorkerContext)
	if _, err := contracts.DecodeStrict[AllocationSpec](encoded); err == nil {
		t.Fatal("summarized effective Worker without context window was accepted")
	}
}

func TestPrivateAllocationSpecComposesValidatedSettingsAndProvenance(t *testing.T) {
	t.Parallel()

	active, err := contracts.DecodeStrict[AllocationSpec](contractstest.ReadFixture(t, "valid", "allocation-spec.json"))
	if err != nil {
		t.Fatal(err)
	}
	settings, err := contracts.DecodePrivateStrict[contracts.RuntimeSettings](
		contractstest.ReadFixture(t, "valid", "runtime-settings-combined.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	provenance, err := contracts.DecodePrivateStrict[contracts.ResolvedRuntimeConfigProvenance](
		contractstest.ReadFixture(t, "valid", "runtime-provenance.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	value := AllocationSpec{
		APIVersion: active.APIVersion, AllocationID: active.AllocationID, RunID: active.RunID,
		StageExecutionID: active.StageExecutionID, LogicalAgentName: active.LogicalAgentName,
		Namespace: active.Namespace, WorkerSessionMode: active.WorkerSessionMode,
		RunMetadataLabels: active.RunMetadataLabels.Clone(),
		LeaseExpiresAt:    active.LeaseExpiresAt,
		AgentTemplate:     active.AgentTemplate, ResolvedSkills: active.ResolvedSkills,
		ModelPolicy:     active.ModelPolicy,
		RuntimeSettings: settings, ResolvedRuntimeConfigProvenance: provenance,
	}
	canonical, err := contracts.MarshalPrivateCanonical(value)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := contracts.DecodePrivateStrict[AllocationSpec](canonical); err != nil {
		t.Fatalf("round-trip AllocationSpec: %v", err)
	}
	missingLabels := value
	missingLabels.RunMetadataLabels = nil
	missingCanonical, err := contracts.MarshalPrivateCanonical(missingLabels)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := contracts.DecodePrivateStrict[AllocationSpec](missingCanonical); err == nil {
		t.Fatal("AllocationSpec accepted missing runMetadataLabels")
	}
	if !bytes.Contains(canonical, []byte(`"resolvedSkills":[]`)) {
		t.Fatalf("private allocation omitted mandatory empty resolvedSkills: %s", canonical)
	}
	if !bytes.Contains(canonical, []byte(`"runMetadataLabels":{"eval.id":"eval_01","purpose":"eval"}`)) {
		t.Fatalf("private allocation omitted canonical Run metadata labels: %s", canonical)
	}
	formatted := fmt.Sprintf("%v %+v %#v", value.RuntimeSettings, value.RuntimeSettings, value.RuntimeSettings)
	for _, secret := range []string{"gateway-secret", "telemetry-secret", "proxy-bearer-secret", "caido-bearer-secret"} {
		if strings.Contains(formatted, secret) {
			t.Fatalf("formatted RuntimeSettings leaked %q", secret)
		}
		if !bytes.Contains(canonical, []byte(secret)) {
			t.Fatalf("private wire omitted required secret %q", secret)
		}
	}
}

func TestToolWorkerRejectsNullModelFields(t *testing.T) {
	for _, path := range [][]string{
		{"modelPolicy"}, {"agentTemplate", "modelPolicy"},
		{"agentTemplate", "instructions"}, {"agentTemplate", "summarizer"},
		{"runtimeSettings", "llmGatewayUrl"},
	} {
		t.Run(path[len(path)-1], func(t *testing.T) {
			var raw map[string]any
			if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "allocation-spec-tool.json"), &raw); err != nil {
				t.Fatal(err)
			}
			target := raw
			if len(path) > 1 {
				target = raw[path[0]].(map[string]any)
			}
			target[path[len(path)-1]] = nil
			data, err := json.Marshal(raw)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := contracts.DecodePrivateStrict[AllocationSpec](data); err == nil {
				t.Fatal("explicit null accepted as model absence")
			}
		})
	}
}

func TestPerformanceGoldenAllocationRequestIsOptional(t *testing.T) {
	var spec map[string]json.RawMessage
	if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "allocation-spec.json"), &spec); err != nil {
		t.Fatal(err)
	}
	spec["runtimeSettings"] = contractstest.ReadFixture(t, "valid", "runtime-settings-empty.json")
	spec["resolvedRuntimeConfigProvenance"] = contractstest.ReadFixture(t, "valid", "runtime-provenance.json")
	raw, _ := json.Marshal(spec)
	withoutMetrics, err := contracts.DecodePrivateStrict[AllocationSpec](raw)
	if err != nil || withoutMetrics.PerformanceMetrics != nil {
		t.Fatalf("allocation without metrics: %v", err)
	}
	canonical, err := contracts.MarshalPrivateCanonical(withoutMetrics)
	if err != nil || bytes.Contains(canonical, []byte("performanceMetrics")) {
		t.Fatalf("omitted request must remain disabled: %v", err)
	}
	spec["performanceMetrics"] = json.RawMessage(`{"version":1,"intervalSeconds":15}`)
	raw, _ = json.Marshal(spec)
	current, err := contracts.DecodePrivateStrict[AllocationSpec](raw)
	if err != nil || current.PerformanceMetrics == nil {
		t.Fatalf("new allocation: %v", err)
	}
	canonical, err = contracts.MarshalPrivateCanonical(current)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := contracts.DecodePrivateStrict[AllocationSpec](canonical); err != nil {
		t.Fatal(err)
	}
}
