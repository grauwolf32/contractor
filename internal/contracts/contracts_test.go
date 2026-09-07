package contracts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestValidGoldenFixtures(t *testing.T) {
	t.Parallel()

	cases := map[string]func([]byte) ([]byte, error){
		"agent-state-snapshot.json":                    roundTrip[AgentStateSnapshot],
		"agent-registration.json":                      roundTrip[AgentRegistration],
		"agent-registration-response.json":             roundTrip[AgentRegistrationResponse],
		"agent-heartbeat.json":                         roundTrip[AgentHeartbeat],
		"heartbeat-response.json":                      roundTrip[HeartbeatResponse],
		"llm-gateway-config.json":                      roundTrip[ResolvedLLMGatewayConfig],
		"allocation-spec.json":                         roundTrip[AllocationSpec],
		"allocation-spec-summarizer.json":              roundTrip[AllocationSpec],
		"allocation-spec-summarizer-instructions.json": roundTrip[AllocationSpec],
		"allocation-spec-skills.json":                  roundTrip[AllocationSpec],
		"allocation-final-response.json":               roundTrip[AllocationFinalResponse],
		"finalize-allocation.json":                     roundTrip[FinalizeAllocationRequest],
		"abort-allocation.json":                        roundTrip[AbortAllocationRequest],
		"release-allocation.json":                      roundTrip[ReleaseAllocationRequest],
		"artifact-read-result.json":                    roundTrip[ArtifactReadResult],
		"artifact-list-result.json":                    roundTrip[ArtifactListResult],
		"stage-content-request.json":                   roundTrip[StageContentRequest],
		"stage-content-result-success.json":            roundTrip[StageContentResult],
		"stage-content-result-failure.json":            roundTrip[StageContentResult],
		"worker-completion-success.json":               roundTrip[WorkerCompletion],
		"worker-completion-failure.json":               roundTrip[WorkerCompletion],
		"worker-completion-empty-observations.json":    roundTrip[WorkerCompletion],
	}

	for filename, decode := range cases {
		filename, decode := filename, decode
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			input := readFixture(t, "valid", filename)
			output, err := decode(input)
			if err != nil {
				t.Fatalf("decode valid fixture: %v", err)
			}
			assertSemanticJSONEqual(t, input, output)
		})
	}
}

func TestInvalidGoldenFixtures(t *testing.T) {
	t.Parallel()

	cases := map[string]func([]byte) error{
		"registration-missing-labels.json":        reject[AgentRegistration],
		"registration-missing-adapters.json":      reject[AgentRegistration],
		"allocation-spec-missing-provenance.json": reject[AllocationSpec],

		"agent-state-zero-revision.json":                      reject[AgentStateSnapshot],
		"agent-state-echoed-allocation.json":                  reject[AgentStateSnapshot],
		"agent-registration-idle-with-allocation.json":        reject[AgentRegistration],
		"agent-registration-oversized-software-version.json":  reject[AgentRegistration],
		"agent-heartbeat-missing-allocation.json":             reject[AgentHeartbeat],
		"heartbeat-response-unknown-action.json":              reject[HeartbeatResponse],
		"llm-gateway-config-secret-field.json":                reject[ResolvedLLMGatewayConfig],
		"allocation-spec-bad-api-version.json":                reject[AllocationSpec],
		"allocation-spec-resolved-skill-versionless.json":     reject[AllocationSpec],
		"stage-content-request-unknown-field.json":            reject[StageContentRequest],
		"stage-content-request-versioned-result-binding.json": reject[StageContentRequest],
		"stage-content-result-unversioned-artifact.json":      reject[StageContentResult],
		"stage-content-result-success-with-error.json":        reject[StageContentResult],
		"worker-completion-both-variants.json":                reject[WorkerCompletion],
		"worker-completion-no-variant.json":                   reject[WorkerCompletion],
		"artifact-read-result-unversioned.json":               reject[ArtifactReadResult],
	}

	for filename, decode := range cases {
		filename, decode := filename, decode
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			if err := decode(readFixture(t, "invalid", filename)); err == nil {
				t.Fatal("invalid fixture was accepted")
			}
		})
	}
}

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
	path := filepath.Join("..", "..", "api", "testdata", "v1alpha1", "run-metadata-label-cases.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	var baseline map[string]json.RawMessage
	if err := json.Unmarshal(readFixture(t, "valid", "allocation-spec.json"), &baseline); err != nil {
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
		_, decodeErr := DecodeStrict[AllocationSpec](encoded)
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
	normalized, err := NormalizeRunMetadataLabels(source)
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
	path := filepath.Join("..", "..", "api", "testdata", "v1alpha1", "worker-session-mode-cases.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	var baseline map[string]json.RawMessage
	if err := json.Unmarshal(readFixture(t, "valid", "allocation-spec.json"), &baseline); err != nil {
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
		_, decodeErr := DecodeStrict[AllocationSpec](encoded)
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
	if err := json.Unmarshal(readFixture(t, "valid", "allocation-spec-skills.json"), &baseline); err != nil {
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
			if _, err := DecodeStrict[AllocationSpec](encoded); err == nil {
				t.Fatal("invalid resolvedSkills manifest was accepted")
			}
		})
	}
}

func TestRuntimeSettingsRedactFormattingButSerializeOnWire(t *testing.T) {
	t.Parallel()

	const token = "recognizable-secret-token"
	secret := NewSecretString(token)
	settings := RuntimeSettings{
		LLMGatewayURL:         "https://gateway.example/v1",
		LLMGatewayToken:       &secret,
		ArtifactAPIURL:        "https://server.example/private/v1",
		RequestTimeoutSeconds: 30,
	}
	formatted := fmt.Sprintf("%v %+v %#v %s", settings, settings, settings, settings.LLMGatewayToken)
	if strings.Contains(formatted, token) {
		t.Fatalf("formatted RuntimeSettings leaked token: %s", formatted)
	}
	wire, err := json.Marshal(settings)
	if err != nil {
		t.Fatalf("marshal RuntimeSettings: %v", err)
	}
	if !strings.Contains(string(wire), token) {
		t.Fatalf("wire JSON did not contain required private token: %s", wire)
	}
}

func TestRuntimeSettingsAllowsExplicitUnauthenticatedGateway(t *testing.T) {
	t.Parallel()
	settings := RuntimeSettings{
		LLMGatewayURL:         "http://127.0.0.1:4000/v1",
		LLMGatewayToken:       nil,
		ArtifactAPIURL:        "https://server.example/private/v1",
		RequestTimeoutSeconds: 30,
	}
	if err := settings.Validate(); err != nil {
		t.Fatalf("unauthenticated RuntimeSettings were rejected: %v", err)
	}
}

func TestAllocationRuntimeAdapterMetricsAreTypedAndBounded(t *testing.T) {
	t.Parallel()

	value, err := DecodeStrict[AllocationFinalResponse](
		readFixture(t, "valid", "allocation-final-response.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	value.Report.Runtime.Adapters = map[RuntimeAdapterRef]RuntimeAdapterMetrics{
		RuntimeAdapterOTLPHTTP: {Operations: 2, FailedOperations: 1},
	}
	if err := value.Validate(); err != nil {
		t.Fatalf("valid adapter metrics were rejected: %v", err)
	}
	value.Report.Runtime.Adapters[RuntimeAdapterOTLPHTTP] = RuntimeAdapterMetrics{
		Operations: 1, FailedOperations: 2,
	}
	if err := value.Validate(); err == nil {
		t.Fatal("adapter metrics with failures above operations were accepted")
	}
	value.Report.Runtime.Adapters = map[RuntimeAdapterRef]RuntimeAdapterMetrics{
		"unknown@1": {Operations: 1},
	}
	if err := value.Validate(); err == nil {
		t.Fatal("unknown Runtime adapter metric key was accepted")
	}
}

func TestExecutionReportValidatesWorkerSummarizerMetrics(t *testing.T) {
	t.Parallel()

	valid := WorkerSummarizerMetrics{
		Attempts: 2, Succeeded: 1, Failed: 1, ModelCalls: 2,
		InputTokens: 10, OutputTokens: 4, TotalTokens: 14,
		FailureCodes: map[string]uint64{"gateway_unavailable": 1},
	}
	report := ExecutionReport{
		ReportID: "worker-summary", Complete: true,
		Metrics:   ExecutionMetrics{Tools: map[string]ToolMetrics{}, Summarizer: &valid},
		ToolCalls: []ToolCallRecord{}, Errors: []ExecutionError{},
	}
	if err := report.Validate(); err != nil {
		t.Fatalf("valid Worker summarizer metrics were rejected: %v", err)
	}

	tests := map[string]func(*WorkerSummarizerMetrics){
		"zero attempts":        func(value *WorkerSummarizerMetrics) { value.Attempts = 0 },
		"terminal mismatch":    func(value *WorkerSummarizerMetrics) { value.Failed = 0 },
		"too many calls":       func(value *WorkerSummarizerMetrics) { value.ModelCalls = 3 },
		"missing usage excess": func(value *WorkerSummarizerMetrics) { value.TokenUsageUnavailable = 3 },
		"nil failure map":      func(value *WorkerSummarizerMetrics) { value.FailureCodes = nil },
		"failure mismatch":     func(value *WorkerSummarizerMetrics) { value.FailureCodes = map[string]uint64{} },
		"invalid failure code": func(value *WorkerSummarizerMetrics) {
			value.FailureCodes = map[string]uint64{"bad-code": 1}
		},
	}
	for name, mutate := range tests {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			candidate := valid
			candidate.FailureCodes = map[string]uint64{"gateway_unavailable": 1}
			mutate(&candidate)
			report.Metrics.Summarizer = &candidate
			if err := report.Validate(); err == nil {
				t.Fatal("inconsistent Worker summarizer metrics were accepted")
			}
		})
	}
}

func TestMalformedRuntimeAdapterMetricsBecomeIncompleteInsteadOfBlockingReport(t *testing.T) {
	t.Parallel()

	var wire map[string]any
	if err := json.Unmarshal(readFixture(t, "valid", "allocation-final-response.json"), &wire); err != nil {
		t.Fatal(err)
	}
	runtime := wire["report"].(map[string]any)["runtime"].(map[string]any)
	runtime["adapters"] = map[string]any{
		"otlp-http@1": map[string]any{
			"operations": 1, "failedOperations": 2,
		},
		"unknown@1": map[string]any{
			"operations": -1, "failedOperations": 0, "secretField": "must-not-reflect",
		},
	}
	encoded, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	value, err := DecodeStrict[AllocationFinalResponse](encoded)
	if err != nil {
		t.Fatalf("semantic final report was blocked by optional adapter metrics: %v", err)
	}
	if value.Report.Runtime.Complete || len(value.Report.Runtime.Adapters) != 0 {
		t.Fatalf("malformed metrics were retained: %+v", value.Report.Runtime)
	}
}

func TestDecodeStrictRejectsTrailingJSON(t *testing.T) {
	t.Parallel()

	input := append(readFixture(t, "valid", "stage-content-request.json"), []byte(" {}")...)
	if _, err := DecodeStrict[StageContentRequest](input); err == nil {
		t.Fatal("trailing JSON value was accepted")
	}
}

func TestStageContentRequestDoesNotExposeSessionControls(t *testing.T) {
	t.Parallel()

	typeOfRequest := reflect.TypeOf(StageContentRequest{})
	for _, field := range []string{"Session", "SessionID", "SessionMode", "WorkerSessionMode"} {
		if _, ok := typeOfRequest.FieldByName(field); ok {
			t.Fatalf("StageContentRequest unexpectedly exposes %s", field)
		}
	}
}

func TestAllocationSpecRequiresBoundedWorkerBudgets(t *testing.T) {
	t.Parallel()

	var baseline map[string]any
	if err := json.Unmarshal(readFixture(t, "valid", "allocation-spec.json"), &baseline); err != nil {
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
		{"too many model calls", "maxModelCalls", float64(MaxWorkerModelCalls + 1), false},
		{"negative tool calls", "maxToolCalls", float64(-1), false},
		{"too many tool calls", "maxToolCalls", float64(MaxWorkerToolCalls + 1), false},
		{"missing total tokens", "maxTotalTokens", nil, true},
		{"too many total tokens", "maxTotalTokens", float64(MaxWorkerTotalTokens + 1), false},
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
				if _, err := DecodeStrict[AllocationSpec](encoded); err == nil {
					t.Fatal("invalid Worker budget was accepted")
				}
			})
		}
	}
}

func TestAllocationSpecRequiresValidWorkerSummarizer(t *testing.T) {
	t.Parallel()

	var baseline map[string]any
	if err := json.Unmarshal(readFixture(t, "valid", "allocation-spec-summarizer.json"), &baseline); err != nil {
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
			if _, err := DecodeStrict[AllocationSpec](encoded); err == nil {
				t.Fatal("invalid Worker summarizer was accepted")
			}
		})
	}
	encoded, _ := json.Marshal(baseline)
	var effectiveBound map[string]any
	_ = json.Unmarshal(encoded, &effectiveBound)
	effectiveBound["modelPolicy"].(map[string]any)["maxTotalTokens"] = float64(20000)
	encoded, _ = json.Marshal(effectiveBound)
	if _, err := DecodeStrict[AllocationSpec](encoded); err == nil {
		t.Fatal("summarizer cumulative budget at effective Worker hard bound was accepted")
	}

	encoded, _ = json.Marshal(baseline)
	var missingWorkerContext map[string]any
	_ = json.Unmarshal(encoded, &missingWorkerContext)
	delete(missingWorkerContext["modelPolicy"].(map[string]any), "contextWindowTokens")
	encoded, _ = json.Marshal(missingWorkerContext)
	if _, err := DecodeStrict[AllocationSpec](encoded); err == nil {
		t.Fatal("summarized effective Worker without context window was accepted")
	}
}

func TestSchemaFilesContainJSONObjects(t *testing.T) {
	t.Parallel()

	matches, err := filepath.Glob(filepath.Join("..", "..", "api", "v1alpha1", "*.schema.json"))
	if err != nil {
		t.Fatalf("glob schemas: %v", err)
	}
	if len(matches) < 5 {
		t.Fatalf("found %d schema files, want at least 5", len(matches))
	}
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		var value map[string]any
		if err := json.Unmarshal(data, &value); err != nil {
			t.Fatalf("parse %s: %v", path, err)
		}
		if value["$schema"] == nil {
			t.Fatalf("%s has no $schema", path)
		}
	}
}

func roundTrip[T Validatable](data []byte) ([]byte, error) {
	value, err := DecodeStrict[T](data)
	if err != nil {
		return nil, err
	}
	return json.Marshal(value)
}

func reject[T Validatable](data []byte) error {
	_, err := DecodeStrict[T](data)
	return err
}

func readFixture(t *testing.T, kind, filename string) []byte {
	t.Helper()
	path := filepath.Join("..", "..", "api", "testdata", "v1alpha1", kind, filename)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", path, err)
	}
	return data
}

func assertSemanticJSONEqual(t *testing.T, left, right []byte) {
	t.Helper()
	var leftValue, rightValue any
	if err := json.Unmarshal(left, &leftValue); err != nil {
		t.Fatalf("decode left JSON: %v", err)
	}
	if err := json.Unmarshal(right, &rightValue); err != nil {
		t.Fatalf("decode right JSON: %v", err)
	}
	if !reflect.DeepEqual(leftValue, rightValue) {
		t.Fatalf("semantic JSON differs\nleft:  %s\nright: %s", left, right)
	}
}
