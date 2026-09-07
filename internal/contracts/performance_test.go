package contracts

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

type performanceCase struct {
	Name  string          `json:"name"`
	Value json.RawMessage `json:"value"`
}
type performanceCases struct {
	ValidResources      []performanceCase `json:"validResources"`
	InvalidResources    []performanceCase `json:"invalidResources"`
	ValidRequests       []json.RawMessage `json:"validRequests"`
	InvalidRequests     []json.RawMessage `json:"invalidRequests"`
	ValidCapabilities   []json.RawMessage `json:"validCapabilities"`
	InvalidCapabilities []json.RawMessage `json:"invalidCapabilities"`
}

func readPerformanceCases(t *testing.T) performanceCases {
	t.Helper()
	raw, err := os.ReadFile("../../testdata/contracts/private-v2/performance-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases performanceCases
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	return cases
}

func TestPerformanceGoldenResources(t *testing.T) {
	cases := readPerformanceCases(t)
	for _, valid := range []bool{true, false} {
		entries := cases.InvalidResources
		if valid {
			entries = cases.ValidResources
		}
		for _, entry := range entries {
			t.Run(entry.Name, func(t *testing.T) {
				value, err := DecodePrivateV2Strict[RuntimeResources](entry.Value)
				if (err == nil) != valid {
					t.Fatalf("valid=%v, err=%v", valid, err)
				}
				if valid {
					encoded, err := MarshalPrivateV2Canonical(value)
					if err != nil {
						t.Fatal(err)
					}
					assertSemanticJSONEqual(t, entry.Value, encoded)
					wire, _ := json.Marshal(map[string]any{"complete": true, "adapters": map[string]any{}, "resources": value})
					report, err := DecodePrivateV2Strict[RuntimeReportV2](wire)
					if err != nil || report.Resources == nil || report.ResourcesError != nil {
						t.Fatalf("valid resources lost on report: %v", err)
					}
					var ordinary RuntimeReport
					if err := json.Unmarshal(wire, &ordinary); err != nil || ordinary.Resources == nil || ordinary.ResourcesError != nil {
						t.Fatalf("valid resources lost on ordinary report: %v", err)
					}
				} else if strings.Contains(err.Error(), "secret-canary") {
					t.Fatal("unsafe resource error")
				}
			})
		}
	}
	for _, number := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), -1} {
		value := RuntimeResources{Version: 1, Scope: "runtime_process", Status: ResourcePartial, CPUUserSeconds: &number}
		if value.Validate() == nil {
			t.Fatal("accepted invalid float")
		}
	}
}

func TestPerformanceGoldenRequestsAndCapabilities(t *testing.T) {
	cases := readPerformanceCases(t)
	for _, valid := range []bool{true, false} {
		requests, capabilities := cases.InvalidRequests, cases.InvalidCapabilities
		if valid {
			requests, capabilities = cases.ValidRequests, cases.ValidCapabilities
		}
		for _, raw := range requests {
			_, err := DecodePrivateV2Strict[PerformanceMetricsRequest](raw)
			if (err == nil) != valid {
				t.Fatalf("request validity %v: %v", valid, err)
			}
		}
		for _, raw := range capabilities {
			var registration map[string]json.RawMessage
			if err := json.Unmarshal(readPrivateFixture(t, "valid", "agent-registration.json"), &registration); err != nil {
				t.Fatal(err)
			}
			registration["supportedPerformanceMetricsVersions"] = raw
			encoded, _ := json.Marshal(registration)
			value, err := DecodePrivateV2Strict[AgentRegistrationV2](encoded)
			if (err == nil) != valid {
				t.Fatalf("capability %s validity %v: %v", raw, valid, err)
			}
			if valid && len(value.SupportedPerformanceMetricsVersions) != 0 {
				clone := NormalizeAgentRegistrationV2(value)
				clone.SupportedPerformanceMetricsVersions[0] = 2
				if value.SupportedPerformanceMetricsVersions[0] != 1 {
					t.Fatal("normalization aliases capability")
				}
				fingerprint, _ := AgentRegistrationFingerprintV2(value)
				value.SupportedPerformanceMetricsVersions = nil
				legacyFingerprint, _ := AgentRegistrationFingerprintV2(value)
				if fingerprint == legacyFingerprint {
					t.Fatal("capability missing from fingerprint")
				}
			}
		}
	}
}

func TestPerformanceCollectionPolicyPinsOnlyAllocationDecisions(t *testing.T) {
	for _, policy := range []PerformanceCollectionPolicy{
		PerformanceCollectionRequested,
		PerformanceCollectionDisabled,
		PerformanceCollectionUnsupported,
	} {
		if err := policy.ValidatePinned(); err != nil {
			t.Fatalf("pinned policy %q: %v", policy, err)
		}
		request := policy.Request()
		if (request != nil) != (policy == PerformanceCollectionRequested) {
			t.Fatalf("policy %q request = %+v", policy, request)
		}
		if request != nil && request.Validate() != nil {
			t.Fatalf("policy %q produced invalid request: %+v", policy, request)
		}
	}
	for _, policy := range []PerformanceCollectionPolicy{"", PerformanceCollectionLegacy, "unknown"} {
		if policy.ValidatePinned() == nil {
			t.Fatalf("read-only or unknown policy %q accepted as a pin", policy)
		}
	}
}

func TestPerformanceGoldenAllocationRequestIsOptional(t *testing.T) {
	var spec map[string]json.RawMessage
	if err := json.Unmarshal(readFixture(t, "valid", "allocation-spec.json"), &spec); err != nil {
		t.Fatal(err)
	}
	spec["runtimeSettings"] = readPrivateFixture(t, "valid", "runtime-settings-empty.json")
	spec["resolvedRuntimeConfigProvenance"] = readPrivateFixture(t, "valid", "runtime-provenance.json")
	raw, _ := json.Marshal(spec)
	legacy, err := DecodePrivateV2Strict[AllocationSpecV2](raw)
	if err != nil || legacy.PerformanceMetrics != nil {
		t.Fatalf("legacy allocation: %v", err)
	}
	canonical, err := MarshalPrivateV2Canonical(legacy)
	if err != nil || bytes.Contains(canonical, []byte("performanceMetrics")) {
		t.Fatalf("omitted request must remain disabled: %v", err)
	}
	spec["performanceMetrics"] = json.RawMessage(`{"version":1,"intervalSeconds":15}`)
	raw, _ = json.Marshal(spec)
	current, err := DecodePrivateV2Strict[AllocationSpecV2](raw)
	if err != nil || current.PerformanceMetrics == nil {
		t.Fatalf("new allocation: %v", err)
	}
	canonical, err = MarshalPrivateV2Canonical(current)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodePrivateV2Strict[AllocationSpecV2](canonical); err != nil {
		t.Fatal(err)
	}
}

func TestPerformanceGoldenInvalidResourcesDoNotPoisonFinalReports(t *testing.T) {
	base, err := DecodePrivateV2Strict[AllocationFinalResponse](readFixture(t, "valid", "allocation-final-response.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range readPerformanceCases(t).InvalidResources {
		t.Run(entry.Name, func(t *testing.T) {
			var envelope map[string]any
			if err := json.Unmarshal(readFixture(t, "valid", "allocation-final-response.json"), &envelope); err != nil {
				t.Fatal(err)
			}
			envelope["report"].(map[string]any)["runtime"].(map[string]any)["resources"] = entry.Value
			raw, _ := json.Marshal(envelope)
			value, err := DecodePrivateV2Strict[AllocationFinalResponse](raw)
			if err != nil {
				t.Fatalf("resource error poisoned final report: %v", err)
			}
			if value.Report.Runtime.Resources != nil || value.Report.Runtime.ResourcesError == nil || *value.Report.Runtime.ResourcesError != ResourceInvalidReport {
				t.Fatal("missing isolated resource diagnostic")
			}
			if !value.Report.Runtime.Complete || !reflect.DeepEqual(value.Report.Worker, base.Report.Worker) {
				t.Fatal("resource error changed execution truth")
			}
			encoded, _ := json.Marshal(value)
			if bytes.Contains(encoded, []byte("secret-canary")) {
				t.Fatal("invalid resource content escaped")
			}
			runtimeRaw, _ := json.Marshal(envelope["report"].(map[string]any)["runtime"])
			v2, err := DecodePrivateV2Strict[RuntimeReportV2](runtimeRaw)
			if err != nil || v2.Resources != nil || v2.ResourcesError == nil || !v2.Complete {
				t.Fatalf("v2 resource isolation: %v", err)
			}
		})
	}
}

func TestPerformanceResourceIsolationKeepsEnvelopeStrict(t *testing.T) {
	for _, raw := range []string{
		`{"complete":true,"adapters":{},"resources":{"version":1,"version":2}}`,
		`{"complete":true,"adapters":{},"resources":{"cpuUserSeconds":NaN}}`,
		`{"complete":true,"adapters":{},"resources":{},"unknown":"secret-canary"}`,
		`{"complete":true,"adapters":{},"resources":{}} {}`,
	} {
		if _, err := DecodePrivateV2Strict[RuntimeReportV2]([]byte(raw)); err == nil {
			t.Fatal("malformed envelope accepted")
		}
		var value RuntimeReport
		if err := json.Unmarshal([]byte(raw), &value); err == nil {
			t.Fatal("ordinary report decoder accepted malformed envelope")
		}
	}
	var report map[string]any
	if err := json.Unmarshal(readFixture(t, "valid", "allocation-final-response.json"), &report); err != nil {
		t.Fatal(err)
	}
	report["report"].(map[string]any)["runtime"].(map[string]any)["resources"] = map[string]string{"unknown": strings.Repeat("x", 1024*1024)}
	raw, _ := json.Marshal(report)
	if _, err := DecodePrivateV2Strict[AllocationFinalResponse](raw); err == nil {
		t.Fatal("resource isolation bypassed original report byte limit")
	}
}
