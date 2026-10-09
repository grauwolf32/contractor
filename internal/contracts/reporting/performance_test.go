package reporting

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

type performanceCase struct {
	Name  string          `json:"name"`
	Value json.RawMessage `json:"value"`
}

type performanceCases struct {
	ValidResources   []performanceCase `json:"validResources"`
	InvalidResources []performanceCase `json:"invalidResources"`
	ValidRequests    []json.RawMessage `json:"validRequests"`
	InvalidRequests  []json.RawMessage `json:"invalidRequests"`
}

func readPerformanceCases(t *testing.T) performanceCases {
	t.Helper()
	raw, err := os.ReadFile(contractstest.Path(t, "api", "testdata", "v1alpha1", "performance-cases.json"))
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
				value, err := contracts.DecodePrivateStrict[RuntimeResources](entry.Value)
				if (err == nil) != valid {
					t.Fatalf("valid=%v, err=%v", valid, err)
				}
				if valid {
					encoded, err := contracts.MarshalPrivateCanonical(value)
					if err != nil {
						t.Fatal(err)
					}
					contractstest.AssertSemanticJSONEqual(t, entry.Value, encoded)
					wire, _ := json.Marshal(map[string]any{"complete": true, "adapters": map[string]any{}, "resources": value})
					report, err := contracts.DecodePrivateStrict[RuntimeReport](wire)
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

func TestPerformanceGoldenRequests(t *testing.T) {
	cases := readPerformanceCases(t)
	for _, valid := range []bool{true, false} {
		requests := cases.InvalidRequests
		if valid {
			requests = cases.ValidRequests
		}
		for _, raw := range requests {
			_, err := contracts.DecodePrivateStrict[PerformanceMetricsRequest](raw)
			if (err == nil) != valid {
				t.Fatalf("request validity %v: %v", valid, err)
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
	for _, policy := range []PerformanceCollectionPolicy{"", "legacy", "unknown"} {
		if policy.ValidatePinned() == nil {
			t.Fatalf("read-only or unknown policy %q accepted as a pin", policy)
		}
	}
}

func TestPerformanceGoldenInvalidResourcesDoNotPoisonFinalReports(t *testing.T) {
	base, err := contracts.DecodePrivateStrict[AllocationFinalResponse](contractstest.ReadFixture(t, "valid", "allocation-final-response.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range readPerformanceCases(t).InvalidResources {
		t.Run(entry.Name, func(t *testing.T) {
			var envelope map[string]any
			if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "allocation-final-response.json"), &envelope); err != nil {
				t.Fatal(err)
			}
			envelope["report"].(map[string]any)["runtime"].(map[string]any)["resources"] = entry.Value
			raw, _ := json.Marshal(envelope)
			value, err := contracts.DecodePrivateStrict[AllocationFinalResponse](raw)
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
			runtime, err := contracts.DecodePrivateStrict[RuntimeReport](runtimeRaw)
			if err != nil || runtime.Resources != nil || runtime.ResourcesError == nil || !runtime.Complete {
				t.Fatalf("runtime resource isolation: %v", err)
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
		if _, err := contracts.DecodePrivateStrict[RuntimeReport]([]byte(raw)); err == nil {
			t.Fatal("malformed envelope accepted")
		}
		var value RuntimeReport
		if err := json.Unmarshal([]byte(raw), &value); err == nil {
			t.Fatal("ordinary report decoder accepted malformed envelope")
		}
	}
	var report map[string]any
	if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "allocation-final-response.json"), &report); err != nil {
		t.Fatal(err)
	}
	report["report"].(map[string]any)["runtime"].(map[string]any)["resources"] = map[string]string{"unknown": strings.Repeat("x", 1024*1024)}
	raw, _ := json.Marshal(report)
	if _, err := contracts.DecodePrivateStrict[AllocationFinalResponse](raw); err == nil {
		t.Fatal("resource isolation bypassed original report byte limit")
	}
}
