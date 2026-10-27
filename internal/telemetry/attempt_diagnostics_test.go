package telemetry

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedRuntimeDiagnosticDecimalsRemainExact(t *testing.T) {
	var cases struct {
		Cases []struct {
			Value    string `json:"value"`
			Accepted bool   `json:"accepted"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "public", "runtime-diagnostic-decimals.json"), &cases); err != nil {
		t.Fatal(err)
	}
	for _, item := range cases.Cases {
		t.Run(item.Value, func(t *testing.T) {
			data := fmt.Appendf(nil, `{"complete":true,"adapters":{"otlp-http@1":{"operations":%s,"failedOperations":0}}}`, item.Value)
			got := projectRuntimeDiagnostic("reviewer", data, false)
			valid := got.Status == "available"
			if valid != item.Accepted {
				t.Fatalf("status=%s, expected accepted=%t", got.Status, item.Accepted)
			}
			if valid && got.Adapters[0].Operations != item.Value {
				t.Fatalf("counter rounded: %s", got.Adapters[0].Operations)
			}
		})
	}
}

func TestRuntimeDiagnosticProjectionPreservesMissingAndPartialFacts(t *testing.T) {
	data := []byte(`{"complete":false,"stopReason":"failed https://provider.example/token","adapters":{"otlp-http@1":{"operations":1,"failedOperations":1,"flushAttempted":true,"flushSucceeded":false,"lastErrorCode":"flush_failed","droppedSpans":{"shutdown":18446744073709551615}}}}`)
	got := projectRuntimeDiagnostic("reviewer", data, false)
	if got.Status != "partial" || got.DurationMS != nil || got.Complete == nil || *got.Complete || got.StopReason == nil || strings.Contains(*got.StopReason, "provider.example") {
		t.Fatalf("unexpected projection: %+v", got)
	}
	if len(got.Adapters) != 1 || got.Adapters[0].FlushSucceeded == nil || *got.Adapters[0].FlushSucceeded || got.Adapters[0].DroppedSpans.Shutdown != "18446744073709551615" {
		t.Fatalf("delivery facts lost: %+v", got.Adapters)
	}
	if bad := projectRuntimeDiagnostic("reviewer", []byte(`{"complete":"yes","adapters":{}}`), false); bad.Status != "unavailable" || bad.Reason != "invalid_report" {
		t.Fatalf("bad report became valid: %+v", bad)
	}
}
