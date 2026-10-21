package telemetry

import (
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
)

// The deployment smoke gate supplies a disposable real OTLP collector.
func TestPlannerOTLPCollectorProjection(t *testing.T) {
	endpoint := os.Getenv("CONTRACTOR_TEST_OTLP_ENDPOINT")
	if endpoint == "" {
		t.Skip("CONTRACTOR_TEST_OTLP_ENDPOINT is unset")
	}
	registry, _ := NewBuiltinPlannerAdapterRegistry()
	adapter, err := registry.Create(PlannerAdapterOTLPHTTP, PlannerAdapterSettings{
		Endpoint: endpoint, FlushTimeout: 10 * time.Second, CaptureContent: true,
		RunMetadataLabels: runlabels.RunMetadataLabels{},
		Resource:          PlannerResource{RunID: "go-smoke", StageExecutionID: "go-stage", PlannerRef: "streamline@1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	span := adapter.Instrumentation().StartSpan(PlannerSpanModel, PlannerSpanAttributes{ModelAlias: "smoke-model"})
	CapturePlannerInput(span, func() any {
		return map[string]any{
			"systemInstruction": "Go system instruction",
			"contents":          []any{map[string]any{"role": "user", "parts": []any{map[string]any{"text": "synthetic Go input"}}}},
		}
	})
	CapturePlannerOutput(span, func() any {
		return map[string]any{"role": "model", "parts": []any{map[string]any{"text": "synthetic Go output"}}}
	})
	span.End("succeeded", PlannerSpanAttributes{ModelUsage: &PlannerModelUsage{InputTokens: 2, OutputTokens: 3, CachedInputTokens: 1}})
	if result := adapter.Flush(t.Context()); !result.Succeeded {
		t.Fatalf("OTLP delivery: %+v", result)
	}
}
