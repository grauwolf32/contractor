package telemetry

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	collectortracev1 "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	tracev1 "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/protobuf/proto"
)

func captureGenAISpan(t *testing.T, name PlannerSpanName, field string, value any, usage *PlannerModelUsage, ack any) (*tracev1.Span, PlannerExportResult) {
	t.Helper()
	var payload []byte
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ack)
	}))
	t.Cleanup(collector.Close)
	registry, _ := NewBuiltinPlannerAdapterRegistry()
	adapter, err := registry.Create(PlannerAdapterOTLPHTTP, PlannerAdapterSettings{Endpoint: collector.URL, FlushTimeout: time.Second, CaptureContent: true, RunMetadataLabels: runlabels.RunMetadataLabels{}, Resource: PlannerResource{RunID: "fixture-run", StageExecutionID: "fixture-stage", PlannerRef: "streamline@1"}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(adapter.Close)
	span := adapter.Instrumentation().StartSpan(name, PlannerSpanAttributes{ModelAlias: "fixture-model"})
	if field == "input" {
		CapturePlannerInput(span, func() any { return value })
	}
	if field == "output" {
		CapturePlannerOutput(span, func() any { return value })
	}
	span.End("succeeded", PlannerSpanAttributes{ModelUsage: usage})
	result := adapter.Flush(t.Context())
	var request collectortracev1.ExportTraceServiceRequest
	if err := proto.Unmarshal(payload, &request); err != nil {
		t.Fatal(err)
	}
	return request.ResourceSpans[0].ScopeSpans[0].Spans[0], result
}

func TestSharedGenAITelemetry(t *testing.T) {
	var corpus struct {
		Content []struct {
			Name     string
			Model    bool
			Field    string
			Value    any
			Expected map[string]any
		}
		Usage []struct {
			Name     string
			Input    *int64
			Output   *int64
			Cached   *int64
			Expected map[string]any
		}
		Acknowledgements []struct {
			Body     any
			Accepted bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "telemetry-genai-cases.json"), &corpus); err != nil {
		t.Fatal(err)
	}
	for _, tc := range corpus.Content {
		t.Run(tc.Name, func(t *testing.T) {
			name := PlannerSpanModel
			if !tc.Model {
				name = PlannerSpanWorker
			}
			span, result := captureGenAISpan(t, name, tc.Field, tc.Value, nil, map[string]any{})
			if !result.Succeeded {
				t.Fatalf("delivery: %+v", result)
			}
			actual := map[string]any{}
			for _, attr := range span.Attributes {
				if strings.HasPrefix(attr.Key, "langfuse.") || strings.HasPrefix(attr.Key, "openinference.") {
					t.Fatal("vendor attribute emitted")
				}
				if attr.Key == "contractor."+tc.Field || strings.HasPrefix(attr.Key, "gen_ai.input.") || strings.HasPrefix(attr.Key, "gen_ai.output.") || attr.Key == "gen_ai.system_instructions" || attr.Key == "gen_ai.tool.definitions" {
					var value any
					if err := json.Unmarshal([]byte(attr.Value.GetStringValue()), &value); err != nil {
						t.Fatal(err)
					}
					actual[attr.Key] = value
				}
			}
			if !reflect.DeepEqual(actual, tc.Expected) {
				t.Fatalf("attributes: %#v, expected %#v", actual, tc.Expected)
			}
		})
	}
	for _, tc := range corpus.Usage {
		t.Run("usage-"+tc.Name, func(t *testing.T) {
			var usage *PlannerModelUsage
			if tc.Input != nil {
				usage = &PlannerModelUsage{*tc.Input, *tc.Output, *tc.Cached}
			}
			span, _ := captureGenAISpan(t, PlannerSpanModel, "", nil, usage, map[string]any{})
			actual := map[string]any{}
			for _, attr := range span.Attributes {
				if strings.HasPrefix(attr.Key, "gen_ai.usage.") {
					actual[attr.Key] = float64(attr.Value.GetIntValue())
				}
			}
			if !reflect.DeepEqual(actual, tc.Expected) {
				t.Fatalf("usage: %#v, expected %#v", actual, tc.Expected)
			}
		})
	}
	for _, tc := range corpus.Acknowledgements {
		_, result := captureGenAISpan(t, PlannerSpanModel, "", nil, nil, tc.Body)
		if result.Succeeded != tc.Accepted {
			t.Fatalf("acknowledgement %#v: %+v", tc.Body, result)
		}
	}
}
