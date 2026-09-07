package telemetry

import (
	"context"
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/contracts"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	collectortracev1 "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	"google.golang.org/protobuf/proto"
)

func TestPlannerTrustedContentOptIn(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "disabled", true: "enabled"}[enabled], func(t *testing.T) {
			var payload []byte
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				payload, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"name":"otel-ingestion-job","id":"1"}`))
			}))
			defer server.Close()
			registry, _ := NewBuiltinPlannerAdapterRegistry()
			adapter, err := registry.Create(PlannerAdapterOTLPHTTP, PlannerAdapterSettings{Endpoint: server.URL, FlushTimeout: time.Second, CaptureContent: enabled, RunMetadataLabels: contracts.RunMetadataLabels{}, Resource: PlannerResource{RunID: "run-test", StageExecutionID: "stage-test", PlannerRef: "router@1"}})
			if err != nil {
				t.Fatal(err)
			}
			defer adapter.Close()
			span := adapter.Instrumentation().StartSpan(PlannerSpanModel, PlannerSpanAttributes{ModelAlias: "test-model"})
			calls := 0
			CapturePlannerInput(span, func() any { calls++; return map[string]string{"prompt": "raw-secret-canary"} })
			CapturePlannerOutput(span, func() any { return strings.Repeat("界", MaxContentBytes) })
			span.End("succeeded", PlannerSpanAttributes{})
			if result := adapter.Flush(context.Background()); result.ErrorCode != "" {
				t.Fatalf("flush: %+v", result)
			}
			if !enabled {
				if calls != 0 || strings.Contains(string(payload), "raw-secret-canary") {
					t.Fatal("disabled content inspected/exported")
				}
				return
			}
			var decoded collectortracev1.ExportTraceServiceRequest
			if err := proto.Unmarshal(payload, &decoded); err != nil {
				t.Fatal(err)
			}
			attrs := map[string]string{}
			for _, a := range decoded.ResourceSpans[0].ScopeSpans[0].Spans[0].Attributes {
				attrs[a.Key] = a.Value.GetStringValue()
			}
			if !strings.Contains(attrs["langfuse.observation.input"], "raw-secret-canary") || attrs["langfuse.observation.type"] != "generation" {
				t.Fatal("missing captured input/generation")
			}
			output := attrs["langfuse.observation.output"]
			var value map[string]any
			if len(output) > MaxContentBytes || json.Unmarshal([]byte(output), &value) != nil || value["truncated"] != true {
				t.Fatal("invalid bounded output")
			}
		})
	}
}

func TestPlannerContentQueueRetainsLateSpansBeyondTwoMiB(t *testing.T) {
	var payload []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	registry, _ := NewBuiltinPlannerAdapterRegistry()
	adapter, err := registry.Create(PlannerAdapterOTLPHTTP, PlannerAdapterSettings{
		Endpoint: server.URL, FlushTimeout: time.Second, CaptureContent: true,
		RunMetadataLabels: contracts.RunMetadataLabels{},
		Resource:          PlannerResource{RunID: "run-test", StageExecutionID: "stage-test", PlannerRef: "streamline@1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	for range 16 {
		span := adapter.Instrumentation().StartSpan(PlannerSpanModel, PlannerSpanAttributes{ModelAlias: "test-model"})
		CapturePlannerInput(span, func() any { return strings.Repeat("x", 192*1024) })
		span.End("succeeded", PlannerSpanAttributes{})
	}
	last := adapter.Instrumentation().StartSpan(PlannerSpanFinish, PlannerSpanAttributes{})
	CapturePlannerOutput(last, func() any { return "late-finish-canary" })
	last.End("succeeded", PlannerSpanAttributes{})
	if result := adapter.Flush(t.Context()); !result.Succeeded {
		t.Fatalf("flush: %+v", result)
	}
	if len(payload) <= 2*1024*1024 || len(payload) > maxPlannerPendingBytes {
		t.Fatalf("unexpected payload size: %d", len(payload))
	}
	var decoded collectortracev1.ExportTraceServiceRequest
	if err := proto.Unmarshal(payload, &decoded); err != nil {
		t.Fatal(err)
	}
	spans := decoded.ResourceSpans[0].ScopeSpans[0].Spans
	if len(spans) != 17 || !strings.Contains(string(payload), "late-finish-canary") {
		t.Fatal("late span dropped from content-heavy trace")
	}
}

func TestPlannerOTLPJSONAndPartialAcknowledgements(t *testing.T) {
	for _, tc := range []struct {
		body string
		ok   bool
	}{
		{`{}`, true}, {`{"partialSuccess":{"rejectedSpans":"0"}}`, true},
		{`{"partialSuccess":{"rejectedSpans":"1"}}`, false},
		{`{"error":"failed"}`, false}, {`{"name":"otel-ingestion-job"}`, false},
	} {
		t.Run(tc.body, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()
			registry, _ := NewBuiltinPlannerAdapterRegistry()
			adapter, err := registry.Create(PlannerAdapterOTLPHTTP, PlannerAdapterSettings{Endpoint: server.URL, FlushTimeout: time.Second, RunMetadataLabels: contracts.RunMetadataLabels{}, Resource: PlannerResource{RunID: "run-test", StageExecutionID: "stage-test", PlannerRef: "router@1"}})
			if err != nil {
				t.Fatal(err)
			}
			defer adapter.Close()
			adapter.Instrumentation().StartSpan(PlannerSpanModel, PlannerSpanAttributes{}).End("succeeded", PlannerSpanAttributes{})
			if result := adapter.Flush(context.Background()); (result.ErrorCode == "") != tc.ok {
				t.Fatalf("ack: %+v", result)
			}
		})
	}
}
