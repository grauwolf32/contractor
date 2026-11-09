package telemetry

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	collectortracev1 "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	tracev1 "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/protobuf/proto"
)

func TestPlannerCredentialReplacementPrecedesEncodingAndQueueing(t *testing.T) {
	secret := "pw\"\\\nключ"
	var delivered []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		delivered, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	adapter, err := NewOTLPHTTPPlannerAdapterFactory().Create(PlannerAdapterSettings{
		Endpoint: server.URL, FlushTimeout: time.Second,
		RunMetadataLabels: runlabels.RunMetadataLabels{},
		Resource:          PlannerResource{RunID: "run-test", StageExecutionID: "stage-test", PlannerRef: "passthrough@1"},
		KnownCredentials:  []contracts.SecretString{contracts.NewSecretString(secret), contracts.NewSecretString("short")},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	exporter := adapter.(*otlpHTTPPlannerTelemetry)
	span := adapter.Instrumentation().StartSpan(PlannerSpanWorker, PlannerSpanAttributes{})
	value := map[string]any{"message": "HTTP /invoices " + secret + " short", "nested": []string{secret}, "name": "ordinary"}
	CapturePlannerInput(span, func() any { return value })
	CapturePlannerOutput(span, func() any { return strings.Repeat("x", MaxContentBytes-8) + secret })
	span.End("succeeded", PlannerSpanAttributes{})
	if value["message"] != "HTTP /invoices "+secret+" short" {
		t.Fatal("telemetry changed original content")
	}
	for _, queued := range exporter.spans {
		raw, e := proto.Marshal(queued)
		if e != nil {
			t.Fatal(e)
		}
		if strings.Contains(string(raw), secret) || strings.Contains(string(raw), "short") {
			t.Fatal("queue retained credential")
		}
	}
	if result := adapter.Flush(context.Background()); !result.Succeeded {
		t.Fatal("collector export failed")
	}
	var batch collectortracev1.ExportTraceServiceRequest
	if err = proto.Unmarshal(delivered, &batch); err != nil {
		t.Fatal(err)
	}
	attrs := map[string]string{}
	for _, a := range batch.ResourceSpans[0].ScopeSpans[0].Spans[0].Attributes {
		attrs[a.Key] = a.Value.GetStringValue()
	}
	var input map[string]any
	if json.Unmarshal([]byte(attrs["contractor.input"]), &input) != nil {
		t.Fatal("invalid JSON content")
	}
	if input["message"] != "HTTP /invoices [REDACTED] [REDACTED]" || input["name"] != "ordinary" {
		t.Fatal("missing sanitized useful content")
	}
	if !strings.Contains(attrs["contractor.output"], `"truncated":true`) {
		t.Fatal("missing bounded truncation")
	}
}

func TestPlannerDataKeysSurviveWithoutCollisions(t *testing.T) {
	adapter, err := NewOTLPHTTPPlannerAdapterFactory().Create(PlannerAdapterSettings{
		Endpoint: "http://127.0.0.1:1/v1/traces", FlushTimeout: time.Second,
		RunMetadataLabels: runlabels.RunMetadataLabels{},
		Resource:          PlannerResource{RunID: "run-test", StageExecutionID: "stage-test", PlannerRef: "passthrough@1"},
		KnownCredentials:  []contracts.SecretString{contracts.NewSecretString("one"), contracts.NewSecretString("two")},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	span := adapter.Instrumentation().StartSpan(PlannerSpanWorker, PlannerSpanAttributes{})
	CapturePlannerInput(span, func() any { return map[string]string{"one": "one", "two": "two"} })
	span.End("succeeded", PlannerSpanAttributes{})
	queued := adapter.(*otlpHTTPPlannerTelemetry).spans
	if len(queued) != 1 {
		t.Fatal("lost span")
	}
	for _, a := range queued[0].Attributes {
		if a.Key == "contractor.input" {
			var input map[string]string
			if json.Unmarshal([]byte(a.Value.GetStringValue()), &input) != nil ||
				input["one"] != "[REDACTED]" || input["two"] != "[REDACTED]" {
				t.Fatal("lost keys or unmasked values")
			}
			return
		}
	}
	t.Fatal("lost input")
}

func TestPlannerLargeContentIsProjectedBeforeByteBounds(t *testing.T) {
	secret := "credential-boundary-canary-" + strings.Repeat("q", 256)
	text := strings.Repeat("x", MaxContentBytes-128) + secret + strings.Repeat("x", 1024)
	value := map[string]any{
		"contents": []any{map[string]any{"role": "user", "parts": []any{map[string]any{"text": text}}}},
		"tools":    []any{map[string]any{"functionDeclarations": []any{map[string]any{"name": "lookup"}}}},
	}
	span, result := captureGenAISpan(t, PlannerSpanModel, "input", value, nil, map[string]any{}, secret)
	if !result.Succeeded {
		t.Fatal("export failed")
	}
	foundTools := false
	for _, attr := range span.Attributes {
		if strings.HasPrefix(attr.Key, "gen_ai.") && (strings.HasSuffix(attr.Key, "messages") || strings.HasSuffix(attr.Key, "definitions")) {
			raw := attr.Value.GetStringValue()
			if len(raw) > MaxContentBytes || !json.Valid([]byte(raw)) || strings.Contains(raw, "credential-boundary-canary-") {
				t.Fatal("invalid or unmasked bounded content")
			}
			if attr.Key == "gen_ai.tool.definitions" {
				foundTools = strings.Contains(raw, `"name":"lookup"`)
			}
		}
	}
	if !foundTools {
		t.Fatal("SDK truncation removed tool declarations")
	}
	if value["contents"].([]any)[0].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"] != text {
		t.Fatal("mutated original request")
	}
}

func TestPlannerCaptureFailureOmitsOnlyAffectedContent(t *testing.T) {
	deep := any("deep")
	for range 66 {
		deep = []any{deep}
	}
	for name, value := range map[string]any{"unsupported": make(chan int), "depth": deep, "values": make([]any, 100001)} {
		t.Run(name, func(t *testing.T) {
			adapter, err := NewOTLPHTTPPlannerAdapterFactory().Create(PlannerAdapterSettings{
				Endpoint: "http://127.0.0.1:1/v1/traces", FlushTimeout: time.Second,
				RunMetadataLabels: runlabels.RunMetadataLabels{},
				Resource:          PlannerResource{RunID: "run-test", StageExecutionID: "stage-test", PlannerRef: "passthrough@1"},
			})
			if err != nil {
				t.Fatal(err)
			}
			defer adapter.Close()
			span := adapter.Instrumentation().StartSpan(PlannerSpanModel, PlannerSpanAttributes{})
			CapturePlannerInput(span, func() any { return value })
			CapturePlannerOutput(span, func() any { return "useful output" })
			span.End("succeeded", PlannerSpanAttributes{})
			queued := adapter.(*otlpHTTPPlannerTelemetry).spans
			if len(queued) != 1 {
				t.Fatal("lost span")
			}
			foundOutput := false
			for _, attr := range queued[0].Attributes {
				if attr.Key == "gen_ai.input.messages" {
					t.Fatal("exported invalid input")
				}
				if attr.Key == "gen_ai.output.messages" {
					foundOutput = strings.Contains(attr.Value.GetStringValue(), "useful output")
				}
			}
			if !foundOutput {
				t.Fatal("failed input suppressed output")
			}
		})
	}
}

func TestPlannerStatusUsesOriginalProtocolOutcome(t *testing.T) {
	for _, outcome := range []string{"succeeded", "recovered", "failed", "unavailable", "invalid"} {
		t.Run(outcome, func(t *testing.T) {
			adapter, err := NewOTLPHTTPPlannerAdapterFactory().Create(PlannerAdapterSettings{
				Endpoint: "http://127.0.0.1:1/v1/traces", FlushTimeout: time.Second,
				RunMetadataLabels: runlabels.RunMetadataLabels{},
				Resource:          PlannerResource{RunID: "run-test", StageExecutionID: "stage-test", PlannerRef: "passthrough@1"},
				KnownCredentials:  []contracts.SecretString{contracts.NewSecretString("s"), contracts.NewSecretString(outcome)},
			})
			if err != nil {
				t.Fatal(err)
			}
			defer adapter.Close()
			span := adapter.Instrumentation().StartSpan(PlannerSpanModel, PlannerSpanAttributes{})
			span.End(outcome, PlannerSpanAttributes{})
			queued := adapter.(*otlpHTTPPlannerTelemetry).spans
			if len(queued) != 1 {
				t.Fatal("lost span")
			}
			want := outcome
			if outcome == "invalid" {
				want = "failed"
			}
			if keyValueMap(queued[0].Attributes)["outcome"] != want {
				t.Fatal("outcome changed")
			}
			status := tracev1.Status_STATUS_CODE_ERROR
			if outcome == "succeeded" || outcome == "recovered" {
				status = tracev1.Status_STATUS_CODE_OK
			}
			if queued[0].Status.Code != status {
				t.Fatal("status changed")
			}
		})
	}
}
