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
	value := map[string]any{"message": "HTTP /invoices " + secret + " short", "nested": []string{secret}, secret: "ordinary"}
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
	if input["message"] != "HTTP /invoices [REDACTED] [REDACTED]" || input["[REDACTED]"] != "ordinary" {
		t.Fatal("missing sanitized useful content")
	}
	if !strings.Contains(attrs["contractor.output"], `"truncated":true`) {
		t.Fatal("missing bounded truncation")
	}
}

func TestPlannerMaskedKeyCollisionDropsOnlyAffectedContent(t *testing.T) {
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
	CapturePlannerInput(span, func() any { return map[string]int{"one": 1, "two": 2} })
	CapturePlannerOutput(span, func() any { return "useful" })
	span.End("succeeded", PlannerSpanAttributes{})
	queued := adapter.(*otlpHTTPPlannerTelemetry).spans
	if len(queued) != 1 {
		t.Fatal("lost span")
	}
	for _, a := range queued[0].Attributes {
		if a.Key == "contractor.input" {
			t.Fatal("exported colliding input")
		}
	}
}
