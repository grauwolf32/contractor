package telemetry

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	collectortracev1 "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonv1 "go.opentelemetry.io/proto/otlp/common/v1"
	"google.golang.org/protobuf/proto"
)

func TestPlannerOTLPHTTPExportsOnlyClosedSafeProjection(t *testing.T) {
	const headerSecret = "planner-otlp-secret-canary"
	const contentCanary = "prompt-response-objective-secret-canary"
	var mu sync.Mutex
	var payload []byte
	var requestHeaders http.Header
	collector := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(io.LimitReader(request.Body, maxPlannerPendingBytes+1))
		if err != nil {
			t.Errorf("read OTLP body: %v", err)
			response.WriteHeader(http.StatusBadRequest)
			return
		}
		mu.Lock()
		payload = append([]byte(nil), body...)
		requestHeaders = request.Header.Clone()
		mu.Unlock()
		response.Header().Set("Content-Type", "application/x-protobuf")
		response.WriteHeader(http.StatusOK)
	}))
	defer collector.Close()

	registry, err := NewBuiltinPlannerAdapterRegistry()
	if err != nil {
		t.Fatal(err)
	}
	metadataLabels := contracts.RunMetadataLabels{
		"purpose": "eval", "eval.id": "eval_01", "eval.leg": "a",
		"eval.case": "case_1", "eval.note": "left = right/β",
	}
	adapter, err := registry.Create(PlannerAdapterOTLPHTTP, PlannerAdapterSettings{
		Endpoint: collector.URL + "/v1/traces",
		Headers: map[string]contracts.SecretString{
			"x-contractor-token": contracts.NewSecretString(headerSecret),
		},
		FlushTimeout:      time.Second,
		RunMetadataLabels: metadataLabels,
		Resource: PlannerResource{
			RunID: "run-safe", StageExecutionID: "stage-safe", PlannerRef: "router@1",
			ModelAlias:     "qwen/qwen3.8-27b",
			ModelPolicyRef: "planner@1", LLMGatewayRef: "local-litellm@1",
			LLMCredentialID: "planner-credential", RuntimeCredentialID: "otel-planner",
			RuntimeConfigRefs:    []string{"debug@1", "default-config@1"},
			RuntimeConfigDigests: []string{"sha256:" + strings.Repeat("a", 64), "sha256:" + strings.Repeat("b", 64)},
			RunLabels:            []string{"debug"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	metadataLabels["eval.id"] = "changed-after-create"
	defer adapter.Close()
	instrumentation := adapter.Instrumentation()
	spans := []struct {
		name       PlannerSpanName
		attributes PlannerSpanAttributes
	}{
		{PlannerSpanInvocation, PlannerSpanAttributes{Operation: "planner.run"}},
		{PlannerSpanSession, PlannerSpanAttributes{Operation: "session.begin", SessionID: "session-safe"}},
		{PlannerSpanModel, PlannerSpanAttributes{Operation: "model.generate", ModelAlias: "qwen/qwen3.8-27b"}},
		{PlannerSpanWorker, PlannerSpanAttributes{Operation: "a2a.invoke", WorkerName: "reviewer", SubtaskID: "2"}},
		{PlannerSpanSubtask, PlannerSpanAttributes{Operation: "planner.plan_changed", SubtaskID: "2", PlanRevision: 3}},
		{PlannerSpanFinish, PlannerSpanAttributes{Operation: "finish", ToolName: "finish"}},
	}
	for _, candidate := range spans {
		span := instrumentation.StartSpan(candidate.name, candidate.attributes)
		span.End("succeeded", PlannerSpanAttributes{})
	}
	unknown := instrumentation.StartSpan(PlannerSpanName("unsafe."+contentCanary), PlannerSpanAttributes{
		ErrorCode: "unsafe error text with spaces",
	})
	unknown.End("not-an-outcome", PlannerSpanAttributes{})

	result := adapter.Flush(t.Context())
	if !result.Attempted || !result.Succeeded || result.ErrorCode != "" {
		t.Fatalf("flush result = %+v", result)
	}
	mu.Lock()
	gotPayload := append([]byte(nil), payload...)
	gotHeaders := requestHeaders.Clone()
	mu.Unlock()
	if gotHeaders.Get("x-contractor-token") != headerSecret ||
		gotHeaders.Get("Content-Type") != "application/x-protobuf" ||
		gotHeaders.Get("Accept") != "application/x-protobuf" {
		t.Fatalf("OTLP request headers = %v", gotHeaders)
	}
	for _, forbidden := range []string{headerSecret, collector.URL, contentCanary} {
		if bytes.Contains(gotPayload, []byte(forbidden)) {
			t.Fatalf("OTLP payload contains forbidden value %q", forbidden)
		}
	}
	var decoded collectortracev1.ExportTraceServiceRequest
	if err := proto.Unmarshal(gotPayload, &decoded); err != nil {
		t.Fatalf("decode OTLP request: %v", err)
	}
	if len(decoded.ResourceSpans) != 1 || len(decoded.ResourceSpans[0].ScopeSpans) != 1 {
		t.Fatalf("OTLP resource shape = %+v", decoded.ResourceSpans)
	}
	resource := keyValueMap(decoded.ResourceSpans[0].Resource.Attributes)
	for key, expected := range map[string]string{
		"service.name": "contractor-server-planner", "contractor.run.id": "run-safe",
		"contractor.stage_execution.id": "stage-safe", "contractor.planner.ref": "router@1",
		"contractor.model_policy.ref": "planner@1", "contractor.llm_gateway.ref": "local-litellm@1",
		"contractor.llm_credential.id": "planner-credential", "contractor.runtime_credential.id": "otel-planner",
	} {
		if resource[key] != expected {
			t.Errorf("resource %s = %#v, want %q", key, resource[key], expected)
		}
	}
	for key := range resource {
		if strings.HasPrefix(key, "contractor.run.label.") {
			t.Fatalf("Run metadata label %q was exported as a Resource attribute", key)
		}
	}
	gotSpans := decoded.ResourceSpans[0].ScopeSpans[0].Spans
	if len(gotSpans) != len(spans)+1 {
		t.Fatalf("span count = %d, want %d", len(gotSpans), len(spans)+1)
	}
	if gotSpans[len(gotSpans)-1].Name != string(PlannerSpanError) ||
		keyValueMap(gotSpans[len(gotSpans)-1].Attributes)["outcome"] != "failed" {
		t.Fatalf("unknown span was not reduced to safe error: %+v", gotSpans[len(gotSpans)-1])
	}
	invocationAttributes := keyValueMap(gotSpans[0].Attributes)
	for key, expected := range map[string]string{
		"contractor.run.label.purpose":   "eval",
		"contractor.run.label.eval.id":   "eval_01",
		"contractor.run.label.eval.leg":  "a",
		"contractor.run.label.eval.case": "case_1",
		"contractor.run.label.eval.note": "left = right/β",
	} {
		if invocationAttributes[key] != expected {
			t.Errorf("invocation attribute %s = %#v, want %q", key, invocationAttributes[key], expected)
		}
		for _, span := range gotSpans[1:] {
			if _, present := keyValueMap(span.Attributes)[key]; present {
				t.Errorf("non-invocation span %q contains Run metadata label %s", span.Name, key)
			}
		}
	}
}

func TestPlannerOTLPHTTPIsBoundedAndFlushFailureIsSupplementary(t *testing.T) {
	requestCount := 0
	collector := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requestCount++
		body, _ := io.ReadAll(request.Body)
		var decoded collectortracev1.ExportTraceServiceRequest
		if proto.Unmarshal(body, &decoded) != nil {
			response.WriteHeader(http.StatusBadRequest)
			return
		}
		if got := len(decoded.ResourceSpans[0].ScopeSpans[0].Spans); got != maxPlannerPendingSpans {
			t.Errorf("bounded span count = %d", got)
		}
		response.WriteHeader(http.StatusOK)
	}))
	defer collector.Close()
	adapter := newPlannerTestAdapter(t, collector.URL+"/v1/traces")
	defer adapter.Close()
	for index := 0; index < maxPlannerPendingSpans+32; index++ {
		span := adapter.Instrumentation().StartSpan(PlannerSpanModel, PlannerSpanAttributes{Operation: "model.generate"})
		span.End("succeeded", PlannerSpanAttributes{})
	}
	result := adapter.Flush(t.Context())
	if !result.Attempted || result.Succeeded || result.ErrorCode != "queue_overflow" || requestCount != 1 {
		t.Fatalf("bounded flush = %+v, requests=%d", result, requestCount)
	}

	failing := newPlannerTestAdapter(t, "http://127.0.0.1:1/v1/traces")
	defer failing.Close()
	span := failing.Instrumentation().StartSpan(PlannerSpanInvocation, PlannerSpanAttributes{Operation: "planner.run"})
	span.End("succeeded", PlannerSpanAttributes{})
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	result = failing.Flush(ctx)
	if !result.Attempted || result.Succeeded || result.ErrorCode == "" {
		t.Fatalf("failed delivery result = %+v", result)
	}
}

func TestPlannerTelemetryRequiresExplicitValidRunMetadataLabels(t *testing.T) {
	registry, err := NewBuiltinPlannerAdapterRegistry()
	if err != nil {
		t.Fatal(err)
	}
	for name, labels := range map[string]contracts.RunMetadataLabels{
		"missing":  nil,
		"reserved": {"contractor.internal": "value"},
	} {
		t.Run(name, func(t *testing.T) {
			adapter, createErr := registry.Create(PlannerAdapterOTLPHTTP, PlannerAdapterSettings{
				Endpoint: "https://telemetry.example/v1/traces",
				Headers:  map[string]contracts.SecretString{}, FlushTimeout: time.Second,
				Resource: PlannerResource{
					RunID: "run-test", StageExecutionID: "stage-test", PlannerRef: "streamline@1",
				},
				RunMetadataLabels: labels,
			})
			if createErr == nil || adapter != nil {
				t.Fatalf("invalid Planner metadata labels = (%v, %v)", adapter, createErr)
			}
		})
	}
}

func TestPlannerOTLPHTTPHangingBackendHonorsCallerDeadline(t *testing.T) {
	collector := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		<-time.After(time.Second)
	}))
	defer collector.Close()
	adapter := newPlannerTestAdapter(t, collector.URL+"/v1/traces")
	defer adapter.Close()
	span := adapter.Instrumentation().StartSpan(PlannerSpanInvocation, PlannerSpanAttributes{Operation: "planner.run"})
	span.End("succeeded", PlannerSpanAttributes{})
	ctx, cancel := context.WithTimeout(t.Context(), 25*time.Millisecond)
	defer cancel()
	started := time.Now()
	result := adapter.Flush(ctx)
	if elapsed := time.Since(started); elapsed > 250*time.Millisecond {
		t.Fatalf("bounded flush took %s", elapsed)
	}
	if !result.Attempted || result.Succeeded || result.ErrorCode != "flush_timeout" {
		t.Fatalf("hanging delivery result = %+v", result)
	}
}

func newPlannerTestAdapter(t *testing.T, endpoint string) PlannerTelemetry {
	t.Helper()
	registry, err := NewBuiltinPlannerAdapterRegistry()
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := registry.Create(PlannerAdapterOTLPHTTP, PlannerAdapterSettings{
		Endpoint: endpoint, Headers: map[string]contracts.SecretString{}, FlushTimeout: time.Second,
		RunMetadataLabels: contracts.RunMetadataLabels{},
		Resource:          PlannerResource{RunID: "run-test", StageExecutionID: "stage-test", PlannerRef: "passthrough@1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return adapter
}

func keyValueMap(values []*commonv1.KeyValue) map[string]any {
	result := make(map[string]any, len(values))
	for _, item := range values {
		switch value := item.Value.Value.(type) {
		case *commonv1.AnyValue_StringValue:
			result[item.Key] = value.StringValue
		case *commonv1.AnyValue_IntValue:
			result[item.Key] = value.IntValue
		case *commonv1.AnyValue_ArrayValue:
			items := make([]string, 0, len(value.ArrayValue.Values))
			for _, current := range value.ArrayValue.Values {
				items = append(items, current.GetStringValue())
			}
			result[item.Key] = items
		}
	}
	return result
}
