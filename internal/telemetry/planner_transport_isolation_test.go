package telemetry

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	collectortracev1 "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	"google.golang.org/protobuf/proto"
)

// Keep this test serial: the factory clones the process default transport.
// The proxy control proves the trap can intercept even loopback destinations.
func TestPlannerOTLPInvocationsBypassAmbientProxyAndExportOnce(t *testing.T) {
	type capturedSink struct {
		calls atomic.Int64
		mu    sync.Mutex
		body  []byte
	}
	newSink := func() (*httptest.Server, *capturedSink) {
		captured := &capturedSink{}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			captured.calls.Add(1)
			body, err := io.ReadAll(io.LimitReader(r.Body, maxPlannerPendingBytes+1))
			if err != nil || len(body) > maxPlannerPendingBytes {
				http.Error(w, "bounded fixture body required", http.StatusBadRequest)
				return
			}
			captured.mu.Lock()
			captured.body = append([]byte(nil), body...)
			captured.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{}`)
		}))
		t.Cleanup(server.Close)
		return server, captured
	}
	left, leftCapture := newSink()
	right, rightCapture := newSink()
	var proxyCalls atomic.Int64
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		proxyCalls.Add(1)
		http.Error(w, "controlled ambient proxy", http.StatusBadGateway)
	}))
	t.Cleanup(proxy.Close)
	proxyURL, err := url.Parse(proxy.URL)
	if err != nil {
		t.Fatal(err)
	}
	original := http.DefaultTransport
	configured := original.(*http.Transport).Clone()
	configured.Proxy = func(*http.Request) (*url.URL, error) { return proxyURL, nil }
	http.DefaultTransport = configured
	t.Cleanup(func() {
		http.DefaultTransport = original
		configured.CloseIdleConnections()
	})

	controlTransport := configured.Clone()
	t.Cleanup(controlTransport.CloseIdleConnections)
	control := &http.Client{Transport: controlTransport, Timeout: time.Second}
	request, err := http.NewRequestWithContext(t.Context(), http.MethodGet, left.URL+"/proxy-control", nil)
	if err != nil {
		t.Fatal(err)
	}
	response, err := control.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusBadGateway || proxyCalls.Load() != 1 || leftCapture.calls.Load() != 0 {
		t.Fatal("controlled proxy did not intercept its control request")
	}

	registry, err := NewBuiltinPlannerAdapterRegistry()
	if err != nil {
		t.Fatal(err)
	}
	var traces [][]byte
	for _, invocation := range []struct {
		runID   string
		server  *httptest.Server
		capture *capturedSink
	}{
		{"run-left", left, leftCapture},
		{"run-right", right, rightCapture},
	} {
		adapter, err := registry.Create(PlannerAdapterOTLPHTTP, PlannerAdapterSettings{
			Endpoint: invocation.server.URL + "/v1/traces", FlushTimeout: time.Second,
			RunMetadataLabels: runlabels.RunMetadataLabels{},
			Resource:          PlannerResource{RunID: invocation.runID, StageExecutionID: "stage-" + invocation.runID, PlannerRef: "streamline@1"},
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(adapter.Close)
		adapter.Instrumentation().StartSpan(PlannerSpanModel, PlannerSpanAttributes{ModelAlias: "fixture-model"}).End("succeeded", PlannerSpanAttributes{})
		if result := adapter.Flush(t.Context()); !result.Attempted || !result.Succeeded || result.ErrorCode != "" {
			t.Fatalf("direct invocation export failed: %+v", result)
		}
		if result := adapter.Flush(t.Context()); result.Attempted || result.Succeeded || result.ErrorCode != "" {
			t.Fatalf("empty flush attempted to export again: %+v", result)
		}
		if invocation.capture.calls.Load() != 1 || proxyCalls.Load() != 1 {
			t.Fatalf("invocation export routes: sink=%d proxy=%d", invocation.capture.calls.Load(), proxyCalls.Load())
		}
		invocation.capture.mu.Lock()
		body := append([]byte(nil), invocation.capture.body...)
		invocation.capture.mu.Unlock()
		var decoded collectortracev1.ExportTraceServiceRequest
		if err := proto.Unmarshal(body, &decoded); err != nil {
			t.Fatal(err)
		}
		if len(decoded.ResourceSpans) != 1 || len(decoded.ResourceSpans[0].ScopeSpans) != 1 || len(decoded.ResourceSpans[0].ScopeSpans[0].Spans) != 1 {
			t.Fatal("invocation emitted extra resources or spans")
		}
		resource := decoded.ResourceSpans[0]
		if keyValueMap(resource.Resource.Attributes)["contractor.run.id"] != invocation.runID || resource.ScopeSpans[0].Spans[0].Name != string(PlannerSpanModel) {
			t.Fatal("invocations mixed identities or exporter vocabulary")
		}
		if len(resource.ScopeSpans[0].Spans[0].TraceId) != 16 {
			t.Fatal("invocation trace identity is absent or invalid")
		}
		traces = append(traces, append([]byte(nil), resource.ScopeSpans[0].Spans[0].TraceId...))
	}
	if leftCapture.calls.Load() != 1 || rightCapture.calls.Load() != 1 || bytes.Equal(traces[0], traces[1]) {
		t.Fatal("invocations shared export state or repeated a request")
	}
}
