package telemetry

import (
	"bytes"
	"fmt"
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

func TestPlannerOTLPEverySpanCarriesOnlyKnownRunIdentities(t *testing.T) {
	const secret = "identity-secret-canary"
	registry, err := NewBuiltinPlannerAdapterRegistry()
	if err != nil {
		t.Fatal(err)
	}
	// Reuse one factory across Runs, including an unbound Run after an Audit.
	for _, test := range []struct {
		name, projectID, auditID string
	}{
		{"audit", "project-1", "audit-1"},
		{"project-only", "project-2", ""},
		{"unbound", "", ""},
		{"audit-only", "", "audit-2"},
		{"maximum-project", "P" + strings.Repeat(".", contracts.MaxResourceIDLength-1), "audit-3"},
		{"redacted", secret, secret},
	} {
		t.Run(test.name, func(t *testing.T) {
			payloads := make(chan []byte, 1)
			collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(io.LimitReader(r.Body, maxPlannerPendingBytes+1))
				if err != nil {
					t.Error(err)
				}
				payloads <- body
				w.WriteHeader(http.StatusOK)
			}))
			defer collector.Close()
			labels := runlabels.RunMetadataLabels{"project.id": "descriptive-project"}
			if test.auditID != "" {
				labels["audit.id"] = test.auditID
			}
			for len(labels) < runlabels.MaxRunMetadataLabels {
				labels[fmt.Sprintf("label.%d", len(labels))] = "value"
			}
			adapter, err := registry.Create(PlannerAdapterOTLPHTTP, PlannerAdapterSettings{
				Endpoint: collector.URL, FlushTimeout: time.Second,
				ProjectID: test.projectID, RunMetadataLabels: labels,
				Resource:         PlannerResource{RunID: "run-identity", StageExecutionID: "stage-identity", PlannerRef: "router@1"},
				KnownCredentials: []contracts.SecretString{contracts.NewSecretString(secret)},
			})
			if err != nil {
				t.Fatal(err)
			}
			defer adapter.Close()
			names := []PlannerSpanName{PlannerSpanInvocation, PlannerSpanSession, PlannerSpanModel, PlannerSpanWorker, PlannerSpanSubtask, PlannerSpanFinish, PlannerSpanError, "unknown-span"}
			for _, name := range names {
				adapter.Instrumentation().StartSpan(name, PlannerSpanAttributes{}).End("succeeded", PlannerSpanAttributes{})
			}
			if result := adapter.Flush(t.Context()); !result.Succeeded {
				t.Fatalf("flush = %+v", result)
			}
			payload := <-payloads
			if bytes.Contains(payload, []byte(secret)) || bytes.Contains(payload, []byte("descriptive-project")) {
				t.Fatal("telemetry exposed a secret or used descriptive Project membership")
			}
			var decoded collectortracev1.ExportTraceServiceRequest
			if err := proto.Unmarshal(payload, &decoded); err != nil {
				t.Fatal(err)
			}
			resource := decoded.ResourceSpans[0]
			for _, attribute := range resource.Resource.Attributes {
				if strings.HasPrefix(attribute.Key, "contractor.run.label.") {
					t.Fatalf("identity exported as a Resource attribute: %s", attribute.Key)
				}
			}
			spans := resource.ScopeSpans[0].Spans
			if len(spans) != len(names) {
				t.Fatalf("got %d spans, want %d", len(spans), len(names))
			}
			for _, span := range spans {
				attributes := keyValueMap(span.Attributes)
				for key, want := range map[string]string{"project.id": test.projectID, "audit.id": test.auditID} {
					if want == secret {
						want = ""
					}
					got, exists := attributes["contractor.run.label."+key]
					if want == "" && exists || want != "" && got != want {
						t.Errorf("%s %s = (%v, %t), want %q", span.Name, key, got, exists, want)
					}
				}
				if len(span.Attributes) > 64 || len(attributes) != len(span.Attributes) {
					t.Fatalf("span %s exceeds the attribute bound or has duplicate keys", span.Name)
				}
				_, genericLabel := attributes["contractor.run.label.label.2"]
				if genericLabel != (span.Name == string(PlannerSpanInvocation)) {
					t.Errorf("generic label projection changed on %s", span.Name)
				}
			}
		})
	}
}
