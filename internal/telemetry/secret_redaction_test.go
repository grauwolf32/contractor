package telemetry

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// The Planner OTLP exporter and the telemetry secret holders redact
// themselves for every fmt verb, flag and nesting, for log/slog and for
// encoding/json (SEC-R88).
func TestTelemetrySecretHoldersRedactForEveryVerbAndEncoder(t *testing.T) {
	t.Parallel()
	exporter := &otlpHTTPPlannerTelemetry{
		headers:      map[string]contracts.SecretString{"authorization": contracts.NewSecretString("hunter2-header")},
		secretValues: []string{"hunter2-header"},
	}
	contractstest.AssertRedacted(t, exporter, "hunter2-header", "telemetry.otlpHTTPPlannerTelemetry([REDACTED])")
	contractstest.AssertRedacted(t, otlpPlannerInstrumentation{owner: exporter}, "hunter2-header",
		"telemetry.otlpPlannerInstrumentation([REDACTED])")
	contractstest.AssertRedacted(t, &otlpPlannerSpan{owner: exporter}, "hunter2-header", "telemetry.otlpPlannerSpan([REDACTED])")
	contractstest.AssertRedacted(t, Policy{secrets: []string{"hunter2-policy"}}, "hunter2-policy", "telemetry.Policy([REDACTED])")
	envelope := AllocationReportEnvelope{Secrets: []string{"hunter2-report"}}
	contractstest.AssertRedacted(t, envelope, "hunter2-report", "telemetry.AllocationReportEnvelope([REDACTED])")
	contractstest.AssertRedacted(t, StoredAllocationReport{AllocationReportEnvelope: envelope}, "hunter2-report",
		"telemetry.AllocationReportEnvelope([REDACTED])")
}
