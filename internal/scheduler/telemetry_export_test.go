package scheduler

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestSchedulerMaterializesPinnedTelemetryExportSettings(t *testing.T) {
	harness := newSchedulerHarness(t)
	resolved, err := fallbackResolvedWorkerConfig(
		harness.workflow.Stages[harness.workflow.EntryStage].ExecutionConfig.Agents["builder"],
	)
	if err != nil {
		t.Fatal(err)
	}
	export := contracts.TelemetryExportSettings{
		BatchSizeBytes: 1048576, MaxAttempts: 3, MaxPendingSpans: 16, MaxPendingBytes: 2097152,
	}
	resolved.WorkerTelemetry = &runtimeconfig.TelemetryConfig{
		Adapter: "otlp-http@1", Endpoint: "https://collector.example/v1/traces",
		FlushTimeoutSeconds: 10, Export: &export,
	}
	settings, err := harness.scheduler.materializeRuntimeSettings(t.Context(), resolved)
	if err != nil {
		t.Fatal(err)
	}
	if settings.Telemetry == nil || settings.Telemetry.Export == nil || *settings.Telemetry.Export != export || settings.Telemetry.FlushTimeoutSeconds != 10 {
		t.Fatalf("materialized export settings = %+v", settings.Telemetry)
	}
	settings.Telemetry.Export.MaxAttempts = 1
	if resolved.WorkerTelemetry.Export.MaxAttempts != 3 {
		t.Fatal("private settings retained mutable pinned configuration")
	}
}
