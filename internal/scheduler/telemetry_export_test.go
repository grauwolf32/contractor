package scheduler

import (
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestSchedulerMaterializesPinnedTelemetryExportSettings(t *testing.T) {
	harness := newSchedulerHarness(t)
	resolved, err := schedulerTestResolvedWorkerConfig(
		harness.workflow.Stages[harness.workflow.EntryStage].ExecutionConfig.Agents["builder"],
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	export := runtimesettings.TelemetryExportSettings{
		BatchSizeBytes: 1048576, MaxAttempts: 3, MaxPendingSpans: 16, MaxPendingBytes: 2097152,
		Retry: &runtimesettings.TelemetryRetrySettings{InitialBackoffMilliseconds: 17, MaxBackoffMilliseconds: 43},
	}
	resolved.WorkerTelemetry = &runtimeconfig.TelemetryConfig{
		Adapter: "otlp-http@1", Endpoint: "https://collector.example/v1/traces",
		FlushTimeoutSeconds: 10, Export: &export,
	}
	settings, err := harness.scheduler.materializeRuntimeSettings(t.Context(), resolved)
	if err != nil {
		t.Fatal(err)
	}
	if settings.Telemetry == nil || settings.Telemetry.Export == nil || !reflect.DeepEqual(*settings.Telemetry.Export, export) || settings.Telemetry.FlushTimeoutSeconds != 10 {
		t.Fatalf("materialized export settings = %+v", settings.Telemetry)
	}
	if settings.Telemetry.Export.Retry == nil || settings.Telemetry.Export.Retry.InitialBackoffMilliseconds != 17 {
		t.Fatal("retry settings did not reach Runtime")
	}
	settings.Telemetry.Export.Retry.InitialBackoffMilliseconds = 1
	if resolved.WorkerTelemetry.Export.Retry.InitialBackoffMilliseconds != 17 {
		t.Fatal("retry settings alias resolved config")
	}
	settings.Telemetry.Export.MaxAttempts = 1
	if resolved.WorkerTelemetry.Export.MaxAttempts != 3 {
		t.Fatal("private settings retained mutable pinned configuration")
	}
}

// TestSchedulerPinsTelemetryExportDefaults checks that a Worker RuntimeConfig
// without an export or retry block still sends complete export settings: the
// Server's defaults (RCF-R99, RCF-R102, RCF-R103), which
// api/testdata/v1alpha1/runtime-settings-bounds-cases.json pins for the
// Runtime, while the resolved configuration keeps its omissions.
func TestSchedulerPinsTelemetryExportDefaults(t *testing.T) {
	harness := newSchedulerHarness(t)
	defaultRetry := runtimesettings.DefaultTelemetryRetrySettings()
	withoutRetry := runtimesettings.TelemetryExportSettings{
		BatchSizeBytes: 1048576, MaxAttempts: 3, MaxPendingSpans: 16, MaxPendingBytes: 2097152,
	}
	for _, test := range []struct {
		name       string
		configured *runtimesettings.TelemetryExportSettings
		want       runtimesettings.TelemetryExportSettings
	}{
		{"no export block", nil, runtimesettings.DefaultTelemetryExportSettings()},
		{"export without retry", &withoutRetry, withoutRetry},
	} {
		t.Run(test.name, func(t *testing.T) {
			resolved, err := schedulerTestResolvedWorkerConfig(
				harness.workflow.Stages[harness.workflow.EntryStage].ExecutionConfig.Agents["builder"],
				false,
			)
			if err != nil {
				t.Fatal(err)
			}
			resolved.WorkerTelemetry = &runtimeconfig.TelemetryConfig{
				Adapter: "otlp-http@1", Endpoint: "https://collector.example/v1/traces",
				FlushTimeoutSeconds: 10, Export: test.configured,
			}
			settings, err := harness.scheduler.materializeRuntimeSettings(t.Context(), resolved)
			if err != nil {
				t.Fatal(err)
			}
			want := test.want
			want.Retry = &defaultRetry
			if settings.Telemetry == nil || settings.Telemetry.Export == nil ||
				!reflect.DeepEqual(*settings.Telemetry.Export, want) {
				t.Fatalf("export settings = %+v, want %+v", settings.Telemetry, want)
			}
			if err := settings.Validate(); err != nil {
				t.Fatalf("materialized settings are not valid on the wire: %v", err)
			}
			if resolved.WorkerTelemetry.Export != test.configured ||
				test.configured != nil && test.configured.Retry != nil {
				t.Fatal("materializing the settings changed the resolved configuration")
			}
		})
	}
}
