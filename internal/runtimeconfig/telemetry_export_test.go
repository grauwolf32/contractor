package runtimeconfig

import (
	"context"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestNormalizeWorkerTelemetryExportSettings(t *testing.T) {
	for _, source := range []string{
		`{"maxAttempts":3,"batchSizeBytes":1048576,"maxPendingSpans":16,"maxPendingBytes":2097152}`,
		`{}`,
	} {
		prepared, err := PreparePublication([]byte(fmt.Sprintf(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"debug","version":"2"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces","export":%s}}}}`, source)))
		if err != nil {
			t.Fatal(err)
		}
		version, err := prepared.Resolve(context.Background(), nil)
		if err != nil {
			t.Fatal(err)
		}
		stored, err := DecodeStoredDocument(version.CanonicalDocument)
		if err != nil {
			t.Fatal(err)
		}
		got := stored.Spec.Worker.Telemetry.Value
		want := contracts.DefaultTelemetryExportSettings()
		if source != `{}` {
			want = contracts.TelemetryExportSettings{BatchSizeBytes: 1048576, MaxAttempts: 3, MaxPendingSpans: 16, MaxPendingBytes: 2097152}
		}
		if got.FlushTimeoutSeconds != 10 || got.Export == nil || *got.Export != want {
			t.Fatalf("normalized telemetry = %+v, export = %+v", got, got.Export)
		}
		cloned := cloneTelemetry(&got)
		cloned.Export.MaxAttempts = 1
		if got.Export.MaxAttempts != want.MaxAttempts {
			t.Fatal("cloning telemetry retained mutable export settings")
		}
	}
}

func TestTelemetryExportPreservesLegacyStoredConfiguration(t *testing.T) {
	document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"debug","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","captureContent":false,"endpoint":"https://otel.example/v1/traces","flushTimeoutSeconds":3}}}}`)
	stored, err := DecodeStoredDocument(document)
	if err != nil {
		t.Fatal(err)
	}
	if stored.Spec.Worker.Telemetry.Value.Export != nil || stored.Spec.Worker.Telemetry.Value.FlushTimeoutSeconds != 3 || stored.Ref.Digest != digest(document) {
		t.Fatal("legacy telemetry configuration was rewritten")
	}
}

func TestNormalizeRejectsInvalidWorkerTelemetryExport(t *testing.T) {
	for _, source := range []string{
		`null`, `{"maxAttempts":0}`, `{"maxAttempts":11}`, `{"maxAttempts":null}`,
		`{"maxAttempts":true}`, `{"maxAttempts":1.5}`, `{"batchSizeBytes":1}`,
		`{"maxPendingBytes":1048576}`, `{"maxPendingBytes":67108865}`,
		`{"maxPendingSpans":0}`, `{"maxPendingSpans":2049}`, `{"unknown":1}`,
	} {
		t.Run(source, func(t *testing.T) {
			_, err := PreparePublication([]byte(fmt.Sprintf(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"debug","version":"2"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces","export":%s}}}}`, source)))
			if err == nil {
				t.Fatal("invalid export settings accepted")
			}
		})
	}
	_, err := PreparePublication([]byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"debug","version":"2"},"spec":{"planner":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces","export":{}}}}}`))
	if err == nil {
		t.Fatal("Worker-only export settings accepted for Planner")
	}
}
