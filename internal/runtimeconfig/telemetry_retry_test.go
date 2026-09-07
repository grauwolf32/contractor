package runtimeconfig

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestTelemetryRetryPublicationAndDetachedResolution(t *testing.T) {
	for _, encoded := range []string{`{}`, `{"initialBackoffMilliseconds":17,"maxBackoffMilliseconds":43}`} {
		prepared, err := PreparePublication([]byte(fmt.Sprintf(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"retry","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://collector.example/v1/traces","export":{"retry":%s}}}}}`, encoded)))
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
		telemetry := stored.Spec.Worker.Telemetry.Value
		expected := contracts.DefaultTelemetryRetrySettings()
		if encoded != `{}` {
			expected.InitialBackoffMilliseconds, expected.MaxBackoffMilliseconds = 17, 43
		}
		if telemetry.Export == nil || telemetry.Export.Retry == nil || *telemetry.Export.Retry != expected {
			t.Fatalf("retry did not round trip: %+v", telemetry.Export)
		}
		clone := cloneTelemetry(&telemetry)
		clone.Export.Retry.InitialBackoffMilliseconds = 1
		if *telemetry.Export.Retry != expected {
			t.Fatal("cloned settings share retry state")
		}
	}
}

func TestTelemetryRetryDoesNotRewriteStoredExportBlocks(t *testing.T) {
	source := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"debug","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","captureContent":false,"endpoint":"https://otel.example/v1/traces","flushTimeoutSeconds":3,"export":{"batchSizeBytes":1048576,"maxAttempts":2,"maxPendingSpans":16,"maxPendingBytes":2097152}}}}}`)
	var document any
	if err := json.Unmarshal(source, &document); err != nil {
		t.Fatal(err)
	}
	source, err := canonicalize(document)
	if err != nil {
		t.Fatal(err)
	}
	stored, err := DecodeStoredDocument(source)
	if err != nil {
		t.Fatal(err)
	}
	if stored.Ref.Digest != digest(source) || stored.Spec.Worker.Telemetry.Value.Export.Retry != nil {
		t.Fatal("legacy stored export was rewritten")
	}
}

func TestTelemetryRetryRejectsInvalidValues(t *testing.T) {
	for _, encoded := range []string{
		`null`, `{"initialBackoffMilliseconds":null}`, `{"maxBackoffMilliseconds":null}`,
		`{"initialBackoffMilliseconds":0}`, `{"maxBackoffMilliseconds":0}`,
		`{"initialBackoffMilliseconds":-1}`, `{"maxBackoffMilliseconds":60001}`,
		`{"initialBackoffMilliseconds":1001}`, `{"initialBackoffMilliseconds":true}`,
		`{"initialBackoffMilliseconds":1.5}`, `{"unknown":1}`,
	} {
		t.Run(encoded, func(t *testing.T) {
			_, err := PreparePublication([]byte(fmt.Sprintf(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"retry","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://collector.example/v1/traces","export":{"retry":%s}}}}}`, encoded)))
			if err == nil {
				t.Fatal("invalid retry settings accepted")
			}
		})
	}
}
