package contracts

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
)

func TestRuntimeReportRetainsDroppedSpans(t *testing.T) {
	raw := []byte(`{"adapters":{"otlp-http@1":{"droppedSpans":{"cancelled":1,"collectorRejected":2,"deadlineExceeded":3,"encodingFailed":4,"nonRetryable":5,"queueOverflow":18446744073709551615,"retryExhausted":6,"shutdown":7},"failedOperations":1,"operations":2}},"complete":true,"durationMs":1200,"stopReason":"finalized"}`)
	var report RuntimeReport
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	drops := report.Adapters["otlp-http@1"].DroppedSpans
	if drops == nil || drops.QueueOverflow != ^uint64(0) || drops.CollectorRejected != 2 || drops.Shutdown != 7 {
		t.Fatalf("dropped counters lost: %+v", drops)
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	var decoded RuntimeReport
	err = json.Unmarshal(encoded, &decoded)
	if err != nil || !reflect.DeepEqual(report, decoded) {
		t.Fatalf("report did not round trip: %s, %v", encoded, err)
	}
}

func TestRuntimeReportRejectsInvalidDroppedSpanCounts(t *testing.T) {
	for _, value := range []string{`-1`, `18446744073709551616`, `true`, `1.5`, `"3"`} {
		raw := []byte(fmt.Sprintf(`{"adapters":{"otlp-http@1":{"droppedSpans":{"queueOverflow":%s},"failedOperations":1,"operations":2}},"complete":true,"durationMs":1,"stopReason":"finalized"}`, value))
		var report RuntimeReport
		err := json.Unmarshal(raw, &report)
		if err != nil || report.Complete || len(report.Adapters) != 0 {
			t.Fatalf("invalid diagnostics must make report incomplete: %s, %+v, %v", value, report, err)
		}
	}
}
