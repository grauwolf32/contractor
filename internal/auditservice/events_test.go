package auditservice

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestEventSummaryPublicProjection(t *testing.T) {
	result := safeEventSummary(json.RawMessage(`{
 "from":"paused","to":"active","round":2,"items":50,"reviews":3,"members":4,"count":5,
 "role":"check","workflowRole":"check","outcome":"failed","disposition":"missing-output",
 "subjectKind":"audit-item-action","subjectId":"item-1","findingId":"finding-1","runId":"run-1",
 "kind":"active-check-approval","action":"approve","verdict":"needs_evidence",
 "message":"secret","rationale":"private","requestDigest":"digest",
 "previousStopReason":{"Message":"secret"},"reason":"sensitive text","future":{"secret":"value"}
}`))
	for _, key := range []string{"message", "rationale", "requestDigest", "previousStopReason", "reason", "future"} {
		if _, ok := result[key]; ok {
			t.Fatalf("unsafe summary member %q exposed", key)
		}
	}
	if len(result) != 18 || result["action"] != "approve" || result["items"] != int64(50) {
		t.Fatalf("structured summary = %#v", result)
	}
	for _, raw := range []string{`null`, `{`, `[]`, `{"items":-1,"count":2147483648,"runId":"https://secret","kind":"unknown","to":"arbitrary text","action":"execute","verdict":"unsupported","role":"unknown"}`} {
		if result := safeEventSummary(json.RawMessage(raw)); !reflect.DeepEqual(result, map[string]any{}) {
			t.Fatalf("invalid summary %s projected as %#v", raw, result)
		}
	}
}
