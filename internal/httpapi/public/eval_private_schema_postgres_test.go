package public

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestSharedEvalPostgresRejectsUnusedPrivateSchemaBeforeEffects(t *testing.T) {
	h := newEvalAPIHarness(t)
	_, supported := h.dataset(t, "workflow")
	raw, err := json.Marshal(supported)
	if err != nil {
		t.Fatal(err)
	}
	var unsupported map[string]any
	if err := json.Unmarshal(raw, &unsupported); err != nil {
		t.Fatal(err)
	}
	unsupported["privateChecks"].([]any)[0].(map[string]any)["schema"] = "json@1"
	before := evalReceiptEffects(t, h)
	answer := apiDecode[errorResponse](t, h.request(t, "POST", "/v1/projects/evaluation/eval-datasets", unsupported, "unused-private-schema", "", 422))
	if answer.Code != "eval_invalid" || answer.Retryable {
		t.Fatalf("removed private schema accepted: %+v", answer)
	}
	if after := evalReceiptEffects(t, h); !bytes.Equal(before, after) {
		t.Fatal("rejected private schema changed the dataset or recorded a mutation receipt")
	}
	// Rejection did not consume the key; the supported private rubric remains
	// accepted without rewriting its expected values or visible case content.
	h.request(t, "POST", "/v1/projects/evaluation/eval-datasets", supported, "unused-private-schema", "", 201)
}
