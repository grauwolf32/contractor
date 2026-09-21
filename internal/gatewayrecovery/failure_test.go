package gatewayrecovery

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// classificationFixture is shared with runtime/tests/test_gateway_error_classification.py
// so the planner and worker classifiers cannot drift apart.
type classificationFixture struct {
	Declared contracts.GatewayFailureSignatures `json:"declared"`
	Cases    []struct {
		Name       string            `json:"name"`
		Status     int               `json:"status"`
		Signatures string            `json:"signatures"`
		Headers    map[string]string `json:"headers"`
		Body       json.RawMessage   `json:"body"`
		BodyText   *string           `json:"bodyText"`
		Code       string            `json:"code"`
		Retryable  bool              `json:"retryable"`
	} `json:"cases"`
}

func TestClassifyGatewayFailureMatchesSharedFixture(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "v1alpha1", "gateway-failure-classification-cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture classificationFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if err := fixture.Declared.Validate(); err != nil {
		t.Fatalf("declared fixture signatures are invalid: %v", err)
	}
	sets := map[string]contracts.GatewayFailureSignatures{
		"declared": fixture.Declared,
		"default":  contracts.DefaultGatewayFailureSignatures(),
		"empty":    {},
	}
	covered := map[[2]any]bool{}
	for _, test := range fixture.Cases {
		signatures, ok := sets[test.Signatures]
		if !ok {
			t.Fatalf("%s: unknown signature set %q", test.Name, test.Signatures)
		}
		header := http.Header{}
		for name, value := range test.Headers {
			header.Set(name, value)
		}
		body := []byte(test.Body)
		if test.BodyText != nil {
			body = []byte(*test.BodyText)
		}
		got := Classify(test.Status, header, body, signatures)
		if got.Code != test.Code || got.Retryable != test.Retryable {
			t.Errorf("%s: HTTP %d = %+v, want %s retryable=%t", test.Name, test.Status, got, test.Code, test.Retryable)
		}
		covered[[2]any{test.Signatures, test.Retryable}] = true
	}
	for name := range sets {
		if !covered[[2]any{name, true}] || !covered[[2]any{name, false}] {
			t.Errorf("fixture lacks both verdicts for signature set %q", name)
		}
	}
}

func TestPythonReprMatchesCPythonQuoting(t *testing.T) {
	for text, want := range map[string]string{
		"Model is unloaded.":       "'Model is unloaded.'",
		"model 'worker' not found": `"model 'worker' not found"`,
		`say "hi"`:                 `'say "hi"'`,
		`both ' and "`:             `'both \' and "'`,
		`back\slash`:               `'back\\slash'`,
	} {
		if got := pythonRepr(text); got != want {
			t.Errorf("pythonRepr(%q) = %s, want %s", text, got, want)
		}
	}
}

func TestRecoveryDelayUsesConfiguredCap(t *testing.T) {
	policy := Policy{RequestTimeout: time.Minute, InitialDelay: 3 * time.Second, MaxDelay: 19 * time.Second, AutomaticWindow: time.Minute}
	for i, want := range []time.Duration{3, 6, 12, 19, 19} {
		if got := policy.failureDelay(int64(i)); got != want*time.Second {
			t.Fatalf("failure %d delay=%s", i, got)
		}
	}
	for _, raw := range []string{"NaN", "+Inf", "-Inf", "garbage", "-1"} {
		if got := retryAfter(http.Header{"Retry-After": []string{raw}}); got != 0 {
			t.Fatalf("invalid hint %q: %f", raw, got)
		}
	}
}
