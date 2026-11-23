package secretmatch

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedExactReplacementFixture(t *testing.T) {
	raw := contractstest.ReadFile(t, "api", "testdata", "telemetry-credential-redaction.json")
	var cases []struct {
		Name     string
		Secrets  []string
		Input    string
		Expected string
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("empty shared fixture")
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			if got := ExactReplacer(c.Secrets)(c.Input); got != c.Expected {
				t.Fatalf("unexpected replacement: %q", got)
			}
		})
	}
}
