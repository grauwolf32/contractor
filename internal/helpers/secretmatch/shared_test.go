package secretmatch_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/secretmatch"
)

// The Runtime's contractor_runtime.redaction reads the same cases.
func TestSharedSecretMatchingCases(t *testing.T) {
	t.Parallel()
	var fixture struct {
		MinSubstringBytes int    `json:"minSubstringBytes"`
		Marker            string `json:"marker"`
		Cases             []struct {
			Name     string   `json:"name"`
			Secrets  []string `json:"secrets"`
			Text     string   `json:"text"`
			Redacted string   `json:"redacted"`
			Contains bool     `json:"contains"`
		} `json:"cases"`
	}
	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "secret-matching-cases.json")
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.MinSubstringBytes != secretmatch.MinSubstringBytes || len(fixture.Cases) == 0 {
		t.Fatalf("fixture bound %d, cases %d", fixture.MinSubstringBytes, len(fixture.Cases))
	}
	for _, test := range fixture.Cases {
		if got := secretmatch.Redact(test.Text, test.Secrets, fixture.Marker); got != test.Redacted {
			t.Errorf("%s: Redact = %q, want %q", test.Name, got, test.Redacted)
		}
		if got := secretmatch.Contains(test.Text, test.Secrets); got != test.Contains {
			t.Errorf("%s: Contains = %v, want %v", test.Name, got, test.Contains)
		}
	}
}
