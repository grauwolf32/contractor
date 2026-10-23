package httpx

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// The Server reads If-Match and renders ETag with the one strong entity-tag
// grammar the Runtime also reads and writes (API-R70, ART-R45).
func TestSharedStrongETagCases(t *testing.T) {
	t.Parallel()
	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "strong-etag-cases.json")
	var cases struct {
		Valid []struct {
			Header   string `json:"header"`
			Revision string `json:"revision"`
		} `json:"valid"`
		InvalidHeaders   []string `json:"invalidHeaders"`
		InvalidRevisions []string `json:"invalidRevisions"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Valid) == 0 || len(cases.InvalidHeaders) == 0 || len(cases.InvalidRevisions) == 0 {
		t.Fatal("the shared strong ETag cases are empty")
	}
	for _, valid := range cases.Valid {
		if got, err := ParseStrongETag(valid.Header); err != nil || got != valid.Revision {
			t.Errorf("ParseStrongETag(%q) = %q, %v; want %q", valid.Header, got, err, valid.Revision)
		}
		if got := QuotedETag(&valid.Revision); got != valid.Header {
			t.Errorf("QuotedETag(%q) = %q; want %q", valid.Revision, got, valid.Header)
		}
	}
	for _, header := range cases.InvalidHeaders {
		if got, err := ParseStrongETag(header); err == nil {
			t.Errorf("ParseStrongETag(%q) = %q; want an error", header, got)
		}
	}
	for _, revision := range cases.InvalidRevisions {
		if got, err := ParseStrongETag(`"` + revision + `"`); err == nil && got == revision {
			t.Errorf("revision %q is carried verbatim by a strong ETag", revision)
		}
	}
}
