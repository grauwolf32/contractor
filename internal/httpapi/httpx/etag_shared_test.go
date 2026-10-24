package httpx

import (
	"encoding/json"
	"errors"
	"net/http"
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

// ParseRevisionETag is the one reader of the public counter preconditions
// (API-R70): it accepts exactly the shared valid cases, and RequiredIfMatch
// accepts exactly one If-Match header without an If-None-Match header.
func TestSharedRevisionETagCases(t *testing.T) {
	t.Parallel()
	cases := contractstest.ReadRevisionETagCases(t)
	for _, valid := range cases.Valid {
		if revision, err := ParseRevisionETag(valid.Header); err != nil || revision != valid.Revision {
			t.Errorf("ParseRevisionETag(%q) = %d, %v; want %d", valid.Header, revision, err, valid.Revision)
		}
		header := http.Header{}
		header.Set("If-Match", valid.Header)
		if raw, err := RequiredIfMatch(header); err != nil || raw != valid.Header {
			t.Errorf("RequiredIfMatch(%q) = %q, %v", valid.Header, raw, err)
		}
		header.Set("If-None-Match", "*")
		if _, err := RequiredIfMatch(header); !errors.Is(err, ErrIfMatchCount) {
			t.Errorf("RequiredIfMatch with If-None-Match = %v; want ErrIfMatchCount", err)
		}
	}
	for _, raw := range cases.Rejected() {
		if revision, err := ParseRevisionETag(raw); err == nil {
			t.Errorf("ParseRevisionETag(%q) = %d; want an error", raw, revision)
		}
	}
	for _, values := range append(cases.RepeatedHeaders, nil) {
		if raw, err := RequiredIfMatch(http.Header{"If-Match": values}); !errors.Is(err, ErrIfMatchCount) {
			t.Errorf("RequiredIfMatch(%q) = %q, %v; want ErrIfMatchCount", values, raw, err)
		}
	}
}

// ParseRevision accepts one canonical decimal spelling per counter, zero
// included, which the owner Queue control uses before its first change.
func TestParseRevisionAcceptsOnlyCanonicalDecimals(t *testing.T) {
	t.Parallel()
	for value, want := range map[string]uint64{"0": 0, "7": 7, "18446744073709551615": 1<<64 - 1} {
		if got, err := ParseRevision(value); err != nil || got != want {
			t.Errorf("ParseRevision(%q) = %d, %v; want %d", value, got, err, want)
		}
	}
	for _, value := range []string{"", "00", "01", "+1", "-0", " 1", "1 ", "1e3", "18446744073709551616"} {
		if got, err := ParseRevision(value); !errors.Is(err, ErrRevision) {
			t.Errorf("ParseRevision(%q) = %d, %v; want ErrRevision", value, got, err)
		}
	}
}
