package contractstest

import (
	"bytes"
	"encoding/json"
	"testing"
)

// RevisionETagCase is one valid counter precondition and the revision it
// carries.
type RevisionETagCase struct {
	Header   string `json:"header"`
	Revision uint64 `json:"revision"`
}

// RevisionETagCases is api/testdata/public/revision-etag-cases.json: the
// counter preconditions of the public API that its one reader accepts
// (Valid) or rejects (Invalid, OutOfRange), and If-Match header sets that
// are not one header (RepeatedHeaders).
type RevisionETagCases struct {
	Valid           []RevisionETagCase
	Invalid         []string
	OutOfRange      []string
	RepeatedHeaders [][]string
}

// Rejected returns every single header value the reader rejects.
func (c RevisionETagCases) Rejected() []string {
	return append(append([]string(nil), c.Invalid...), c.OutOfRange...)
}

// ReadRevisionETagCases strictly reads the shared counter precondition
// cases. An unknown key or an empty group fails the test.
func ReadRevisionETagCases(t testing.TB) RevisionETagCases {
	t.Helper()
	var fixture struct {
		Description     string             `json:"description"`
		Valid           []RevisionETagCase `json:"valid"`
		Invalid         []string           `json:"invalid"`
		OutOfRange      []string           `json:"outOfRange"`
		RepeatedHeaders [][]string         `json:"repeatedHeaders"`
	}
	decoder := json.NewDecoder(bytes.NewReader(ReadFile(t, "api", "testdata", "public", "revision-etag-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatalf("decode shared revision ETag cases: %v", err)
	}
	if len(fixture.Valid) == 0 || len(fixture.Invalid) == 0 || len(fixture.OutOfRange) == 0 || len(fixture.RepeatedHeaders) == 0 {
		t.Fatal("shared revision ETag cases need valid, invalid, out-of-range and repeated header cases")
	}
	return RevisionETagCases{
		Valid: fixture.Valid, Invalid: fixture.Invalid,
		OutOfRange: fixture.OutOfRange, RepeatedHeaders: fixture.RepeatedHeaders,
	}
}
