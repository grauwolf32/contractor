package contractstest

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// IdempotencyKeyCases is api/testdata/public/idempotency-key-cases.json with
// its boundary keys resolved: Valid and Invalid include a key of MaxLength
// bytes ending with each allowed punctuation character and the same key one
// byte longer. DuplicateHeaders are header value sets that are not exactly
// one Idempotency-Key header.
type IdempotencyKeyCases struct {
	MaxLength        int
	Valid            []string
	Invalid          []string
	DuplicateHeaders [][]string
}

// ReadIdempotencyKeyCases strictly reads the shared Idempotency-Key cases
// that the UI reads too (ui/src/mutations/idempotency.test.ts). An unknown
// key or an empty group fails the test.
func ReadIdempotencyKeyCases(t testing.TB) IdempotencyKeyCases {
	t.Helper()
	var fixture struct {
		Description      string     `json:"description"`
		MaxLength        int        `json:"maxLength"`
		Valid            []string   `json:"valid"`
		Invalid          []string   `json:"invalid"`
		DuplicateHeaders [][]string `json:"duplicateHeaders"`
	}
	decoder := json.NewDecoder(bytes.NewReader(ReadFile(t, "api", "testdata", "public", "idempotency-key-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatalf("decode shared Idempotency-Key cases: %v", err)
	}
	if fixture.MaxLength < 2 || len(fixture.Valid) == 0 || len(fixture.Invalid) == 0 || len(fixture.DuplicateHeaders) == 0 {
		t.Fatal("shared Idempotency-Key cases need a maxLength and valid, invalid and duplicate header cases")
	}
	cases := IdempotencyKeyCases{
		MaxLength: fixture.MaxLength, Valid: fixture.Valid, Invalid: fixture.Invalid,
		DuplicateHeaders: fixture.DuplicateHeaders,
	}
	for _, last := range []string{"a", ".", "_", ":", "-"} {
		key := strings.Repeat("a", fixture.MaxLength-1) + last
		cases.Valid = append(cases.Valid, key)
		cases.Invalid = append(cases.Invalid, "a"+key)
	}
	return cases
}
