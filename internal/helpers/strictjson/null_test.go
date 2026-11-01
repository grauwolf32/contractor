package strictjson

import (
	"errors"
	"testing"
)

func TestDecodeNonNullSeparatesPrivateOmissionFromPublicNullableFields(t *testing.T) {
	type document struct {
		Optional *int `json:"optional,omitempty"`
	}
	var public, private document
	if err := Decode([]byte(`{"optional":null}`), &public); err != nil {
		t.Fatalf("public nullable field rejected: %v", err)
	}
	if err := DecodeNonNull([]byte(`{"optional":null}`), &private); err == nil {
		t.Fatal("private optional null accepted")
	}
	if err := DecodeNonNull([]byte(`{}`), &private); err != nil {
		t.Fatalf("omitted optional field rejected: %v", err)
	}
	if err := DecodeNonNull([]byte(`{"optional":null,"optional":null}`), &private); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate key lost precedence: %v", err)
	}
}

func TestDecodeNonNullPreservesDeclaredNullsAndUntypedJSON(t *testing.T) {
	type document struct {
		Required *int           `json:"required" strictjson:"required,nullable"`
		Nullable *int           `json:"nullable,omitempty" strictjson:"nullable"`
		Payload  map[string]any `json:"payload"`
	}
	var value document
	if err := DecodeNonNull([]byte(`{"required":null,"nullable":null,"payload":{"a":null,"nested":[null,{"b":null}]}}`), &value); err != nil {
		t.Fatalf("schema-declared null or untyped JSON rejected: %v", err)
	}
}

func TestDecodeNonNullRejectsTypedMapAndSequenceNulls(t *testing.T) {
	type document struct {
		Counters map[string]uint64 `json:"counters,omitempty"`
		Labels   map[string]string `json:"labels,omitempty"`
		Values   []int             `json:"values,omitempty"`
	}
	for _, raw := range []string{
		`{"counters":{"a":null}}`, `{"labels":{"a":null}}`, `{"values":[null]}`,
	} {
		var value document
		if err := DecodeNonNull([]byte(raw), &value); err == nil {
			t.Errorf("accepted %s", raw)
		}
	}
}
