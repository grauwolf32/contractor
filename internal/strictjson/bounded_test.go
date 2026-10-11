package strictjson

import (
	"encoding/json"
	"errors"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

var wideBounds = Bounds{MaxDepth: 64, MaxNodes: 1000}

func floatNumber(literal string) (float64, bool) {
	value, err := strconv.ParseFloat(literal, 64)
	return value, err == nil
}

func TestDecodeBoundedBuildsGenericTree(t *testing.T) {
	got, err := DecodeBounded([]byte(` {"a":[1.5,"x",true,false,null,{}],"b":[],"c":{"d":"é"}} `), wideBounds, floatNumber)
	want := map[string]any{
		"a": []any{1.5, "x", true, false, nil, map[string]any{}},
		"b": []any{},
		"c": map[string]any{"d": "é"},
	}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("DecodeBounded = %#v, %v; want %#v", got, err, want)
	}
	if empty := got.(map[string]any)["b"].([]any); empty == nil {
		t.Fatal("an empty array decoded to a nil slice")
	}
}

func TestDecodeBoundedPassesNumberLiterals(t *testing.T) {
	var literals []string
	keep := func(literal string) (json.Number, bool) {
		literals = append(literals, literal)
		return json.Number(literal), true
	}
	got, err := DecodeBounded([]byte(`[1.50e+2,-0,9007199254740993]`), wideBounds, keep)
	if err != nil || !reflect.DeepEqual(got, []any{json.Number("1.50e+2"), json.Number("-0"), json.Number("9007199254740993")}) {
		t.Fatalf("DecodeBounded = %#v, %v", got, err)
	}
	if !reflect.DeepEqual(literals, []string{"1.50e+2", "-0", "9007199254740993"}) {
		t.Fatalf("number saw %q, want the exact literals", literals)
	}
	reject := func(string) (float64, bool) { return 0, false }
	if _, err := DecodeBounded([]byte(`{"a":1}`), wideBounds, reject); !errors.Is(err, ErrInvalid) {
		t.Fatalf("rejected number = %v, want ErrInvalid", err)
	}
}

func TestDecodeBoundedEnforcesDepth(t *testing.T) {
	bounds := Bounds{MaxDepth: 3, MaxNodes: 100}
	// An empty container at the limit has no deeper node.
	for _, input := range []string{`[[1]]`, `{"a":{"b":1}}`, `[[{}]]`} {
		if _, err := DecodeBounded([]byte(input), bounds, floatNumber); err != nil {
			t.Errorf("DecodeBounded(%s) within depth 3 = %v", input, err)
		}
	}
	for _, input := range []string{`[[[1]]]`, `{"a":{"b":{"c":1}}}`, `[[[{}]]]`} {
		if _, err := DecodeBounded([]byte(input), bounds, floatNumber); !errors.Is(err, ErrBounds) {
			t.Errorf("DecodeBounded(%s) = %v, want ErrBounds", input, err)
		}
	}
}

func TestDecodeBoundedEnforcesNodes(t *testing.T) {
	// The array and its three elements are four nodes.
	if _, err := DecodeBounded([]byte(`[1,2,3]`), Bounds{MaxDepth: 4, MaxNodes: 4}, floatNumber); err != nil {
		t.Fatalf("four nodes = %v", err)
	}
	if _, err := DecodeBounded([]byte(`[1,2,3,4]`), Bounds{MaxDepth: 4, MaxNodes: 4}, floatNumber); !errors.Is(err, ErrBounds) {
		t.Fatalf("five nodes = %v, want ErrBounds", err)
	}
	// Reading stops at the first node beyond the bound, before the invalid
	// remainder.
	if _, err := DecodeBounded([]byte(`[1,2,3,4,`+strings.Repeat("x", 100)), Bounds{MaxDepth: 4, MaxNodes: 4}, floatNumber); !errors.Is(err, ErrBounds) {
		t.Fatalf("nodes beyond the bound before invalid input = %v, want ErrBounds", err)
	}
}

func TestDecodeBoundedCountsNamesOnlyWhenAsked(t *testing.T) {
	values := Bounds{MaxDepth: 2, MaxNodes: 2}
	if _, err := DecodeBounded([]byte(`{"a":1}`), values, floatNumber); err != nil {
		t.Fatalf("object and value = %v", err)
	}
	names := Bounds{MaxDepth: 2, MaxNodes: 2, NamesAreNodes: true}
	if _, err := DecodeBounded([]byte(`{"a":1}`), names, floatNumber); !errors.Is(err, ErrBounds) {
		t.Fatalf("object, name and value within two nodes = %v, want ErrBounds", err)
	}
	names.MaxNodes = 3
	if _, err := DecodeBounded([]byte(`{"a":1}`), names, floatNumber); err != nil {
		t.Fatalf("object, name and value within three nodes = %v", err)
	}
	// A name is a node at its value's depth.
	shallow := Bounds{MaxDepth: 1, MaxNodes: 10, NamesAreNodes: true}
	if _, err := DecodeBounded([]byte(`{}`), shallow, floatNumber); err != nil {
		t.Fatalf("empty object at depth one = %v", err)
	}
	if _, err := DecodeBounded([]byte(`{"a":1}`), shallow, floatNumber); !errors.Is(err, ErrBounds) {
		t.Fatalf("member name below depth one = %v, want ErrBounds", err)
	}
}

func TestDecodeBoundedEnforcesStringBytes(t *testing.T) {
	bounds := Bounds{MaxDepth: 4, MaxNodes: 10, MaxStringBytes: 4}
	for _, input := range []string{`"abcd"`, `{"abcd":"éé"}`} {
		if _, err := DecodeBounded([]byte(input), bounds, floatNumber); err != nil {
			t.Errorf("DecodeBounded(%s) = %v", input, err)
		}
	}
	// Decoded bytes count, so an escape longer than its result passes and a
	// multi-byte result of a short escape does not.
	for _, input := range []string{`"abcde"`, `{"abcde":1}`, `"ééé"`} {
		if _, err := DecodeBounded([]byte(input), bounds, floatNumber); !errors.Is(err, ErrBounds) {
			t.Errorf("DecodeBounded(%s) = %v, want ErrBounds", input, err)
		}
	}
	if _, err := DecodeBounded([]byte(`"ab"`), bounds, floatNumber); err != nil {
		t.Fatalf("escaped two-byte string = %v", err)
	}
	if _, err := DecodeBounded([]byte(`"`+strings.Repeat("x", 1000)+`"`), wideBounds, floatNumber); err != nil {
		t.Fatalf("unbounded string = %v", err)
	}
}

func TestDecodeBoundedRejectsDuplicatesTrailingDataAndSyntax(t *testing.T) {
	for _, input := range []string{
		`{"a":1,"a":2}`,
		`{"a":1,"a":2}`,
		`[{"x":{"y":null,"y":null}}]`,
	} {
		if _, err := DecodeBounded([]byte(input), wideBounds, floatNumber); !errors.Is(err, ErrDuplicateKey) {
			t.Errorf("DecodeBounded(%s) = %v, want ErrDuplicateKey", input, err)
		}
	}
	for _, input := range []string{`{} {}`, `1 2`, `"a""b"`} {
		if _, err := DecodeBounded([]byte(input), wideBounds, floatNumber); !errors.Is(err, ErrTrailingData) {
			t.Errorf("DecodeBounded(%s) = %v, want ErrTrailingData", input, err)
		}
	}
	for _, input := range []string{``, ` `, `{`, `{"a":}`, `[1,]`, `{"a" 1}`, `{}x`, `01`, `nul`} {
		if value, err := DecodeBounded([]byte(input), wideBounds, floatNumber); err == nil {
			t.Errorf("DecodeBounded(%q) = %#v, want an error", input, value)
		}
	}
}
