package auditdomain

import (
	"strconv"
	"strings"
	"testing"
)

// TestStrictJSONNumberSemantics pins the number rules of the two strict JSON
// parsers: canonical encodings of Server values accept every finite float64,
// while JSON source documents keep the safe-integer bound of their YAML form.
func TestStrictJSONNumberSemantics(t *testing.T) {
	for _, literal := range []string{"0", "-0", "1.5", "9007199254740993", "1e308", "-1e308", "5e-324"} {
		value, err := parseStrictJSON([]byte(`{"n":` + literal + `}`))
		want, _ := strconv.ParseFloat(literal, 64)
		if err != nil || value.(map[string]any)["n"] != want {
			t.Errorf("canonical number %s = %#v, %v; want %v", literal, value, err, want)
		}
	}
	for _, literal := range []string{"1e309", "-1e309"} {
		if value, err := parseStrictJSON([]byte(`{"n":` + literal + `}`)); err == nil {
			t.Errorf("canonical number %s = %#v, want an error", literal, value)
		}
	}
	for _, literal := range []string{"9007199254740991", "-9007199254740991", "1.25", "1e3"} {
		root, err := parseJSONOrYAML([]byte(`{"n":`+literal+`}`), "application/json")
		want, _ := strconv.ParseFloat(literal, 64)
		if err != nil || root["n"] != want {
			t.Errorf("source number %s = %#v, %v; want %v", literal, root, err, want)
		}
	}
	for _, literal := range []string{"9007199254740992", "9007199254740993", "9007199254740991.1", "1e300", "1e309"} {
		if _, err := parseJSONOrYAML([]byte(`{"n":`+literal+`}`), "application/json"); ErrorCode(err) != CodeInvalid {
			t.Errorf("source number %s error = %v, want %s", literal, err, CodeInvalid)
		}
	}
}

// TestStrictJSONBoundsAndStrictness pins the structure rules both parsers
// share: depth, node and string bounds, duplicate members and trailing data.
func TestStrictJSONBoundsAndStrictness(t *testing.T) {
	nested := func(depth int) string {
		return strings.Repeat("[", depth-1) + "0" + strings.Repeat("]", depth-1)
	}
	if _, err := parseStrictJSON([]byte(nested(MaximumJSONDepth))); err != nil {
		t.Fatalf("depth %d = %v", MaximumJSONDepth, err)
	}
	long := strings.Repeat("x", MaximumStringBytes)
	for name, input := range map[string]string{
		"depth":            nested(MaximumJSONDepth + 1),
		"string":           `{"s":"` + long + `x"}`,
		"member name":      `{"` + long + `x":1}`,
		"duplicate member": `{"a":1,"a":2}`,
		"trailing value":   `{} {}`,
		"lone surrogate":   `{"s":"\ud800"}`,
	} {
		if value, err := parseStrictJSON([]byte(input)); err == nil {
			t.Errorf("%s = %#v, want an error", name, value)
		}
	}
	if _, err := parseStrictJSON([]byte(`{"s":"` + long + `"}`)); err != nil {
		t.Fatalf("string at the byte bound = %v", err)
	}
	// An array and its elements are nodes; member names are not.
	elements := func(count int) string {
		return "[" + strings.TrimSuffix(strings.Repeat("0,", count), ",") + "]"
	}
	if _, err := parseStrictJSON([]byte(elements(MaximumJSONNodes - 1))); err != nil {
		t.Fatalf("%d nodes = %v", MaximumJSONNodes, err)
	}
	if _, err := parseStrictJSON([]byte(elements(MaximumJSONNodes))); err == nil {
		t.Fatalf("%d nodes were accepted", MaximumJSONNodes+1)
	}
}
