package contracts_test

import (
	"bytes"
	"encoding/json"
	"sort"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedExecutionMetricIntegerBounds holds the allocation schema and the
// Server's private decoder to the execution report integer bound the Runtime
// model applies to the same cases (OBS-R111, LIM-OBS-17).
func TestSharedExecutionMetricIntegerBounds(t *testing.T) {
	var cases struct {
		Fixture string `json:"fixture"`
		Cases   []struct {
			Name  string                 `json:"name"`
			Set   map[string]json.Number `json:"set"`
			Valid bool                   `json:"valid"`
		} `json:"cases"`
	}
	raw := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "execution-metric-integer-cases.json")
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Cases) == 0 {
		t.Fatal("execution metric integer cases are empty")
	}
	entry, ok := readFixtureIndex(t).Valid[cases.Fixture]
	if !ok {
		t.Fatalf("case fixture %q is not a valid indexed fixture", cases.Fixture)
	}
	compiler, ids := privateSchemaCompiler(t)
	schema, err := compiler.Compile(ids[entry.Schema])
	if err != nil {
		t.Fatal(err)
	}
	fixture := contractstest.ReadFixture(t, "valid", cases.Fixture)
	var valid, invalid int
	for _, item := range cases.Cases {
		t.Run(item.Name, func(t *testing.T) {
			pointers := make([]string, 0, len(item.Set))
			for pointer := range item.Set {
				pointers = append(pointers, pointer)
			}
			sort.Strings(pointers)
			document := fixture
			for _, pointer := range pointers {
				document = withValue(t, document, pointer, item.Set[pointer])
			}
			if got := schema.Validate(decodeNumbers(t, document)) == nil; got != item.Valid {
				t.Errorf("schema accepted=%t, want %t", got, item.Valid)
			}
			if err := fixtureCodecs[entry.Type].privateReject(document); (err == nil) != item.Valid {
				t.Errorf("Go accepted=%t, want %t: %v", err == nil, item.Valid, err)
			}
		})
		if item.Valid {
			valid++
		} else {
			invalid++
		}
	}
	if valid == 0 || invalid == 0 {
		t.Fatalf("cases must hold both outcomes: %d valid, %d invalid", valid, invalid)
	}
}
