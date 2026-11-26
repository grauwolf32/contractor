package contracts_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPlannerAllocationNullSettings(t *testing.T) {
	base := contractstest.ReadFixture(t, "valid", "allocation-spec-planner.json")
	value := decodeNumbers(t, base).(map[string]any)
	patch := decodeNumbers(t, contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "planner-allocation-settings.json")).(map[string]any)
	for name, field := range patch {
		value["runtimeSettings"].(map[string]any)[name] = field
	}
	compiler, ids := privateSchemaCompiler(t)
	schema, err := compiler.Compile(ids["allocation.schema.json"])
	if err != nil {
		t.Fatal(err)
	}
	if err := schema.Validate(value); err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(value)
	got, err := fixtureCodecs["AllocationSpec"].roundTrip(raw)
	if err != nil {
		t.Fatal(err)
	}
	contractstest.AssertSemanticJSONEqual(t, base, got)
}
