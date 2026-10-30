package scanservice

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedSweepParameterBoundsCountUTF8Bytes checks a sweep parameter name
// and value (LIM-SCN-94) with each shared character at and one byte over the
// declared bound.
func TestSharedSweepParameterBoundsCountUTF8Bytes(t *testing.T) {
	t.Parallel()
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	checked := map[string]bool{}
	for _, field := range fields {
		if field.Rule != "LIM-SCN-94" {
			continue
		}
		name := field.Path[len(field.Path)-1]
		bound := MaximumParameterValueBytes
		if name == "propertyNames" {
			bound = MaximumParameterNameBytes
		}
		if field.MaxBytes != bound {
			t.Fatalf("%s %v: shared bound %d, Go bound %d", field.Schema, field.Path, field.MaxBytes, bound)
		}
		checked[name] = true
		for _, value := range field.Values(characters) {
			parameters := map[string]string{"severity": value.Value}
			if name == "propertyNames" {
				parameters = map[string]string{value.Value: "high"}
			}
			if _, err := normalize(Request{AllProjects: true, Parameters: parameters}); (err == nil) != value.Within {
				t.Errorf("%s of %d bytes of %q: %v, within %v", name, len(value.Value), value.Character, err, value.Within)
			}
		}
	}
	if len(checked) != 2 {
		t.Fatalf("shared sweep parameter bounds checked: %v", checked)
	}
}
