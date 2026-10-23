package projectstore

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedProjectTextBoundsCountUTF8Bytes holds the Project name and
// description to their shared UTF-8 byte bounds: a value of exactly the bound
// in any character is valid and one byte more is not.
func TestSharedProjectTextBoundsCountUTF8Bytes(t *testing.T) {
	t.Parallel()
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	bounds := map[string]int{"LIM-PRJ-3": MaxNameBytes, "LIM-PRJ-5": MaxDescriptionBytes}
	checked := map[string]bool{}
	for _, field := range fields {
		bound, owned := bounds[field.Rule]
		if !owned {
			continue
		}
		if bound != field.MaxBytes {
			t.Fatalf("%s %v: shared bound %d, Go bound %d", field.Schema, field.Path, field.MaxBytes, bound)
		}
		checked[field.Rule] = true
		for _, value := range field.Values(characters) {
			name, description := value.Value, ""
			if field.Rule == "LIM-PRJ-5" {
				name, description = "Project", value.Value
			}
			if err := validateMetadata(name, description); (err == nil) != value.Within || (err != nil && !errors.Is(err, ErrInvalid)) {
				t.Errorf("%s of %d bytes of %q: error %v, within %v", field.Rule, len(value.Value), value.Character, err, value.Within)
			}
		}
	}
	if len(checked) != len(bounds) {
		t.Fatalf("shared cases cover %v of the Project text rules", checked)
	}
}
