package auditservice

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedAuditTextBoundsCountUTF8Bytes holds each Audit scope field
// (LIM-AUD-29) and every review rationale (LIM-REV-11) to their shared UTF-8
// byte bounds: a value of exactly the bound in any character is valid and one
// byte more is not.
func TestSharedAuditTextBoundsCountUTF8Bytes(t *testing.T) {
	t.Parallel()
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	bounds := map[string]int{"LIM-AUD-29": maximumScopeValue, "LIM-REV-11": MaxFindingRationale}
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
			var valid bool
			if field.Rule == "LIM-REV-11" {
				valid = validRationale(value.Value)
			} else {
				scope := Scope{}
				switch field.Path[len(field.Path)-1] {
				case "objective":
					scope.Objective = value.Value
				case "target":
					scope.Target = value.Value
				case "authorizationScope":
					scope.AuthorizationScope = value.Value
				default:
					t.Fatalf("unknown Audit scope field %v", field.Path)
				}
				_, err := normalizeScope(scope)
				if err != nil && !errors.Is(err, ErrInvalid) {
					t.Fatalf("Audit scope error %v is not ErrInvalid", err)
				}
				valid = err == nil
			}
			if valid != value.Within {
				t.Errorf("%s %v of %d bytes of %q: valid %v, within %v", field.Schema, field.Path, len(value.Value), value.Character, valid, value.Within)
			}
		}
	}
	if len(checked) != len(bounds) {
		t.Fatalf("shared cases cover %v of the Audit text rules", checked)
	}
}
