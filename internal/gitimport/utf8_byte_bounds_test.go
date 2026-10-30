package gitimport

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedGitImportBoundsCountUTF8Bytes checks the requested ref
// (LIM-GIT-22) and the repository URL as the caller sends it (LIM-GIT-23)
// with each shared character at and one byte over the declared bound.
func TestSharedGitImportBoundsCountUTF8Bytes(t *testing.T) {
	t.Parallel()
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	checked := map[string]bool{}
	for _, field := range fields {
		var bound int
		var valid func(string) bool
		values := field.Values(characters)
		switch field.Rule {
		case "LIM-GIT-22":
			bound, valid = MaximumRefBytes, validRef
		case "LIM-GIT-23":
			bound = MaximumRemoteURLBytes
			valid = func(raw string) bool {
				_, err := ParseRemote(raw)
				return err == nil
			}
			values = field.EmbeddedValues(characters, "https://git.example.com/", "")
		default:
			continue
		}
		if bound != field.MaxBytes {
			t.Fatalf("%s %v: shared bound %d, Go bound %d", field.Schema, field.Path, field.MaxBytes, bound)
		}
		checked[field.Rule] = true
		for _, value := range values {
			if valid(value.Value) != value.Within {
				t.Errorf("%s of %d bytes of %q: valid %v, within %v", field.Rule, len(value.Value), value.Character, !value.Within, value.Within)
			}
		}
	}
	if len(checked) != 2 {
		t.Fatalf("shared Git import byte bounds checked: %v", checked)
	}
}
