package preview

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedArchivePathCountsUTF8Bytes checks an archive file path
// (LIM-API-27) with each shared character at and one byte over the declared
// bound.
func TestSharedArchivePathCountsUTF8Bytes(t *testing.T) {
	t.Parallel()
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	checked := false
	for _, field := range fields {
		if field.Rule != "LIM-API-27" {
			continue
		}
		if field.MaxBytes != MaximumPathBytes {
			t.Fatalf("%s: shared bound %d, Go bound %d", field.Schema, field.MaxBytes, MaximumPathBytes)
		}
		checked = true
		for _, value := range field.Values(characters) {
			if ValidPath(value.Value) != value.Within {
				t.Errorf("path of %d bytes of %q: within %v", len(value.Value), value.Character, value.Within)
			}
		}
	}
	if !checked {
		t.Fatal("no shared archive path bound")
	}
}
