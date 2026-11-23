package runlabels

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedRunMetadataLabelValuesCountUTF8Bytes holds a Run metadata label
// value to its shared UTF-8 byte bound (LIM-RUN-5).
func TestSharedRunMetadataLabelValuesCountUTF8Bytes(t *testing.T) {
	t.Parallel()
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	checked := false
	for _, field := range fields {
		if field.Rule != "LIM-RUN-5" {
			continue
		}
		if field.MaxBytes != MaxRunMetadataLabelValue {
			t.Fatalf("shared label value bound %d, Go bound %d", field.MaxBytes, MaxRunMetadataLabelValue)
		}
		checked = true
		for _, value := range field.Values(characters) {
			if err := ValidateRunMetadataLabel("team", value.Value); (err == nil) != value.Within {
				t.Errorf("label value of %d bytes of %q: error %v, within %v", len(value.Value), value.Character, err, value.Within)
			}
		}
	}
	if !checked {
		t.Fatal("shared cases have no Run metadata label value bound")
	}
}
