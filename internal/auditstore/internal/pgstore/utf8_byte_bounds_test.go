package pgstore

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedSubjectKeyCountsUTF8Bytes checks the item subject filter
// (LIM-AUD-40) with each shared character at and one byte over the declared
// bound.
func TestSharedSubjectKeyCountsUTF8Bytes(t *testing.T) {
	t.Parallel()
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	checked := false
	for _, field := range fields {
		if field.Rule != "LIM-AUD-40" {
			continue
		}
		if field.MaxBytes != auditstore.MaxSubjectKeyBytes {
			t.Fatalf("%s: shared bound %d, Go bound %d", field.Schema, field.MaxBytes, auditstore.MaxSubjectKeyBytes)
		}
		checked = true
		for _, value := range field.Values(characters) {
			subject := value.Value
			err := validateListItems(auditstore.ListItemsParams{OwnerID: "user-1", AuditID: "audit-1", SubjectKey: &subject, Limit: 10})
			if (err == nil) != value.Within {
				t.Errorf("subject of %d bytes of %q: %v, within %v", len(value.Value), value.Character, err, value.Within)
			}
		}
	}
	if !checked {
		t.Fatal("no shared subject key bound")
	}
}
