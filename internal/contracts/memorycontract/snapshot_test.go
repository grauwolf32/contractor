package memorycontract

import (
	"fmt"
	"strings"
	"testing"
)

func TestSnapshotSelectionHasOneBoundedExactGrammar(t *testing.T) {
	selection, err := SnapshotOptions(map[string]any{"namespace": "writer", "notes": "second,first"})
	if err != nil || strings.Join(selection.Names, ",") != "first,second" {
		t.Fatal(selection, err)
	}
	names := make([]string, maximumSnapshotNotes)
	for i := range names {
		names[i] = fmt.Sprintf("note_%d", i)
	}
	if _, err := SnapshotOptions(map[string]any{"namespace": "writer", "notes": strings.Join(names, ",")}); err != nil {
		t.Fatal(err)
	}
	for _, notes := range []string{"", "a,a", "a, b", "a,", "a/b", "note-name", "Note", "é", strings.Repeat("a", MaximumNoteNameBytes+1), strings.Join(append(names, "extra"), ",")} {
		if _, err := SnapshotOptions(map[string]any{"namespace": "writer", "notes": notes}); err == nil {
			t.Fatalf("invalid note selection accepted: %q", notes)
		}
	}
	if !ValidNoteName(strings.Repeat("a", MaximumNoteNameBytes)) {
		t.Fatal("maximum logical name rejected")
	}
}
