package contentdigest

import (
	"errors"
	"strings"
	"testing"
)

func TestCheckPinnedSelection(t *testing.T) {
	value := map[string]string{"workflow": "copy@1"}
	if err := CheckPinnedSelection("", value); err != nil {
		t.Fatalf("empty expectation error = %v", err)
	}
	pinned, err := JSON(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckPinnedSelection(pinned, value); err != nil {
		t.Fatalf("unchanged selection error = %v", err)
	}
	if err := CheckPinnedSelection("sha256:"+strings.Repeat("0", 64), value); !errors.Is(err, ErrPinnedSelectionChanged) {
		t.Fatalf("changed selection error = %v", err)
	}
	if err := CheckPinnedSelection(pinned, func() {}); err == nil || errors.Is(err, ErrPinnedSelectionChanged) {
		t.Fatalf("unencodable selection error = %v", err)
	}
}
