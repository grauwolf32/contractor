package clone

import "testing"

func TestPointerCopiesValue(t *testing.T) {
	if Pointer[int](nil) != nil {
		t.Fatal("Pointer(nil) is not nil")
	}
	source := "value"
	copied := Pointer(&source)
	*copied = "changed"
	if source != "value" {
		t.Fatal("Pointer aliases its source")
	}
}

func TestMapIsNeverNilAndDoesNotAlias(t *testing.T) {
	empty := Map[map[string]string](nil)
	if empty == nil {
		t.Fatal("Map(nil) is nil")
	}
	empty["added"] = "ok"
	source := map[string]string{"a": "1"}
	copied := Map(source)
	copied["a"] = "2"
	if source["a"] != "1" {
		t.Fatal("Map aliases its source")
	}
}
