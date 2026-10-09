package evaldomain

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestEvalMediaAndSelectorMatchSharedCases(t *testing.T) {
	type group struct {
		Valid, Invalid []string
		MaxLength      int
	}
	read := func(path string, target any) {
		t.Helper()
		raw, err := os.ReadFile("../../api/testdata/v1alpha1/" + path)
		if err != nil || json.Unmarshal(raw, target) != nil {
			t.Fatalf("read shared cases %s: %v", path, err)
		}
	}
	check := func(kind, value string, valid bool) {
		t.Helper()
		raw, _ := json.Marshal(value)
		if err := Validate(kind, raw); (err == nil) != valid {
			t.Errorf("%s %q: valid=%t error=%v", kind, value, valid, err)
		}
	}
	var media group
	read("media-type-cases.json", &media)
	for _, value := range media.Valid {
		check("Media", value, true)
	}
	for _, value := range media.Invalid {
		check("Media", value, false)
	}
	var identity struct{ ID, Version group }
	read("config-identity-cases.json", &identity)
	for _, part := range []struct {
		cases group
		join  func(string) string
	}{
		{identity.ID, func(value string) string { return value + "@1" }},
		{identity.Version, func(value string) string { return "a@" + value }},
	} {
		for _, value := range part.cases.Valid {
			check("Selector", part.join(value), true)
		}
		for _, value := range part.cases.Invalid {
			check("Selector", part.join(value), false)
		}
		check("Selector", part.join(strings.Repeat("a", part.cases.MaxLength)), true)
		check("Selector", part.join(strings.Repeat("a", part.cases.MaxLength+1)), false)
	}
	check("Selector", strings.Repeat("a", identity.ID.MaxLength)+"@"+strings.Repeat("V", identity.Version.MaxLength), true)
}
