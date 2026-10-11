package contracts

import (
	"encoding/json"
	"os"
	"testing"
)

func TestArtifactNameCases(t *testing.T) {
	data, err := os.ReadFile("../../api/testdata/v1alpha1/artifact-name-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases struct {
		Valid   []string
		Invalid []string
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, group := range []struct {
		names []string
		valid bool
	}{{cases.Valid, true}, {cases.Invalid, false}} {
		for _, name := range group.names {
			for _, ref := range []ArtifactRef{{Namespace: "worker", Name: name}, {Namespace: name, Name: "report"}} {
				if err := ref.Validate(); (err == nil) != group.valid {
					t.Errorf("%+v: valid=%v, error=%v", ref, group.valid, err)
				}
			}
		}
	}
}

func TestAcceptsMediaType(t *testing.T) {
	t.Parallel()
	cases := []struct {
		accepted []string
		actual   string
		want     bool
	}{
		{[]string{"*/*"}, "application/zip", true},
		{[]string{"*/*"}, "text/html", true},
		{[]string{"text/markdown", "text/plain"}, "text/plain", true},
		{[]string{"text/markdown"}, "text/plain", false},
		{[]string{"text/*"}, "text/plain", false},
		{nil, "text/plain", false},
	}
	for _, tc := range cases {
		if got := AcceptsMediaType(tc.accepted, tc.actual); got != tc.want {
			t.Errorf("AcceptsMediaType(%v, %q) = %v, want %v", tc.accepted, tc.actual, got, tc.want)
		}
	}
}

func TestArtifactListCursorCases(t *testing.T) {
	data, err := os.ReadFile("../../api/testdata/v1alpha1/artifact-list-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases struct {
		MaxCursorLength int `json:"maxCursorLength"`
		Cursors         struct {
			Valid   []string `json:"valid"`
			Invalid []string `json:"invalid"`
		} `json:"cursors"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if cases.MaxCursorLength != MaxArtifactListCursorLength || len(cases.Cursors.Valid) == 0 || len(cases.Cursors.Invalid) == 0 {
		t.Fatalf("shared cursor cases = %+v, Go maximum %d", cases, MaxArtifactListCursorLength)
	}
	for _, group := range []struct {
		cursors []string
		valid   bool
	}{{cases.Cursors.Valid, true}, {cases.Cursors.Invalid, false}} {
		for _, cursor := range group.cursors {
			if err := ValidateArtifactListCursor(cursor); (err == nil) != group.valid {
				t.Errorf("cursor %q: valid=%v, error=%v", cursor, group.valid, err)
			}
			refs := []ArtifactRef{{Namespace: "analysis", Name: "report"}}
			result := ArtifactListResult{APIVersion: APIVersion, Artifacts: refs, NextCursor: &cursor}
			if err := result.Validate(); (err == nil) != group.valid {
				t.Errorf("list result cursor %q: valid=%v, error=%v", cursor, group.valid, err)
			}
		}
	}
}
