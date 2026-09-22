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
