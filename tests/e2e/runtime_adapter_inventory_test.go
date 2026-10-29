package e2e

import (
	"slices"
	"testing"
)

// Runtime CLI selection controls configurable adapters; the controlled HTTP
// adapter is always advertised. Compare the entire sorted inventory, including
// that builtin. RuntimeConfig provenance contains only configured adapters and
// must not use this expectation.
func matchesCurrentRuntimeAdapters(advertised []string, selected ...string) bool {
	want := append([]string{"pentest-target-http@1"}, selected...)
	slices.Sort(want)
	return slices.Equal(advertised, want)
}

func TestCurrentRuntimeAdapterInventory(t *testing.T) {
	cases := []struct {
		name       string
		advertised []string
		selected   []string
		want       bool
	}{
		{"http", []string{"http-proxy@1", "pentest-target-http@1"}, []string{"http-proxy@1"}, true},
		{"caido", []string{"caido-graphql@1", "pentest-target-http@1"}, []string{"caido-graphql@1"}, true},
		{"telemetry", []string{"otlp-http@1", "pentest-target-http@1"}, []string{"otlp-http@1"}, true},
		{"builtin_only", []string{"pentest-target-http@1"}, nil, true},
		{"missing_builtin", []string{"http-proxy@1"}, []string{"http-proxy@1"}, false},
		{"missing_specialization", []string{"pentest-target-http@1"}, []string{"http-proxy@1"}, false},
		{"wrong_specialization", []string{"otlp-http@1", "pentest-target-http@1"}, []string{"http-proxy@1"}, false},
		{"extra_known", []string{"http-proxy@1", "otlp-http@1", "pentest-target-http@1"}, []string{"http-proxy@1"}, false},
		{"unknown", []string{"http-proxy@1", "pentest-target-http@1", "unknown@1"}, []string{"http-proxy@1"}, false},
		{"duplicate", []string{"http-proxy@1", "pentest-target-http@1", "pentest-target-http@1"}, []string{"http-proxy@1"}, false},
		{"unsorted", []string{"pentest-target-http@1", "http-proxy@1"}, []string{"http-proxy@1"}, false},
		{"empty", nil, nil, false},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if got := matchesCurrentRuntimeAdapters(test.advertised, test.selected...); got != test.want {
				t.Fatalf("inventory %v with selected %v matches = %v, want %v", test.advertised, test.selected, got, test.want)
			}
		})
	}
}
