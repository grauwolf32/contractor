//go:build e2e

package e2e

import (
	"slices"
	"testing"
)

func supportsRuntimeAdapters(supported []string, required ...string) bool {
	for _, adapter := range required {
		if !slices.Contains(supported, adapter) {
			return false
		}
	}
	return true
}

func findPrincipalWithAdapters(
	principals []observedRuntimePrincipal,
	adapters ...string,
) (observedRuntimePrincipal, bool) {
	var selected observedRuntimePrincipal
	found := false
	for _, principal := range principals {
		if principal.Live == nil || !supportsRuntimeAdapters(principal.Live.SupportedRuntimeAdapters, adapters...) {
			continue
		}
		if found {
			return observedRuntimePrincipal{}, false
		}
		selected, found = principal, true
	}
	return selected, found
}

func TestRuntimeAdapterSelectionAllowsAdditionalCapabilities(t *testing.T) {
	t.Run("required subset", func(t *testing.T) {
		for _, test := range []struct {
			name      string
			supported []string
			required  []string
			want      bool
		}{
			{
				name: "additional built-in adapter", want: true,
				supported: []string{"http-proxy@1", "pentest-target-http@1"},
				required:  []string{"http-proxy@1"},
			},
			{
				name: "required adapters in different order", want: true,
				supported: []string{"pentest-target-http@1", "otlp-http@1", "http-proxy@1"},
				required:  []string{"http-proxy@1", "otlp-http@1"},
			},
			{
				name:      "one requirement missing",
				supported: []string{"http-proxy@1", "pentest-target-http@1"},
				required:  []string{"http-proxy@1", "caido-graphql@1"},
			},
			{
				name: "no capabilities", required: []string{"http-proxy@1"},
			},
		} {
			t.Run(test.name, func(t *testing.T) {
				if got := supportsRuntimeAdapters(test.supported, test.required...); got != test.want {
					t.Fatalf("supportsRuntimeAdapters(%v, %v) = %v, want %v", test.supported, test.required, got, test.want)
				}
			})
		}
	})

	proxy := observedRuntimePrincipal{
		RuntimeAgentID: "proxy",
		Live: &observedRuntimeAgent{
			SupportedRuntimeAdapters: []string{"pentest-target-http@1", "http-proxy@1"},
		},
	}
	caido := observedRuntimePrincipal{
		RuntimeAgentID: "caido",
		Live: &observedRuntimeAgent{
			SupportedRuntimeAdapters: []string{"caido-graphql@1", "pentest-target-http@1"},
		},
	}
	principals := []observedRuntimePrincipal{{RuntimeAgentID: "offline"}, caido, proxy}
	t.Run("live principal with additional capability", func(t *testing.T) {
		if got := principalWithAdapters(t, principals, "http-proxy@1"); got.RuntimeAgentID != proxy.RuntimeAgentID {
			t.Fatalf("selected principal = %s, want %s", got.RuntimeAgentID, proxy.RuntimeAgentID)
		}
	})
	t.Run("missing principal", func(t *testing.T) {
		if _, ok := findPrincipalWithAdapters(principals, "otlp-http@1"); ok {
			t.Fatal("selected a principal without the required adapter")
		}
	})
	t.Run("ambiguous principal", func(t *testing.T) {
		if _, ok := findPrincipalWithAdapters(principals, "pentest-target-http@1"); ok {
			t.Fatal("selected a principal when two live principals match")
		}
	})
}
