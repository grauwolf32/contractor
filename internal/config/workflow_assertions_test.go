package config

import (
	"slices"
	"sort"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func selectedToolsets(selections []contracts.ToolsetSelection) map[string][]string {
	result := make(map[string][]string, len(selections))
	for _, selection := range selections {
		ref := selection.Ref.ToolsetID + "@" + selection.Ref.Version
		result[ref] = selection.Tools
	}
	return result
}

func assertExactTemplateTools(t *testing.T, template contracts.ResolvedAgentTemplate, want map[string][]string) {
	t.Helper()
	got := make(map[string][]string, len(template.Toolsets))
	for _, selection := range template.Toolsets {
		ref := selection.Ref.ToolsetID + "@" + selection.Ref.Version
		got[ref] = append([]string(nil), selection.Tools...)
		sort.Strings(got[ref])
	}
	for ref := range want {
		sort.Strings(want[ref])
	}
	if len(got) != len(want) {
		t.Fatalf("%s@%s Toolsets = %v, want %v", template.Ref.TemplateID, template.Ref.Version, got, want)
	}
	for ref, operations := range want {
		if !slices.Equal(got[ref], operations) {
			t.Errorf("%s@%s %s tools = %v, want %v", template.Ref.TemplateID, template.Ref.Version, ref, got[ref], operations)
		}
	}
}

func assertNext(t *testing.T, action TransitionAction, want string) {
	t.Helper()
	if action.Kind != TransitionNext || action.NextStage != want {
		t.Fatalf("next transition = %+v, want %q", action, want)
	}
}

func assertBoundedRetry(t *testing.T, action TransitionAction, maxAttempts int) {
	t.Helper()
	if action.Kind != TransitionRetry || action.Retry == nil || action.Retry.MaxAttempts != maxAttempts || action.Retry.Then.Kind != TransitionFail {
		t.Fatalf("retry transition = %+v, want maxAttempts=%d then fail", action, maxAttempts)
	}
}
