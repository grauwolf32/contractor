package config

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

func assertNext(t *testing.T, action resolved.TransitionAction, want string) {
	t.Helper()
	if action.Kind != resolved.TransitionNext || action.NextStage != want {
		t.Fatalf("next transition = %+v, want %q", action, want)
	}
}

func assertBoundedRetry(t *testing.T, action resolved.TransitionAction, maxAttempts int) {
	t.Helper()
	if action.Kind != resolved.TransitionRetry || action.Retry == nil || action.Retry.MaxAttempts != maxAttempts || action.Retry.Then.Kind != resolved.TransitionFail {
		t.Fatalf("retry transition = %+v, want maxAttempts=%d then fail", action, maxAttempts)
	}
}
