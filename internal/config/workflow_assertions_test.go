package config

import "testing"

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
