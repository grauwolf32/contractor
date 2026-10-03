package evalservice

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestCredentialPreflightSeparatesInvalidDependenciesFromLookupOutages(t *testing.T) {
	for _, test := range []struct {
		name     string
		cause    error
		notReady bool
	}{
		{"missing LLM", credentials.ErrNotFound, true},
		{"missing Runtime", credentials.ErrRuntimeCredentialNotFound, true},
		{"wrong kind", credentials.ErrRuntimeCredentialInvalid, true},
		{"identity mismatch", runtimeconfig.ErrInvalid, true},
		{"recovery", credentials.ErrRecoveryRequired, false},
		{"storage", errors.New("storage unavailable"), false},
	} {
		t.Run(test.name, func(t *testing.T) {
			classified := preflightDependencyError(test.cause)
			if !errors.Is(classified, test.cause) || evaldomain.IsCode(classified, "eval_not_ready") != test.notReady {
				t.Fatalf("preflight classification = %v; want notReady=%t", classified, test.notReady)
			}
		})
	}
}
