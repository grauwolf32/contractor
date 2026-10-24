package evalservice

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/apierror"
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

// A Run Skill failure has its own public code (SKL-R117), but preflight keeps
// answering a definite one with eval_not_ready, which comes first in the
// joined error; a transient one keeps its own retryable code.
func TestSkillPreflightFailuresKeepTheEvalAnswer(t *testing.T) {
	for _, test := range []struct {
		cause *agentskills.RunSkillError
		code  string
	}{
		{&agentskills.RunSkillError{Code: agentskills.CodeArtifactNotFound, Name: "review"}, "eval_not_ready"},
		{&agentskills.RunSkillError{Code: agentskills.CodeLimitExceeded, Name: "review"}, "eval_not_ready"},
		{&agentskills.RunSkillError{Code: agentskills.CodeArtifactUnavailable, Name: "review", Retryable: true},
			agentskills.CodeArtifactUnavailable},
	} {
		t.Run(test.cause.Code, func(t *testing.T) {
			classified, ok := apierror.Of(preflightDependencyError(test.cause))
			if !ok || classified.ErrorKind().Code() != test.code {
				t.Fatalf("preflight answer = %v %t; want %s", classified, ok, test.code)
			}
		})
	}
}
