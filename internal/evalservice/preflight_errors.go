package evalservice

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// Dependency errors retain their cause. Only documented validation/not-found
// outcomes require editing the draft; unknown I/O and database failures retry.
func definiteConfigurationError(cause error) bool {
	var skill *agentskills.RunSkillError
	var validation *agentskills.ValidationError
	if errors.As(cause, &skill) && !skill.Retryable || errors.As(cause, &validation) {
		return true
	}
	for _, known := range []error{
		runtimeconfig.ErrInvalid, runtimeconfig.ErrNotFound, runtimeconfig.ErrConflict,
		credentials.ErrInvalid, credentials.ErrNotFound, credentials.ErrConflict,
		credentials.ErrRuntimeCredentialInvalid, credentials.ErrRuntimeCredentialNotFound,
		auditstandards.ErrInvalid, auditstandards.ErrNotFound, auditstandards.ErrDrift,
	} {
		if errors.Is(cause, known) {
			return true
		}
	}
	return false
}

func preflightDependencyError(cause error) error {
	if definiteConfigurationError(cause) {
		return errors.Join(evaldomain.Failure("eval_not_ready"), cause)
	}
	if errors.Is(cause, artifacts.ErrArtifactNotFound) || errors.Is(cause, artifacts.ErrArtifactIntegrity) || errors.Is(cause, artifacts.ErrBlobMissing) {
		return errors.Join(evaldomain.Failure("eval_evidence_unavailable"), cause)
	}
	if errors.Is(cause, projectstore.ErrNotFound) {
		return errors.Join(evaldomain.Failure("eval_not_found"), cause)
	}
	return cause
}

// A Workflow selection that fails validation needs a draft edit. A credential
// lookup outage keeps its cause, so only a confirmed configuration failure
// returns the experiment to draft and recovery or storage failures retry.
func workflowSelectionError(cause error) error {
	if errors.Is(cause, resolved.ErrCredentialLookup) {
		return preflightDependencyError(cause)
	}
	return errors.Join(evaldomain.Failure("eval_not_ready"), cause)
}
