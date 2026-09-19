package evalservice

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// Dependency errors retain their cause. Only documented validation/not-found
// outcomes require editing the draft; unknown I/O and database failures retry.
func preflightDependencyError(cause error) error {
	var skill *agentskills.RunSkillError
	var validation *agentskills.ValidationError
	if errors.As(cause, &skill) && !skill.Retryable || errors.As(cause, &validation) {
		return errors.Join(evaldomain.Failure("eval_not_ready"), cause)
	}
	for _, known := range []error{runtimeconfig.ErrInvalid, runtimeconfig.ErrNotFound, runtimeconfig.ErrConflict, credentials.ErrInvalid, credentials.ErrNotFound, auditstandards.ErrInvalid, auditstandards.ErrNotFound, auditstandards.ErrDrift} {
		if errors.Is(cause, known) {
			return errors.Join(evaldomain.Failure("eval_not_ready"), cause)
		}
	}
	if errors.Is(cause, artifacts.ErrArtifactNotFound) || errors.Is(cause, artifacts.ErrArtifactIntegrity) || errors.Is(cause, artifacts.ErrBlobMissing) {
		return errors.Join(evaldomain.Failure("eval_evidence_unavailable"), cause)
	}
	if errors.Is(cause, projectstore.ErrNotFound) {
		return errors.Join(evaldomain.Failure("eval_not_found"), cause)
	}
	return cause
}

// Catalog validation intentionally uses content-free credential messages. Keep
// its dependency failure separately so a failed lookup cannot become a draft
// error merely because the catalog hid the underlying database error.
type preflightCredentialLookup struct {
	config.CredentialLookup
	failure error
}

func (l *preflightCredentialLookup) LookupLLMCredential(ctx context.Context, id string) (config.CredentialMetadata, error) {
	value, err := l.CredentialLookup.LookupLLMCredential(ctx, id)
	if err != nil {
		l.failure = err
	}
	return value, err
}
func (l *preflightCredentialLookup) catalogError(cause error) error {
	if l.failure != nil {
		return preflightDependencyError(l.failure)
	}
	return errors.Join(evaldomain.Failure("eval_not_ready"), cause)
}
