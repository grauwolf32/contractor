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
	"github.com/jackc/pgx/v5"
)

// Dependency errors retain their cause. Only documented validation/not-found
// outcomes require editing the draft; unknown I/O and database failures retry.
func preflightDependencyError(cause error) error {
	var skill *agentskills.RunSkillError
	var validation *agentskills.ValidationError
	if errors.As(cause, &skill) && !skill.Retryable || errors.As(cause, &validation) {
		return errors.Join(evaldomain.Failure("eval_not_ready"), cause)
	}
	for _, known := range []error{
		runtimeconfig.ErrInvalid, runtimeconfig.ErrNotFound, runtimeconfig.ErrConflict,
		credentials.ErrInvalid, credentials.ErrNotFound, credentials.ErrConflict,
		credentials.ErrRuntimeCredentialInvalid, credentials.ErrRuntimeCredentialNotFound,
		auditstandards.ErrInvalid, auditstandards.ErrNotFound, auditstandards.ErrDrift,
	} {
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

// Catalog validation hides lookup errors, and runtime pinning adds ErrInvalid
// even to infrastructure failures. Retain the dependency error before either
// boundary so only confirmed configuration failures return the experiment to draft.
type preflightCredentials struct {
	config.CredentialLookup
	runtime runtimeconfig.RuntimeCredentialValidator
	failure error
}

func bindPreflightCredentials(tx pgx.Tx, factory runtimeconfig.TransactionLLMCredentialLookupFactory) (*preflightCredentials, runtimeconfig.TransactionLLMCredentialLookup, error) {
	tracker := &preflightCredentials{runtime: credentials.NewRuntimeCredentialRepository(tx)}
	tracked := runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(func(tx pgx.Tx) (config.CredentialLookup, error) {
		lookup, err := factory.ForTransaction(tx)
		if err != nil || lookup == nil {
			return lookup, err
		}
		tracker.CredentialLookup = lookup
		return tracker, nil
	})
	lookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, tracked)
	return tracker, lookup, err
}

func (l *preflightCredentials) LookupLLMCredential(ctx context.Context, id string) (config.CredentialMetadata, error) {
	value, err := l.CredentialLookup.LookupLLMCredential(ctx, id)
	if err != nil {
		l.failure = err
	}
	return value, err
}

func (l *preflightCredentials) ValidateRuntimeCredential(ctx context.Context, id string, kinds ...string) error {
	err := l.runtime.ValidateRuntimeCredential(ctx, id, kinds...)
	if err != nil {
		l.failure = err
	}
	return err
}

func (l *preflightCredentials) dependencyError(cause error) error {
	if l.failure != nil {
		return preflightDependencyError(l.failure)
	}
	return preflightDependencyError(cause)
}

func (l *preflightCredentials) catalogError(cause error) error {
	if l.failure != nil {
		return preflightDependencyError(l.failure)
	}
	return errors.Join(evaldomain.Failure("eval_not_ready"), cause)
}
