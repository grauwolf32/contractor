package runtimeconfig

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/jackc/pgx/v5"
)

const (
	CredentialKindOTLPHeaders = "otlp-headers@1"
	CredentialKindProxyBasic  = "http-proxy-basic@1"
	CredentialKindProxyBearer = "http-proxy-bearer@1"
	CredentialKindCaidoBearer = "caido-bearer@1"
)

type RuntimeCredentialValidator interface {
	ValidateRuntimeCredential(context.Context, string, ...string) error
}

// TransactionLLMCredentialLookupFactory binds every database-backed metadata
// lookup to the transaction which will persist the resulting Run snapshot.
// Implementations may also include process-local providers, but must not
// acquire another pooled PostgreSQL connection.
type TransactionLLMCredentialLookupFactory interface {
	ForTransaction(pgx.Tx) (config.CredentialLookup, error)
}

// TransactionLLMCredentialLookup is deliberately constructible only through
// BindTransactionLLMCredentialLookup. PinRunSnapshot accepts this concrete
// wrapper so a caller cannot accidentally pass the process-wide, pool-backed
// credential provider while already holding a transaction connection.
type TransactionLLMCredentialLookup struct {
	lookup config.CredentialLookup
}

func BindTransactionLLMCredentialLookup(
	tx pgx.Tx,
	factory TransactionLLMCredentialLookupFactory,
) (TransactionLLMCredentialLookup, error) {
	if tx == nil || factory == nil {
		return TransactionLLMCredentialLookup{}, errors.New("transaction LLM credential lookup is not configured")
	}
	lookup, err := factory.ForTransaction(tx)
	if err != nil {
		return TransactionLLMCredentialLookup{}, fmt.Errorf("bind transaction LLM credential lookup: %w", err)
	}
	if lookup == nil {
		return TransactionLLMCredentialLookup{}, errors.New("transaction LLM credential lookup factory returned nil")
	}
	return TransactionLLMCredentialLookup{lookup: lookup}, nil
}

func (l TransactionLLMCredentialLookup) LookupLLMCredential(
	ctx context.Context,
	credentialID string,
) (config.CredentialMetadata, error) {
	if l.lookup == nil {
		return config.CredentialMetadata{}, errors.New("transaction LLM credential lookup is not bound")
	}
	return l.lookup.LookupLLMCredential(ctx, credentialID)
}

// TransactionLLMCredentialLookupFactoryFunc keeps tests and non-PostgreSQL
// composition explicit without weakening the production transaction boundary.
type TransactionLLMCredentialLookupFactoryFunc func(pgx.Tx) (config.CredentialLookup, error)

func (f TransactionLLMCredentialLookupFactoryFunc) ForTransaction(
	tx pgx.Tx,
) (config.CredentialLookup, error) {
	if f == nil {
		return nil, errors.New("transaction LLM credential lookup factory is nil")
	}
	return f(tx)
}

type CredentialReferenceBarrier interface {
	WithCredentialReferences(context.Context, func() error) error
}

// RuntimeCredentialCatalog combines validation and its reference fence so a
// caller cannot accidentally validate through one service while committing
// under a different lifecycle barrier.
type RuntimeCredentialCatalog interface {
	RuntimeCredentialValidator
	CredentialReferenceBarrier
}

func validateSpecRuntimeCredentials(
	ctx context.Context, spec Spec, validator RuntimeCredentialValidator,
) error {
	type requirement struct {
		credentialID string
		allowedKinds []string
	}
	requirements := make([]requirement, 0, 3)
	if spec.Worker.Telemetry.Present && !spec.Worker.Telemetry.Clear && spec.Worker.Telemetry.Value.Credential != "" {
		requirements = append(requirements, requirement{
			credentialID: spec.Worker.Telemetry.Value.Credential,
			allowedKinds: []string{CredentialKindOTLPHeaders},
		})
	}
	if spec.Worker.HTTPProxy.Present && !spec.Worker.HTTPProxy.Clear && spec.Worker.HTTPProxy.Value.Credential != "" {
		requirements = append(requirements, requirement{
			credentialID: spec.Worker.HTTPProxy.Value.Credential,
			allowedKinds: []string{CredentialKindProxyBasic, CredentialKindProxyBearer},
		})
	}
	if spec.Worker.Caido.Present && !spec.Worker.Caido.Clear && spec.Worker.Caido.Value.Credential != "" {
		requirements = append(requirements, requirement{
			credentialID: spec.Worker.Caido.Value.Credential,
			allowedKinds: []string{CredentialKindCaidoBearer},
		})
	}
	if spec.Planner.Telemetry.Present && !spec.Planner.Telemetry.Clear && spec.Planner.Telemetry.Value.Credential != "" {
		requirements = append(requirements, requirement{
			credentialID: spec.Planner.Telemetry.Value.Credential,
			allowedKinds: []string{CredentialKindOTLPHeaders},
		})
	}
	if len(requirements) == 0 {
		return nil
	}
	if validator == nil {
		return invalid("Runtime credential validator is not configured")
	}
	for _, requirement := range requirements {
		if err := validator.ValidateRuntimeCredential(ctx, requirement.credentialID, requirement.allowedKinds...); err != nil {
			if contextError := ctx.Err(); contextError != nil {
				return contextError
			}
			return fmt.Errorf("%w: RuntimeConfig references an unavailable or incompatible credential", ErrInvalid)
		}
	}
	return nil
}
