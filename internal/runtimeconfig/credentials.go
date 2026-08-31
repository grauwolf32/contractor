package runtimeconfig

import (
	"context"
	"fmt"
)

const (
	CredentialKindOTLPHeaders = "otlp-headers@1"
	CredentialKindProxyBasic  = "http-proxy-basic@1"
	CredentialKindProxyBearer = "http-proxy-bearer@1"
)

type RuntimeCredentialValidator interface {
	ValidateRuntimeCredential(context.Context, string, ...string) error
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
