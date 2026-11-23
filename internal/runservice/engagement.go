package runservice

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/engagement"
)

// selectRunEngagement runs inside the creation transaction and lifecycle guard.
// The returned exact source is also used by the ordinary input fork.
func selectRunEngagement(ctx context.Context, service *artifacts.Service, validator RuntimeCredentialValidator, workflow resolved.ResolvedWorkflow, params PublicCreateParams, slot string) (*engagement.Pin, error) {
	ref, exists := params.Inputs[slot]
	if slot == "" || !exists {
		return nil, nil
	}
	for _, stage := range workflow.Stages {
		for _, agent := range stage.Agents {
			if resolved.TemplateRequiresUncontrolledNetwork(agent.Template) {
				return nil, fmt.Errorf("%w: engagement does not authorize uncontrolled network tools", ErrInvalid)
			}
		}
	}
	scope, err := inputSourceScope(params.OwnerID, params.ProjectID, workflow.Inputs[slot].SourceScope)
	if err != nil {
		return nil, err
	}
	store, err := service.Scoped(scope)
	if err != nil {
		return nil, err
	}
	metadata, err := store.Metadata(ctx, ref)
	if err != nil {
		return nil, err
	}
	document, err := engagement.Read(ctx, store, metadata.Exact())
	if errors.Is(err, contracts.ErrValidation) {
		return nil, fmt.Errorf("%w: engagement failed validation", ErrInvalid)
	}
	if err != nil {
		return nil, err
	}
	for _, identity := range document.Identities {
		if identity.Credential == nil {
			continue
		}
		if validator == nil {
			return nil, ErrNotConfigured
		}
		err := validator.ValidateRuntimeCredentialUse(ctx, credentials.RuntimeCredentialUser{
			UserID: params.OwnerID, Operations: params.OperationsPrincipal,
		}, identity.Credential.CredentialID, string(identity.Credential.Kind))
		if errors.Is(err, credentials.ErrRuntimeCredentialInvalid) {
			return nil, credentials.ErrRuntimeCredentialNotFound
		}
		if err != nil {
			return nil, err
		}
	}
	return &engagement.Pin{Artifact: metadata.Exact(), Document: document}, nil
}
