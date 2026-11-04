package auditservice

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/jackc/pgx/v5"
)

type engagementSelection struct {
	artifact      auditstore.ExactArtifact
	document      engagementcontract.Document
	credentialIDs []string
}

func readEngagement(ctx context.Context, service *artifacts.Service, projectID string, profile auditprofile.PinnedProgram, selection DraftSelection) (*engagementSelection, error) {
	var selected *engagementSelection
	for name, exact := range selection.Inputs {
		if exact.MediaType != engagementcontract.MediaType {
			continue
		}
		slot, ok := profile.Program.Inputs[name]
		if !ok || !slices.Contains(slot.MediaTypes, engagementcontract.MediaType) || len(slot.MediaTypes) != 1 || selected != nil {
			return nil, fmt.Errorf("%w: engagement needs exactly one explicit typed input slot", ErrInvalid)
		}
		selected = &engagementSelection{artifact: exact, credentialIDs: []string{}}
	}
	if selected == nil {
		return nil, nil
	}
	store, err := service.Project(projectID)
	if err != nil {
		return nil, err
	}
	// Check metadata before loading bytes; an oversized engagement is rejected
	// even when selected by an earlier draft.
	if selected.artifact.SizeBytes > engagementcontract.MaxDocumentBytes {
		return nil, fmt.Errorf("%w: engagement exceeds its byte limit", ErrInvalid)
	}
	read, err := store.ReadExact(ctx, selected.artifact)
	if err != nil {
		return nil, err
	}
	document, err := engagementcontract.Decode(read.Payload.Data)
	if err != nil {
		return nil, fmt.Errorf("%w: engagement engagement failed validation", ErrInvalid)
	}
	selected.document = document
	for _, identity := range document.Identities {
		if identity.Credential != nil {
			selected.credentialIDs = append(selected.credentialIDs, identity.Credential.CredentialID)
		}
	}
	selected.credentialIDs = mergeIDs(selected.credentialIDs)
	return selected, nil
}

func validateEngagementCredentials(ctx context.Context, tx pgx.Tx, user credentials.RuntimeCredentialUser, selected *engagementSelection) error {
	if selected == nil {
		return nil
	}
	repository := credentials.NewRuntimeCredentialRepository(tx)
	for _, identity := range selected.document.Identities {
		if identity.Credential == nil {
			continue
		}
		if err := repository.ValidateRuntimeCredentialUse(ctx, user, identity.Credential.CredentialID, string(identity.Credential.Kind)); err != nil {
			// Wrong kind and foreign IDs share the safe missing-credential outcome.
			if errors.Is(err, credentials.ErrRuntimeCredentialNotFound) || errors.Is(err, credentials.ErrRuntimeCredentialInvalid) {
				return credentials.ErrRuntimeCredentialNotFound
			}
			return err
		}
	}
	return nil
}

func engagementCompatibility(profile auditprofile.PinnedProgram, activeChecks contracts.ActiveChecksPolicy, selected *engagementSelection) []CompatibilityReason {
	reasons := []CompatibilityReason{}
	if selected == nil {
		return reasons
	}
	missing := false
	for _, capability := range append(slices.Clone(selected.document.Policy.RequiredCapabilities), selected.document.Policy.OptionalCapabilities...) {
		// Controlled identity preflight has a placed Runtime consumer. Other
		// execution capabilities still need their own production consumer.
		if capability != engagementcontract.Capability && capability != string(runtimesettings.RuntimeAdapterHTTPTarget) {
			missing = true
		}
	}
	network := false
	for _, binding := range profile.Workflows {
		for _, stage := range binding.Stages {
			for _, agent := range stage.Agents {
				if slices.Contains(selected.document.Policy.RequiredCapabilities, string(runtimesettings.RuntimeAdapterHTTPTarget)) {
					network = network || resolved.TemplateRequiresUncontrolledNetwork(agent.Template)
				} else {
					network = network || resolved.TemplateRequiresNetwork(agent.Template)
				}
			}
		}
	}
	if missing || network {
		reasons = append(reasons, ReasonEngagementCapabilityUnsupported)
	}
	if activeChecks != contracts.ActiveChecksAutomatic || profile.Program.Policy.ActiveOperations != "allowed" {
		reasons = append(reasons, ReasonEngagementActiveChecksUnsupported)
	}
	return reasons
}

func validateEngagementBaseline(baseline BaselineSnapshot) error {
	engagement := baseline.Engagement
	if engagement == nil {
		return nil
	}
	matched := false
	for _, input := range baseline.Inputs {
		matched = matched || (input.Verify(engagement.Artifact) == nil && input.MediaType == engagementcontract.MediaType)
	}
	if !matched {
		return fmt.Errorf("%w: engagement is not an exact baseline input", ErrInvalid)
	}
	if engagement.Schema != auditbaseline.EngagementSchema || engagement.Artifact.Validate() != nil || !engagement.Artifact.HasMetadata() || engagement.Artifact.MediaType != engagementcontract.MediaType || engagement.Document.Validate() != nil || engagement.AuthorizedBy == "" || engagement.AuthorizedAt.IsZero() {
		return fmt.Errorf("%w: engagement baseline is invalid", ErrInvalid)
	}
	digest, err := engagement.Document.Policy.Digest()
	if err != nil || engagement.PolicyDigest != digest {
		return fmt.Errorf("%w: engagement policy digest differs", ErrInvalid)
	}
	for _, identity := range engagement.Document.Identities {
		if identity.Credential != nil && !slices.Contains(baseline.RuntimeCredentialIDs, identity.Credential.CredentialID) {
			return fmt.Errorf("%w: engagement credential is not held", ErrInvalid)
		}
	}
	return nil
}
