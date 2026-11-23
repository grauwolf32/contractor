package control

import "github.com/grauwolf32/contractor/internal/contracts"

// These groups retain AllocationSpec.Validate's evaluation order and errors.
func (s AllocationSpec) validateEngagement() error {
	if len(s.RuntimeSettings.HTTPIdentities) > 0 && (s.Engagement == nil || s.Engagement.HTTP == nil) {
		return contracts.Invalidf("engagement material requires an assigned identity")
	}
	if s.Engagement != nil && s.Engagement.HTTP != nil {
		expected := 0
		for _, identity := range s.Engagement.HTTP.Identities {
			if identity.Credential == nil {
				continue
			}
			expected++
			material, ok := s.RuntimeSettings.HTTPIdentities[identity.ID]
			if !ok || material.Credential != *identity.Credential {
				return contracts.Invalidf("engagement private credential differs from assignment")
			}
		}
		if expected != len(s.RuntimeSettings.HTTPIdentities) {
			return contracts.Invalidf("unassigned engagement private material")
		}
	}

	if s.Engagement != nil {
		if err := s.Engagement.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func (s AllocationSpec) validateIdentity() error {
	if s.ProjectID != nil {
		if err := contracts.ValidateResourceID("projectId", *s.ProjectID); err != nil {
			return err
		}
	}
	if err := contracts.ValidateAPIVersion(s.APIVersion); err != nil {
		return err
	}
	for field, value := range map[string]string{
		"allocationId": s.AllocationID, "runId": s.RunID,
		"stageExecutionId": s.StageExecutionID, "namespace": s.Namespace,
	} {
		if err := contracts.ValidateOpaqueID(field, value); err != nil {
			return err
		}
	}
	if !contracts.ValidName(s.LogicalAgentName) {
		return contracts.Invalidf("logicalAgentName must be a name of at most %d characters", contracts.MaxNameLength)
	}
	if contracts.ValidateArtifactName(s.Namespace) != nil || s.LeaseExpiresAt.IsZero() {
		return contracts.Invalidf("allocation namespace or lease is invalid")
	}
	if err := s.WorkerSessionMode.Validate(); err != nil {
		return err
	}
	if err := s.RunMetadataLabels.Validate(); err != nil {
		return err
	}
	return nil
}

func (s AllocationSpec) validateWorkerConfiguration() error {
	if s.AgentTemplate.IsToolWorker() {
		if !s.ModelPolicy.IsZero() || s.SummarizerModelPolicy != nil || s.RuntimeSettings.LLMGatewayURL != "" || s.RuntimeSettings.LLMGatewayToken != nil ||
			s.ResolvedRuntimeConfigProvenance.LLMGatewayConfig != nil || s.ResolvedRuntimeConfigProvenance.LLMCredential != nil || s.Workspace != nil || s.CompletionContract != nil {
			return contracts.Invalidf("tool@1 allocation forbids model access, project workspace and completion contracts")
		}
	} else {
		if s.RuntimeSettings.LLMGatewayURL == "" {
			return contracts.Invalidf("modeled allocation requires llmGatewayUrl")
		}
		if err := s.ModelPolicy.ValidateForWorker(len(s.AgentTemplate.Toolsets) > 0 || len(s.AgentTemplate.Skills) > 0); err != nil {
			return err
		}
	}
	if err := s.ContextManagement.ValidateAllocation(contracts.AllocationContext{
		Template: s.AgentTemplate, SessionMode: s.WorkerSessionMode,
		SummarizerModelPolicy: s.SummarizerModelPolicy, WorkerPolicy: s.ModelPolicy,
		CompactorModelPolicy: s.CompactorModelPolicy,
	}); err != nil {
		return err
	}
	if s.SummarizerModelPolicy != nil {
		if err := s.SummarizerModelPolicy.ValidateForWorkerSummarizer(); err != nil {
			return err
		}
	}
	// The compact@1 branch wholly replaces the template summarizer, so its
	// terminal limits do not constrain the effective Worker policy.
	if s.ContextManagement.Strategy == contracts.ContextStrategyTerminal {
		if err := s.AgentTemplate.Summarizer.ValidateForWorker(s.ModelPolicy); err != nil {
			return err
		}
	}
	return nil
}
