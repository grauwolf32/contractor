package config

import (
	"context"
	"fmt"
	"reflect"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// ResolvePinnedRunWorkflow validates a trusted retained Eval binding without
// consulting mutable deployment defaults. Catalog dependency digests must still
// exist exactly, and credentials are rechecked under the caller's lifecycle guard.
func (s *Snapshot) validatePinnedWorkflow(pinned resolved.ResolvedWorkflow) error {
	current, err := s.Workflow(pinned.Ref.Name + "@" + pinned.Ref.Version)
	if err != nil || current.Ref != pinned.Ref {
		return fmt.Errorf("%w: pinned Workflow no longer matches the catalog", contentdigest.ErrPinnedSelectionChanged)
	}
	checkPolicy := func(policy contracts.ResolvedModelPolicy) error {
		if policy.IsZero() {
			return nil
		}
		current, err := s.ModelPolicy(policy.Ref.PolicyID + "@" + policy.Ref.Version)
		if err != nil || !reflect.DeepEqual(current, policy) {
			return fmt.Errorf("%w: pinned ModelPolicy no longer matches the catalog", contentdigest.ErrPinnedSelectionChanged)
		}
		return nil
	}
	checkSelection := func(selection resolved.ResolvedConsumerExecutionConfig) error {
		if err := checkPolicy(selection.ModelPolicy); err != nil {
			return err
		}
		if selection.SummarizerModelPolicy != nil {
			if err := checkPolicy(*selection.SummarizerModelPolicy); err != nil {
				return err
			}
		}
		if selection.LLMGateway != nil {
			ref := selection.LLMGateway.Ref
			current, err := s.LLMGateway(ref.GatewayID + "@" + ref.Version)
			if err != nil || !reflect.DeepEqual(current, *selection.LLMGateway) {
				return fmt.Errorf("%w: pinned LLMGatewayConfig no longer matches the catalog", contentdigest.ErrPinnedSelectionChanged)
			}
		}
		return nil
	}
	for stageName, stage := range pinned.Stages {
		for _, policy := range stage.ModelDefaults {
			if err := checkPolicy(policy); err != nil {
				return err
			}
		}
		baseline, ok := current.Stages[stageName]
		if !ok {
			return fmt.Errorf("%w: pinned Stage no longer matches the catalog", contentdigest.ErrPinnedSelectionChanged)
		}
		for name, binding := range stage.Agents {
			original, ok := baseline.Agents[name]
			if !ok || !reflect.DeepEqual(original.Template, binding.Template) {
				return fmt.Errorf("%w: pinned AgentTemplate no longer matches the catalog", contentdigest.ErrPinnedSelectionChanged)
			}
			if err := checkSelection(stage.ExecutionConfig.Agents[name]); err != nil {
				return err
			}
		}
		if stage.ExecutionConfig.Planner != nil {
			if err := checkSelection(*stage.ExecutionConfig.Planner); err != nil {
				return err
			}
		}

		for _, item := range stage.On.Actions() {
			if item.Action.Escalate == nil {
				continue
			}
			effective := item.Action.Escalate.ExecutionConfig.Effective
			if effective.Planner != nil {
				if err := checkSelection(*effective.Planner); err != nil {
					return err
				}
			}
			for _, selection := range effective.Agents {
				if err := checkSelection(selection); err != nil {
					return err
				}
			}
		}

	}
	if err := resolved.ValidateWorkflowExecutionConfigs(pinned); err != nil {
		return err
	}
	if err := resolved.ValidateWorkflowGraph(pinned); err != nil {
		return err
	}
	return nil
}

func (s *Snapshot) ResolvePinnedRunWorkflow(ctx context.Context, pinned resolved.ResolvedWorkflow, credentials resolved.CredentialLookup) (resolved.ResolvedWorkflow, error) {
	if err := s.validatePinnedWorkflow(pinned); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	if err := resolved.ValidateResolvedWorkflowCredentials(ctx, pinned, credentials); err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	return pinned.Clone(), nil
}

// ResolvePinnedAuditProfile preserves a prepared Eval's exact Workflow model
// bindings while rejecting catalog dependency drift. Credential use is checked
// by the ordinary Audit start transaction, just as for a newly selected draft.
func (s *Snapshot) ResolvePinnedAuditProfile(_ context.Context, pinned auditprofile.ResolvedAuditProfile) (auditprofile.ResolvedAuditProfile, error) {
	seal, err := auditprofile.ProfileClosureDigest(pinned)
	if err != nil || seal != pinned.ClosureDigest {
		return auditprofile.ResolvedAuditProfile{}, contentdigest.ErrPinnedSelectionChanged
	}
	current, err := s.AuditProfile(pinned.Ref.Name + "@" + pinned.Ref.Version)
	if err != nil || current.Ref != pinned.Ref || len(current.Workflows) != len(pinned.Workflows) {
		return auditprofile.ResolvedAuditProfile{}, contentdigest.ErrPinnedSelectionChanged
	}
	for role, binding := range pinned.Workflows {
		if _, ok := current.Workflows[role]; !ok {
			return auditprofile.ResolvedAuditProfile{}, contentdigest.ErrPinnedSelectionChanged
		}
		if err := s.validatePinnedWorkflow(binding.Workflow); err != nil {
			return auditprofile.ResolvedAuditProfile{}, err
		}
	}
	return pinned.Clone(), nil
}

func (m *Manager) ResolvePinnedAuditProfile(ctx context.Context, pinned auditprofile.ResolvedAuditProfile) (auditprofile.ResolvedAuditProfile, error) {
	return m.Snapshot().ResolvePinnedAuditProfile(ctx, pinned)
}

func (m *Manager) ResolvePinnedRunWorkflow(ctx context.Context, pinned resolved.ResolvedWorkflow, credentials resolved.CredentialLookup) (resolved.ResolvedWorkflow, error) {
	return m.Snapshot().ResolvePinnedRunWorkflow(ctx, pinned, credentials)
}
