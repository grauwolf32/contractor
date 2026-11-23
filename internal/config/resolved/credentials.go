package resolved

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/credentialerrors"
)

// CredentialMetadata is the safe lookup result used while a Run is resolved.
// It deliberately contains no token or provider response.
type CredentialMetadata struct {
	Ref           llmgateway.LLMCredentialRef
	LLMGateway    llmgateway.LLMGatewayConfigRef
	ModelPolicies []contracts.ModelPolicyRef
	Models        []string
	// Unrestricted is reserved for explicit process-local development
	// credentials. Managed credentials always carry their immutable effective
	// policy above.
	Unrestricted bool
}

type CredentialLookup interface {
	LookupLLMCredential(context.Context, string) (CredentialMetadata, error)
}

// ErrCredentialLookup classifies an LLM credential lookup that failed for a
// reason other than confirmed absence: credential recovery in progress, a
// database transaction conflict or outage, or cancellation. ResolveRunWorkflow
// and ValidateResolvedWorkflowCredentials return it only for such lookups, so
// every other error they return is a deterministic validation failure of the
// selection. The lookup cause stays reachable through errors.Is/As and SQLState
// for retry classification, while Error() renders only a static message, so no
// provider, database or credential text escapes.
var ErrCredentialLookup = errors.New("LLM credential lookup failed")

type credentialLookupError struct {
	message string
	cause   error
}

func (e *credentialLookupError) Error() string { return e.message }

func (e *credentialLookupError) Unwrap() []error { return []error{ErrCredentialLookup, e.cause} }

// credentialLookupFailure is the one rule for a failed credential lookup during
// Workflow resolution. Only confirmed absence rejects the selection, and that
// cause is dropped on purpose: a missing credential is an invalid selection,
// not a missing request resource. Every other failure, and any failure after
// the caller's context ended, keeps its cause behind ErrCredentialLookup.
func credentialLookupFailure(ctx context.Context, subject string, err error) error {
	if contextErr := ctx.Err(); contextErr != nil {
		return &credentialLookupError{message: subject + " credential lookup failed", cause: contextErr}
	}
	if errors.Is(err, credentialerrors.LLMNotFound) {
		return fmt.Errorf("%s credential is unavailable", subject)
	}
	return &credentialLookupError{message: subject + " credential lookup failed", cause: err}
}

func validateRunCredentials(
	ctx context.Context,
	workflow ResolvedWorkflow,
	lookup CredentialLookup,
) error {
	for stageName, stage := range workflow.Stages {
		configs := map[string]ResolvedStageExecutionConfig{"base": stage.ExecutionConfig}
		for _, named := range stage.On.Actions() {
			action := named.Action
			if action.Kind == TransitionEscalate && action.Escalate != nil {
				configs[named.Outcome+" escalation"] = action.Escalate.ExecutionConfig.Effective
			}
		}
		for _, variant := range slices.Sorted(maps.Keys(configs)) {
			if err := validateStageExecutionConfigCredentials(
				ctx, stageName, variant, configs[variant], lookup,
			); err != nil {
				return err
			}
		}
	}
	return nil
}

// ValidateResolvedWorkflowCredentials revalidates every base and escalation
// credential selected by an already-resolved Workflow. Audit start uses this
// without consulting the mutable Workflow catalog again. The Workflow comes
// from a catalog Snapshot, ResolveRunWorkflow or a snapshot decoder, which
// have validated its graph, so only the credentials are checked here. Its
// errors follow the ErrCredentialLookup classification of ResolveRunWorkflow.
func ValidateResolvedWorkflowCredentials(
	ctx context.Context, workflow ResolvedWorkflow, lookup CredentialLookup,
) error {
	return validateRunCredentials(ctx, workflow, lookup)
}

// ResolvedWorkflowCredentialIDs returns the canonical exact credential-ID set
// retained by an already validated Workflow, including escalation variants.
func ResolvedWorkflowCredentialIDs(workflow ResolvedWorkflow) []string {
	selected := make(map[string]struct{})
	for _, stageName := range slices.Sorted(maps.Keys(workflow.Stages)) {
		stage := workflow.Stages[stageName]
		configs := []ResolvedStageExecutionConfig{stage.ExecutionConfig}
		for _, named := range stage.On.Actions() {
			action := named.Action
			if action.Kind == TransitionEscalate && action.Escalate != nil {
				configs = append(configs, action.Escalate.ExecutionConfig.Effective)
			}
		}
		for _, execution := range configs {
			if execution.Planner != nil && execution.Planner.Credential != nil {
				selected[execution.Planner.Credential.CredentialID] = struct{}{}
			}
			for _, consumer := range execution.Agents {
				if consumer.Credential != nil {
					selected[consumer.Credential.CredentialID] = struct{}{}
				}
			}
		}
	}
	result := make([]string, 0, len(selected))
	for credentialID := range selected {
		result = append(result, credentialID)
	}
	sort.Strings(result)
	return result
}

func validateStageExecutionConfigCredentials(
	ctx context.Context,
	stageName string,
	variant string,
	config ResolvedStageExecutionConfig,
	lookup CredentialLookup,
) error {
	selections := make(map[string]ResolvedConsumerExecutionConfig, len(config.Agents)+1)
	if config.Planner != nil {
		selections["planner"] = *config.Planner
	}
	for logicalName, selection := range config.Agents {
		selections["agent "+logicalName] = selection
	}
	for _, consumer := range slices.Sorted(maps.Keys(selections)) {
		selection := selections[consumer]
		if selection.Credential == nil {
			continue
		}
		if lookup == nil {
			return fmt.Errorf(
				"Stage %q %s %s selects unavailable credential %q",
				stageName, variant, consumer, selection.Credential.CredentialID,
			)
		}
		metadata, err := lookup.LookupLLMCredential(ctx, selection.Credential.CredentialID)
		if err != nil {
			return credentialLookupFailure(
				ctx, fmt.Sprintf("Stage %q %s %s", stageName, variant, consumer), err,
			)
		}
		if metadata.Ref != *selection.Credential ||
			selection.LLMGateway != nil && metadata.LLMGateway != selection.LLMGateway.Ref {
			return fmt.Errorf(
				"Stage %q %s %s credential is bound to another LLMGatewayConfig",
				stageName, variant, consumer,
			)
		}
	}
	return nil
}
