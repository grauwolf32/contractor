package runservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// CreatePublic preserves the public create/replay contract while keeping all
// mutable resolution behind a replay-first boundary. A committed response-loss
// replay therefore does not need the original Workflow credential, Runtime
// label binding, Skill binding, or Project target to remain current.
func (s *Service) CreatePublic(ctx context.Context, params PublicCreateParams) (CreateResult, error) {
	normalized, err := normalizePublicCreate(params)
	if err != nil {
		return CreateResult{}, err
	}
	stored, replayed, err := s.runs.LookupRunIdempotency(
		ctx, normalized.OwnerID, normalized.IdempotencyKey, normalized.RequestDigest,
	)
	if err != nil {
		return CreateResult{}, err
	}
	if replayed {
		return CreateResult{Run: stored, Replayed: true}, nil
	}
	runID, err := normalized.NewRunID()
	if err != nil {
		return CreateResult{}, fmt.Errorf("generate WorkflowRun identity: %w", err)
	}

	created := false
	createPinned := func() error {
		return s.credentialGuard.WithRunCreation(ctx, func() error {
			workflow, resolveErr := s.workflows.ResolveRunWorkflow(
				ctx, normalized.Workflow, normalized.ExecutionConfig, s.llmCredentials,
			)
			if resolveErr != nil {
				return fmt.Errorf("%w: invalid Workflow or executionConfig selection: %v", ErrInvalid, resolveErr)
			}
			if validationErr := validateWorkflowInputs(workflow, normalized.Parameters, normalized.Inputs); validationErr != nil {
				return validationErr
			}
			skillRefs, resolveErr := config.WorkflowSkillRefs(workflow)
			if resolveErr != nil {
				return fmt.Errorf("%w: invalid Workflow Skill selection: %v", ErrInvalid, resolveErr)
			}
			if len(skillRefs) > 0 && !s.skillInitializationAvailable {
				return fmt.Errorf("%w: Run Skill initialization is unavailable", ErrNotConfigured)
			}
			workflowSnapshot, encodeErr := json.Marshal(workflow)
			if encodeErr != nil {
				return fmt.Errorf("encode resolved Workflow: %w", encodeErr)
			}

			var projectTarget *contracts.HTTPOriginTargetRef
			if normalized.ProjectID != nil {
				project, projectErr := s.projects.Get(ctx, normalized.OwnerID, *normalized.ProjectID)
				if projectErr != nil {
					return projectErr
				}
				if project.Lifecycle == projectstore.LifecycleDeleting {
					return projectstore.ErrDeleting
				}
				projectTarget = cloneHTTPOriginTarget(project.HTTPTarget)
				if projectTarget != nil && projectTarget.Credential != nil {
					if credentialErr := s.runtimeCredentials.ValidateRuntimeCredential(
						ctx, projectTarget.Credential.CredentialID,
						string(projectTarget.Credential.Kind),
					); credentialErr != nil {
						return credentialErr
					}
				}
			}

			return s.publicTransaction(ctx, func(runs PublicRunWriter, artifactService *artifacts.Service) error {
				catalog, catalogErr := agentskills.NewCatalog(artifactService)
				if catalogErr != nil {
					return catalogErr
				}
				runtimeSnapshot, pinErr := runs.PinRuntimeLabels(ctx, normalized.RuntimeLabels, s.llmCredentials)
				if pinErr != nil {
					return pinErr
				}
				var selectedSkills []contracts.RunSkillSnapshot
				if len(skillRefs) > 0 {
					selectedSkills, catalogErr = catalog.SelectRunSources(ctx, normalized.OwnerID, skillRefs)
					if catalogErr != nil {
						return catalogErr
					}
					if catalogErr = agentskills.ValidateSelectedLimits(
						selectedSkills, config.WorkflowSkillSets(workflow),
					); catalogErr != nil {
						return catalogErr
					}
				}
				stored, created, err = runs.CreateRunIdempotent(ctx, runstore.CreateRunIdempotentParams{
					CreateRunParams: runstore.CreateRunParams{
						RunID: runID, OwnerID: normalized.OwnerID, ProjectID: normalized.ProjectID,
						WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
						WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: workflowSnapshot,
						Parameters: cloneParameters(normalized.Parameters), MetadataLabels: normalized.MetadataLabels,
						RuntimeConfig: runtimeSnapshot, ProjectHTTPTarget: projectTarget,
					},
					IdempotencyKey: normalized.IdempotencyKey, RequestDigest: normalized.RequestDigest,
				})
				if err != nil || !created {
					return err
				}
				if len(selectedSkills) > 0 {
					if err := runs.SetRunSkillSelections(ctx, runID, selectedSkills); err != nil {
						return err
					}
					if err := catalog.PinRunSources(ctx, normalized.OwnerID, runID, selectedSkills); err != nil {
						return err
					}
				}
				for _, slot := range sortedInputSlots(normalized.Inputs) {
					var fork artifacts.ForkResult
					var forkErr error
					if normalized.ProjectID == nil {
						fork, forkErr = artifactService.ForkInput(
							ctx, normalized.OwnerID, normalized.Inputs[slot], runID, slot,
						)
					} else {
						fork, forkErr = artifactService.ForkProjectInput(
							ctx, *normalized.ProjectID, normalized.Inputs[slot], runID, slot,
						)
					}
					if forkErr != nil {
						return forkErr
					}
					if !acceptsMediaType(workflow.Inputs[slot].MediaTypes, fork.MediaType) {
						return fmt.Errorf("%w: input %q has unsupported media type", ErrInvalid, slot)
					}
				}
				if len(selectedSkills) == 0 {
					stored, err = runs.TransitionRun(
						ctx, runID, runstore.RunInitializing, runstore.RunRunning,
						runstore.Reason{Code: "initialized"},
					)
					return err
				}
				stored.SkillSnapshot = append([]contracts.RunSkillSnapshot(nil), selectedSkills...)
				stored.StateReason = runstore.Reason{Code: runstore.SkillInitializationPendingReason}
				return nil
			})
		})
	}

	err = createPinned()
	if err != nil {
		if !errors.Is(err, runstore.ErrConflict) && persistencepostgres.SQLState(err) != "40001" {
			return CreateResult{}, err
		}
		var recovered bool
		stored, recovered, _ = s.runs.LookupRunIdempotency(
			ctx, normalized.OwnerID, normalized.IdempotencyKey, normalized.RequestDigest,
		)
		if !recovered {
			return CreateResult{}, err
		}
		created = false
	}
	return CreateResult{Run: stored, Created: created, Replayed: !created}, nil
}

func normalizePublicCreate(params PublicCreateParams) (PublicCreateParams, error) {
	if params.OwnerID == "" || params.Workflow == "" || params.IdempotencyKey == "" ||
		params.RequestDigest == "" || params.NewRunID == nil {
		return PublicCreateParams{}, fmt.Errorf("%w: required public Run field is missing", ErrInvalid)
	}
	labels, err := runtimeconfig.NormalizeRunLabels(params.RuntimeLabels)
	if err != nil {
		return PublicCreateParams{}, err
	}
	metadata, err := runstore.NormalizeRunMetadataLabels(params.MetadataLabels)
	if err != nil {
		return PublicCreateParams{}, err
	}
	params.RuntimeLabels = labels
	params.MetadataLabels = metadata
	params.Parameters = cloneParameters(params.Parameters)
	params.Inputs = cloneRefs(params.Inputs)
	if params.ProjectID != nil {
		projectID := *params.ProjectID
		if projectID == "" {
			return PublicCreateParams{}, fmt.Errorf("%w: Project identity is empty", ErrInvalid)
		}
		params.ProjectID = &projectID
	}
	return params, nil
}

func validateWorkflowInputs(
	workflow config.ResolvedWorkflow,
	parameters map[string]string,
	inputs map[string]contracts.ArtifactRef,
) error {
	for name := range parameters {
		if _, ok := workflow.Parameters[name]; !ok {
			return fmt.Errorf("%w: unknown parameter %q", ErrInvalid, name)
		}
	}
	for name, slot := range workflow.Parameters {
		if _, ok := parameters[name]; slot.Required && !ok {
			return fmt.Errorf("%w: required parameter %q is missing", ErrInvalid, name)
		}
	}
	for name, ref := range inputs {
		if _, ok := workflow.Inputs[name]; !ok {
			return fmt.Errorf("%w: unknown input artifact %q", ErrInvalid, name)
		}
		if err := ref.Validate(); err != nil {
			return fmt.Errorf("%w: invalid input artifact %q", ErrInvalid, name)
		}
	}
	for name, slot := range workflow.Inputs {
		if _, ok := inputs[name]; slot.Required && !ok {
			return fmt.Errorf("%w: required input artifact %q is missing", ErrInvalid, name)
		}
	}
	return nil
}

func sortedInputSlots[T any](source map[string]T) []string {
	result := make([]string, 0, len(source))
	for slot := range source {
		result = append(result, slot)
	}
	sort.Strings(result)
	return result
}
