package runservice

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/clone"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// CreateAudit creates and binds one ordinary WorkflowRun through the trusted
// Audit authority. Check executions may contain a bounded ordered set of
// independently attributable ExecutionItems.
func (s *Service) CreateAudit(ctx context.Context, params AuditCreateParams) (CreateResult, error) {
	if s.auditTransaction == nil {
		return CreateResult{}, ErrNotConfigured
	}
	normalized, workflowSnapshot, err := normalizeAuditCreate(params)
	if err != nil {
		return CreateResult{}, err
	}
	var result CreateResult
	err = s.credentialGuard.WithRunCreation(ctx, func() error {
		return s.auditTransaction(ctx, func(
			runs AuditRunWriter,
			artifactService *artifacts.Service,
			audits AuditExecutionWriter,
		) error {
			intent, loadErr := audits.GetRunCreationIntent(ctx, normalized.Claim, normalized.ExecutionID)
			if loadErr != nil {
				return loadErr
			}
			if validationErr := validateAuditIntent(intent, normalized); validationErr != nil {
				return validationErr
			}
			if err := validateRoundResultsAuthority(ctx, audits, intent, normalized); err != nil {
				return err
			}
			if intent.Execution.RunID != nil {
				run, getErr := runs.GetRun(ctx, *intent.Execution.RunID)
				if getErr != nil {
					return getErr
				}
				if !matchingAuditRun(run, intent) {
					return fmt.Errorf("%w: stored Audit Run association is inconsistent", ErrInvalid)
				}
				if run.State == runstore.RunInitializing && normalized.MaterializeInputs != nil {
					run, getErr = completeAuditInputSlice(ctx, runs, artifactService, intent, normalized, run, time.Now())
					if getErr != nil {
						return getErr
					}
				} else if run.State == runstore.RunInitializing && run.StateReason.Code == runstore.AuditInputInitializationPendingReason {
					return fmt.Errorf("%w: Audit input initialization has no declared materializer", ErrInvalid)
				}
				result = CreateResult{Run: run, Replayed: true}
				return nil
			}
			if intent.Execution.State != auditstore.ExecutionIntent {
				return auditstore.ErrPrecondition
			}
			runID, idErr := normalized.NewRunID()
			if idErr != nil {
				return fmt.Errorf("generate Audit WorkflowRun identity: %w", idErr)
			}
			completion, completionErr := prepareAuditCompletion(ctx, artifactService, intent, normalized)
			if completionErr != nil {
				return completionErr
			}
			labels, labelErr := auditRunLabels(intent)
			if labelErr != nil {
				return labelErr
			}
			run, createErr := runs.CreateAuditRun(ctx, runstore.CreateAuditRunParams{
				CreateRunParams: runstore.CreateRunParams{
					RequiredPlacementLabels: append([]string{}, normalized.RequiredPlacementLabels...),
					RunID:                   runID, OwnerID: intent.OwnerID, ProjectID: &intent.ProjectID,
					WorkflowName:          normalized.Workflow.Ref.Name,
					WorkflowVersion:       normalized.Workflow.Ref.Version,
					WorkflowSchemaVersion: contracts.APIVersion,
					WorkflowSnapshot:      workflowSnapshot,
					Parameters:            clone.Map(normalized.Parameters), MetadataLabels: labels,
					RuntimeConfig:     normalized.RuntimeConfig,
					ProjectHTTPTarget: normalized.ProjectHTTPTarget.Clone(),
					LaneClass:         resolved.EntryLaneClass(normalized.Workflow),
				},
				AuditExecutionID:   intent.Execution.ExecutionID,
				AuditSubmissionKey: intent.Execution.SubmissionKey,
			})
			if createErr != nil {
				return createErr
			}
			if _, bindErr := audits.BindRun(ctx, auditstore.BindRunParams{
				Claim: normalized.Claim, ExecutionID: normalized.ExecutionID, RunID: runID,
			}); bindErr != nil {
				return bindErr
			}

			catalog, catalogErr := agentskills.NewCatalog(artifactService)
			if catalogErr != nil {
				return catalogErr
			}
			if len(normalized.Skills) > 0 {
				if err := runs.SetRunSkillSelections(ctx, runID, normalized.Skills); err != nil {
					return err
				}
				if err := catalog.RequireRunSources(ctx, intent.OwnerID, normalized.Skills); err != nil {
					return err
				}
			}
			projectScope, scopeErr := artifacts.ProjectScope(intent.ProjectID)
			if scopeErr != nil {
				return scopeErr
			}
			if _, metadataErr := verifyExactProjectArtifact(
				ctx, artifactService, intent.ProjectID, normalized.ExecutionManifest,
			); metadataErr != nil {
				return metadataErr
			}
			if err := artifactService.RequireExact(ctx, projectScope, normalized.ExecutionManifest.Ref); err != nil {
				return err
			}
			forks := make(map[string]contracts.ArtifactRef, len(normalized.Inputs))
			for _, slot := range sortedInputSlots(normalized.Inputs) {
				if _, materialized := normalized.RoundResults[slot]; materialized {
					// The Audit-owned initializer writes the consumer descriptor
					// only after every required local page is present.
					continue
				}
				descriptor := normalized.Inputs[slot]
				if _, metadataErr := verifyExactProjectArtifact(
					ctx, artifactService, intent.ProjectID, descriptor,
				); metadataErr != nil {
					return metadataErr
				}
				fork, forkErr := forkInput(ctx, artifactService, projectScope, descriptor.Ref, runID, slot)
				if forkErr != nil {
					return forkErr
				}
				forks[slot] = fork.TargetRef
				if !contracts.AcceptsMediaType(normalized.Workflow.Inputs[slot].MediaTypes, fork.MediaType) {
					return fmt.Errorf("%w: Audit input %q has unsupported media type", ErrInvalid, slot)
				}
			}
			if completion != nil {
				completion.Contract.Task = forks[completion.Contract.Task.Name]
				completion.Contract.ExecutionManifest = forks[completion.Contract.ExecutionManifest.Name]
				if err := runs.SetAuditCompletion(ctx, runID, *completion); err != nil {
					return err
				}
				run.AuditCompletion = completion
			}
			if normalized.MaterializeInputs != nil {
				run, err = completeAuditInputSlice(ctx, runs, artifactService, intent, normalized, run, time.Now())
				if err != nil {
					return err
				}
			} else if len(normalized.Skills) == 0 {
				run, err = runs.TransitionRun(
					ctx, runID, runstore.RunInitializing, runstore.RunPending,
					runstore.Reason{Code: "initialized"},
				)
				if err != nil {
					return err
				}
			} else {
				run.SkillSnapshot = agentskills.CloneRunSkills(normalized.Skills)
				run.StateReason = runstore.Reason{Code: runstore.SkillInitializationPendingReason}
			}
			result = CreateResult{Run: run, Created: true}
			return nil
		})
	})
	if err != nil {
		if normalized.MaterializeInputs != nil && permanentAuditInputFailure(err) {
			failed, handled, recoveryErr := s.failBoundAuditInput(ctx, normalized)
			if recoveryErr != nil {
				return CreateResult{}, recoveryErr
			}
			if handled {
				return failed, nil
			}
		}
		return CreateResult{}, err
	}
	return result, nil
}

func normalizeAuditCreate(params AuditCreateParams) (AuditCreateParams, json.RawMessage, error) {
	if (params.MaterializeInputs == nil) != (params.InputMaterializationID == "") ||
		params.InputMaterializationID != "" && !contentdigest.Valid(params.InputMaterializationID) ||
		len(params.RoundResults) > 1 || len(params.RoundResults) != 0 && params.MaterializeInputs == nil {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: Audit input materialization identity is invalid", ErrInvalid)
	}
	if err := contracts.ValidatePlacementLabels(params.RequiredPlacementLabels); err != nil {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: Audit placement requirements are invalid", ErrInvalid)
	}
	params.RequiredPlacementLabels = append([]string{}, params.RequiredPlacementLabels...)
	if params.ExecutionID == "" || params.RequestDigest == "" || params.NewRunID == nil ||
		params.Claim.AuditID == "" || params.Claim.HolderID == "" || params.Claim.Epoch == 0 {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: Audit Run identity is incomplete", ErrInvalid)
	}
	if !contentdigest.Valid(params.RequestDigest) || params.ExecutionManifest.Ref.ValidateExact() != nil ||
		!contentdigest.Valid(params.ExecutionManifest.Digest) || params.ExecutionManifest.MediaType == "" ||
		params.ExecutionManifest.SizeBytes < 0 {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: Audit execution manifest is invalid", ErrInvalid)
	}
	if err := params.RuntimeConfig.Validate(); err != nil {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: pinned RuntimeConfig is invalid", ErrInvalid)
	}
	if params.ProjectHTTPTarget != nil {
		if err := params.ProjectHTTPTarget.Validate(); err != nil {
			return AuditCreateParams{}, nil, fmt.Errorf("%w: pinned Project HTTP target is invalid", ErrInvalid)
		}
	}
	workflowSnapshot, err := resolved.EncodeResolvedWorkflowSnapshot(params.Workflow)
	if err != nil {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: pinned Workflow snapshot: %v", ErrInvalid, err)
	}
	workflow, err := resolved.DecodeResolvedWorkflowSnapshot(workflowSnapshot)
	if err != nil {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: pinned Workflow is invalid", ErrInvalid)
	}
	params.Workflow = workflow
	refs := make(map[string]contracts.ArtifactRef, len(params.Inputs))
	params.Inputs = contracts.CloneExactArtifacts(params.Inputs)
	for slot, descriptor := range params.Inputs {
		if descriptor.Ref.ValidateExact() != nil || !contentdigest.Valid(descriptor.Digest) ||
			descriptor.MediaType == "" || descriptor.SizeBytes < 0 {
			return AuditCreateParams{}, nil, fmt.Errorf("%w: Audit input %q is invalid", ErrInvalid, slot)
		}
		refs[slot] = descriptor.Ref
	}
	params.Parameters = clone.Map(params.Parameters)
	if err := validateWorkflowInputs(params.Workflow, params.Parameters, refs); err != nil {
		return AuditCreateParams{}, nil, err
	}
	params.ExecutionManifest = params.ExecutionManifest.Clone()
	params.RoundResults = contracts.CloneExactArtifacts(params.RoundResults)
	params.ProjectHTTPTarget = params.ProjectHTTPTarget.Clone()
	params.Skills = agentskills.CloneRunSkills(params.Skills)
	workflowSkillRefs, err := resolved.WorkflowSkillRefs(params.Workflow)
	if err != nil || !matchingPinnedSkills(workflowSkillRefs, params.Skills) {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: pinned Skill set does not match Workflow", ErrInvalid)
	}
	if err := agentskills.ValidateSelectedLimits(params.Skills, resolved.WorkflowSkillSets(params.Workflow)); err != nil {
		return AuditCreateParams{}, nil, err
	}
	return params, workflowSnapshot, nil
}

func validateAuditIntent(intent auditstore.RunCreationIntent, params AuditCreateParams) error {
	if !slices.Equal(intent.RequiredPlacementLabels, params.RequiredPlacementLabels) {
		return fmt.Errorf("%w: Audit placement requirements differ from the immutable baseline", auditstore.ErrConflict)
	}
	execution := intent.Execution
	if execution.ExecutionID != params.ExecutionID || execution.AuditID != params.Claim.AuditID ||
		execution.RequestDigest != params.RequestDigest ||
		!sameExactIdentity(execution.Manifest, params.ExecutionManifest) {
		return fmt.Errorf("%w: Audit execution intent does not match pinned submission", auditstore.ErrConflict)
	}
	if execution.Role == auditstore.ExecutionPrepare {
		var binding auditprofile.ResolvedAuditWorkflowBinding
		if execution.Preparation == nil || execution.RoundID != nil || len(intent.Items) != 0 ||
			json.Unmarshal(intent.WorkflowBinding, &binding) != nil || binding.Kind != auditprofile.AuditWorkflowPrepare ||
			!maps.Equal(execution.Preparation.Parameters, params.Parameters) || len(execution.Preparation.Inputs) != len(params.Inputs) {
			return fmt.Errorf("%w: preparation submission differs from its immutable intent", auditstore.ErrConflict)
		}
		// Both values are resolved.ResolvedWorkflow values that a JSON decode
		// just produced, so re-encoding is not expected to fail; a failure is
		// still a conflict, so two failed encodings never compare equal.
		pinned, pinnedErr := json.Marshal(binding.Workflow)
		submitted, submittedErr := json.Marshal(params.Workflow)
		if pinnedErr != nil || submittedErr != nil || !bytes.Equal(pinned, submitted) {
			return fmt.Errorf("%w: preparation Workflow differs from its pinned binding", auditstore.ErrConflict)
		}
		for slot, pinned := range execution.Preparation.Inputs {
			input, exists := params.Inputs[slot]
			if !exists || !sameExactIdentity(pinned, input) || pinned.MediaType != input.MediaType || pinned.SizeBytes != input.SizeBytes {
				return fmt.Errorf("%w: preparation input %q differs from its immutable intent", auditstore.ErrConflict, slot)
			}
		}
		return nil
	}
	allowed := make(map[string]string, 1+len(intent.Items)*2)
	addAllowedArtifact(allowed, execution.Manifest)
	for _, item := range intent.Items {
		addAllowedArtifact(allowed, item.Task)
		for _, input := range item.Inputs {
			addAllowedArtifact(allowed, input)
		}
	}
	for slot, input := range params.Inputs {
		if root, materialized := params.RoundResults[slot]; materialized && sameExactIdentity(root, input) {
			// validateRoundResultsAuthority proves this additional role input
			// against the exact accepted Round root before any Run is created.
			continue
		}
		if digest, ok := allowed[input.Ref.Key()]; !ok || digest != input.Digest {
			return fmt.Errorf("%w: Audit input %q is outside immutable execution intent", auditstore.ErrConflict, slot)
		}
	}
	return nil
}

func auditRunLabels(intent auditstore.RunCreationIntent) (runstore.RunMetadataLabels, error) {
	labels := runstore.RunMetadataLabels(intent.Labels).Clone()
	labels["audit.id"] = intent.Execution.AuditID
	labels["audit.role"] = string(intent.Execution.Role)
	if intent.Execution.RoundID != nil {
		labels["audit.round"] = *intent.Execution.RoundID
	}
	if len(intent.Items) == 1 {
		labels["audit.item"] = intent.Items[0].ItemID
	}
	return runstore.NormalizeRunMetadataLabels(labels)
}

func matchingAuditRun(run runstore.WorkflowRun, intent auditstore.RunCreationIntent) bool {
	labels, err := auditRunLabels(intent)
	return err == nil && maps.Equal(labels, run.MetadataLabels) && slices.Equal(run.RequiredPlacementLabels, intent.RequiredPlacementLabels) &&
		run.PublicationMode == runstore.PublicationAuditManaged &&
		run.AuditExecutionID != nil && *run.AuditExecutionID == intent.Execution.ExecutionID &&
		run.AuditSubmissionKey != nil && *run.AuditSubmissionKey == intent.Execution.SubmissionKey &&
		run.OwnerID == intent.OwnerID && run.ProjectID != nil && *run.ProjectID == intent.ProjectID
}

func verifyExactProjectArtifact(
	ctx context.Context,
	service *artifacts.Service,
	projectID string,
	descriptor auditstore.ExactArtifact,
) (artifacts.Metadata, error) {
	store, err := service.Project(projectID)
	if err != nil {
		return artifacts.Metadata{}, err
	}
	metadata, err := store.Metadata(ctx, descriptor.Ref)
	if err != nil {
		return artifacts.Metadata{}, err
	}
	if !descriptor.HasMetadata() || descriptor.Verify(metadata.Exact()) != nil {
		return artifacts.Metadata{}, fmt.Errorf("%w: pinned Audit artifact failed integrity validation", ErrInvalid)
	}
	return metadata, nil
}

func matchingPinnedSkills(refs []contracts.ArtifactRef, skills []agentskills.RunSkillSnapshot) bool {
	if len(refs) != len(skills) {
		return false
	}
	for index := range refs {
		if refs[index].Namespace != contracts.AgentSkillNamespace || refs[index].Name != skills[index].Name ||
			skills[index].Source == nil || skills[index].Initialized() || skills[index].Validate() != nil {
			return false
		}
	}
	return true
}

func sameExactIdentity(left, right auditstore.ExactArtifact) bool {
	return left.Ref.Key() == right.Ref.Key() && left.Digest == right.Digest
}

func addAllowedArtifact(target map[string]string, artifact auditstore.ExactArtifact) {
	if artifact.Ref.Revision != nil {
		target[artifact.Ref.Key()] = artifact.Digest
	}
}
