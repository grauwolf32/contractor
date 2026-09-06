package runservice

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
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
			if intent.Execution.RunID != nil {
				run, getErr := runs.GetRun(ctx, *intent.Execution.RunID)
				if getErr != nil {
					return getErr
				}
				if !matchingAuditRun(run, intent) {
					return fmt.Errorf("%w: stored Audit Run association is inconsistent", ErrInvalid)
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
			labels, labelErr := auditRunLabels(intent)
			if labelErr != nil {
				return labelErr
			}
			run, createErr := runs.CreateAuditRun(ctx, runstore.CreateAuditRunParams{
				CreateRunParams: runstore.CreateRunParams{
					RunID: runID, OwnerID: intent.OwnerID, ProjectID: &intent.ProjectID,
					WorkflowName:          normalized.Workflow.Ref.Name,
					WorkflowVersion:       normalized.Workflow.Ref.Version,
					WorkflowSchemaVersion: contracts.APIVersion,
					WorkflowSnapshot:      workflowSnapshot,
					Parameters:            cloneParameters(normalized.Parameters), MetadataLabels: labels,
					RuntimeConfig:     normalized.RuntimeConfig,
					ProjectHTTPTarget: cloneHTTPOriginTarget(normalized.ProjectHTTPTarget),
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
				if err := catalog.PinRunSources(ctx, intent.OwnerID, runID, normalized.Skills); err != nil {
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
			if pinErr := artifactService.PinExact(
				ctx, runID, projectScope, normalized.ExecutionManifest.Ref,
				artifacts.PinRunInput, runID+":audit-execution-manifest",
			); pinErr != nil {
				return pinErr
			}
			for _, slot := range sortedInputSlots(normalized.Inputs) {
				descriptor := normalized.Inputs[slot]
				if _, metadataErr := verifyExactProjectArtifact(
					ctx, artifactService, intent.ProjectID, descriptor,
				); metadataErr != nil {
					return metadataErr
				}
				fork, forkErr := artifactService.ForkProjectInput(
					ctx, intent.ProjectID, descriptor.Ref, runID, slot,
				)
				if forkErr != nil {
					return forkErr
				}
				if !acceptsMediaType(normalized.Workflow.Inputs[slot].MediaTypes, fork.MediaType) {
					return fmt.Errorf("%w: Audit input %q has unsupported media type", ErrInvalid, slot)
				}
			}
			if len(normalized.Skills) == 0 {
				run, err = runs.TransitionRun(
					ctx, runID, runstore.RunInitializing, runstore.RunRunning,
					runstore.Reason{Code: "initialized"},
				)
				if err != nil {
					return err
				}
			} else {
				run.SkillSnapshot = cloneSkills(normalized.Skills)
				run.StateReason = runstore.Reason{Code: runstore.SkillInitializationPendingReason}
			}
			result = CreateResult{Run: run, Created: true}
			return nil
		})
	})
	if err != nil {
		return CreateResult{}, err
	}
	return result, nil
}

func normalizeAuditCreate(params AuditCreateParams) (AuditCreateParams, json.RawMessage, error) {
	if params.ExecutionID == "" || params.RequestDigest == "" || params.NewRunID == nil ||
		params.Claim.AuditID == "" || params.Claim.HolderID == "" || params.Claim.Epoch == 0 {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: Audit Run identity is incomplete", ErrInvalid)
	}
	if !validDigest(params.RequestDigest) || params.ExecutionManifest.Ref.ValidateExact() != nil ||
		!validDigest(params.ExecutionManifest.Digest) || params.ExecutionManifest.MediaType == "" ||
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
	workflowSnapshot, err := json.Marshal(params.Workflow)
	if err != nil {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: encode pinned Workflow", ErrInvalid)
	}
	workflow, err := config.DecodeResolvedWorkflowSnapshot(workflowSnapshot)
	if err != nil || config.ValidateWorkflowGraph(workflow) != nil {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: pinned Workflow is invalid", ErrInvalid)
	}
	params.Workflow = workflow
	refs := make(map[string]contracts.ArtifactRef, len(params.Inputs))
	params.Inputs = cloneExactArtifacts(params.Inputs)
	for slot, descriptor := range params.Inputs {
		if descriptor.Ref.ValidateExact() != nil || !validDigest(descriptor.Digest) ||
			descriptor.MediaType == "" || descriptor.SizeBytes < 0 {
			return AuditCreateParams{}, nil, fmt.Errorf("%w: Audit input %q is invalid", ErrInvalid, slot)
		}
		refs[slot] = descriptor.Ref
	}
	params.Parameters = cloneParameters(params.Parameters)
	if err := validateWorkflowInputs(params.Workflow, params.Parameters, refs); err != nil {
		return AuditCreateParams{}, nil, err
	}
	params.ExecutionManifest = cloneExactArtifact(params.ExecutionManifest)
	params.ProjectHTTPTarget = cloneHTTPOriginTarget(params.ProjectHTTPTarget)
	params.Skills = cloneSkills(params.Skills)
	workflowSkillRefs, err := config.WorkflowSkillRefs(params.Workflow)
	if err != nil || !matchingPinnedSkills(workflowSkillRefs, params.Skills) {
		return AuditCreateParams{}, nil, fmt.Errorf("%w: pinned Skill set does not match Workflow", ErrInvalid)
	}
	if err := agentskills.ValidateSelectedLimits(params.Skills, config.WorkflowSkillSets(params.Workflow)); err != nil {
		return AuditCreateParams{}, nil, err
	}
	return params, workflowSnapshot, nil
}

func validateAuditIntent(intent auditstore.RunCreationIntent, params AuditCreateParams) error {
	execution := intent.Execution
	if execution.ExecutionID != params.ExecutionID || execution.AuditID != params.Claim.AuditID ||
		execution.RequestDigest != params.RequestDigest ||
		!sameExactIdentity(execution.Manifest, params.ExecutionManifest) {
		return fmt.Errorf("%w: Audit execution intent does not match pinned submission", auditstore.ErrConflict)
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
		if digest, ok := allowed[exactRefKey(input.Ref)]; !ok || digest != input.Digest {
			return fmt.Errorf("%w: Audit input %q is outside immutable execution intent", auditstore.ErrConflict, slot)
		}
	}
	return nil
}

func auditRunLabels(intent auditstore.RunCreationIntent) (runstore.RunMetadataLabels, error) {
	labels := runstore.RunMetadataLabels{
		"audit.id":   intent.Execution.AuditID,
		"audit.role": string(intent.Execution.Role),
	}
	if intent.Execution.RoundID != nil {
		labels["audit.round"] = *intent.Execution.RoundID
	}
	if len(intent.Items) == 1 {
		labels["audit.item"] = intent.Items[0].ItemID
	}
	return runstore.NormalizeRunMetadataLabels(labels)
}

func matchingAuditRun(run runstore.WorkflowRun, intent auditstore.RunCreationIntent) bool {
	return run.PublicationMode == runstore.PublicationAuditManaged &&
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
	if metadata.Ref.Revision == nil || descriptor.Ref.Revision == nil ||
		*metadata.Ref.Revision != *descriptor.Ref.Revision || metadata.Digest != descriptor.Digest ||
		metadata.MediaType != descriptor.MediaType || metadata.Size != descriptor.SizeBytes {
		return artifacts.Metadata{}, fmt.Errorf("%w: pinned Audit artifact failed integrity validation", ErrInvalid)
	}
	return metadata, nil
}

func matchingPinnedSkills(refs []contracts.ArtifactRef, skills []contracts.RunSkillSnapshot) bool {
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

func cloneSkills(source []contracts.RunSkillSnapshot) []contracts.RunSkillSnapshot {
	result := append([]contracts.RunSkillSnapshot(nil), source...)
	for index := range result {
		if result[index].Source != nil {
			ref := *result[index].Source
			if ref.Revision != nil {
				revision := *ref.Revision
				ref.Revision = &revision
			}
			result[index].Source = &ref
		}
		if result[index].Artifact != nil {
			ref := *result[index].Artifact
			if ref.Revision != nil {
				revision := *ref.Revision
				ref.Revision = &revision
			}
			result[index].Artifact = &ref
		}
	}
	return result
}

func cloneExactArtifacts(source map[string]auditstore.ExactArtifact) map[string]auditstore.ExactArtifact {
	result := make(map[string]auditstore.ExactArtifact, len(source))
	for name, value := range source {
		result[name] = cloneExactArtifact(value)
	}
	return result
}

func cloneExactArtifact(source auditstore.ExactArtifact) auditstore.ExactArtifact {
	if source.Ref.Revision != nil {
		revision := *source.Ref.Revision
		source.Ref.Revision = &revision
	}
	return source
}

func sameExactIdentity(left, right auditstore.ExactArtifact) bool {
	return exactRefKey(left.Ref) == exactRefKey(right.Ref) && left.Digest == right.Digest
}

func addAllowedArtifact(target map[string]string, artifact auditstore.ExactArtifact) {
	if artifact.Ref.Revision != nil {
		target[exactRefKey(artifact.Ref)] = artifact.Digest
	}
}

func exactRefKey(ref contracts.ArtifactRef) string {
	revision := ""
	if ref.Revision != nil {
		revision = *ref.Revision
	}
	return ref.Namespace + "\x00" + ref.Name + "\x00" + revision
}

func validDigest(value string) bool {
	if !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil
}
