package runservice

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/clone"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// CreateAudit creates and binds one ordinary WorkflowRun through the trusted
// Audit authority. Check executions may contain a bounded ordered set of
// independently attributable ExecutionItems.
func (s *Service) CreateAudit(ctx context.Context, params AuditCreateParams) (CreateResult, error) {
	if s.auditTransaction == nil {
		return CreateResult{}, ErrNotConfigured
	}
	normalized := auditRunCreate{AuditCreateParams: params}
	var workflowSnapshot json.RawMessage
	var err error
	if !validTaskRunIdentity(params) {
		return CreateResult{}, ErrInvalid
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
			normalized, workflowSnapshot, loadErr = normalizeTaskRunCreate(intent, params)
			if loadErr != nil {
				return loadErr
			}
			if validationErr := validateTaskRunIntent(intent, normalized); validationErr != nil {
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
			completion, completionErr := prepareAuditCompletion(intent, normalized)
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
			selected := map[string]contracts.ArtifactRef{}
			excluded := map[string]bool{}
			pin, e := auditprofile.DecodePinnedProgram(intent.ProgramSnapshot)
			if e != nil {
				return e
			}
			for slot, m := range normalized.Inputs {
				selected[slot] = m.Ref
				mapping := pin.Program.Workflows[intent.Execution.WorkflowRole].Inputs[slot]
				excluded[slot] = mapping.Source == "task-context" || mapping.Source == "prior-results"
			}
			budget := &contracts.MaterialBudget{}
			checked, e := checkWorkflowInputs(ctx, artifactService, normalized.Workflow, selected, artifacts.MaterialAuthority{OwnerID: intent.OwnerID, ProjectID: intent.ProjectID, AuditID: intent.Execution.AuditID}, func(string) (artifacts.Scope, error) { return artifacts.AuditScope(intent.Execution.AuditID) }, excluded, normalized.Inputs, budget)
			if e != nil {
				return e
			}
			prepared, e := artifactService.MaterializeInputs(ctx, runID, checked)
			if e != nil {
				return e
			}
			if err := materializeTaskRunContext(ctx, artifactService, intent, normalized, runID, &prepared); err != nil {
				return err
			}
			forks := prepared.Refs()

			if completion != nil {
				completion.Contract.Context = forks[completion.Contract.Context.Name]
				if err := runs.SetAuditCompletion(ctx, runID, *completion); err != nil {
					return err
				}
				run.AuditCompletion = completion
			}
			if len(normalized.Skills) == 0 {
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
			if intent.Execution.Role == auditstore.ExecutionTask {
				final, err := audits.GetRunCreationIntent(ctx, normalized.Claim, normalized.ExecutionID)
				if err != nil {
					return err
				}
				if !final.CanDispatch {
					return auditstore.ErrPrecondition
				}
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

func auditRunLabels(intent auditstore.RunCreationIntent) (runstore.RunMetadataLabels, error) {
	labels := runstore.RunMetadataLabels(intent.Labels).Clone()
	labels["audit.id"] = intent.Execution.AuditID
	labels["audit.role"] = string(intent.Execution.Role)
	if len(intent.Tasks) == 1 {
		labels["audit.task"] = intent.Tasks[0].Document.TaskID
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
