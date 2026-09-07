package public

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runrepeat"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

const (
	repeatAuthorityOrdinary = "ordinary"
	repeatAuthorityAudit    = "audit-managed"
	repeatStatusAvailable   = "available"
	repeatStatusUnavailable = "unavailable"
)

func (h *handler) getRunRepeatDraft(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	run, err := h.ownedRun(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if !runstore.RunLifecycleTerminal.Includes(run.State) {
		h.handleError(w, fmt.Errorf("%w: only terminal Runs have a repeat draft", runstore.ErrConflict))
		return
	}
	response := runRepeatDraftResponse{
		SourceRunID: run.RunID,
		Workflow: config.WorkflowRef{
			Name: run.WorkflowName, Version: run.WorkflowVersion,
		},
		ProjectID: cloneStringPointer(run.ProjectID),
		Notices:   []runRepeatDraftNotice{},
	}
	if run.PublicationMode == runstore.PublicationAuditManaged {
		response.Authority = repeatAuthorityAudit
		if auditID := run.MetadataLabels["audit.id"]; auditID != "" {
			response.AuditID = &auditID
		}
		response.Notices = append(response.Notices, runRepeatDraftNotice{
			Code: "audit_managed_run", Severity: "blocking", Field: "authority",
			Message: "This Run belongs to an Audit. Continue from the owning Audit instead of creating an untracked copy.",
		})
		writeJSON(w, http.StatusOK, response)
		return
	}
	response.Authority = repeatAuthorityOrdinary
	if run.ProjectID != nil {
		project, projectErr := h.dependencies.Projects.Get(
			r.Context(), run.OwnerID, *run.ProjectID,
		)
		if errors.Is(projectErr, projectstore.ErrNotFound) {
			response.Notices = append(response.Notices, runRepeatDraftNotice{
				Code: "project_unavailable", Severity: "blocking", Field: "projectId",
				Message: "The owning Project is no longer available. Its ProjectScope inputs cannot be used for another Run.",
			})
		} else if projectErr != nil {
			h.handleError(w, projectErr)
			return
		} else if project.Lifecycle == projectstore.LifecycleDeleting {
			response.Notices = append(response.Notices, runRepeatDraftNotice{
				Code: "project_deleting", Severity: "blocking", Field: "projectId",
				Message: "The owning Project is deleting and cannot accept another Run.",
			})
		}
	}
	draft := &runRepeatDraft{
		Parameters:    cloneParameters(run.Parameters),
		RuntimeLabels: append([]string{}, run.RuntimeLabels...),
		Labels:        run.MetadataLabels.Clone(),
		ExecutionConfig: runRepeatExecutionConfig{
			Status: repeatStatusUnavailable,
		},
		Inputs: make(map[string]runRepeatInputSelection),
	}
	response.Draft = draft

	selector := run.WorkflowName + "@" + run.WorkflowVersion
	workflowAvailable := true
	if _, workflowErr := h.dependencies.Config.Workflow(selector); workflowErr != nil {
		workflowAvailable = false
		response.Notices = append(response.Notices, runRepeatDraftNotice{
			Code: "workflow_unavailable", Severity: "blocking", Field: "workflow",
			Message: "The exact Workflow version is no longer available. Publish or restore it before configuring another Run.",
		})
	}

	snapshot, retained, snapshotErr := h.loadRunRepeatSnapshot(r.Context(), run)
	if snapshotErr != nil {
		if !errors.Is(snapshotErr, runrepeat.ErrInvalidSnapshot) &&
			!errors.Is(snapshotErr, artifacts.ErrArtifactIntegrity) {
			h.handleError(w, snapshotErr)
			return
		}
		response.Notices = append(response.Notices, runRepeatDraftNotice{
			Code: "repeat_request_invalid", Severity: "blocking", Field: "executionConfig",
			Message: "The retained repeat-request snapshot is invalid. Historical execution overrides cannot be trusted.",
		})
	}
	if retained && snapshotErr == nil {
		patch := snapshot.ExecutionConfig
		draft.ExecutionConfig = runRepeatExecutionConfig{
			Status: repeatStatusAvailable, Value: &patch,
		}
		draft.Inputs, err = h.repeatInputsFromSnapshot(r.Context(), run, snapshot)
		if err != nil {
			h.handleError(w, err)
			return
		}
		if workflowAvailable {
			if _, resolveErr := h.dependencies.Config.ResolveRunWorkflow(
				r.Context(), selector, patch, h.dependencies.Credentials,
			); resolveErr != nil {
				response.Notices = append(response.Notices, runRepeatDraftNotice{
					Code: "execution_configuration_unavailable", Severity: "blocking", Field: "executionConfig",
					Message: "At least one retained model, Gateway or credential selection is no longer available. Review the execution overrides before submitting.",
				})
			}
		}
	} else {
		legacyInputs, inputErr := h.repeatInputsFromLineage(r.Context(), run)
		if inputErr != nil {
			h.handleError(w, inputErr)
			return
		}
		draft.Inputs = legacyInputs
		response.Notices = append(response.Notices, runRepeatDraftNotice{
			Code: "execution_configuration_not_retained", Severity: "warning", Field: "executionConfig",
			Message: "This historical Run predates repeat-request retention. Current defaults are not substituted; explicitly review execution settings before submitting.",
		})
	}

	for slot, input := range draft.Inputs {
		if input.Status == repeatStatusUnavailable {
			response.Notices = append(response.Notices, runRepeatDraftNotice{
				Code: input.Code, Severity: "blocking", Field: "inputs." + slot,
				Message: input.Message,
			})
		}
	}
	if hasEvaluationLabels(draft.Labels) {
		response.Notices = append(response.Notices, runRepeatDraftNotice{
			Code: "evaluation_labels_require_review", Severity: "warning", Field: "labels",
			Message: "Evaluation labels were copied exactly. Review eval identity, sample and leg values before submitting to avoid duplicate records.",
		})
	}
	if err := h.appendRuntimeBindingNotices(r.Context(), run, &response.Notices); err != nil {
		h.handleError(w, err)
		return
	}
	sort.SliceStable(response.Notices, func(left, right int) bool {
		if response.Notices[left].Field != response.Notices[right].Field {
			return response.Notices[left].Field < response.Notices[right].Field
		}
		return response.Notices[left].Code < response.Notices[right].Code
	})
	writeJSON(w, http.StatusOK, response)
}

func (h *handler) loadRunRepeatSnapshot(
	ctx context.Context, run runstore.WorkflowRun,
) (runrepeat.Snapshot, bool, error) {
	retained, err := h.dependencies.Artifacts.ReadRunRepeatRequest(ctx, run.RunID)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		return runrepeat.Snapshot{}, false, nil
	}
	if err != nil {
		return runrepeat.Snapshot{}, true, err
	}
	snapshot, err := runrepeat.Decode(retained.Payload.Data)
	if err != nil {
		return runrepeat.Snapshot{}, true, err
	}
	if snapshot.Workflow.Name != run.WorkflowName || snapshot.Workflow.Version != run.WorkflowVersion ||
		!sameOptionalString(snapshot.ProjectID, run.ProjectID) {
		return runrepeat.Snapshot{}, true, fmt.Errorf("%w: snapshot does not match its Run", runrepeat.ErrInvalidSnapshot)
	}
	return snapshot, true, nil
}

func (h *handler) repeatInputsFromSnapshot(
	ctx context.Context, run runstore.WorkflowRun, snapshot runrepeat.Snapshot,
) (map[string]runRepeatInputSelection, error) {
	result := make(map[string]runRepeatInputSelection, len(snapshot.Inputs))
	for _, slot := range sortedArtifactSlots(snapshot.Inputs) {
		selection, err := h.resolveRepeatSource(ctx, run, snapshot.Inputs[slot])
		if err != nil {
			return nil, err
		}
		result[slot] = selection
	}
	return result, nil
}

func (h *handler) repeatInputsFromLineage(
	ctx context.Context, run runstore.WorkflowRun,
) (map[string]runRepeatInputSelection, error) {
	runArtifacts, err := h.dependencies.Artifacts.Run(run.RunID)
	if err != nil {
		return nil, err
	}
	namespace := "inputs"
	refs, err := runArtifacts.List(ctx, &namespace)
	if err != nil {
		return nil, err
	}
	result := make(map[string]runRepeatInputSelection, len(refs))
	for _, ref := range refs {
		metadata, metadataErr := runArtifacts.Metadata(ctx, ref)
		if metadataErr != nil {
			return nil, metadataErr
		}
		lineage, lineageErr := runArtifacts.ListLineage(
			ctx, metadata.Ref, artifacts.LineagePageQuery{Limit: 2},
		)
		if lineageErr != nil {
			return nil, lineageErr
		}
		var source *contracts.ArtifactRef
		for _, edge := range lineage {
			if edge.Kind != artifacts.LineageInputFork || !sameArtifactRef(edge.Target, metadata.Ref) ||
				!repeatSourceScopeMatchesRun(edge.SourceScope, edge.SourceScopeID, run) {
				continue
			}
			candidate := edge.Source
			if source != nil && !sameArtifactRef(*source, candidate) {
				source = nil
				break
			}
			source = &candidate
		}
		if source == nil {
			result[ref.Name] = unavailableRepeatInput(nil, "input_provenance_unavailable",
				"The original source for this input is not available. Select a replacement explicitly.")
			continue
		}
		selection, resolveErr := h.resolveRepeatSource(ctx, run, *source)
		if resolveErr != nil {
			return nil, resolveErr
		}
		result[ref.Name] = selection
	}
	return result, nil
}

func (h *handler) resolveRepeatSource(
	ctx context.Context, run runstore.WorkflowRun, ref contracts.ArtifactRef,
) (runRepeatInputSelection, error) {
	copy := cloneArtifactRef(ref)
	var (
		store artifacts.ScopedStore
		err   error
		kind  artifacts.ScopeKind
	)
	if run.ProjectID == nil {
		kind = artifacts.ScopeUser
		store, err = h.dependencies.Artifacts.User(run.OwnerID)
	} else {
		kind = artifacts.ScopeProject
		store, err = h.dependencies.Artifacts.Project(*run.ProjectID)
	}
	if err != nil {
		return runRepeatInputSelection{}, err
	}
	metadata, err := store.Metadata(ctx, ref)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		return unavailableRepeatInput(&copy, "input_source_unavailable",
			"The exact original Artifact revision is unavailable. Select a replacement explicitly."), nil
	}
	if err != nil {
		return runRepeatInputSelection{}, err
	}
	if !sameArtifactRef(metadata.Ref, ref) {
		return unavailableRepeatInput(&copy, "input_source_mismatch",
			"The exact original Artifact revision could not be verified. Select a replacement explicitly."), nil
	}
	metadataCopy := metadata
	return runRepeatInputSelection{
		Status: repeatStatusAvailable, SourceScope: kind,
		Artifact: &copy, Metadata: &metadataCopy,
	}, nil
}

func unavailableRepeatInput(
	ref *contracts.ArtifactRef, code string, message string,
) runRepeatInputSelection {
	return runRepeatInputSelection{
		Status: repeatStatusUnavailable, Artifact: ref, Code: code, Message: message,
	}
}

func (h *handler) appendRuntimeBindingNotices(
	ctx context.Context, run runstore.WorkflowRun, notices *[]runRepeatDraftNotice,
) error {
	pins := append([]runtimeconfig.PinnedLabel{run.RuntimeConfig.Default}, run.RuntimeConfig.Labels...)
	for _, pin := range pins {
		binding, err := h.dependencies.RuntimeConfigs.GetBinding(ctx, pin.Label)
		field := "runtimeLabels." + pin.Label
		if errors.Is(err, runtimeconfig.ErrNotFound) {
			*notices = append(*notices, runRepeatDraftNotice{
				Code: "runtime_binding_unavailable", Severity: "blocking", Field: field,
				Message: fmt.Sprintf("Runtime label %q is no longer bound. Remove it or restore its binding before submitting.", pin.Label),
			})
			continue
		}
		if err != nil {
			return err
		}
		if binding.Revision != pin.BindingRevision || binding.Ref != pin.Config {
			*notices = append(*notices, runRepeatDraftNotice{
				Code: "runtime_binding_changed", Severity: "warning", Field: field,
				Message: fmt.Sprintf("Runtime label %q now resolves to a different immutable configuration. Review it before submitting.", pin.Label),
			})
		}
	}
	return nil
}

func repeatSourceScopeMatchesRun(kind artifacts.ScopeKind, id string, run runstore.WorkflowRun) bool {
	if run.ProjectID == nil {
		return kind == artifacts.ScopeUser && id == run.OwnerID
	}
	return kind == artifacts.ScopeProject && id == *run.ProjectID
}

func hasEvaluationLabels(labels runstore.RunMetadataLabels) bool {
	if labels["purpose"] == "eval" {
		return true
	}
	for key := range labels {
		if strings.HasPrefix(key, "eval.") && len(key) > len("eval.") {
			return true
		}
	}
	return false
}

func sameArtifactRef(left, right contracts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name &&
		sameOptionalString(left.Revision, right.Revision)
}

func sameOptionalString(left, right *string) bool {
	return left == nil && right == nil || left != nil && right != nil && *left == *right
}

func cloneArtifactRef(source contracts.ArtifactRef) contracts.ArtifactRef {
	result := source
	result.Revision = cloneStringPointer(source.Revision)
	return result
}

func cloneStringPointer(source *string) *string {
	if source == nil {
		return nil
	}
	result := *source
	return &result
}
