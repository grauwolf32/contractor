// Run repeat projections use retained Run settings and exact original input lineage.
package runservice

import (
	"context"
	"errors"
	"fmt"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/clone"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"sort"
	"strings"
)

const (
	RepeatAuthorityOrdinary = "ordinary"
	RepeatAuthorityAudit    = "audit-managed"
	RepeatStatusAvailable   = "available"
	RepeatStatusUnavailable = "unavailable"
)

// RepeatWorkflowResolver checks that a retained Workflow selection still
// resolves.
type RepeatWorkflowResolver interface {
	Workflow(string) (resolved.ResolvedWorkflow, error)
	ResolveRunWorkflow(
		context.Context, string, resolved.CredentialLookup,
	) (resolved.ResolvedWorkflow, error)
}

// RepeatArtifacts reads original input provenance and exact source metadata.
type RepeatArtifacts interface {
	Run(string) (artifacts.ScopedStore, error)
	User(string) (artifacts.ScopedStore, error)
	Project(string) (artifacts.ScopedStore, error)
}

// RuntimeBindingReader reads the current binding of a Runtime label.
type RuntimeBindingReader interface {
	GetBinding(context.Context, string) (runtimeconfig.Binding, error)
}

type RepeatRunReader interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
}

type RepeatOptions struct {
	Runs           RepeatRunReader
	Projects       ProjectReader
	Config         RepeatWorkflowResolver
	Credentials    resolved.CredentialLookup
	Artifacts      RepeatArtifacts
	RuntimeConfigs RuntimeBindingReader
}

type RepeatReader struct{ RepeatOptions }

func NewRepeatReader(options RepeatOptions) *RepeatReader { return &RepeatReader{options} }

type RepeatProjection struct {
	SourceRunID string
	Authority   string
	Workflow    resolved.WorkflowRef
	ProjectID   *string
	AuditID     *string
	Draft       *RepeatDraft
	Notices     []RepeatNotice
}
type RepeatDraft struct {
	RequiredPlacementLabels []string
	Parameters              map[string]string
	RuntimeLabels           []string
	Labels                  runstore.RunMetadataLabels
	Inputs                  map[string]RepeatInput
}
type RepeatInput struct {
	Status        string
	SourceScope   artifacts.ScopeKind
	Artifact      *contracts.ArtifactRef
	Metadata      *artifacts.Metadata
	CanReplace    bool
	Code, Message string
}
type RepeatNotice struct {
	Code     string `json:"code"`
	Severity string `json:"severity"`
	Field    string `json:"field,omitempty"`
	Message  string `json:"message"`
}

// Read rechecks ownership and current availability without changing retained
// settings or replacing exact input refs with mutable current bindings.
func (h *RepeatReader) Read(ctx context.Context, ownerID, runID string) (RepeatProjection, error) {
	run, err := h.Runs.GetRun(ctx, runID)
	if err != nil {
		return RepeatProjection{}, err
	}
	if run.OwnerID != ownerID {
		return RepeatProjection{}, runstore.ErrNotFound
	}
	if !runstore.RunLifecycleTerminal.Includes(run.State) {
		return RepeatProjection{}, fmt.Errorf("%w: only terminal Runs have a repeat draft", runstore.ErrConflict)
	}
	response := RepeatProjection{
		SourceRunID: run.RunID,
		Workflow: resolved.WorkflowRef{
			Name: run.WorkflowName, Version: run.WorkflowVersion,
		},
		ProjectID: clone.Pointer(run.ProjectID),
		Notices:   []RepeatNotice{},
	}
	if run.PublicationMode == runstore.PublicationAuditManaged {
		response.Authority = RepeatAuthorityAudit
		if auditID := run.MetadataLabels["audit.id"]; auditID != "" {
			response.AuditID = &auditID
		}
		response.Notices = append(response.Notices, RepeatNotice{
			Code: "audit_managed_run", Severity: "blocking", Field: "authority",
			Message: "This Run belongs to an Audit. Continue from the owning Audit instead of creating an untracked copy.",
		})
		return response, nil
	}
	response.Authority = RepeatAuthorityOrdinary
	projectUploadsAllowed := false
	if run.ProjectID != nil {
		project, projectErr := h.Projects.Get(
			ctx, run.OwnerID, *run.ProjectID,
		)
		if errors.Is(projectErr, projectstore.ErrNotFound) {
			response.Notices = append(response.Notices, RepeatNotice{
				Code: "project_unavailable", Severity: "blocking", Field: "projectId",
				Message: "The owning Project is no longer available. Its ProjectScope inputs cannot be used for another Run.",
			})
		} else if projectErr != nil {
			return RepeatProjection{}, projectErr
		} else if project.Lifecycle == projectstore.LifecycleDeleting {
			response.Notices = append(response.Notices, RepeatNotice{
				Code: "project_deleting", Severity: "blocking", Field: "projectId",
				Message: "The owning Project is deleting and cannot accept another Run.",
			})
		} else {
			projectUploadsAllowed = true
		}
	}
	selector := run.WorkflowName + "@" + run.WorkflowVersion
	workflowAvailable := true
	currentWorkflow, workflowErr := h.Config.Workflow(selector)
	if workflowErr != nil {
		workflowAvailable = false
		response.Notices = append(response.Notices, RepeatNotice{
			Code: "workflow_unavailable", Severity: "blocking", Field: "workflow",
			Message: "The exact Workflow version is no longer available. Publish or restore it before configuring another Run.",
		})
	}

	inputs, err := h.originalInputSources(ctx, run)
	if err != nil {
		if !errors.Is(err, blob.ErrIntegrity) {
			return RepeatProjection{}, err
		}
		response.Notices = append(response.Notices, RepeatNotice{
			Code: "input_provenance_invalid", Severity: "blocking", Field: "inputs",
			Message: "The original Run input sources could not be verified. Configure a new Run from the Workflow.",
		})
		return response, nil
	}

	draft := &RepeatDraft{
		Parameters:              clone.Map(run.Parameters),
		RuntimeLabels:           append([]string{}, run.RuntimeLabels...),
		RequiredPlacementLabels: append([]string{}, run.RequiredPlacementLabels...),
		Labels:                  run.MetadataLabels.Clone(),
	}
	response.Draft = draft
	draft.Inputs, err = h.repeatInputsFromSources(ctx, run, inputs, projectUploadsAllowed)
	if err != nil {
		return RepeatProjection{}, err
	}
	if workflowAvailable {
		for slot, source := range inputs {
			expected := artifacts.ScopeUser
			if run.ProjectID != nil && currentWorkflow.Inputs[slot].SourceScope != "user" {
				expected = artifacts.ScopeProject
			}
			if source.SourceScope != expected {
				draft.Inputs[slot] = RepeatInput{Status: RepeatStatusUnavailable, Code: "input_source_scope_changed", Message: "The Workflow now selects a different source scope. Select the input again before creating another Run."}
			}
		}
		if _, resolveErr := h.Config.ResolveRunWorkflow(
			ctx, selector, h.Credentials,
		); errors.Is(resolveErr, resolved.ErrCredentialLookup) {
			// An outage is not an unavailable selection; let the client retry.
			return RepeatProjection{}, resolveErr
		} else if resolveErr != nil {
			response.Notices = append(response.Notices, RepeatNotice{
				Code: "execution_configuration_unavailable", Severity: "blocking", Field: "workflow",
				Message: "The Workflow's execution configuration is unavailable. Review the Workflow before submitting.",
			})
		}
	}

	for slot, input := range draft.Inputs {
		if input.Status == RepeatStatusUnavailable {
			response.Notices = append(response.Notices, RepeatNotice{
				Code: input.Code, Severity: "blocking", Field: "inputs." + slot,
				Message: input.Message,
			})
		}
	}
	if hasEvaluationLabels(draft.Labels) {
		response.Notices = append(response.Notices, RepeatNotice{
			Code: "evaluation_labels_require_review", Severity: "warning", Field: "labels",
			Message: "Evaluation labels were copied exactly. Review eval identity, sample and leg values before submitting to avoid duplicate records.",
		})
	}
	if err := h.appendRuntimeBindingNotices(ctx, run, &response.Notices); err != nil {
		return RepeatProjection{}, err
	}
	sort.SliceStable(response.Notices, func(left, right int) bool {
		if response.Notices[left].Field != response.Notices[right].Field {
			return response.Notices[left].Field < response.Notices[right].Field
		}
		return response.Notices[left].Code < response.Notices[right].Code
	})
	return response, nil
}

func (h *RepeatReader) originalInputSources(ctx context.Context, run runstore.WorkflowRun) (map[string]artifacts.LineageEdge, error) {
	store, err := h.Artifacts.Run(run.RunID)
	if err != nil {
		return nil, err
	}
	return runstore.OriginalInputSources(ctx, run, store)
}

func (h *RepeatReader) repeatInputsFromSources(
	ctx context.Context, run runstore.WorkflowRun, inputs map[string]artifacts.LineageEdge, projectUploadsAllowed bool,
) (map[string]RepeatInput, error) {
	result := make(map[string]RepeatInput, len(inputs))
	for _, slot := range sortedArtifactSlots(inputs) {
		selection, err := h.resolveRepeatSource(ctx, run, inputs[slot], projectUploadsAllowed)
		if err != nil {
			return nil, err
		}
		result[slot] = selection
	}
	return result, nil
}

func (h *RepeatReader) resolveRepeatSource(
	ctx context.Context, run runstore.WorkflowRun, source artifacts.LineageEdge, projectUploadsAllowed bool,
) (RepeatInput, error) {
	ref := source.Source
	copy := ref.Clone()
	var (
		store artifacts.ScopedStore
		err   error
		kind  artifacts.ScopeKind
	)
	if source.SourceScope == artifacts.ScopeUser {
		kind = artifacts.ScopeUser
		store, err = h.Artifacts.User(run.OwnerID)
	} else {
		kind = artifacts.ScopeProject
		store, err = h.Artifacts.Project(*run.ProjectID)
	}
	if err != nil {
		return RepeatInput{}, err
	}
	metadata, err := store.Metadata(ctx, ref)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		return unavailableRepeatInput(&copy, "input_source_unavailable",
			"The exact original Artifact revision is unavailable. Select a replacement explicitly."), nil
	}
	if err != nil {
		return RepeatInput{}, err
	}
	if !metadata.Ref.Equal(ref) {
		return unavailableRepeatInput(&copy, "input_source_mismatch",
			"The exact original Artifact revision could not be verified. Select a replacement explicitly."), nil
	}
	canReplace := (kind == artifacts.ScopeUser || projectUploadsAllowed) && store.CanReplace(metadata)
	return RepeatInput{
		Status: RepeatStatusAvailable, SourceScope: kind,
		Artifact: &copy, Metadata: &metadata, CanReplace: canReplace,
	}, nil
}

func unavailableRepeatInput(
	ref *contracts.ArtifactRef, code string, message string,
) RepeatInput {
	return RepeatInput{
		Status: RepeatStatusUnavailable, Artifact: ref, Code: code, Message: message,
	}
}

func (h *RepeatReader) appendRuntimeBindingNotices(
	ctx context.Context, run runstore.WorkflowRun, notices *[]RepeatNotice,
) error {
	pins := append([]runtimeconfig.PinnedLabel{run.RuntimeConfig.Default}, run.RuntimeConfig.Labels...)
	for _, pin := range pins {
		binding, err := h.RuntimeConfigs.GetBinding(ctx, pin.Label)
		field := "runtimeLabels." + pin.Label
		if errors.Is(err, runtimeconfig.ErrNotFound) {
			*notices = append(*notices, RepeatNotice{
				Code: "runtime_binding_unavailable", Severity: "blocking", Field: field,
				Message: fmt.Sprintf("Runtime label %q is no longer bound. Remove it or restore its binding before submitting.", pin.Label),
			})
			continue
		}
		if err != nil {
			return err
		}
		if binding.Revision != pin.BindingRevision || binding.Ref != pin.Config {
			*notices = append(*notices, RepeatNotice{
				Code: "runtime_binding_changed", Severity: "warning", Field: field,
				Message: fmt.Sprintf("Runtime label %q now resolves to a different immutable configuration. Review it before submitting.", pin.Label),
			})
		}
	}
	return nil
}

func sortedArtifactSlots[T any](source map[string]T) []string {
	result := make([]string, 0, len(source))
	for slot := range source {
		result = append(result, slot)
	}
	sort.Strings(result)
	return result
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
