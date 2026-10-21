package config

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/securefile"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

// ModelChoice describes a deployed immutable route and the compatible role policies.
// Digest identifies operator metadata, never an observed provider weights revision.
type ModelChoice struct {
	ID                  string            `json:"id" strictjson:"required"`
	Digest              string            `json:"digest" strictjson:"required"`
	UpstreamModel       string            `json:"upstreamModel" strictjson:"required"`
	ContextWindowTokens int               `json:"contextWindowTokens" strictjson:"required"`
	MaxOutputTokens     int               `json:"maxOutputTokens" strictjson:"required"`
	Policies            map[string]string `json:"policies" strictjson:"required"`
}

type modelChoiceCatalog struct {
	SchemaVersion    string                            `json:"schemaVersion" strictjson:"required"`
	InitialSelection settingsstore.ModelSelectionValue `json:"initialSelection" strictjson:"required"`
	Models           []ModelChoice                     `json:"models" strictjson:"required"`
}

type ModelSelectionStore interface {
	InitializeModelSelection(context.Context, settingsstore.ModelSelectionValue) (settingsstore.ModelSelectionSettings, error)
	GetModelSelection(context.Context) (settingsstore.ModelSelectionSettings, error)
	UpdateModelSelection(context.Context, settingsstore.ModelSelectionValue, uint64) (settingsstore.ModelSelectionSettings, error)
}

type deploymentModelSelection struct {
	catalog     modelChoiceCatalog
	fingerprint string
	store       ModelSelectionStore
}

// ModelSelectionInfo is the server-confirmed Operations view. An absent compiled
// catalog disables this optional deployment facility without inventing a model.
type ModelSelectionInfo struct {
	Configured    bool                               `json:"configured"`
	Models        []ModelChoice                      `json:"models"`
	Selection     *settingsstore.ModelSelectionValue `json:"selection,omitempty"`
	Revision      string                             `json:"revision,omitempty"`
	CatalogDigest string                             `json:"catalogDigest,omitempty"`
	UpdatedAt     *time.Time                         `json:"updatedAt,omitempty"`
}

var modelRoles = []string{"planner", "worker", "summarizer", "audit_completion_worker"}

func readModelChoiceCatalog(root string, snapshot *Snapshot) (*deploymentModelSelection, error) {
	data, err := securefile.ReadBeneath(root, "model-selection.json", 1<<20)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read model-choice catalog: %w", err)
	}
	var catalog modelChoiceCatalog
	if err := strictjson.Decode(data, &catalog); err != nil {
		return nil, fmt.Errorf("decode model-choice catalog: %w", err)
	}
	if catalog.SchemaVersion != "1" || len(catalog.Models) == 0 || len(catalog.Models) > 32 {
		return nil, fmt.Errorf("invalid model-choice catalog bounds or schema")
	}
	current := &deploymentModelSelection{catalog: catalog}
	seen := map[string]bool{}
	policies := map[string]bool{}
	for _, model := range catalog.Models {
		if seen[model.ID] || (settingsstore.ModelSelectionValue{DefaultModel: model.ID, Roles: map[string]string{}}).Validate() != nil {
			return nil, fmt.Errorf("invalid or duplicate model choice")
		}
		seen[model.ID] = true
		hash, err := hex.DecodeString(strings.TrimPrefix(model.Digest, "sha256:"))
		if err != nil || len(hash) != sha256.Size || model.Digest != "sha256:"+hex.EncodeToString(hash) {
			return nil, fmt.Errorf("invalid model profile digest")
		}
		if len(model.Policies) == 0 || len(model.Policies) > 4 || model.UpstreamModel == "" || len(model.UpstreamModel) > 160 || model.ContextWindowTokens <= 0 || model.ContextWindowTokens > contracts.MaxModelContextTokens || model.MaxOutputTokens <= 0 || model.MaxOutputTokens >= model.ContextWindowTokens {
			return nil, fmt.Errorf("invalid model choice capabilities")
		}
		for role, selector := range model.Policies {
			if !isModelRole(role) || policies[selector] {
				return nil, fmt.Errorf("invalid or duplicate model role policy")
			}
			policies[selector] = true
			policy, err := snapshot.ModelPolicy(selector)
			if err != nil {
				return nil, err
			}
			if policy.ContextWindowTokens != model.ContextWindowTokens || policy.MaxOutputTokens > model.MaxOutputTokens || !strings.HasPrefix(policy.Model, model.UpstreamModel+"@"+hex.EncodeToString(hash)[:16]+"/") {
				return nil, fmt.Errorf("model choice capability or route does not match policy")
			}
			switch role {
			case "planner":
				err = resolved.ValidatePlannerModelPolicy(policy)
			case "summarizer":
				err = policy.ValidateForWorkerSummarizer()
			default:
				err = policy.ValidateForWorker(true)
			}
			if err != nil {
				return nil, err
			}
		}
	}
	if err := current.validate(catalog.InitialSelection); err != nil {
		return nil, err
	}
	canonical, err := json.Marshal(catalog.Models)
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(canonical)
	current.fingerprint = "sha256:" + hex.EncodeToString(digest[:])
	return current, nil
}

// ValidateModelChoiceCatalog validates the optional compiled operator artifact
// against the same immutable policies the Server will execute.
func ValidateModelChoiceCatalog(root string, snapshot *Snapshot) error {
	_, err := readModelChoiceCatalog(root, snapshot)
	return err
}

func isModelRole(role string) bool {
	switch role {
	case "planner", "worker", "summarizer", "audit_completion_worker":
		return true
	}
	return false
}

func (m *deploymentModelSelection) selected(value settingsstore.ModelSelectionValue, role string) (ModelChoice, error) {
	id := value.DefaultModel
	if override, ok := value.Roles[role]; ok {
		id = override
	}
	for _, model := range m.catalog.Models {
		if model.ID == id && model.Policies[role] != "" {
			return model, nil
		}
	}
	return ModelChoice{}, settingsstore.ErrModelSelectionInvalid
}

func (m *deploymentModelSelection) validate(value settingsstore.ModelSelectionValue) error {
	if err := value.Validate(); err != nil {
		return err
	}
	knownDefault := false
	for _, model := range m.catalog.Models {
		if model.ID == value.DefaultModel {
			knownDefault = true
			break
		}
	}
	if !knownDefault {
		return settingsstore.ErrModelSelectionInvalid
	}
	for _, role := range modelRoles {
		if _, err := m.selected(value, role); err != nil {
			return err
		}
	}
	return nil
}

// ConfigureModelSelection is called once before serving. PostgreSQL initializes
// an absent setting only; the compiled selection never overrides a saved choice.
func (m *Manager) ConfigureModelSelection(ctx context.Context, store ModelSelectionStore) error {
	selection, err := readModelChoiceCatalog(m.operatorRoot, m.Snapshot())
	if err != nil {
		return err
	}
	if selection == nil {
		return nil
	}
	if store == nil {
		return fmt.Errorf("model selection store is required")
	}
	selection.store = store
	saved, err := store.InitializeModelSelection(ctx, selection.catalog.InitialSelection)
	if err != nil {
		return err
	}
	if err := selection.validate(saved.Value); err != nil {
		return fmt.Errorf("saved model selection is not deployed: %w", err)
	}
	m.modelSelection = selection
	return nil
}

func (m *deploymentModelSelection) info(saved settingsstore.ModelSelectionSettings) ModelSelectionInfo {
	models := make([]ModelChoice, len(m.catalog.Models))
	for i, model := range m.catalog.Models {
		model.Policies = maps.Clone(model.Policies)
		models[i] = model
	}
	value := saved.Value.Clone()
	updated := saved.UpdatedAt
	return ModelSelectionInfo{Configured: true, Models: models, Selection: &value, Revision: strconv.FormatUint(saved.Revision, 10), CatalogDigest: m.fingerprint, UpdatedAt: &updated}
}

func (m *Manager) GetModelSelection(ctx context.Context) (ModelSelectionInfo, error) {
	if m.modelSelection == nil {
		return ModelSelectionInfo{Models: []ModelChoice{}}, nil
	}
	saved, err := m.modelSelection.store.GetModelSelection(ctx)
	if err != nil {
		return ModelSelectionInfo{}, err
	}
	if err := m.modelSelection.validate(saved.Value); err != nil {
		return ModelSelectionInfo{}, err
	}
	return m.modelSelection.info(saved), nil
}

func (m *Manager) ReplaceModelSelection(ctx context.Context, value settingsstore.ModelSelectionValue, revision uint64, catalogDigest string) (ModelSelectionInfo, error) {
	selection := m.modelSelection
	if selection == nil {
		return ModelSelectionInfo{}, settingsstore.ErrModelSelectionInvalid
	}
	if selection.fingerprint != catalogDigest {
		return ModelSelectionInfo{}, settingsstore.ErrPrecondition
	}
	if err := selection.validate(value); err != nil {
		return ModelSelectionInfo{}, err
	}
	saved, err := selection.store.UpdateModelSelection(ctx, value, revision)
	if err != nil {
		return ModelSelectionInfo{}, err
	}
	return selection.info(saved), nil
}

func (m *Manager) modelDefaults(ctx context.Context) (map[string]contracts.ResolvedModelPolicy, error) {
	if m.modelSelection == nil {
		return nil, nil
	}
	value, err := m.modelSelection.store.GetModelSelection(ctx)
	if err != nil {
		return nil, err
	}
	if err := m.modelSelection.validate(value.Value); err != nil {
		return nil, err
	}
	defaults := make(map[string]contracts.ResolvedModelPolicy, 4)
	snapshot := m.Snapshot()
	for _, role := range modelRoles {
		model, err := m.modelSelection.selected(value.Value, role)
		if err != nil {
			return nil, err
		}
		policy, err := snapshot.ModelPolicy(model.Policies[role])
		if err != nil {
			return nil, err
		}
		defaults[role] = policy
	}
	return defaults, nil
}

func applyModelDefaults(workflow *resolved.ResolvedWorkflow, defaults map[string]contracts.ResolvedModelPolicy) {
	if len(defaults) == 0 {
		return
	}
	for name, stage := range workflow.Stages {
		modeled := stage.ExecutionConfig.Planner != nil
		for _, binding := range stage.Agents {
			if !binding.Template.IsToolWorker() {
				modeled = true
				break
			}
		}
		if !modeled {
			continue
		}
		stage.ModelDefaults = make(map[string]contracts.ResolvedModelPolicy, len(defaults))
		for role, policy := range defaults {
			stage.ModelDefaults[role] = policy.Clone()
		}
		resolved.ApplyStageModelDefaults(&stage)
		workflow.Stages[name] = stage
	}
}

// ResolveAuditProfile freezes deployment model defaults into a new Audit's
// Workflow closures; its later child Runs use those pinned closures.
func (m *Manager) ResolveAuditProfile(ctx context.Context, raw string) (auditprofile.ResolvedAuditProfile, error) {
	profile, err := m.Snapshot().AuditProfile(raw)
	if err != nil {
		return profile, err
	}
	defaults, err := m.modelDefaults(ctx)
	if err != nil {
		return profile, err
	}
	for role, binding := range profile.Workflows {
		applyModelDefaults(&binding.Workflow, defaults)
		if err := resolved.ValidateWorkflowExecutionConfigs(binding.Workflow); err != nil {
			return profile, err
		}
		if err := resolved.ResolveEscalationVariants(&binding.Workflow); err != nil {
			return profile, err
		}
		profile.Workflows[role] = binding
	}
	return profile, nil
}

// ResolveAuditProfile on an offline Snapshot has no deployment setting.
func (s *Snapshot) ResolveAuditProfile(_ context.Context, raw string) (auditprofile.ResolvedAuditProfile, error) {
	return s.AuditProfile(raw)
}
