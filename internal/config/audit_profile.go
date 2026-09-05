package config

import (
	"fmt"
	"sort"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	MaxAuditProfileStandards  = 16
	MaxAuditProfileInputs     = 32
	MaxAuditProfileWorkflows  = 16
	MaxAuditRounds            = 32
	MaxAuditBatchSize         = 64
	MaxAuditItemsPerRound     = 10_000
	MaxAuditItemsTotal        = 100_000
	MaxAuditSubmittedRuns     = 1_000_000
	MaxAuditItemRunAttempts   = 10
	MaxAuditDeadlineSeconds   = 365 * 24 * 60 * 60
	MaxAuditEvidenceBytes     = int64(1 << 30)
	MaxAuditLiteralParamBytes = 4096

	auditTaskPackageMediaType = "application/zip"
)

type AuditProfileMode string

const (
	AuditModeRiskAssessment           AuditProfileMode = "risk-assessment"
	AuditModeRequirementsVerification AuditProfileMode = "requirements-verification"
	AuditModeCustomChecklist          AuditProfileMode = "custom-checklist"
	AuditModeOperationTracing         AuditProfileMode = "operation-tracing"
	AuditModeFindingVerification      AuditProfileMode = "finding-verification"
)

func (m AuditProfileMode) valid() bool {
	switch m {
	case AuditModeRiskAssessment, AuditModeRequirementsVerification,
		AuditModeCustomChecklist, AuditModeOperationTracing,
		AuditModeFindingVerification:
		return true
	default:
		return false
	}
}

type AuditProfileRef struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}

type AuditStandardRef struct {
	Scheme  string `json:"scheme"`
	Version string `json:"version"`
}

type AuditProfileInput struct {
	Required   bool     `json:"required"`
	MediaTypes []string `json:"mediaTypes"`
}

type AuditInventory struct {
	Implementation   string `json:"implementation"`
	SourceInput      string `json:"sourceInput"`
	ItemWorkflowRole string `json:"itemWorkflowRole"`
}

type AuditWorkflowInputSource string

const (
	AuditInputFromAudit          AuditWorkflowInputSource = "audit-input"
	AuditInputFromItemPackage    AuditWorkflowInputSource = "item-package"
	AuditInputFromRetainedOutput AuditWorkflowInputSource = "retained-output"
)

type AuditWorkflowInputMapping struct {
	Source AuditWorkflowInputSource `json:"source"`
	Name   string                   `json:"name,omitempty"`
	Role   string                   `json:"role,omitempty"`
}

type AuditWorkflowParameterSource string

const (
	AuditParameterLiteral    AuditWorkflowParameterSource = "literal"
	AuditParameterItemField  AuditWorkflowParameterSource = "item-field"
	AuditParameterScopeField AuditWorkflowParameterSource = "scope-field"
)

type AuditWorkflowParameterMapping struct {
	Source AuditWorkflowParameterSource `json:"source"`
	Name   string                       `json:"name,omitempty"`
	Value  string                       `json:"value,omitempty"`
}

type ResolvedAuditWorkflowBinding struct {
	Workflow   ResolvedWorkflow                         `json:"workflow"`
	Inputs     map[string]AuditWorkflowInputMapping     `json:"inputs"`
	Parameters map[string]AuditWorkflowParameterMapping `json:"parameters"`
	Outputs    map[string]string                        `json:"outputs"`
}

type AuditRoundMode string

const AuditRoundFixedBarrier AuditRoundMode = "fixed-barrier"

type AuditIncompleteRoundPolicy string

const (
	AuditIncompleteAssessWithGaps AuditIncompleteRoundPolicy = "assess-with-gaps"
	AuditIncompleteFail           AuditIncompleteRoundPolicy = "fail"
)

type AuditExecutionPolicy struct {
	RoundMode          AuditRoundMode             `json:"roundMode"`
	MaxRounds          int                        `json:"maxRounds"`
	BatchSize          int                        `json:"batchSize"`
	MaxItemsPerRound   int                        `json:"maxItemsPerRound"`
	MaxItemsTotal      int                        `json:"maxItemsTotal"`
	MaxSubmittedRuns   int                        `json:"maxSubmittedRuns"`
	MaxItemRunAttempts int                        `json:"maxItemRunAttempts"`
	DeadlineSeconds    int                        `json:"deadlineSeconds"`
	MaxEvidenceBytes   int64                      `json:"maxEvidenceBytes"`
	IncompleteRound    AuditIncompleteRoundPolicy `json:"incompleteRound"`
}

type AuditActiveChecksPolicy string
type AuditFindingConfirmationPolicy string
type AuditNotApplicablePolicy string
type AuditReportAcceptancePolicy string

const (
	AuditActiveChecksProhibited       AuditActiveChecksPolicy = "prohibited"
	AuditActiveChecksAutomatic        AuditActiveChecksPolicy = "automatic"
	AuditActiveChecksApprovalRequired AuditActiveChecksPolicy = "approval-required"

	AuditFindingDisabled      AuditFindingConfirmationPolicy = "disabled"
	AuditFindingHumanRequired AuditFindingConfirmationPolicy = "human-required"

	AuditNotApplicableHumanRequired AuditNotApplicablePolicy = "human-required"
	AuditNotApplicableProfileRule   AuditNotApplicablePolicy = "profile-rule"

	AuditReportAutomatic     AuditReportAcceptancePolicy = "automatic"
	AuditReportHumanRequired AuditReportAcceptancePolicy = "human-required"
)

type AuditInteractionPolicy struct {
	ActiveChecks        AuditActiveChecksPolicy        `json:"activeChecks"`
	FindingConfirmation AuditFindingConfirmationPolicy `json:"findingConfirmation"`
	NotApplicable       AuditNotApplicablePolicy       `json:"notApplicable"`
	ReportAcceptance    AuditReportAcceptancePolicy    `json:"reportAcceptance"`
}

// ResolvedAuditProfile is the immutable Server-side authority retained when an
// Audit starts. Workers never receive this value and cannot select a profile.
type ResolvedAuditProfile struct {
	Ref         AuditProfileRef                         `json:"ref"`
	Mode        AuditProfileMode                        `json:"mode"`
	Standards   []AuditStandardRef                      `json:"standards"`
	Inputs      map[string]AuditProfileInput            `json:"inputs"`
	Inventory   AuditInventory                          `json:"inventory"`
	Workflows   map[string]ResolvedAuditWorkflowBinding `json:"workflows"`
	Execution   AuditExecutionPolicy                    `json:"execution"`
	Interaction AuditInteractionPolicy                  `json:"interaction"`
}

type auditProfileDocument struct {
	APIVersion string                  `yaml:"apiVersion"`
	Kind       string                  `yaml:"kind"`
	Metadata   *metadataSource         `yaml:"metadata"`
	Spec       *auditProfileSpecSource `yaml:"spec"`
}

type auditProfileSpecSource struct {
	Mode        string                                 `yaml:"mode"`
	Standards   []auditStandardRefSource               `yaml:"standards,omitempty"`
	Inputs      *map[string]auditProfileInputSource    `yaml:"inputs"`
	Inventory   *auditInventorySource                  `yaml:"inventory"`
	Workflows   *map[string]auditWorkflowBindingSource `yaml:"workflows"`
	Execution   *auditExecutionPolicySource            `yaml:"execution"`
	Interaction *auditInteractionPolicySource          `yaml:"interaction"`
}

type auditStandardRefSource struct {
	Scheme  string `yaml:"scheme"`
	Version string `yaml:"version"`
}

type auditProfileInputSource struct {
	Required   *bool    `yaml:"required"`
	MediaTypes []string `yaml:"mediaTypes"`
}

type auditInventorySource struct {
	Implementation   string `yaml:"implementation"`
	SourceInput      string `yaml:"sourceInput"`
	ItemWorkflowRole string `yaml:"itemWorkflowRole"`
}

type auditWorkflowBindingSource struct {
	Ref        string                                      `yaml:"ref"`
	Inputs     *map[string]auditWorkflowInputMappingSource `yaml:"inputs"`
	Parameters *map[string]auditParameterMappingSource     `yaml:"parameters"`
	Outputs    *map[string]string                          `yaml:"outputs"`
}

type auditWorkflowInputMappingSource struct {
	Source string `yaml:"source"`
	Name   string `yaml:"name,omitempty"`
	Role   string `yaml:"role,omitempty"`
}

type auditParameterMappingSource struct {
	Source string  `yaml:"source"`
	Name   string  `yaml:"name,omitempty"`
	Value  *string `yaml:"value,omitempty"`
}

type auditExecutionPolicySource struct {
	RoundMode          string `yaml:"roundMode"`
	MaxRounds          int    `yaml:"maxRounds"`
	BatchSize          int    `yaml:"batchSize"`
	MaxItemsPerRound   int    `yaml:"maxItemsPerRound"`
	MaxItemsTotal      int    `yaml:"maxItemsTotal"`
	MaxSubmittedRuns   int    `yaml:"maxSubmittedRuns"`
	MaxItemRunAttempts int    `yaml:"maxItemRunAttempts"`
	DeadlineSeconds    int    `yaml:"deadlineSeconds"`
	MaxEvidenceBytes   int64  `yaml:"maxEvidenceBytes"`
	IncompleteRound    string `yaml:"incompleteRound"`
}

type auditInteractionPolicySource struct {
	ActiveChecks        string `yaml:"activeChecks"`
	FindingConfirmation string `yaml:"findingConfirmation"`
	NotApplicable       string `yaml:"notApplicable"`
	ReportAcceptance    string `yaml:"reportAcceptance"`
}

func (l *loader) loadAuditProfiles() error {
	files, err := l.discover("audit-profiles")
	if err != nil {
		return err
	}
	for _, file := range files {
		if file.source != ConfigurationSourceOperator {
			return fmt.Errorf("%s: AuditProfile must be operator-authored", file.relative)
		}
		document, decodeErr := decodeOne[auditProfileDocument](file)
		if decodeErr != nil {
			return decodeErr
		}
		selector, resolveErr := validateEnvelope(
			document.APIVersion, document.Kind, auditProfileKind, document.Metadata,
		)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		if _, exists := l.auditProfiles[selector.String()]; exists {
			return fmt.Errorf("%s: duplicate AuditProfile identity %s", file.relative, selector)
		}
		profile, resolveErr := l.resolveAuditProfile(selector, document.Spec)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		l.auditProfiles[selector.String()] = profile
	}
	return nil
}

func (l *loader) resolveAuditProfile(
	selector Selector, spec *auditProfileSpecSource,
) (ResolvedAuditProfile, error) {
	if spec == nil {
		return ResolvedAuditProfile{}, fmt.Errorf("spec is required")
	}
	mode := AuditProfileMode(spec.Mode)
	if !mode.valid() {
		return ResolvedAuditProfile{}, fmt.Errorf("spec.mode is invalid")
	}
	standards, err := resolveAuditStandards(spec.Standards)
	if err != nil {
		return ResolvedAuditProfile{}, err
	}
	inputs, err := resolveAuditProfileInputs(spec.Inputs)
	if err != nil {
		return ResolvedAuditProfile{}, err
	}
	workflows, err := l.resolveAuditWorkflowBindings(spec.Workflows, inputs)
	if err != nil {
		return ResolvedAuditProfile{}, err
	}
	inventory, err := resolveAuditInventory(spec.Inventory, inputs, workflows)
	if err != nil {
		return ResolvedAuditProfile{}, err
	}
	if err := validateAuditModeInventory(mode, inventory.Implementation); err != nil {
		return ResolvedAuditProfile{}, err
	}
	if err := validateRetainedOutputDependencies(workflows); err != nil {
		return ResolvedAuditProfile{}, err
	}
	execution, err := resolveAuditExecutionPolicy(spec.Execution)
	if err != nil {
		return ResolvedAuditProfile{}, err
	}
	interaction, err := resolveAuditInteractionPolicy(spec.Interaction)
	if err != nil {
		return ResolvedAuditProfile{}, err
	}

	profile := ResolvedAuditProfile{
		Ref:  AuditProfileRef{Name: selector.ID, Version: selector.Version},
		Mode: mode, Standards: standards, Inputs: inputs, Inventory: inventory,
		Workflows: workflows, Execution: execution, Interaction: interaction,
	}
	digest, err := auditProfileDigest(selector, profile)
	if err != nil {
		return ResolvedAuditProfile{}, err
	}
	profile.Ref.Digest = digest
	return profile, nil
}

func resolveAuditStandards(source []auditStandardRefSource) ([]AuditStandardRef, error) {
	if len(source) > MaxAuditProfileStandards {
		return nil, fmt.Errorf("spec.standards may contain at most %d entries", MaxAuditProfileStandards)
	}
	result := make([]AuditStandardRef, len(source))
	seen := make(map[string]struct{}, len(source))
	for index, candidate := range source {
		if !idPattern.MatchString(candidate.Scheme) {
			return nil, fmt.Errorf("spec.standards[%d].scheme is invalid", index)
		}
		if !versionPattern.MatchString(candidate.Version) {
			return nil, fmt.Errorf("spec.standards[%d].version is invalid", index)
		}
		key := candidate.Scheme + "\x00" + candidate.Version
		if _, duplicate := seen[key]; duplicate {
			return nil, fmt.Errorf("spec.standards contains duplicate %s@%s", candidate.Scheme, candidate.Version)
		}
		seen[key] = struct{}{}
		result[index] = AuditStandardRef{Scheme: candidate.Scheme, Version: candidate.Version}
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Scheme == result[j].Scheme {
			return result[i].Version < result[j].Version
		}
		return result[i].Scheme < result[j].Scheme
	})
	return result, nil
}

func resolveAuditProfileInputs(
	source *map[string]auditProfileInputSource,
) (map[string]AuditProfileInput, error) {
	if source == nil || len(*source) == 0 {
		return nil, fmt.Errorf("spec.inputs must be a non-empty mapping")
	}
	if len(*source) > MaxAuditProfileInputs {
		return nil, fmt.Errorf("spec.inputs may contain at most %d entries", MaxAuditProfileInputs)
	}
	result := make(map[string]AuditProfileInput, len(*source))
	for _, name := range sortedMapKeys(*source) {
		candidate := (*source)[name]
		if err := validateAuditMapKey("spec.inputs slot", name); err != nil {
			return nil, err
		}
		if candidate.Required == nil {
			return nil, fmt.Errorf("spec.inputs.%s.required is required", name)
		}
		mediaTypes, err := validateMediaTypes("spec.inputs."+name, candidate.MediaTypes)
		if err != nil {
			return nil, err
		}
		sort.Strings(mediaTypes)
		result[name] = AuditProfileInput{Required: *candidate.Required, MediaTypes: mediaTypes}
	}
	return result, nil
}

func (l *loader) resolveAuditWorkflowBindings(
	source *map[string]auditWorkflowBindingSource,
	profileInputs map[string]AuditProfileInput,
) (map[string]ResolvedAuditWorkflowBinding, error) {
	if source == nil || len(*source) == 0 {
		return nil, fmt.Errorf("spec.workflows must be a non-empty mapping")
	}
	if len(*source) > MaxAuditProfileWorkflows {
		return nil, fmt.Errorf("spec.workflows may contain at most %d entries", MaxAuditProfileWorkflows)
	}
	result := make(map[string]ResolvedAuditWorkflowBinding, len(*source))
	for _, role := range sortedMapKeys(*source) {
		candidate := (*source)[role]
		if err := validateAuditMapKey("spec.workflows role", role); err != nil {
			return nil, err
		}
		selector, err := ParseSelector(candidate.Ref)
		if err != nil {
			return nil, fmt.Errorf("spec.workflows.%s.ref: %w", role, err)
		}
		workflow, ok := l.workflows[selector.String()]
		if !ok {
			return nil, fmt.Errorf("spec.workflows.%s.ref selects unknown Workflow %q", role, selector)
		}
		inputs, err := resolveAuditWorkflowInputs(role, candidate.Inputs, workflow, profileInputs)
		if err != nil {
			return nil, err
		}
		parameters, err := resolveAuditWorkflowParameters(role, candidate.Parameters, workflow)
		if err != nil {
			return nil, err
		}
		outputs, err := resolveAuditWorkflowOutputs(role, candidate.Outputs, workflow)
		if err != nil {
			return nil, err
		}
		result[role] = ResolvedAuditWorkflowBinding{
			Workflow: cloneWorkflow(workflow), Inputs: inputs,
			Parameters: parameters, Outputs: outputs,
		}
	}
	return result, nil
}

func resolveAuditWorkflowInputs(
	role string,
	source *map[string]auditWorkflowInputMappingSource,
	workflow ResolvedWorkflow,
	profileInputs map[string]AuditProfileInput,
) (map[string]AuditWorkflowInputMapping, error) {
	field := "spec.workflows." + role + ".inputs"
	if source == nil {
		return nil, fmt.Errorf("%s is required (use {} for none)", field)
	}
	result := make(map[string]AuditWorkflowInputMapping, len(*source))
	for _, slotName := range sortedMapKeys(*source) {
		candidate := (*source)[slotName]
		slot, exists := workflow.Inputs[slotName]
		if !exists {
			return nil, fmt.Errorf("%s contains unknown Workflow input %q", field, slotName)
		}
		mapping, err := resolveAuditWorkflowInputMapping(field+"."+slotName, candidate, slot, profileInputs)
		if err != nil {
			return nil, err
		}
		result[slotName] = mapping
	}
	for _, slotName := range sortedMapKeys(workflow.Inputs) {
		if workflow.Inputs[slotName].Required {
			if _, exists := result[slotName]; !exists {
				return nil, fmt.Errorf("%s is missing required Workflow input %q", field, slotName)
			}
		}
	}
	return result, nil
}

func resolveAuditWorkflowInputMapping(
	field string,
	source auditWorkflowInputMappingSource,
	slot ArtifactSlot,
	profileInputs map[string]AuditProfileInput,
) (AuditWorkflowInputMapping, error) {
	mapping := AuditWorkflowInputMapping{
		Source: AuditWorkflowInputSource(source.Source), Name: source.Name, Role: source.Role,
	}
	switch mapping.Source {
	case AuditInputFromAudit:
		if mapping.Role != "" || validateAuditMapKey(field+".name", mapping.Name) != nil {
			return AuditWorkflowInputMapping{}, fmt.Errorf("%s audit-input requires name and forbids role", field)
		}
		input, exists := profileInputs[mapping.Name]
		if !exists {
			return AuditWorkflowInputMapping{}, fmt.Errorf("%s names unknown Audit input %q", field, mapping.Name)
		}
		if slot.Required && !input.Required {
			return AuditWorkflowInputMapping{}, fmt.Errorf("%s maps a required Workflow input from an optional Audit input", field)
		}
		if !mediaTypesIntersect(input.MediaTypes, slot.MediaTypes) {
			return AuditWorkflowInputMapping{}, fmt.Errorf("%s media types are incompatible with Workflow input", field)
		}
	case AuditInputFromItemPackage:
		if mapping.Name != "" || mapping.Role != "" {
			return AuditWorkflowInputMapping{}, fmt.Errorf("%s item-package forbids name and role", field)
		}
		if !mediaTypesIntersect([]string{auditTaskPackageMediaType}, slot.MediaTypes) {
			return AuditWorkflowInputMapping{}, fmt.Errorf("%s Workflow input does not accept %s", field, auditTaskPackageMediaType)
		}
	case AuditInputFromRetainedOutput:
		if validateAuditMapKey(field+".role", mapping.Role) != nil ||
			validateAuditMapKey(field+".name", mapping.Name) != nil {
			return AuditWorkflowInputMapping{}, fmt.Errorf("%s retained-output requires role and name", field)
		}
	default:
		return AuditWorkflowInputMapping{}, fmt.Errorf("%s.source is invalid", field)
	}
	return mapping, nil
}

func resolveAuditWorkflowParameters(
	role string,
	source *map[string]auditParameterMappingSource,
	workflow ResolvedWorkflow,
) (map[string]AuditWorkflowParameterMapping, error) {
	field := "spec.workflows." + role + ".parameters"
	if source == nil {
		return nil, fmt.Errorf("%s is required (use {} for none)", field)
	}
	result := make(map[string]AuditWorkflowParameterMapping, len(*source))
	for _, parameterName := range sortedMapKeys(*source) {
		candidate := (*source)[parameterName]
		if _, exists := workflow.Parameters[parameterName]; !exists {
			return nil, fmt.Errorf("%s contains unknown Workflow parameter %q", field, parameterName)
		}
		mapping, err := resolveAuditParameterMapping(field+"."+parameterName, candidate)
		if err != nil {
			return nil, err
		}
		result[parameterName] = mapping
	}
	for _, parameterName := range sortedMapKeys(workflow.Parameters) {
		if workflow.Parameters[parameterName].Required {
			if _, exists := result[parameterName]; !exists {
				return nil, fmt.Errorf("%s is missing required Workflow parameter %q", field, parameterName)
			}
		}
	}
	return result, nil
}

func resolveAuditParameterMapping(
	field string, source auditParameterMappingSource,
) (AuditWorkflowParameterMapping, error) {
	result := AuditWorkflowParameterMapping{Source: AuditWorkflowParameterSource(source.Source), Name: source.Name}
	switch result.Source {
	case AuditParameterLiteral:
		if source.Value == nil || result.Name != "" || !utf8.ValidString(*source.Value) ||
			len([]byte(*source.Value)) > MaxAuditLiteralParamBytes {
			return AuditWorkflowParameterMapping{}, fmt.Errorf("%s literal requires a bounded UTF-8 value and forbids name", field)
		}
		result.Value = *source.Value
	case AuditParameterItemField:
		if source.Value != nil || !auditItemFields[result.Name] {
			return AuditWorkflowParameterMapping{}, fmt.Errorf("%s item-field name is invalid or value is present", field)
		}
	case AuditParameterScopeField:
		if source.Value != nil || !auditScopeFields[result.Name] {
			return AuditWorkflowParameterMapping{}, fmt.Errorf("%s scope-field name is invalid or value is present", field)
		}
	default:
		return AuditWorkflowParameterMapping{}, fmt.Errorf("%s.source is invalid", field)
	}
	return result, nil
}

var auditItemFields = map[string]bool{"itemKey": true, "subjectKey": true, "kind": true}
var auditScopeFields = map[string]bool{"objective": true, "target": true, "authorizationScope": true}

func resolveAuditWorkflowOutputs(
	role string, source *map[string]string, workflow ResolvedWorkflow,
) (map[string]string, error) {
	field := "spec.workflows." + role + ".outputs"
	if source == nil || len(*source) == 0 {
		return nil, fmt.Errorf("%s must be a non-empty mapping", field)
	}
	result := make(map[string]string, len(*source))
	seenWorkflowOutputs := make(map[string]struct{}, len(*source))
	for _, logicalName := range sortedMapKeys(*source) {
		workflowOutput := (*source)[logicalName]
		if err := validateAuditMapKey(field+" logical output", logicalName); err != nil {
			return nil, err
		}
		if err := validateAuditMapKey(field+" workflow output", workflowOutput); err != nil {
			return nil, err
		}
		slot, exists := workflow.Outputs[workflowOutput]
		if !exists {
			return nil, fmt.Errorf("%s.%s names unknown Workflow output %q", field, logicalName, workflowOutput)
		}
		if !slot.Required {
			return nil, fmt.Errorf("%s.%s must select a required Workflow output", field, logicalName)
		}
		if _, duplicate := seenWorkflowOutputs[workflowOutput]; duplicate {
			return nil, fmt.Errorf("%s selects Workflow output %q more than once", field, workflowOutput)
		}
		seenWorkflowOutputs[workflowOutput] = struct{}{}
		result[logicalName] = workflowOutput
	}
	return result, nil
}

func resolveAuditInventory(
	source *auditInventorySource,
	inputs map[string]AuditProfileInput,
	workflows map[string]ResolvedAuditWorkflowBinding,
) (AuditInventory, error) {
	if source == nil {
		return AuditInventory{}, fmt.Errorf("spec.inventory is required")
	}
	acceptedMediaTypes, exists := auditInventoryMediaTypes[source.Implementation]
	if !exists {
		return AuditInventory{}, fmt.Errorf("spec.inventory.implementation is unsupported")
	}
	if err := validateAuditMapKey("spec.inventory.sourceInput", source.SourceInput); err != nil {
		return AuditInventory{}, err
	}
	input, exists := inputs[source.SourceInput]
	if !exists {
		return AuditInventory{}, fmt.Errorf("spec.inventory.sourceInput names unknown Audit input %q", source.SourceInput)
	}
	if !input.Required {
		return AuditInventory{}, fmt.Errorf("spec.inventory.sourceInput must name a required Audit input")
	}
	if !mediaTypesIntersect(input.MediaTypes, acceptedMediaTypes) {
		return AuditInventory{}, fmt.Errorf("spec.inventory.sourceInput media types are incompatible with %s", source.Implementation)
	}
	if err := validateAuditMapKey("spec.inventory.itemWorkflowRole", source.ItemWorkflowRole); err != nil {
		return AuditInventory{}, err
	}
	if _, exists := workflows[source.ItemWorkflowRole]; !exists {
		return AuditInventory{}, fmt.Errorf("spec.inventory.itemWorkflowRole names unknown role %q", source.ItemWorkflowRole)
	}
	itemBinding := workflows[source.ItemWorkflowRole]
	hasItemContext := false
	for _, mapping := range itemBinding.Inputs {
		if mapping.Source == AuditInputFromItemPackage {
			hasItemContext = true
			break
		}
	}
	if !hasItemContext {
		for _, mapping := range itemBinding.Parameters {
			if mapping.Source == AuditParameterItemField {
				hasItemContext = true
				break
			}
		}
	}
	if !hasItemContext {
		return AuditInventory{}, fmt.Errorf("spec.inventory.itemWorkflowRole must receive item-package or item-field context")
	}
	return AuditInventory{
		Implementation: source.Implementation, SourceInput: source.SourceInput,
		ItemWorkflowRole: source.ItemWorkflowRole,
	}, nil
}

var auditInventoryMediaTypes = map[string][]string{
	"openapi-operations@1": {"application/json", "application/yaml", "application/zip"},
	"checklist@1":          {"application/json", "application/yaml", "application/zip"},
	"finding-candidates@1": {"application/json", "application/zip"},
}

func validateAuditModeInventory(mode AuditProfileMode, implementation string) error {
	valid := false
	switch mode {
	case AuditModeOperationTracing:
		valid = implementation == "openapi-operations@1"
	case AuditModeFindingVerification:
		valid = implementation == "finding-candidates@1"
	case AuditModeRiskAssessment, AuditModeRequirementsVerification, AuditModeCustomChecklist:
		valid = implementation == "checklist@1"
	}
	if !valid {
		return fmt.Errorf("spec.inventory.implementation %q is incompatible with mode %q", implementation, mode)
	}
	return nil
}

func validateRetainedOutputDependencies(
	workflows map[string]ResolvedAuditWorkflowBinding,
) error {
	dependencies := make(map[string][]string, len(workflows))
	for _, role := range sortedMapKeys(workflows) {
		binding := workflows[role]
		for _, inputName := range sortedMapKeys(binding.Inputs) {
			mapping := binding.Inputs[inputName]
			if mapping.Source != AuditInputFromRetainedOutput {
				continue
			}
			source, exists := workflows[mapping.Role]
			if !exists {
				return fmt.Errorf("spec.workflows.%s.inputs.%s names unknown retained-output role %q", role, inputName, mapping.Role)
			}
			workflowOutput, exists := source.Outputs[mapping.Name]
			if !exists {
				return fmt.Errorf("spec.workflows.%s.inputs.%s names unknown logical output %q on role %q", role, inputName, mapping.Name, mapping.Role)
			}
			if !mediaTypesIntersect(source.Workflow.Outputs[workflowOutput].MediaTypes, binding.Workflow.Inputs[inputName].MediaTypes) {
				return fmt.Errorf("spec.workflows.%s.inputs.%s retained output media types are incompatible", role, inputName)
			}
			dependencies[role] = append(dependencies[role], mapping.Role)
		}
	}
	state := make(map[string]uint8, len(workflows))
	var visit func(string) error
	visit = func(role string) error {
		switch state[role] {
		case 1:
			return fmt.Errorf("spec.workflows retained-output dependencies contain a cycle at role %q", role)
		case 2:
			return nil
		}
		state[role] = 1
		sort.Strings(dependencies[role])
		for _, dependency := range dependencies[role] {
			if err := visit(dependency); err != nil {
				return err
			}
		}
		state[role] = 2
		return nil
	}
	for _, role := range sortedMapKeys(workflows) {
		if err := visit(role); err != nil {
			return err
		}
	}
	return nil
}

func resolveAuditExecutionPolicy(source *auditExecutionPolicySource) (AuditExecutionPolicy, error) {
	if source == nil {
		return AuditExecutionPolicy{}, fmt.Errorf("spec.execution is required")
	}
	result := AuditExecutionPolicy{
		RoundMode: AuditRoundMode(source.RoundMode), MaxRounds: source.MaxRounds,
		BatchSize:        source.BatchSize,
		MaxItemsPerRound: source.MaxItemsPerRound, MaxItemsTotal: source.MaxItemsTotal,
		MaxSubmittedRuns:   source.MaxSubmittedRuns,
		MaxItemRunAttempts: source.MaxItemRunAttempts, DeadlineSeconds: source.DeadlineSeconds,
		MaxEvidenceBytes: source.MaxEvidenceBytes,
		IncompleteRound:  AuditIncompleteRoundPolicy(source.IncompleteRound),
	}
	if result.RoundMode != AuditRoundFixedBarrier {
		return AuditExecutionPolicy{}, fmt.Errorf("spec.execution.roundMode must be %q", AuditRoundFixedBarrier)
	}
	for _, bound := range []struct {
		name, label string
		value       int
		maximum     int
	}{
		{"maxRounds", "rounds", result.MaxRounds, MaxAuditRounds},
		{"batchSize", "items", result.BatchSize, MaxAuditBatchSize},
		{"maxItemsPerRound", "items", result.MaxItemsPerRound, MaxAuditItemsPerRound},
		{"maxItemsTotal", "items", result.MaxItemsTotal, MaxAuditItemsTotal},
		{"maxSubmittedRuns", "Runs", result.MaxSubmittedRuns, MaxAuditSubmittedRuns},
		{"maxItemRunAttempts", "attempts", result.MaxItemRunAttempts, MaxAuditItemRunAttempts},
		{"deadlineSeconds", "seconds", result.DeadlineSeconds, MaxAuditDeadlineSeconds},
	} {
		if bound.value <= 0 || bound.value > bound.maximum {
			return AuditExecutionPolicy{}, fmt.Errorf("spec.execution.%s must be between 1 and %d %s", bound.name, bound.maximum, bound.label)
		}
	}
	if result.MaxEvidenceBytes <= 0 || result.MaxEvidenceBytes > MaxAuditEvidenceBytes {
		return AuditExecutionPolicy{}, fmt.Errorf("spec.execution.maxEvidenceBytes must be between 1 and %d", MaxAuditEvidenceBytes)
	}
	if result.MaxItemsPerRound > result.MaxItemsTotal {
		return AuditExecutionPolicy{}, fmt.Errorf("spec.execution.maxItemsPerRound cannot exceed maxItemsTotal")
	}
	if result.BatchSize > result.MaxItemsPerRound {
		return AuditExecutionPolicy{}, fmt.Errorf("spec.execution.batchSize cannot exceed maxItemsPerRound")
	}
	minimumInitialRuns := (result.MaxItemsTotal + result.BatchSize - 1) / result.BatchSize
	if result.MaxSubmittedRuns < minimumInitialRuns {
		return AuditExecutionPolicy{}, fmt.Errorf(
			"spec.execution.maxSubmittedRuns cannot cover maxItemsTotal at batchSize %d",
			result.BatchSize,
		)
	}
	if result.IncompleteRound != AuditIncompleteAssessWithGaps && result.IncompleteRound != AuditIncompleteFail {
		return AuditExecutionPolicy{}, fmt.Errorf("spec.execution.incompleteRound is invalid")
	}
	return result, nil
}

func resolveAuditInteractionPolicy(source *auditInteractionPolicySource) (AuditInteractionPolicy, error) {
	if source == nil {
		return AuditInteractionPolicy{}, fmt.Errorf("spec.interaction is required")
	}
	result := AuditInteractionPolicy{
		ActiveChecks:        AuditActiveChecksPolicy(source.ActiveChecks),
		FindingConfirmation: AuditFindingConfirmationPolicy(source.FindingConfirmation),
		NotApplicable:       AuditNotApplicablePolicy(source.NotApplicable),
		ReportAcceptance:    AuditReportAcceptancePolicy(source.ReportAcceptance),
	}
	if result.ActiveChecks != AuditActiveChecksProhibited &&
		result.ActiveChecks != AuditActiveChecksAutomatic &&
		result.ActiveChecks != AuditActiveChecksApprovalRequired {
		return AuditInteractionPolicy{}, fmt.Errorf("spec.interaction.activeChecks is invalid")
	}
	if result.FindingConfirmation != AuditFindingDisabled &&
		result.FindingConfirmation != AuditFindingHumanRequired {
		return AuditInteractionPolicy{}, fmt.Errorf("spec.interaction.findingConfirmation is invalid")
	}
	if result.NotApplicable != AuditNotApplicableHumanRequired && result.NotApplicable != AuditNotApplicableProfileRule {
		return AuditInteractionPolicy{}, fmt.Errorf("spec.interaction.notApplicable is invalid")
	}
	if result.ReportAcceptance != AuditReportAutomatic && result.ReportAcceptance != AuditReportHumanRequired {
		return AuditInteractionPolicy{}, fmt.Errorf("spec.interaction.reportAcceptance is invalid")
	}
	return result, nil
}

func auditProfileDigest(selector Selector, profile ResolvedAuditProfile) (string, error) {
	workflows := make(map[string]any, len(profile.Workflows))
	for role, binding := range profile.Workflows {
		workflows[role] = map[string]any{
			"workflow": binding.Workflow, "inputs": binding.Inputs,
			"parameters": binding.Parameters, "outputs": binding.Outputs,
		}
	}
	return digestJCS(map[string]any{
		"apiVersion": contracts.APIVersion,
		"kind":       auditProfileKind,
		"metadata":   map[string]any{"name": selector.ID, "version": selector.Version},
		"spec": map[string]any{
			"mode": profile.Mode, "standards": profile.Standards,
			"inputs": profile.Inputs, "inventory": profile.Inventory,
			"workflows": workflows, "execution": profile.Execution,
			"interaction": profile.Interaction,
		},
	})
}

func cloneAuditProfile(source ResolvedAuditProfile) ResolvedAuditProfile {
	result := source
	result.Standards = append([]AuditStandardRef{}, source.Standards...)
	result.Inputs = make(map[string]AuditProfileInput, len(source.Inputs))
	for name, input := range source.Inputs {
		input.MediaTypes = append([]string{}, input.MediaTypes...)
		result.Inputs[name] = input
	}
	result.Workflows = make(map[string]ResolvedAuditWorkflowBinding, len(source.Workflows))
	for role, binding := range source.Workflows {
		cloned := binding
		cloned.Workflow = cloneWorkflow(binding.Workflow)
		cloned.Inputs = cloneMap(binding.Inputs)
		cloned.Parameters = cloneMap(binding.Parameters)
		cloned.Outputs = cloneMap(binding.Outputs)
		result.Workflows[role] = cloned
	}
	return result
}

func validateAuditMapKey(field, value string) error {
	if !utf8.ValidString(value) || len([]byte(value)) > 128 {
		return fmt.Errorf("%s must be valid UTF-8 and at most 128 bytes", field)
	}
	return validateMapKey(field, value)
}

func sortedMapKeys[V any](source map[string]V) []string {
	result := make([]string, 0, len(source))
	for key := range source {
		result = append(result, key)
	}
	sort.Strings(result)
	return result
}
