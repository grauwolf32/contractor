package contracts

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

const (
	WorkerStateSchemaVersion    = 2
	MaxAgentStateSnapshotBytes  = 4 * 1024 * 1024
	maxStateWorkspacePaths      = 10_000
	maxStateWorkspacePathBytes  = 2 * 1024 * 1024
	maxStateAllocationCounters  = 10_000
	maxStateInvocationToolNames = 256
)

var stateMetricNamePattern = regexp.MustCompile(`^[a-z0-9_]+(?:\.[a-z0-9_]+)*$`)

// AgentStateSnapshot is the complete bounded Contractor-owned State exported
// by one live Worker. It deliberately contains no allocation identity.
type AgentStateSnapshot struct {
	APIVersion string                `json:"apiVersion"`
	State      ContractorWorkerState `json:"state"`
}

type ContractorWorkerState struct {
	SchemaVersion           int                    `json:"schemaVersion"`
	StateRevision           uint64                 `json:"stateRevision"`
	Metrics                 WorkerAllocationState  `json:"metrics"`
	CurrentInvocation       *WorkerInvocationState `json:"currentInvocation"`
	LastCompletedInvocation *WorkerInvocationState `json:"lastCompletedInvocation"`
	currentPresent          bool
	completedPresent        bool
}

type WorkerAllocationState struct {
	Counters     map[string]uint64     `json:"counters"`
	ToolCalls    []WorkerStateToolCall `json:"toolCalls"`
	Errors       []ExecutionError      `json:"errors"`
	FinalOutcome *string               `json:"finalOutcome"`
	Truncated    bool                  `json:"truncated"`
	WorkerBudget *WorkerStateBudget    `json:"workerBudget,omitempty"`
	finalPresent bool
}

type WorkerStateToolCall struct {
	CallID             string          `json:"callId"`
	Tool               string          `json:"tool"`
	Arguments          map[string]any  `json:"arguments,omitempty"`
	ArgumentsTruncated bool            `json:"argumentsTruncated"`
	Outcome            ToolCallOutcome `json:"outcome"`
	DurationMS         *uint64         `json:"durationMs,omitempty"`
	ResultSizeBytes    *uint64         `json:"resultSizeBytes,omitempty"`
	Error              *ExecutionError `json:"error,omitempty"`
}

type WorkerStateBudget struct {
	MaxModelCalls         uint64  `json:"maxModelCalls"`
	MaxToolCalls          uint64  `json:"maxToolCalls"`
	MaxTotalTokens        uint64  `json:"maxTotalTokens"`
	ObservedModelCalls    uint64  `json:"observedModelCalls"`
	ObservedToolCalls     uint64  `json:"observedToolCalls"`
	ObservedTotalTokens   uint64  `json:"observedTotalTokens"`
	TokenUsageUnavailable uint64  `json:"tokenUsageUnavailable"`
	Exhausted             *string `json:"exhausted,omitempty"`
}

type WorkerInvocationState struct {
	InvocationID string                      `json:"invocationId"`
	SubtaskID    string                      `json:"subtaskId"`
	Phase        string                      `json:"phase"`
	Metrics      WorkerStateInvocationMetric `json:"metrics"`
	Summarizer   WorkerStateSummarizer       `json:"summarizer"`
	Workspace    *WorkerStateWorkspace       `json:"workspace"`
	workspaceSet bool
}

type WorkerStateInvocationMetric struct {
	ModelCalls            uint64                                     `json:"modelCalls"`
	ModelErrors           uint64                                     `json:"modelErrors"`
	InputTokens           uint64                                     `json:"inputTokens"`
	OutputTokens          uint64                                     `json:"outputTokens"`
	TotalTokens           uint64                                     `json:"totalTokens"`
	CachedInputTokens     uint64                                     `json:"cachedInputTokens"`
	TokenUsageUnavailable uint64                                     `json:"tokenUsageUnavailable"`
	LatestPromptTokens    *uint64                                    `json:"latestPromptTokens"`
	ToolCalls             uint64                                     `json:"toolCalls"`
	ToolErrors            uint64                                     `json:"toolErrors"`
	Tools                 map[string]WorkerStateInvocationToolMetric `json:"tools"`
	Truncated             bool                                       `json:"truncated"`
}

type WorkerStateSummarizer struct {
	Phase                 string  `json:"phase"`
	RequestStateRevision  *uint64 `json:"requestStateRevision,omitempty"`
	ModelCalls            uint64  `json:"modelCalls"`
	InputTokens           uint64  `json:"inputTokens"`
	OutputTokens          uint64  `json:"outputTokens"`
	TotalTokens           uint64  `json:"totalTokens"`
	TokenUsageUnavailable uint64  `json:"tokenUsageUnavailable"`
	FailureCode           *string `json:"failureCode,omitempty"`
}

type WorkerStateInvocationToolMetric struct {
	Calls    uint64 `json:"calls"`
	Failures uint64 `json:"failures"`
}

type WorkerStateWorkspace struct {
	WorkspaceDigest string                            `json:"workspaceDigest"`
	ScopePaths      []string                          `json:"scopePaths"`
	ScopeComplete   bool                              `json:"scopeComplete"`
	Interactions    []WorkerStateWorkspaceInteraction `json:"interactions"`
	DetailComplete  bool                              `json:"detailComplete"`
}

type WorkerStateWorkspaceInteraction struct {
	Path           string `json:"path"`
	FirstOrdinal   uint64 `json:"firstOrdinal"`
	LastOrdinal    uint64 `json:"lastOrdinal"`
	DiscoveryCalls uint64 `json:"discoveryCalls"`
	ReadCalls      uint64 `json:"readCalls"`
	MatchCalls     uint64 `json:"matchCalls"`
	MutationCalls  uint64 `json:"mutationCalls"`
}

func (s AgentStateSnapshot) Validate() error {
	if err := validateAPIVersion(s.APIVersion); err != nil {
		return err
	}
	if err := s.State.validate(); err != nil {
		return err
	}
	encoded, err := marshalAgentStateJSON(s)
	if err != nil || len(encoded) > MaxAgentStateSnapshotBytes {
		return invalidf("AgentStateSnapshot exceeds its bounded contract")
	}
	return nil
}

func (s ContractorWorkerState) validate() error {
	if s.SchemaVersion != WorkerStateSchemaVersion || s.StateRevision == 0 {
		return invalidf("Worker State schema version or revision is invalid")
	}
	if !s.currentPresent || !s.completedPresent {
		return invalidf("Worker State invocation slots are required")
	}
	if err := s.Metrics.validate(); err != nil {
		return err
	}
	if s.CurrentInvocation != nil {
		if err := s.CurrentInvocation.validate(); err != nil {
			return err
		}
		if s.CurrentInvocation.Phase != "running" {
			return invalidf("current Worker invocation must be running")
		}
	}
	if s.LastCompletedInvocation != nil {
		if err := s.LastCompletedInvocation.validate(); err != nil {
			return err
		}
		if s.LastCompletedInvocation.Phase == "running" {
			return invalidf("last completed Worker invocation must be terminal")
		}
	}
	return nil
}

func (s WorkerStateSummarizer) validate() error {
	requested := s.Phase == "requested" || s.Phase == "succeeded" || s.Phase == "failed"
	if s.Phase != "disabled" && s.Phase != "not_requested" && !requested {
		return invalidf("Worker State summarizer phase is invalid")
	}
	if requested != (s.RequestStateRevision != nil) ||
		(s.RequestStateRevision != nil && *s.RequestStateRevision == 0) ||
		s.ModelCalls > 1 || s.TokenUsageUnavailable > s.ModelCalls {
		return invalidf("Worker State summarizer request or usage is inconsistent")
	}
	if (s.Phase == "disabled" || s.Phase == "not_requested" || s.Phase == "requested") &&
		(s.ModelCalls != 0 || s.InputTokens != 0 || s.OutputTokens != 0 ||
			s.TotalTokens != 0 || s.TokenUsageUnavailable != 0) {
		return invalidf("inactive Worker State summarizer has usage")
	}
	if s.Phase == "succeeded" && s.ModelCalls != 1 {
		return invalidf("successful Worker State summarizer requires one model call")
	}
	if (s.Phase == "failed") != (s.FailureCode != nil) {
		return invalidf("Worker State summarizer failure code is inconsistent")
	}
	if s.FailureCode != nil && !validWorkerFailureCode(*s.FailureCode) {
		return invalidf("Worker State summarizer failure code is invalid")
	}
	return nil
}

func (s WorkerAllocationState) validate() error {
	if s.Counters == nil || s.ToolCalls == nil || s.Errors == nil || !s.finalPresent {
		return invalidf("Worker allocation State collections and finalOutcome are required")
	}
	if len(s.Counters) > maxStateAllocationCounters || len(s.ToolCalls) > 1000 || len(s.Errors) > 100 {
		return invalidf("Worker allocation State exceeds its collection bounds")
	}
	for name := range s.Counters {
		if len(name) < 1 || len(name) > 128 || !stateMetricNamePattern.MatchString(name) {
			return invalidf("Worker allocation State counter name is invalid")
		}
	}
	if s.FinalOutcome != nil && (len(*s.FinalOutcome) > 64 || !idPattern.MatchString(*s.FinalOutcome)) {
		return invalidf("Worker State final outcome is invalid")
	}
	for index := range s.ToolCalls {
		if err := validateStateToolCall(s.ToolCalls[index]); err != nil {
			return fmt.Errorf("Worker State tool call %d: %w", index, err)
		}
	}
	for _, item := range s.Errors {
		if err := item.Validate(); err != nil {
			return err
		}
	}
	if s.WorkerBudget != nil {
		if err := s.WorkerBudget.validate(); err != nil {
			return err
		}
	}
	return nil
}

func (b WorkerStateBudget) validate() error {
	if b.MaxModelCalls == 0 || b.MaxModelCalls > MaxWorkerModelCalls ||
		b.MaxToolCalls == 0 || b.MaxToolCalls > MaxWorkerToolCalls ||
		b.MaxTotalTokens == 0 || b.MaxTotalTokens > MaxWorkerTotalTokens ||
		b.ObservedModelCalls > b.MaxModelCalls || b.ObservedToolCalls > b.MaxToolCalls {
		return invalidf("Worker State budget is invalid")
	}
	if b.Exhausted == nil {
		return nil
	}
	switch *b.Exhausted {
	case "model_calls":
		if b.ObservedModelCalls != b.MaxModelCalls {
			return invalidf("Worker State model-call exhaustion is inconsistent")
		}
	case "tool_calls":
		if b.ObservedToolCalls != b.MaxToolCalls {
			return invalidf("Worker State tool-call exhaustion is inconsistent")
		}
	case "total_tokens":
		if b.ObservedTotalTokens < b.MaxTotalTokens {
			return invalidf("Worker State token exhaustion is inconsistent")
		}
	default:
		return invalidf("Worker State budget exhaustion dimension is invalid")
	}
	return nil
}

func validateStateToolCall(call WorkerStateToolCall) error {
	if err := validateOpaqueID("Worker State tool call ID", call.CallID); err != nil {
		return err
	}
	if err := validateOpaqueID("Worker State tool name", call.Tool); err != nil {
		return err
	}
	if call.Arguments == nil && call.ArgumentsTruncated {
		return invalidf("absent Worker State arguments cannot be marked truncated")
	}
	if call.Arguments != nil {
		encoded, err := marshalAgentStateJSON(call.Arguments)
		if err != nil || len(encoded) > 4096 {
			return invalidf("Worker State tool arguments exceed their bound")
		}
	}
	if call.Outcome != ToolCallSucceeded && call.Outcome != ToolCallFailed {
		return invalidf("Worker State tool outcome is invalid")
	}
	if (call.Outcome == ToolCallSucceeded) == (call.Error != nil) {
		return invalidf("Worker State tool outcome and error are inconsistent")
	}
	if call.Error != nil {
		return call.Error.Validate()
	}
	return nil
}

func (s WorkerInvocationState) validate() error {
	if !validBoundedOpaqueWorkerID(s.InvocationID) {
		return invalidf("Worker State invocationId is invalid")
	}
	if err := validateWorkerSubtaskID(s.SubtaskID); err != nil {
		return err
	}
	if s.Phase != "running" && s.Phase != "succeeded" && s.Phase != "failed" && s.Phase != "cancelled" {
		return invalidf("Worker State invocation phase is invalid")
	}
	if !s.workspaceSet {
		return invalidf("Worker State invocation workspace is required")
	}
	if err := s.Metrics.validate(); err != nil {
		return err
	}
	if err := s.Summarizer.validate(); err != nil {
		return err
	}
	if s.Phase != "running" && s.Summarizer.Phase == "requested" {
		return invalidf("terminal Worker State invocation has pending summarization")
	}
	if s.Workspace != nil {
		return s.Workspace.validate()
	}
	return nil
}

func (m WorkerStateInvocationMetric) validate() error {
	if m.Tools == nil || len(m.Tools) > maxStateInvocationToolNames ||
		m.ModelErrors > m.ModelCalls || m.ToolErrors > m.ToolCalls {
		return invalidf("Worker State invocation metrics are invalid")
	}
	var detailedCalls uint64
	var detailedFailures uint64
	for name, item := range m.Tools {
		if len(name) < 1 || len(name) > 64 || !idPattern.MatchString(name) || item.Failures > item.Calls {
			return invalidf("Worker State invocation tool metrics are invalid")
		}
		if ^uint64(0)-detailedCalls < item.Calls || ^uint64(0)-detailedFailures < item.Failures {
			return invalidf("Worker State invocation tool metrics overflow")
		}
		detailedCalls += item.Calls
		detailedFailures += item.Failures
	}
	if detailedCalls > m.ToolCalls || detailedFailures > m.ToolErrors ||
		(!m.Truncated && (detailedCalls != m.ToolCalls || detailedFailures != m.ToolErrors)) {
		return invalidf("Worker State invocation tool detail is inconsistent")
	}
	return nil
}

func (s WorkerStateWorkspace) validate() error {
	if err := validateDigest("Worker State workspaceDigest", s.WorkspaceDigest); err != nil {
		return err
	}
	if s.ScopePaths == nil || s.Interactions == nil || len(s.ScopePaths) > maxStateWorkspacePaths ||
		len(s.Interactions) > maxStateWorkspacePaths || !sort.StringsAreSorted(s.ScopePaths) {
		return invalidf("Worker State workspace collections are invalid")
	}
	for index, path := range s.ScopePaths {
		if err := validateStateWorkspacePath(path); err != nil {
			return err
		}
		if index > 0 && path == s.ScopePaths[index-1] {
			return invalidf("Worker State workspace scope paths must be unique")
		}
	}
	if encoded, err := marshalAgentStateJSON(s.ScopePaths); err != nil || len(encoded) > maxStateWorkspacePathBytes {
		return invalidf("Worker State workspace scope paths exceed their bound")
	}
	interactionPaths := make([]string, 0, len(s.Interactions))
	seen := make(map[string]struct{}, len(s.Interactions))
	var previous *WorkerStateWorkspaceInteraction
	for index := range s.Interactions {
		item := &s.Interactions[index]
		if err := item.validate(); err != nil {
			return err
		}
		if _, exists := seen[item.Path]; exists {
			return invalidf("Worker State workspace interaction paths must be unique")
		}
		seen[item.Path] = struct{}{}
		if previous != nil && (item.FirstOrdinal < previous.FirstOrdinal ||
			(item.FirstOrdinal == previous.FirstOrdinal && item.Path < previous.Path)) {
			return invalidf("Worker State workspace interactions are not ordered")
		}
		previous = item
		interactionPaths = append(interactionPaths, item.Path)
	}
	if encoded, err := marshalAgentStateJSON(interactionPaths); err != nil || len(encoded) > maxStateWorkspacePathBytes {
		return invalidf("Worker State workspace interaction paths exceed their bound")
	}
	return nil
}

func (i WorkerStateWorkspaceInteraction) validate() error {
	if err := validateStateWorkspacePath(i.Path); err != nil {
		return err
	}
	if i.FirstOrdinal == 0 || i.LastOrdinal < i.FirstOrdinal ||
		(i.DiscoveryCalls == 0 && i.ReadCalls == 0 && i.MatchCalls == 0 && i.MutationCalls == 0) {
		return invalidf("Worker State workspace interaction is inconsistent")
	}
	return nil
}

func validateStateWorkspacePath(value string) error {
	if value == "" || !utf8.ValidString(value) || value != norm.NFC.String(value) ||
		len([]byte(value)) > 4096 || strings.HasPrefix(value, "/") ||
		strings.ContainsAny(value, "\\\x00") || strings.Contains(value, "://") {
		return invalidf("Worker State workspace path is invalid")
	}
	parts := strings.Split(value, "/")
	if len(parts) > 128 {
		return invalidf("Worker State workspace path is invalid")
	}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." {
			return invalidf("Worker State workspace path is invalid")
		}
		for _, character := range part {
			if character < 0x20 || character == 0x7f {
				return invalidf("Worker State workspace path is invalid")
			}
		}
	}
	first := parts[0]
	if len(first) >= 2 && ((first[0] >= 'A' && first[0] <= 'Z') ||
		(first[0] >= 'a' && first[0] <= 'z')) && first[1] == ':' {
		return invalidf("Worker State workspace path is invalid")
	}
	return nil
}

func (s *ContractorWorkerState) UnmarshalJSON(data []byte) error {
	type wireState struct {
		SchemaVersion           *int                   `json:"schemaVersion"`
		StateRevision           *uint64                `json:"stateRevision"`
		Metrics                 *WorkerAllocationState `json:"metrics"`
		CurrentInvocation       json.RawMessage        `json:"currentInvocation"`
		LastCompletedInvocation json.RawMessage        `json:"lastCompletedInvocation"`
	}
	var wire wireState
	if err := decodeStrictAgentStateJSON(data, &wire); err != nil {
		return err
	}
	if wire.SchemaVersion == nil || wire.StateRevision == nil || wire.Metrics == nil ||
		wire.CurrentInvocation == nil || wire.LastCompletedInvocation == nil {
		return invalidf("Worker State required fields are missing")
	}
	s.SchemaVersion = *wire.SchemaVersion
	s.StateRevision = *wire.StateRevision
	s.Metrics = *wire.Metrics
	s.currentPresent = true
	s.completedPresent = true
	if !bytes.Equal(wire.CurrentInvocation, []byte("null")) {
		var current WorkerInvocationState
		if err := decodeStrictAgentStateJSON(wire.CurrentInvocation, &current); err != nil {
			return err
		}
		s.CurrentInvocation = &current
	}
	if !bytes.Equal(wire.LastCompletedInvocation, []byte("null")) {
		var completed WorkerInvocationState
		if err := decodeStrictAgentStateJSON(wire.LastCompletedInvocation, &completed); err != nil {
			return err
		}
		s.LastCompletedInvocation = &completed
	}
	return nil
}

func (s *WorkerAllocationState) UnmarshalJSON(data []byte) error {
	type wireState struct {
		Counters     map[string]uint64     `json:"counters"`
		ToolCalls    []WorkerStateToolCall `json:"toolCalls"`
		Errors       []ExecutionError      `json:"errors"`
		FinalOutcome json.RawMessage       `json:"finalOutcome"`
		Truncated    *bool                 `json:"truncated"`
		WorkerBudget *WorkerStateBudget    `json:"workerBudget,omitempty"`
	}
	var wire wireState
	if err := decodeStrictAgentStateJSON(data, &wire); err != nil {
		return err
	}
	if wire.Counters == nil || wire.ToolCalls == nil || wire.Errors == nil ||
		wire.FinalOutcome == nil || wire.Truncated == nil {
		return invalidf("Worker allocation State required fields are missing")
	}
	s.Counters = wire.Counters
	s.ToolCalls = wire.ToolCalls
	s.Errors = wire.Errors
	s.Truncated = *wire.Truncated
	s.WorkerBudget = wire.WorkerBudget
	s.finalPresent = true
	if !bytes.Equal(wire.FinalOutcome, []byte("null")) {
		var outcome string
		if err := json.Unmarshal(wire.FinalOutcome, &outcome); err != nil {
			return err
		}
		s.FinalOutcome = &outcome
	}
	return nil
}

func (s *WorkerStateToolCall) UnmarshalJSON(data []byte) error {
	if err := requireAgentStateFields(
		data, "callId", "tool", "argumentsTruncated", "outcome",
	); err != nil {
		return err
	}
	type plain WorkerStateToolCall
	var wire plain
	if err := decodeStrictAgentStateJSON(data, &wire); err != nil {
		return err
	}
	*s = WorkerStateToolCall(wire)
	return nil
}

func (s *WorkerStateBudget) UnmarshalJSON(data []byte) error {
	if err := requireAgentStateFields(
		data,
		"maxModelCalls", "maxToolCalls", "maxTotalTokens", "observedModelCalls",
		"observedToolCalls", "observedTotalTokens", "tokenUsageUnavailable",
	); err != nil {
		return err
	}
	type plain WorkerStateBudget
	var wire plain
	if err := decodeStrictAgentStateJSON(data, &wire); err != nil {
		return err
	}
	*s = WorkerStateBudget(wire)
	return nil
}

func (s *WorkerInvocationState) UnmarshalJSON(data []byte) error {
	type wireState struct {
		InvocationID string                      `json:"invocationId"`
		SubtaskID    string                      `json:"subtaskId"`
		Phase        string                      `json:"phase"`
		Metrics      WorkerStateInvocationMetric `json:"metrics"`
		Summarizer   *WorkerStateSummarizer      `json:"summarizer"`
		Workspace    json.RawMessage             `json:"workspace"`
	}
	var wire wireState
	if err := decodeStrictAgentStateJSON(data, &wire); err != nil {
		return err
	}
	if wire.Workspace == nil || wire.Summarizer == nil {
		return invalidf("Worker State invocation summarizer or workspace is missing")
	}
	s.InvocationID = wire.InvocationID
	s.SubtaskID = wire.SubtaskID
	s.Phase = wire.Phase
	s.Metrics = wire.Metrics
	s.Summarizer = *wire.Summarizer
	s.workspaceSet = true
	if !bytes.Equal(wire.Workspace, []byte("null")) {
		var workspace WorkerStateWorkspace
		if err := decodeStrictAgentStateJSON(wire.Workspace, &workspace); err != nil {
			return err
		}
		s.Workspace = &workspace
	}
	return nil
}

func (s *WorkerStateInvocationMetric) UnmarshalJSON(data []byte) error {
	if err := requireAgentStateFields(
		data,
		"modelCalls", "modelErrors", "inputTokens", "outputTokens", "totalTokens",
		"cachedInputTokens", "tokenUsageUnavailable", "latestPromptTokens", "toolCalls", "toolErrors", "tools",
		"truncated",
	); err != nil {
		return err
	}
	type plain WorkerStateInvocationMetric
	var wire plain
	if err := decodeStrictAgentStateJSON(data, &wire); err != nil {
		return err
	}
	*s = WorkerStateInvocationMetric(wire)
	return nil
}

func (s *WorkerStateSummarizer) UnmarshalJSON(data []byte) error {
	if err := requireAgentStateFields(
		data, "phase", "modelCalls", "inputTokens", "outputTokens", "totalTokens",
		"tokenUsageUnavailable",
	); err != nil {
		return err
	}
	type plain WorkerStateSummarizer
	var wire plain
	if err := decodeStrictAgentStateJSON(data, &wire); err != nil {
		return err
	}
	*s = WorkerStateSummarizer(wire)
	return nil
}

func (s *WorkerStateInvocationToolMetric) UnmarshalJSON(data []byte) error {
	if err := requireAgentStateFields(data, "calls", "failures"); err != nil {
		return err
	}
	type plain WorkerStateInvocationToolMetric
	var wire plain
	if err := decodeStrictAgentStateJSON(data, &wire); err != nil {
		return err
	}
	*s = WorkerStateInvocationToolMetric(wire)
	return nil
}

func (s *WorkerStateWorkspace) UnmarshalJSON(data []byte) error {
	if err := requireAgentStateFields(
		data, "workspaceDigest", "scopePaths", "scopeComplete", "interactions", "detailComplete",
	); err != nil {
		return err
	}
	type plain WorkerStateWorkspace
	var wire plain
	if err := decodeStrictAgentStateJSON(data, &wire); err != nil {
		return err
	}
	*s = WorkerStateWorkspace(wire)
	return nil
}

func (s *WorkerStateWorkspaceInteraction) UnmarshalJSON(data []byte) error {
	if err := requireAgentStateFields(
		data,
		"path", "firstOrdinal", "lastOrdinal", "discoveryCalls", "readCalls", "matchCalls",
		"mutationCalls",
	); err != nil {
		return err
	}
	type plain WorkerStateWorkspaceInteraction
	var wire plain
	if err := decodeStrictAgentStateJSON(data, &wire); err != nil {
		return err
	}
	*s = WorkerStateWorkspaceInteraction(wire)
	return nil
}

func decodeStrictAgentStateJSON(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("decode Agent State JSON: %w", err)
	}
	return ensureJSONEOF(decoder)
}

func requireAgentStateFields(data []byte, required ...string) error {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(data, &object); err != nil {
		return fmt.Errorf("decode Agent State object: %w", err)
	}
	if object == nil {
		return invalidf("Agent State value must be an object")
	}
	for _, field := range required {
		if _, present := object[field]; !present {
			return invalidf("Agent State field %s is required", field)
		}
	}
	return nil
}

func marshalAgentStateJSON(value any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buffer.Bytes(), []byte("\n")), nil
}
