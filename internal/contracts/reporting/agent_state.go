package reporting

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"

	"github.com/grauwolf32/contractor/internal/contracts"
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
	APIVersion string                `json:"apiVersion" strictjson:"required"`
	State      ContractorWorkerState `json:"state" strictjson:"required"`
}

type ContractorWorkerState struct {
	SchemaVersion           int                    `json:"schemaVersion" strictjson:"required"`
	StateRevision           uint64                 `json:"stateRevision" strictjson:"required"`
	Metrics                 WorkerAllocationState  `json:"metrics" strictjson:"required"`
	CurrentInvocation       *WorkerInvocationState `json:"currentInvocation" strictjson:"required,nullable"`
	LastCompletedInvocation *WorkerInvocationState `json:"lastCompletedInvocation" strictjson:"required,nullable"`
}

type WorkerAllocationState struct {
	Completion   *WorkerCompletionDiagnostics `json:"completion,omitempty"`
	Counters     map[string]uint64            `json:"counters" strictjson:"required"`
	ToolCalls    []WorkerStateToolCall        `json:"toolCalls" strictjson:"required"`
	Errors       []ExecutionError             `json:"errors" strictjson:"required"`
	FinalOutcome *string                      `json:"finalOutcome" strictjson:"required,nullable"`
	Truncated    bool                         `json:"truncated" strictjson:"required"`
	WorkerBudget *WorkerStateBudget           `json:"workerBudget,omitempty"`
}

type WorkerStateToolCall struct {
	CallID             string          `json:"callId" strictjson:"required"`
	Tool               string          `json:"tool" strictjson:"required"`
	Arguments          map[string]any  `json:"arguments,omitempty"`
	ArgumentsTruncated bool            `json:"argumentsTruncated" strictjson:"required"`
	Outcome            ToolCallOutcome `json:"outcome" strictjson:"required"`
	DurationMS         *uint64         `json:"durationMs,omitempty"`
	ResultSizeBytes    *uint64         `json:"resultSizeBytes,omitempty"`
	Error              *ExecutionError `json:"error,omitempty"`
}

type WorkerStateBudget struct {
	MaxModelCalls         uint64  `json:"maxModelCalls" strictjson:"required"`
	MaxToolCalls          uint64  `json:"maxToolCalls" strictjson:"required"`
	MaxTotalTokens        uint64  `json:"maxTotalTokens" strictjson:"required"`
	ObservedModelCalls    uint64  `json:"observedModelCalls" strictjson:"required"`
	ObservedToolCalls     uint64  `json:"observedToolCalls" strictjson:"required"`
	ObservedTotalTokens   uint64  `json:"observedTotalTokens" strictjson:"required"`
	TokenUsageUnavailable uint64  `json:"tokenUsageUnavailable" strictjson:"required"`
	Exhausted             *string `json:"exhausted,omitempty"`
}

type WorkerInvocationState struct {
	InvocationID string                      `json:"invocationId" strictjson:"required"`
	SubtaskID    string                      `json:"subtaskId" strictjson:"required"`
	Phase        string                      `json:"phase" strictjson:"required"`
	Metrics      WorkerStateInvocationMetric `json:"metrics" strictjson:"required"`
	Summarizer   WorkerStateSummarizer       `json:"summarizer" strictjson:"required"`
	Workspace    *WorkerStateWorkspace       `json:"workspace" strictjson:"required,nullable"`
}

type WorkerStateInvocationMetric struct {
	ModelCalls            uint64                                     `json:"modelCalls" strictjson:"required"`
	ModelErrors           uint64                                     `json:"modelErrors" strictjson:"required"`
	InputTokens           uint64                                     `json:"inputTokens" strictjson:"required"`
	OutputTokens          uint64                                     `json:"outputTokens" strictjson:"required"`
	TotalTokens           uint64                                     `json:"totalTokens" strictjson:"required"`
	CachedInputTokens     uint64                                     `json:"cachedInputTokens" strictjson:"required"`
	TokenUsageUnavailable uint64                                     `json:"tokenUsageUnavailable" strictjson:"required"`
	LatestPromptTokens    *uint64                                    `json:"latestPromptTokens" strictjson:"required,nullable"`
	ToolCalls             uint64                                     `json:"toolCalls" strictjson:"required"`
	ToolErrors            uint64                                     `json:"toolErrors" strictjson:"required"`
	Tools                 map[string]WorkerStateInvocationToolMetric `json:"tools" strictjson:"required"`
	Truncated             bool                                       `json:"truncated" strictjson:"required"`
}

type WorkerStateSummarizer struct {
	Phase                 string  `json:"phase" strictjson:"required"`
	RequestStateRevision  *uint64 `json:"requestStateRevision,omitempty"`
	ModelCalls            uint64  `json:"modelCalls" strictjson:"required"`
	InputTokens           uint64  `json:"inputTokens" strictjson:"required"`
	OutputTokens          uint64  `json:"outputTokens" strictjson:"required"`
	TotalTokens           uint64  `json:"totalTokens" strictjson:"required"`
	TokenUsageUnavailable uint64  `json:"tokenUsageUnavailable" strictjson:"required"`
	FailureCode           *string `json:"failureCode,omitempty"`
}

type WorkerStateInvocationToolMetric struct {
	Calls    uint64 `json:"calls" strictjson:"required"`
	Failures uint64 `json:"failures" strictjson:"required"`
}

type WorkerStateWorkspace struct {
	WorkspaceDigest string                            `json:"workspaceDigest" strictjson:"required"`
	ScopePaths      []string                          `json:"scopePaths" strictjson:"required"`
	ScopeComplete   bool                              `json:"scopeComplete" strictjson:"required"`
	Interactions    []WorkerStateWorkspaceInteraction `json:"interactions" strictjson:"required"`
	DetailComplete  bool                              `json:"detailComplete" strictjson:"required"`
}

type WorkerStateWorkspaceInteraction struct {
	Path           string `json:"path" strictjson:"required"`
	FirstOrdinal   uint64 `json:"firstOrdinal" strictjson:"required"`
	LastOrdinal    uint64 `json:"lastOrdinal" strictjson:"required"`
	DiscoveryCalls uint64 `json:"discoveryCalls" strictjson:"required"`
	ReadCalls      uint64 `json:"readCalls" strictjson:"required"`
	MatchCalls     uint64 `json:"matchCalls" strictjson:"required"`
	MutationCalls  uint64 `json:"mutationCalls" strictjson:"required"`
}

func (s AgentStateSnapshot) Validate() error {
	if err := contracts.ValidateAPIVersion(s.APIVersion); err != nil {
		return err
	}
	if err := s.State.validate(); err != nil {
		return err
	}
	encoded, err := marshalAgentStateJSON(s)
	if err != nil || len(encoded) > MaxAgentStateSnapshotBytes {
		return contracts.Invalidf("AgentStateSnapshot exceeds its bounded contract")
	}
	return nil
}

func (s ContractorWorkerState) validate() error {
	if s.SchemaVersion != WorkerStateSchemaVersion || s.StateRevision == 0 {
		return contracts.Invalidf("Worker State schema version or revision is invalid")
	}
	if err := s.Metrics.validate(); err != nil {
		return err
	}
	if s.CurrentInvocation != nil {
		if err := s.CurrentInvocation.validate(); err != nil {
			return err
		}
		if s.CurrentInvocation.Phase != "running" {
			return contracts.Invalidf("current Worker invocation must be running")
		}
	}
	if s.LastCompletedInvocation != nil {
		if err := s.LastCompletedInvocation.validate(); err != nil {
			return err
		}
		if s.LastCompletedInvocation.Phase == "running" {
			return contracts.Invalidf("last completed Worker invocation must be terminal")
		}
	}
	return nil
}

func (s WorkerStateSummarizer) validate() error {
	requested := s.Phase == "requested" || s.Phase == "succeeded" || s.Phase == "failed"
	if s.Phase != "disabled" && s.Phase != "not_requested" && !requested {
		return contracts.Invalidf("Worker State summarizer phase is invalid")
	}
	if requested != (s.RequestStateRevision != nil) ||
		(s.RequestStateRevision != nil && *s.RequestStateRevision == 0) ||
		s.ModelCalls > 1 || s.TokenUsageUnavailable > s.ModelCalls {
		return contracts.Invalidf("Worker State summarizer request or usage is inconsistent")
	}
	if (s.Phase == "disabled" || s.Phase == "not_requested" || s.Phase == "requested") &&
		(s.ModelCalls != 0 || s.InputTokens != 0 || s.OutputTokens != 0 ||
			s.TotalTokens != 0 || s.TokenUsageUnavailable != 0) {
		return contracts.Invalidf("inactive Worker State summarizer has usage")
	}
	if s.Phase == "succeeded" && s.ModelCalls != 1 {
		return contracts.Invalidf("successful Worker State summarizer requires one model call")
	}
	if (s.Phase == "failed") != (s.FailureCode != nil) {
		return contracts.Invalidf("Worker State summarizer failure code is inconsistent")
	}
	if s.FailureCode != nil && !contracts.ValidWorkerFailureCode(*s.FailureCode) {
		return contracts.Invalidf("Worker State summarizer failure code is invalid")
	}
	return nil
}

func (s WorkerAllocationState) validate() error {
	if s.Completion != nil {
		if err := s.Completion.Validate(); err != nil {
			return err
		}
	}
	if s.Counters == nil || s.ToolCalls == nil || s.Errors == nil {
		return contracts.Invalidf("Worker allocation State collections are required")
	}
	if len(s.Counters) > maxStateAllocationCounters || len(s.ToolCalls) > 1000 || len(s.Errors) > 100 {
		return contracts.Invalidf("Worker allocation State exceeds its collection bounds")
	}
	for name := range s.Counters {
		if len(name) < 1 || len(name) > 128 || !stateMetricNamePattern.MatchString(name) {
			return contracts.Invalidf("Worker allocation State counter name is invalid")
		}
	}
	if s.FinalOutcome != nil && (len(*s.FinalOutcome) > 64 || !contracts.ValidIdentifier(*s.FinalOutcome)) {
		return contracts.Invalidf("Worker State final outcome is invalid")
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
	if b.MaxModelCalls == 0 || b.MaxModelCalls > contracts.MaxWorkerModelCalls ||
		b.MaxToolCalls == 0 || b.MaxToolCalls > contracts.MaxWorkerToolCalls ||
		b.MaxTotalTokens == 0 || b.MaxTotalTokens > contracts.MaxWorkerTotalTokens ||
		b.ObservedModelCalls > b.MaxModelCalls || b.ObservedToolCalls > b.MaxToolCalls {
		return contracts.Invalidf("Worker State budget is invalid")
	}
	if b.Exhausted == nil {
		return nil
	}
	switch *b.Exhausted {
	case "model_calls":
		if b.ObservedModelCalls != b.MaxModelCalls {
			return contracts.Invalidf("Worker State model-call exhaustion is inconsistent")
		}
	case "tool_calls":
		if b.ObservedToolCalls != b.MaxToolCalls {
			return contracts.Invalidf("Worker State tool-call exhaustion is inconsistent")
		}
	case "total_tokens":
		if b.ObservedTotalTokens < b.MaxTotalTokens {
			return contracts.Invalidf("Worker State token exhaustion is inconsistent")
		}
	default:
		return contracts.Invalidf("Worker State budget exhaustion dimension is invalid")
	}
	return nil
}

func validateStateToolCall(call WorkerStateToolCall) error {
	if err := contracts.ValidateOpaqueID("Worker State tool call ID", call.CallID); err != nil {
		return err
	}
	if err := contracts.ValidateOpaqueID("Worker State tool name", call.Tool); err != nil {
		return err
	}
	if call.Arguments == nil && call.ArgumentsTruncated {
		return contracts.Invalidf("absent Worker State arguments cannot be marked truncated")
	}
	if call.Arguments != nil {
		encoded, err := marshalAgentStateJSON(call.Arguments)
		if err != nil || len(encoded) > 4096 {
			return contracts.Invalidf("Worker State tool arguments exceed their bound")
		}
	}
	if call.Outcome != ToolCallSucceeded && call.Outcome != ToolCallFailed {
		return contracts.Invalidf("Worker State tool outcome is invalid")
	}
	if (call.Outcome == ToolCallSucceeded) == (call.Error != nil) {
		return contracts.Invalidf("Worker State tool outcome and error are inconsistent")
	}
	if call.Error != nil {
		return call.Error.Validate()
	}
	return nil
}

func (s WorkerInvocationState) validate() error {
	if !contracts.ValidBoundedOpaqueWorkerID(s.InvocationID) {
		return contracts.Invalidf("Worker State invocationId is invalid")
	}
	if err := contracts.ValidateWorkerSubtaskID(s.SubtaskID); err != nil {
		return err
	}
	if s.Phase != "running" && s.Phase != "succeeded" && s.Phase != "failed" && s.Phase != "cancelled" {
		return contracts.Invalidf("Worker State invocation phase is invalid")
	}
	if err := s.Metrics.validate(); err != nil {
		return err
	}
	if err := s.Summarizer.validate(); err != nil {
		return err
	}
	if s.Phase != "running" && s.Summarizer.Phase == "requested" {
		return contracts.Invalidf("terminal Worker State invocation has pending summarization")
	}
	if s.Workspace != nil {
		return s.Workspace.validate()
	}
	return nil
}

func (m WorkerStateInvocationMetric) validate() error {
	if m.Tools == nil || len(m.Tools) > maxStateInvocationToolNames ||
		m.ModelErrors > m.ModelCalls || m.ToolErrors > m.ToolCalls {
		return contracts.Invalidf("Worker State invocation metrics are invalid")
	}
	var detailedCalls uint64
	var detailedFailures uint64
	for name, item := range m.Tools {
		if len(name) < 1 || len(name) > 64 || !contracts.ValidIdentifier(name) || item.Failures > item.Calls {
			return contracts.Invalidf("Worker State invocation tool metrics are invalid")
		}
		if ^uint64(0)-detailedCalls < item.Calls || ^uint64(0)-detailedFailures < item.Failures {
			return contracts.Invalidf("Worker State invocation tool metrics overflow")
		}
		detailedCalls += item.Calls
		detailedFailures += item.Failures
	}
	if detailedCalls > m.ToolCalls || detailedFailures > m.ToolErrors ||
		(!m.Truncated && (detailedCalls != m.ToolCalls || detailedFailures != m.ToolErrors)) {
		return contracts.Invalidf("Worker State invocation tool detail is inconsistent")
	}
	return nil
}

func (s WorkerStateWorkspace) validate() error {
	if err := contracts.ValidateDigest("Worker State workspaceDigest", s.WorkspaceDigest); err != nil {
		return err
	}
	if s.ScopePaths == nil || s.Interactions == nil || len(s.ScopePaths) > maxStateWorkspacePaths ||
		len(s.Interactions) > maxStateWorkspacePaths || !sort.StringsAreSorted(s.ScopePaths) {
		return contracts.Invalidf("Worker State workspace collections are invalid")
	}
	for index, path := range s.ScopePaths {
		if err := validateStateWorkspacePath(path); err != nil {
			return err
		}
		if index > 0 && path == s.ScopePaths[index-1] {
			return contracts.Invalidf("Worker State workspace scope paths must be unique")
		}
	}
	if encoded, err := marshalAgentStateJSON(s.ScopePaths); err != nil || len(encoded) > maxStateWorkspacePathBytes {
		return contracts.Invalidf("Worker State workspace scope paths exceed their bound")
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
			return contracts.Invalidf("Worker State workspace interaction paths must be unique")
		}
		seen[item.Path] = struct{}{}
		if previous != nil && (item.FirstOrdinal < previous.FirstOrdinal ||
			(item.FirstOrdinal == previous.FirstOrdinal && item.Path < previous.Path)) {
			return contracts.Invalidf("Worker State workspace interactions are not ordered")
		}
		previous = item
		interactionPaths = append(interactionPaths, item.Path)
	}
	if encoded, err := marshalAgentStateJSON(interactionPaths); err != nil || len(encoded) > maxStateWorkspacePathBytes {
		return contracts.Invalidf("Worker State workspace interaction paths exceed their bound")
	}
	return nil
}

func (i WorkerStateWorkspaceInteraction) validate() error {
	if err := validateStateWorkspacePath(i.Path); err != nil {
		return err
	}
	if i.FirstOrdinal == 0 || i.LastOrdinal < i.FirstOrdinal ||
		(i.DiscoveryCalls == 0 && i.ReadCalls == 0 && i.MatchCalls == 0 && i.MutationCalls == 0) {
		return contracts.Invalidf("Worker State workspace interaction is inconsistent")
	}
	return nil
}

func validateStateWorkspacePath(value string) error {
	if value == "" || !utf8.ValidString(value) || value != norm.NFC.String(value) ||
		len([]byte(value)) > 4096 || strings.HasPrefix(value, "/") ||
		strings.ContainsAny(value, "\\\x00") || strings.Contains(value, "://") {
		return contracts.Invalidf("Worker State workspace path is invalid")
	}
	parts := strings.Split(value, "/")
	if len(parts) > 128 {
		return contracts.Invalidf("Worker State workspace path is invalid")
	}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." {
			return contracts.Invalidf("Worker State workspace path is invalid")
		}
		for _, character := range part {
			if character < 0x20 || character == 0x7f {
				return contracts.Invalidf("Worker State workspace path is invalid")
			}
		}
	}
	first := parts[0]
	if len(first) >= 2 && ((first[0] >= 'A' && first[0] <= 'Z') ||
		(first[0] >= 'a' && first[0] <= 'z')) && first[1] == ':' {
		return contracts.Invalidf("Worker State workspace path is invalid")
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
