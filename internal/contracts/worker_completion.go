package contracts

import (
	"encoding/json"
	"regexp"
	"strings"
	"unicode/utf8"
)

const (
	WorkerObservationProfileLeanV1 = "lean@1"
	MaxWorkerResultBytes           = 64 * 1024
	MaxWorkerFailureMessageBytes   = 4 * 1024
	MaxWorkerResultArtifacts       = 128
	MaxWorkerObservationTools      = 256
	MaxWorkerFilesRead             = 25
	MaxWorkerCompletionBytes       = 256 * 1024
)

var (
	workerSubtaskIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
	workerFailureCode      = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
)

// ToolObservationCount is a content-free, Runtime-authored aggregate for one
// model-visible tool name.
type ToolObservationCount struct {
	Calls    uint64 `json:"calls"`
	Failures uint64 `json:"failures"`
}

// WorkspaceObservationSummary is the bounded lean@1 projection attached by
// Runtime. Detailed live paths remain in allocation-local Worker State.
type WorkspaceObservationSummary struct {
	ScopedFiles        uint64   `json:"scopedFiles"`
	ScopeComplete      bool     `json:"scopeComplete"`
	DiscoveredFiles    uint64   `json:"discoveredFiles"`
	ReadFiles          uint64   `json:"readFiles"`
	MatchedFiles       uint64   `json:"matchedFiles"`
	ModifiedFiles      uint64   `json:"modifiedFiles"`
	DetailComplete     bool     `json:"detailComplete"`
	UnreadFiles        *uint64  `json:"unreadFiles,omitempty"`
	FilesRead          []string `json:"filesRead"`
	FilesReadTruncated bool     `json:"filesReadTruncated"`
}

// WorkerObservations is authored only by Runtime, never by the Worker model.
type WorkerObservations struct {
	Profile   string                          `json:"profile"`
	Tools     map[string]ToolObservationCount `json:"tools"`
	Workspace *WorkspaceObservationSummary    `json:"workspace,omitempty"`
	Truncated bool                            `json:"truncated"`
}

// WorkerResult is the trusted semantic success returned to Planner.
type WorkerResult struct {
	SubtaskID    string                 `json:"subtaskId"`
	Result       string                 `json:"result"`
	Observations WorkerObservations     `json:"observations"`
	Artifacts    map[string]ArtifactRef `json:"artifacts"`
	Summarized   bool                   `json:"summarized"`
}

// WorkerFailure describes a technical Runtime/tool/provider/contract failure.
// It is not a model-authored semantic Stage outcome.
type WorkerFailure struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
}

// WorkerCompletion is the strict private A2A response shared by Go and Python.
// Exactly one of Result and Failure is present.
type WorkerCompletion struct {
	APIVersion    string         `json:"apiVersion"`
	Result        *WorkerResult  `json:"result,omitempty"`
	Failure       *WorkerFailure `json:"failure,omitempty"`
	InvocationID  string         `json:"invocationId"`
	StateRevision uint64         `json:"stateRevision"`
}

func (c ToolObservationCount) validate() error {
	if c.Failures > c.Calls {
		return invalidf("Worker observation failures exceed calls")
	}
	return nil
}

func (s WorkspaceObservationSummary) validate() error {
	if s.FilesRead == nil || len(s.FilesRead) > MaxWorkerFilesRead {
		return invalidf("Worker filesRead must be a non-null list of at most %d paths", MaxWorkerFilesRead)
	}
	seen := make(map[string]struct{}, len(s.FilesRead))
	for _, path := range s.FilesRead {
		if path == "" {
			return invalidf("Worker observed workspace path must not be empty")
		}
		if err := validateWorkspaceTarget(path); err != nil {
			return invalidf("Worker observed workspace path is invalid")
		}
		if _, exists := seen[path]; exists {
			return invalidf("Worker filesRead paths must be unique")
		}
		seen[path] = struct{}{}
	}
	if uint64(len(s.FilesRead)) > s.ReadFiles {
		return invalidf("Worker filesRead detail exceeds readFiles")
	}
	if s.FilesReadTruncated != (uint64(len(s.FilesRead)) < s.ReadFiles) {
		return invalidf("Worker filesRead truncation is inconsistent")
	}
	coverageComplete := s.ScopeComplete && s.DetailComplete
	if coverageComplete != (s.UnreadFiles != nil) {
		return invalidf("Worker unreadFiles completeness is inconsistent")
	}
	if s.UnreadFiles != nil && *s.UnreadFiles > s.ScopedFiles {
		return invalidf("Worker unreadFiles exceeds scopedFiles")
	}
	return nil
}

func (o WorkerObservations) validate() error {
	if o.Profile != WorkerObservationProfileLeanV1 {
		return invalidf("unknown Worker observation profile")
	}
	if o.Tools == nil || len(o.Tools) > MaxWorkerObservationTools {
		return invalidf("Worker observation tools must be a non-null bounded map")
	}
	for name, count := range o.Tools {
		if len(name) > 128 || !idPattern.MatchString(name) {
			return invalidf("Worker observation tool name is invalid")
		}
		if err := count.validate(); err != nil {
			return err
		}
	}
	if o.Workspace != nil {
		if err := o.Workspace.validate(); err != nil {
			return err
		}
		if (!o.Workspace.ScopeComplete || !o.Workspace.DetailComplete ||
			o.Workspace.FilesReadTruncated) && !o.Truncated {
			return invalidf("Worker observation truncation is inconsistent")
		}
	}
	return nil
}

func (r WorkerResult) validate() error {
	if err := validateWorkerSubtaskID(r.SubtaskID); err != nil {
		return err
	}
	if !utf8.ValidString(r.Result) || strings.TrimSpace(r.Result) == "" ||
		len([]byte(r.Result)) > MaxWorkerResultBytes {
		return invalidf("Worker result must contain 1..%d UTF-8 bytes", MaxWorkerResultBytes)
	}
	if err := r.Observations.validate(); err != nil {
		return err
	}
	if r.Artifacts == nil || len(r.Artifacts) > MaxWorkerResultArtifacts {
		return invalidf("Worker result artifacts must be a non-null bounded map")
	}
	for slot, ref := range r.Artifacts {
		if err := validateOpaqueID("Worker result artifact slot", slot); err != nil {
			return err
		}
		if err := ref.ValidateExact(); err != nil {
			return err
		}
		if isReservedWorkerResultBinding(ref.Namespace, ref.Name) {
			return invalidf("Worker result artifact identifies a reserved binding")
		}
	}
	return nil
}

func (f WorkerFailure) validate() error {
	if !workerFailureCode.MatchString(f.Code) {
		return invalidf("Worker failure code is invalid")
	}
	if !utf8.ValidString(f.Message) || strings.TrimSpace(f.Message) == "" ||
		len([]byte(f.Message)) > MaxWorkerFailureMessageBytes {
		return invalidf("Worker failure message must contain 1..%d UTF-8 bytes", MaxWorkerFailureMessageBytes)
	}
	return nil
}

func (c WorkerCompletion) Validate() error {
	if err := validateAPIVersion(c.APIVersion); err != nil {
		return err
	}
	if (c.Result == nil) == (c.Failure == nil) {
		return invalidf("WorkerCompletion requires exactly one result or failure")
	}
	if c.Result != nil {
		if err := c.Result.validate(); err != nil {
			return err
		}
	} else if err := c.Failure.validate(); err != nil {
		return err
	}
	if !validBoundedOpaqueWorkerID(c.InvocationID) {
		return invalidf("Worker invocationId is invalid")
	}
	if c.StateRevision == 0 {
		return invalidf("Worker stateRevision must be positive")
	}
	encoded, err := json.Marshal(c)
	if err != nil || len(encoded) > MaxWorkerCompletionBytes {
		return invalidf("WorkerCompletion exceeds its bounded contract")
	}
	return nil
}

func validateWorkerSubtaskID(value string) error {
	if !workerSubtaskIDPattern.MatchString(value) {
		return invalidf("subtaskId is invalid")
	}
	return nil
}

func validBoundedOpaqueWorkerID(value string) bool {
	return utf8.ValidString(value) && value == strings.TrimSpace(value) &&
		len([]byte(value)) >= 1 && len([]byte(value)) <= 128 &&
		!strings.ContainsAny(value, "\r\n\t\x00")
}

func isReservedWorkerResultBinding(namespace, name string) bool {
	switch namespace {
	case "inputs", "outputs", "skills":
		return true
	default:
		return strings.HasPrefix(name, "memory.")
	}
}
