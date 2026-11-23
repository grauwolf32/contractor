// Package stateview exposes fixed, content-free projections over one modeled
// Planner invocation's live Worker State. It never exposes the raw State or
// accepts physical Runtime selectors from model-facing callers.
package stateview

import (
	"context"
	"crypto/hmac"
	cryptorand "crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/planner"
)

const (
	CodeUnavailable          = "worker_state_unavailable"
	CodeChanged              = "worker_state_changed"
	CodeWorkspaceUnavailable = "worker_workspace_unavailable"
	CodeWorkspaceIncomplete  = "worker_workspace_incomplete"
	CodeCursorInvalid        = "worker_state_cursor_invalid"
	CodeRequestInvalid       = "worker_state_request_invalid"

	maximumPageSize = 100
	defaultPageSize = 100
	maximumCursors  = 4096
	cursorKeyBytes  = 32
)

type Error struct {
	Code      string
	Retryable bool
}

func (e *Error) Error() string { return "Worker State projection failed (" + e.Code + ")" }

type Binding struct {
	LogicalName       string
	Handle            contracts.WorkerHandle
	WorkspaceEligible bool
}

type Options struct {
	// CursorKey exists for deterministic tests. Production leaves it empty and
	// receives a fresh invocation-local random key.
	CursorKey []byte
}

type completionSelector struct {
	InvocationID string
	SubtaskID    string
	Revision     uint64
}

type boundWorker struct {
	handle            contracts.WorkerHandle
	workspaceEligible bool
}

type cacheEntry struct {
	selector  completionSelector
	etag      string
	snapshot  reporting.AgentStateSnapshot
	workspace *workspaceProjection
}

type cursorRecord struct {
	projection string
	worker     string
	selector   completionSelector
	offset     int
}

// Service is allocation-private and must not be shared across StageExecution
// Planner invocations.
type Service struct {
	mu sync.Mutex

	reader    planner.WorkerStateReader
	bindings  map[string]boundWorker
	selectors map[string]completionSelector
	cache     map[string]cacheEntry
	cursors   map[string]cursorRecord
	cursorKey []byte
	closed    bool
}

func New(
	reader planner.WorkerStateReader,
	bindings []Binding,
	options Options,
) (*Service, error) {
	if reader == nil || len(bindings) == 0 {
		return nil, errors.New("Worker State reader and bindings are required")
	}
	key := append([]byte(nil), options.CursorKey...)
	if len(key) == 0 {
		key = make([]byte, cursorKeyBytes)
		if _, err := cryptorand.Read(key); err != nil {
			return nil, errors.New("create Worker State cursor key")
		}
	}
	if len(key) != cursorKeyBytes {
		clear(key)
		return nil, fmt.Errorf("Worker State cursor key must contain %d bytes", cursorKeyBytes)
	}
	resolved := make(map[string]boundWorker, len(bindings))
	for _, binding := range bindings {
		if strings.TrimSpace(binding.LogicalName) == "" || strings.Contains(binding.LogicalName, "/") ||
			strings.TrimSpace(binding.Handle.AllocationID) == "" {
			clear(key)
			return nil, errors.New("Worker State binding is invalid")
		}
		if _, duplicate := resolved[binding.LogicalName]; duplicate {
			clear(key)
			return nil, errors.New("Worker State binding is duplicated")
		}
		resolved[binding.LogicalName] = boundWorker{
			handle:            binding.Handle.Clone(),
			workspaceEligible: binding.WorkspaceEligible,
		}
	}
	return &Service{
		reader: reader, bindings: resolved, selectors: map[string]completionSelector{},
		cache: map[string]cacheEntry{}, cursors: map[string]cursorRecord{}, cursorKey: key,
	}, nil
}

// RecordCompletion remembers only trusted correlation, never result text,
// observations, artifacts, Runtime URLs or other model-visible content.
func (s *Service) RecordCompletion(
	logicalName string,
	expectedSubtaskID string,
	completion contracts.WorkerCompletion,
) error {
	if validation := planner.ValidateWorkerCompletion(completion, expectedSubtaskID); validation != nil {
		return errors.New("invalid Worker completion correlation")
	}
	selector := completionSelector{
		InvocationID: completion.InvocationID,
		SubtaskID:    expectedSubtaskID,
		Revision:     completion.StateRevision,
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return errors.New("Worker State projection service is closed")
	}
	if _, exists := s.bindings[logicalName]; !exists {
		return errors.New("Worker State logical binding is unknown")
	}
	s.selectors[logicalName] = selector
	delete(s.cache, logicalName)
	return nil
}

func (s *Service) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	clear(s.cursorKey)
	s.cursorKey = nil
	clear(s.bindings)
	clear(s.selectors)
	clear(s.cache)
	clear(s.cursors)
}

type WorkspaceCoverage struct {
	ScopedFiles     uint64  `json:"scopedFiles"`
	ScopeComplete   bool    `json:"scopeComplete"`
	DiscoveredFiles uint64  `json:"discoveredFiles"`
	ReadFiles       uint64  `json:"readFiles"`
	MatchedFiles    uint64  `json:"matchedFiles"`
	ModifiedFiles   uint64  `json:"modifiedFiles"`
	DetailComplete  bool    `json:"detailComplete"`
	UnreadFiles     *uint64 `json:"unreadFiles,omitempty"`
}

type FilePage struct {
	Files      []string `json:"files"`
	NextCursor string   `json:"nextCursor,omitempty"`
	Complete   bool     `json:"complete"`
}

type ToolUsageItem struct {
	Name     string `json:"name"`
	Calls    uint64 `json:"calls"`
	Failures uint64 `json:"failures"`
}

type WorkerToolUsage struct {
	ModelCalls            uint64          `json:"modelCalls"`
	ModelErrors           uint64          `json:"modelErrors"`
	InputTokens           uint64          `json:"inputTokens"`
	OutputTokens          uint64          `json:"outputTokens"`
	TotalTokens           uint64          `json:"totalTokens"`
	CachedInputTokens     uint64          `json:"cachedInputTokens"`
	TokenUsageUnavailable uint64          `json:"tokenUsageUnavailable"`
	ToolCalls             uint64          `json:"toolCalls"`
	ToolErrors            uint64          `json:"toolErrors"`
	Tools                 []ToolUsageItem `json:"tools"`
	Truncated             bool            `json:"truncated"`
}

func (s *Service) GetWorkspaceCoverage(
	ctx context.Context,
	logicalName string,
) (WorkspaceCoverage, error) {
	workspace, selector, err := s.loadWorkspace(ctx, logicalName)
	if err != nil {
		return WorkspaceCoverage{}, err
	}
	result := workspace.coverage
	if result.UnreadFiles != nil {
		unread := *result.UnreadFiles
		result.UnreadFiles = &unread
	}
	if err := s.ensureCurrent(logicalName, selector); err != nil {
		return WorkspaceCoverage{}, err
	}
	return result, nil
}

func (s *Service) ListReadFiles(
	ctx context.Context,
	logicalName string,
	cursor string,
	limit int,
) (FilePage, error) {
	workspace, selector, err := s.loadWorkspace(ctx, logicalName)
	if err != nil {
		return FilePage{}, err
	}
	return s.page("read", logicalName, selector, workspace.readFiles, cursor, limit, workspace.coverage.DetailComplete)
}

func (s *Service) ListUnreadFiles(
	ctx context.Context,
	logicalName string,
	cursor string,
	limit int,
) (FilePage, error) {
	workspace, selector, err := s.loadWorkspace(ctx, logicalName)
	if err != nil {
		return FilePage{}, err
	}
	if !workspace.coverage.ScopeComplete || !workspace.coverage.DetailComplete {
		return FilePage{}, projectionError(CodeWorkspaceIncomplete, false)
	}
	return s.page("unread", logicalName, selector, workspace.unreadFiles, cursor, limit, true)
}

func (s *Service) GetWorkerToolUsage(
	ctx context.Context,
	logicalName string,
) (WorkerToolUsage, error) {
	loaded, selector, err := s.load(ctx, logicalName, false)
	if err != nil {
		return WorkerToolUsage{}, err
	}
	invocation := loaded.snapshot.State.LastCompletedInvocation
	if invocation == nil {
		return WorkerToolUsage{}, projectionError(CodeChanged, true)
	}
	metrics := invocation.Metrics
	names := make([]string, 0, len(metrics.Tools))
	for name := range metrics.Tools {
		names = append(names, name)
	}
	sort.Strings(names)
	tools := make([]ToolUsageItem, 0, len(names))
	for _, name := range names {
		item := metrics.Tools[name]
		tools = append(tools, ToolUsageItem{Name: name, Calls: item.Calls, Failures: item.Failures})
	}
	result := WorkerToolUsage{
		ModelCalls: metrics.ModelCalls, ModelErrors: metrics.ModelErrors,
		InputTokens: metrics.InputTokens, OutputTokens: metrics.OutputTokens,
		TotalTokens: metrics.TotalTokens, CachedInputTokens: metrics.CachedInputTokens,
		TokenUsageUnavailable: metrics.TokenUsageUnavailable,
		ToolCalls:             metrics.ToolCalls, ToolErrors: metrics.ToolErrors,
		Tools: tools, Truncated: metrics.Truncated,
	}
	if err := s.ensureCurrent(logicalName, selector); err != nil {
		return WorkerToolUsage{}, err
	}
	return result, nil
}

func (s *Service) loadWorkspace(
	ctx context.Context,
	logicalName string,
) (*workspaceProjection, completionSelector, error) {
	loaded, selector, err := s.load(ctx, logicalName, true)
	if err != nil {
		return nil, completionSelector{}, err
	}
	if loaded.workspace == nil {
		return nil, completionSelector{}, projectionError(CodeWorkspaceUnavailable, false)
	}
	return loaded.workspace, selector, nil
}

func (s *Service) load(
	ctx context.Context,
	logicalName string,
	requireWorkspace bool,
) (cacheEntry, completionSelector, error) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return cacheEntry{}, completionSelector{}, projectionError(CodeUnavailable, true)
	}
	binding, exists := s.bindings[logicalName]
	selector, selected := s.selectors[logicalName]
	cached, hasCache := s.cache[logicalName]
	if !hasCache || cached.selector != selector {
		hasCache = false
	}
	s.mu.Unlock()
	if !exists || !selected {
		return cacheEntry{}, completionSelector{}, projectionError(CodeUnavailable, true)
	}
	if requireWorkspace && !binding.workspaceEligible {
		return cacheEntry{}, completionSelector{}, projectionError(
			CodeWorkspaceUnavailable, false,
		)
	}
	etag := ""
	if hasCache {
		etag = cached.etag
	}
	read, readErr := s.reader.ReadWorkerState(ctx, binding.handle.Clone(), etag)
	if readErr != nil {
		retryable := true
		var typed *reporting.WorkerStateReadError
		if errors.As(readErr, &typed) {
			retryable = typed.Retryable
		}
		return cacheEntry{}, completionSelector{}, projectionError(CodeUnavailable, retryable)
	}
	var loaded cacheEntry
	switch {
	case read.NotModified && read.Snapshot == nil && hasCache && read.ETag == cached.etag:
		loaded = cached
	case !read.NotModified && read.Snapshot != nil:
		// The reader hands over a snapshot it decoded and validated for this
		// read alone; projections only read it, so it is cached as is.
		loaded = cacheEntry{selector: selector, etag: read.ETag, snapshot: *read.Snapshot}
	default:
		return cacheEntry{}, completionSelector{}, projectionError(CodeUnavailable, true)
	}
	snapshot := loaded.snapshot
	expectedETag := fmt.Sprintf("\"contractor-agent-state-v1-%d\"", snapshot.State.StateRevision)
	if read.ETag != expectedETag || !correlates(snapshot, selector) {
		return cacheEntry{}, completionSelector{}, projectionError(CodeChanged, true)
	}
	if requireWorkspace && loaded.workspace == nil {
		if invocation := snapshot.State.LastCompletedInvocation; invocation != nil && invocation.Workspace != nil {
			loaded.workspace = projectWorkspace(*invocation.Workspace)
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.selectors[logicalName] != selector {
		return cacheEntry{}, completionSelector{}, projectionError(CodeChanged, true)
	}
	s.cache[logicalName] = loaded
	return loaded, selector, nil
}

func correlates(snapshot reporting.AgentStateSnapshot, selector completionSelector) bool {
	invocation := snapshot.State.LastCompletedInvocation
	return snapshot.State.StateRevision == selector.Revision && invocation != nil &&
		invocation.InvocationID == selector.InvocationID && invocation.SubtaskID == selector.SubtaskID
}

func countIf(condition bool) uint64 {
	if condition {
		return 1
	}
	return 0
}

// workspaceProjection is immutable and cached with the validated State. Each
// page still revalidates its strong ETag and completion correlation, while
// complete path collections are derived only once per changed revision.
type workspaceProjection struct {
	coverage               WorkspaceCoverage
	readFiles, unreadFiles []string
}

func projectWorkspace(workspace reporting.WorkerStateWorkspace) *workspaceProjection {
	projection := &workspaceProjection{coverage: WorkspaceCoverage{
		ScopedFiles: uint64(len(workspace.ScopePaths)), ScopeComplete: workspace.ScopeComplete,
		DetailComplete: workspace.DetailComplete,
	}}
	read := make(map[string]struct{}, len(workspace.Interactions))
	for _, item := range workspace.Interactions {
		projection.coverage.DiscoveredFiles += countIf(item.DiscoveryCalls > 0)
		projection.coverage.ReadFiles += countIf(item.ReadCalls > 0)
		projection.coverage.MatchedFiles += countIf(item.MatchCalls > 0)
		projection.coverage.ModifiedFiles += countIf(item.MutationCalls > 0)
		if item.ReadCalls > 0 {
			projection.readFiles = append(projection.readFiles, item.Path)
			read[item.Path] = struct{}{}
		}
	}
	if workspace.ScopeComplete && workspace.DetailComplete {
		for _, path := range workspace.ScopePaths {
			if _, observed := read[path]; !observed {
				projection.unreadFiles = append(projection.unreadFiles, path)
			}
		}
		unread := uint64(len(projection.unreadFiles))
		projection.coverage.UnreadFiles = &unread
	}
	return projection
}

func (s *Service) page(
	projection string,
	logicalName string,
	selector completionSelector,
	files []string,
	cursor string,
	limit int,
	complete bool,
) (FilePage, error) {
	if limit == 0 {
		limit = defaultPageSize
	}
	if limit < 1 || limit > maximumPageSize {
		return FilePage{}, projectionError(CodeRequestInvalid, false)
	}
	offset := 0
	if cursor != "" {
		record, err := s.resolveCursor(cursor)
		if err != nil {
			return FilePage{}, err
		}
		if record.projection != projection || record.worker != logicalName || record.selector != selector {
			return FilePage{}, projectionError(CodeChanged, true)
		}
		offset = record.offset
	}
	if offset < 0 || offset > len(files) {
		return FilePage{}, projectionError(CodeCursorInvalid, false)
	}
	end := min(len(files), offset+limit)
	result := FilePage{Files: append([]string(nil), files[offset:end]...), Complete: complete}
	if end < len(files) {
		next, err := s.createCursor(cursorRecord{
			projection: projection, worker: logicalName, selector: selector, offset: end,
		})
		if err != nil {
			return FilePage{}, err
		}
		result.NextCursor = next
	}
	if err := s.ensureCurrent(logicalName, selector); err != nil {
		return FilePage{}, err
	}
	return result, nil
}

func (s *Service) ensureCurrent(
	logicalName string,
	selector completionSelector,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.selectors[logicalName] != selector {
		return projectionError(CodeChanged, true)
	}
	return nil
}

func (s *Service) createCursor(record cursorRecord) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.selectors[record.worker] != record.selector {
		return "", projectionError(CodeChanged, true)
	}
	if len(s.cursors) >= maximumCursors {
		return "", projectionError(CodeUnavailable, true)
	}
	metadata, _ := json.Marshal(struct {
		Projection   string `json:"projection"`
		Worker       string `json:"worker"`
		InvocationID string `json:"invocationId"`
		SubtaskID    string `json:"subtaskId"`
		Revision     uint64 `json:"revision"`
		Offset       int    `json:"offset"`
	}{
		Projection:   record.projection,
		Worker:       record.worker,
		InvocationID: record.selector.InvocationID,
		SubtaskID:    record.selector.SubtaskID,
		Revision:     record.selector.Revision,
		Offset:       record.offset,
	})
	digest := hmacDigest(s.cursorKey, metadata)
	nonce := digest[:16]
	signature := hmacDigest(s.cursorKey, append([]byte("contractor-state-cursor-v1:"), nonce...))
	token := base64.RawURLEncoding.EncodeToString(nonce) + "." +
		base64.RawURLEncoding.EncodeToString(signature)
	s.cursors[token] = record
	return token, nil
}

func (s *Service) resolveCursor(token string) (cursorRecord, error) {
	if len(token) > 128 || strings.Count(token, ".") != 1 {
		return cursorRecord{}, projectionError(CodeCursorInvalid, false)
	}
	encodedNonce, encodedSignature, _ := strings.Cut(token, ".")
	nonce, nonceErr := base64.RawURLEncoding.DecodeString(encodedNonce)
	signature, signatureErr := base64.RawURLEncoding.DecodeString(encodedSignature)
	if nonceErr != nil || signatureErr != nil || len(nonce) != 16 || len(signature) != sha256.Size {
		return cursorRecord{}, projectionError(CodeCursorInvalid, false)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return cursorRecord{}, projectionError(CodeUnavailable, true)
	}
	expected := hmacDigest(s.cursorKey, append([]byte("contractor-state-cursor-v1:"), nonce...))
	if subtle.ConstantTimeCompare(signature, expected) != 1 {
		return cursorRecord{}, projectionError(CodeCursorInvalid, false)
	}
	record, exists := s.cursors[token]
	if !exists {
		return cursorRecord{}, projectionError(CodeCursorInvalid, false)
	}
	return record, nil
}

func hmacDigest(key, value []byte) []byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write(value)
	return mac.Sum(nil)
}

func projectionError(code string, retryable bool) *Error {
	return &Error{Code: code, Retryable: retryable}
}
