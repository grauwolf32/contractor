package privateartifacts

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
)

func TestPrivateArtifactReadDerivesRunScopeOnlyFromAllocation(t *testing.T) {
	repository := newMemoryRepository()
	repository.seed("run-a", "inputs", "source", "revision-a", []byte("run A"))
	repository.seed("run-b", "inputs", "source", "revision-b", []byte("run B secret"))
	registry := &fakeRegistry{grant: testGrant("run-a")}
	handler := newTestHandler(t, registry, repository)

	malicious := trustedRequest(
		http.MethodGet,
		"/private/v1/allocations/allocation-1/artifacts/inputs/source?runId=run-b",
		nil,
	)
	maliciousResponse := httptest.NewRecorder()
	handler.ServeHTTP(maliciousResponse, malicious)
	if maliciousResponse.Code != http.StatusBadRequest || bytes.Contains(maliciousResponse.Body.Bytes(), []byte("run B secret")) {
		t.Fatalf("scope-selector response = %d %s", maliciousResponse.Code, maliciousResponse.Body.String())
	}

	request := trustedRequest(
		http.MethodGet,
		"/private/v1/allocations/allocation-1/artifacts/inputs/source",
		nil,
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Body.String() != "run A" || response.Header().Get("ETag") != `"revision-a"` {
		t.Fatalf("RunScope read = %d %q headers=%v", response.Code, response.Body.String(), response.Header())
	}
}

func TestPrivateArtifactWriteEnforcesCASAndReservedOutputs(t *testing.T) {
	repository := newMemoryRepository()
	registry := &fakeRegistry{grant: testGrant("run-a")}
	handler := newTestHandler(t, registry, repository)

	created := putArtifact(t, handler, "inputs", "source", "*", []byte("first"))
	if created.Code != http.StatusCreated {
		t.Fatalf("create = %d %s", created.Code, created.Body.String())
	}
	var result contracts.ArtifactWriteResult
	if err := json.Unmarshal(created.Body.Bytes(), &result); err != nil || result.Artifact.Revision == nil {
		t.Fatalf("decode exact write result = (%+v, %v)", result, err)
	}

	stale := trustedRequest(
		http.MethodPut,
		"/private/v1/allocations/allocation-1/artifacts/inputs/source",
		strings.NewReader("stale"),
	)
	stale.Header.Set("Content-Type", "text/plain")
	stale.Header.Set("If-Match", `"not-current"`)
	staleResponse := httptest.NewRecorder()
	handler.ServeHTTP(staleResponse, stale)
	if staleResponse.Code != http.StatusConflict {
		t.Fatalf("stale update = %d %s", staleResponse.Code, staleResponse.Body.String())
	}

	output := putArtifact(t, handler, "outputs", "result", "*", []byte("forbidden"))
	if output.Code != http.StatusForbidden {
		t.Fatalf("output write = %d %s", output.Code, output.Body.String())
	}
	if _, exists := repository.current("run-a", "outputs", "result"); exists {
		t.Fatal("reserved output write reached the repository")
	}
}

func TestPrivateArtifactResponseLossRetryCreatesNoSecondRevision(t *testing.T) {
	repository := newMemoryRepository()
	registry := &fakeRegistry{grant: testGrant("run-a")}
	handler := newTestHandler(t, registry, repository)

	created := putArtifact(t, handler, "inputs", "source", "*", []byte("first"))
	if created.Code != http.StatusCreated {
		t.Fatalf("first create = %d %s", created.Code, created.Body.String())
	}
	retry := putArtifact(t, handler, "inputs", "source", "*", []byte("first"))
	if retry.Code != http.StatusConflict || repository.next != 1 {
		t.Fatalf("lost-response retry = status %d revisions %d body %s", retry.Code, repository.next, retry.Body.String())
	}
	current, exists := repository.current("run-a", "inputs", "source")
	if !exists || current.revision != "revision-1" || string(current.data) != "first" {
		t.Fatalf("current after retry = (%+v, %v)", current, exists)
	}
}

func TestPrivateArtifactWriteFenceRejectsLaterWriteWithoutMutation(t *testing.T) {
	repository := newMemoryRepository()
	repository.seed("run-a", "inputs", "source", "revision-a", []byte("before fence"))
	registry := &fakeRegistry{grant: testGrant("run-a")}
	handler := newTestHandler(t, registry, repository)
	registry.mu.Lock()
	registry.grant.WriteFenced = true
	registry.mu.Unlock()
	writesBefore := repository.writeCalls

	request := trustedRequest(
		http.MethodPut,
		"/private/v1/allocations/allocation-1/artifacts/inputs/source",
		strings.NewReader("after fence"),
	)
	request.Header.Set("Content-Type", "text/plain")
	request.Header.Set("If-Match", `"revision-a"`)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusConflict || !strings.Contains(response.Body.String(), "allocation_write_fenced") {
		t.Fatalf("fenced write = %d %s", response.Code, response.Body.String())
	}
	var failure errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &failure); err != nil ||
		failure.RequestID != "artifact-request-fixed" ||
		response.Header().Get("X-Request-ID") != failure.RequestID {
		t.Fatalf("fenced write correlation = headers:%v body:%+v error:%v",
			response.Header(), failure, err)
	}
	current, _ := repository.current("run-a", "inputs", "source")
	if repository.writeCalls != writesBefore || string(current.data) != "before fence" {
		t.Fatalf("fenced write mutated repository: calls=%d current=%q", repository.writeCalls, current.data)
	}
}

func TestPrivateArtifactWriteFenceWaitsForInFlightStoreCommit(t *testing.T) {
	repository := newMemoryRepository()
	repository.writeStarted = make(chan struct{})
	repository.continueWrite = make(chan struct{})
	registry := &fakeRegistry{grant: testGrant("run-a")}
	handler := newTestHandler(t, registry, repository)

	responseDone := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		responseDone <- putArtifact(t, handler, "builder", "report", "*", []byte("committed before fence"))
	}()
	<-repository.writeStarted
	fenceDone := make(chan struct{})
	go func() {
		registry.mu.Lock()
		registry.grant.WriteFenced = true
		registry.mu.Unlock()
		close(fenceDone)
	}()
	select {
	case <-fenceDone:
		t.Fatal("write fence completed before the authorized Store mutation")
	case <-time.After(25 * time.Millisecond):
	}
	close(repository.continueWrite)
	response := <-responseDone
	if response.Code != http.StatusCreated {
		t.Fatalf("in-flight write = %d %s", response.Code, response.Body.String())
	}
	<-fenceDone

	later := putArtifact(t, handler, "builder", "later", "*", []byte("must be rejected"))
	if later.Code != http.StatusConflict || !strings.Contains(later.Body.String(), "allocation_write_fenced") {
		t.Fatalf("post-fence write = %d %s", later.Code, later.Body.String())
	}
}

func TestPrivateArtifactListIsVersionlessAndMTLSRequired(t *testing.T) {
	repository := newMemoryRepository()
	repository.seed("run-a", "analysis", "report", "revision-a", []byte("report"))
	registry := &fakeRegistry{grant: testGrant("run-a")}
	handler := newTestHandler(t, registry, repository)

	untrusted := httptest.NewRequest(
		http.MethodGet, "/private/v1/allocations/allocation-1/artifacts", nil,
	)
	untrustedResponse := httptest.NewRecorder()
	handler.ServeHTTP(untrustedResponse, untrusted)
	if untrustedResponse.Code != http.StatusUnauthorized {
		t.Fatalf("untrusted list status = %d", untrustedResponse.Code)
	}

	request := trustedRequest(
		http.MethodGet, "/private/v1/allocations/allocation-1/artifacts?namespace=analysis", nil,
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	var listed contracts.ArtifactListResult
	if err := json.Unmarshal(response.Body.Bytes(), &listed); err != nil || response.Code != http.StatusOK || len(listed.Artifacts) != 1 {
		t.Fatalf("list = status %d, value %+v, error %v", response.Code, listed, err)
	}
	if listed.Artifacts[0].Revision != nil {
		t.Fatalf("list exposed revision: %+v", listed.Artifacts[0])
	}
}

func newTestHandler(t *testing.T, registry *fakeRegistry, repository *memoryRepository) http.Handler {
	t.Helper()
	handler, err := NewHandler(Dependencies{
		Registry: registry, Artifacts: artifacts.NewService(repository),
		NewRequestID: func() (string, error) { return "artifact-request-fixed", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func trustedRequest(method, target string, body *strings.Reader) *http.Request {
	var request *http.Request
	if body == nil {
		request = httptest.NewRequest(method, target, nil)
	} else {
		request = httptest.NewRequest(method, target, body)
	}
	certificate := &x509.Certificate{}
	request.TLS = &tls.ConnectionState{
		PeerCertificates: []*x509.Certificate{certificate},
		VerifiedChains:   [][]*x509.Certificate{{certificate}},
	}
	return request
}

func putArtifact(
	t *testing.T,
	handler http.Handler,
	namespace string,
	name string,
	precondition string,
	body []byte,
) *httptest.ResponseRecorder {
	t.Helper()
	request := trustedRequest(
		http.MethodPut,
		fmt.Sprintf("/private/v1/allocations/allocation-1/artifacts/%s/%s", namespace, name),
		strings.NewReader(string(body)),
	)
	request.Header.Set("Content-Type", "text/plain")
	request.Header.Set("If-None-Match", precondition)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func testGrant(runID string) controlplane.AllocationGrant {
	return controlplane.AllocationGrant{
		AllocationID: "allocation-1", RuntimeInstanceID: "runtime-1", RunID: runID,
		StageExecutionID: "stage-1", LogicalAgentName: "builder", Namespace: "builder",
		ReadPolicy: controlplane.ReadCurrentRun, WritePolicy: controlplane.WriteInputsAndIntermediates,
	}
}

type fakeRegistry struct {
	mu    sync.Mutex
	grant controlplane.AllocationGrant
}

func (f *fakeRegistry) GetGrant(allocationID string) (controlplane.AllocationGrant, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if allocationID != f.grant.AllocationID {
		return controlplane.AllocationGrant{}, controlplane.ErrAllocationNotFound
	}
	return f.grant, nil
}

func (f *fakeRegistry) WithWriteGrant(
	allocationID string,
	operation func(controlplane.AllocationGrant) error,
) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if allocationID != f.grant.AllocationID {
		return controlplane.ErrAllocationNotFound
	}
	return operation(f.grant)
}

type storedArtifact struct {
	revision  string
	mediaType string
	data      []byte
}

type memoryRepository struct {
	mu             sync.Mutex
	next           int
	writeCalls     int
	bindings       map[string]storedArtifact
	history        map[string]storedArtifact
	writeStarted   chan struct{}
	continueWrite  chan struct{}
	writeStartOnce sync.Once
}

func newMemoryRepository() *memoryRepository {
	return &memoryRepository{bindings: make(map[string]storedArtifact), history: make(map[string]storedArtifact)}
}

func (m *memoryRepository) seed(runID, namespace, name, revision string, data []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()
	value := storedArtifact{revision: revision, mediaType: "text/plain", data: append([]byte(nil), data...)}
	m.bindings[artifactKey("run", runID, namespace, name)] = value
	m.history[historyKey("run", runID, namespace, name, revision)] = value
}

func (m *memoryRepository) current(runID, namespace, name string) (storedArtifact, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	value, exists := m.bindings[artifactKey("run", runID, namespace, name)]
	return value, exists
}

func (m *memoryRepository) Write(
	_ context.Context,
	scope artifacts.Scope,
	target artifacts.ArtifactRef,
	payload artifacts.Payload,
	expected *string,
) (artifacts.WriteResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.writeCalls++
	if m.writeStarted != nil {
		m.writeStartOnce.Do(func() { close(m.writeStarted) })
		<-m.continueWrite
	}
	key := artifactKey(string(scope.Kind()), scope.ID(), target.Namespace, target.Name)
	current, exists := m.bindings[key]
	if expected == nil && exists || expected != nil && (!exists || current.revision != *expected) {
		return artifacts.WriteResult{}, &artifacts.ConflictError{Ref: target, ExpectedRevision: expected}
	}
	m.next++
	revision := fmt.Sprintf("revision-%d", m.next)
	stored := storedArtifact{
		revision: revision, mediaType: payload.MediaType, data: append([]byte(nil), payload.Data...),
	}
	m.bindings[key] = stored
	m.history[historyKey(string(scope.Kind()), scope.ID(), target.Namespace, target.Name, revision)] = stored
	return artifacts.WriteResult{
		Ref:       artifacts.ArtifactRef{Namespace: target.Namespace, Name: target.Name, Revision: &revision},
		MediaType: payload.MediaType, Size: int64(len(payload.Data)),
	}, nil
}

func (m *memoryRepository) Read(
	_ context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef,
) (artifacts.ReadResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var value storedArtifact
	var exists bool
	if ref.Revision == nil {
		value, exists = m.bindings[artifactKey(string(scope.Kind()), scope.ID(), ref.Namespace, ref.Name)]
	} else {
		value, exists = m.history[historyKey(string(scope.Kind()), scope.ID(), ref.Namespace, ref.Name, *ref.Revision)]
	}
	if !exists {
		return artifacts.ReadResult{}, artifacts.ErrArtifactNotFound
	}
	revision := value.revision
	return artifacts.ReadResult{
		Ref:     artifacts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &revision},
		Payload: artifacts.Payload{MediaType: value.mediaType, Data: append([]byte(nil), value.data...)},
	}, nil
}

func (m *memoryRepository) List(
	_ context.Context, scope artifacts.Scope, namespace *string,
) ([]artifacts.ArtifactRef, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	prefix := string(scope.Kind()) + "\x00" + scope.ID() + "\x00"
	result := make([]artifacts.ArtifactRef, 0)
	for key := range m.bindings {
		if !strings.HasPrefix(key, prefix) {
			continue
		}
		parts := strings.Split(key, "\x00")
		if namespace != nil && parts[2] != *namespace {
			continue
		}
		result = append(result, artifacts.ArtifactRef{Namespace: parts[2], Name: parts[3]})
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].Namespace+"/"+result[i].Name < result[j].Namespace+"/"+result[j].Name
	})
	return result, nil
}

func (*memoryRepository) ForkInput(context.Context, artifacts.Scope, artifacts.ArtifactRef, artifacts.Scope, string) (artifacts.ForkResult, error) {
	return artifacts.ForkResult{}, errors.New("not implemented")
}

func (*memoryRepository) BindOutputExact(context.Context, artifacts.Scope, string, artifacts.ArtifactRef, *string) (artifacts.ForkResult, error) {
	return artifacts.ForkResult{}, errors.New("not implemented")
}

func (*memoryRepository) PinExact(context.Context, artifacts.Scope, artifacts.ArtifactRef, artifacts.PinKind, string) error {
	return errors.New("not implemented")
}

func (*memoryRepository) FreezeOutputs(context.Context, artifacts.Scope) error {
	return errors.New("not implemented")
}

func artifactKey(kind, id, namespace, name string) string {
	return strings.Join([]string{kind, id, namespace, name}, "\x00")
}

func historyKey(kind, id, namespace, name, revision string) string {
	return artifactKey(kind, id, namespace, name) + "\x00" + revision
}
