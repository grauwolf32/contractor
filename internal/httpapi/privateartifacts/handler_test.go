package privateartifacts

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/mtls"
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
	assertArtifactTimestampHeaders(t, response.Header(), artifactTestEpoch, artifactTestEpoch)
}

func TestPrivateArtifactSkillReadRequiresTheLiveAllocationGrant(t *testing.T) {
	repository := newMemoryRepository()
	canary := []byte("selected-skill-body-canary")
	repository.seed("run-a", "skills", "likec4", "revision-skill", canary)
	registry := &fakeRegistry{grant: testGrant("run-a")}
	handler := newTestHandler(t, registry, repository)
	target := "/private/v1/allocations/allocation-1/artifacts/skills/likec4?revision=revision-skill"

	live := httptest.NewRecorder()
	handler.ServeHTTP(live, trustedRequest(http.MethodGet, target, nil))
	if live.Code != http.StatusOK || !bytes.Equal(live.Body.Bytes(), canary) {
		t.Fatalf("live exact Skill read = %d %q", live.Code, live.Body.Bytes())
	}

	registry.mu.Lock()
	registry.grant = controlplane.AllocationGrant{}
	registry.mu.Unlock()
	stale := httptest.NewRecorder()
	handler.ServeHTTP(stale, trustedRequest(http.MethodGet, target, nil))
	if stale.Code != http.StatusNotFound || bytes.Contains(stale.Body.Bytes(), canary) {
		t.Fatalf("released allocation Skill read = %d %s", stale.Code, stale.Body.String())
	}
}

func TestPrivateArtifactWriteEnforcesCASAndReservedOutputs(t *testing.T) {
	repository := newMemoryRepository()
	repository.seed(
		"run-a", artifactpolicy.RunSystemNamespace, artifactpolicy.RunRepeatRequestName,
		"revision-system", []byte("internal request"),
	)
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
	assertArtifactTimestampHeaders(
		t, created.Header(), artifactTestEpoch.Add(time.Second), artifactTestEpoch.Add(time.Second),
	)
	var responseBody map[string]any
	if err := json.Unmarshal(created.Body.Bytes(), &responseBody); err != nil || len(responseBody) != 4 {
		t.Fatalf("write JSON body changed with timestamp metadata: %s", created.Body.String())
	}
	for _, field := range []string{"apiVersion", "artifact", "mediaType", "size"} {
		if _, present := responseBody[field]; !present {
			t.Fatalf("write JSON body is missing %q: %s", field, created.Body.String())
		}
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

	skill := putArtifact(t, handler, "skills", "likec4", "*", []byte("forbidden package"))
	if skill.Code != http.StatusForbidden {
		t.Fatalf("Skill write = %d %s", skill.Code, skill.Body.String())
	}
	if _, exists := repository.current("run-a", "skills", "likec4"); exists {
		t.Fatal("reserved Skill write reached the repository")
	}

	proposal := putArtifact(t, handler, "finding-proposals", "forged", "*", []byte(`{}`))
	if proposal.Code != http.StatusForbidden {
		t.Fatalf("finding proposal generic write = %d %s", proposal.Code, proposal.Body.String())
	}
	if _, exists := repository.current("run-a", "finding-proposals", "forged"); exists {
		t.Fatal("reserved finding proposal write reached the repository")
	}

	systemWrite := putArtifact(
		t, handler, artifactpolicy.RunSystemNamespace,
		artifactpolicy.RunRepeatRequestName, "*", []byte("forged request"),
	)
	if systemWrite.Code != http.StatusForbidden {
		t.Fatalf("system write = %d %s", systemWrite.Code, systemWrite.Body.String())
	}
	systemRead := httptest.NewRecorder()
	handler.ServeHTTP(systemRead, trustedRequest(
		http.MethodGet,
		"/private/v1/allocations/allocation-1/artifacts/"+artifactpolicy.RunSystemNamespace+"/"+artifactpolicy.RunRepeatRequestName,
		nil,
	))
	if systemRead.Code != http.StatusForbidden || bytes.Contains(systemRead.Body.Bytes(), []byte("internal request")) {
		t.Fatalf("system read = %d %s", systemRead.Code, systemRead.Body.String())
	}
	list := httptest.NewRecorder()
	handler.ServeHTTP(list, trustedRequest(
		http.MethodGet, "/private/v1/allocations/allocation-1/artifacts", nil,
	))
	if list.Code != http.StatusOK || bytes.Contains(list.Body.Bytes(), []byte(artifactpolicy.RunSystemNamespace)) {
		t.Fatalf("system list visibility = %d %s", list.Code, list.Body.String())
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

func TestFindingReceiptReplaySurvivesFenceButNewSubmissionDoesNot(t *testing.T) {
	repository := newMemoryRepository()
	registry := &fakeRegistry{grant: testGrant("run-a")}
	findings := &fakeFindingIntake{}
	handler, err := NewHandler(Dependencies{
		Registry: registry, Artifacts: artifacts.NewService(repository), Findings: findings,
		NewRequestID: func() (string, error) { return "artifact-request-fixed", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	first := postFinding(t, handler, "candidate-1")
	if first.Code != http.StatusCreated {
		t.Fatalf("new finding = %d %s", first.Code, first.Body.String())
	}
	registry.mu.Lock()
	registry.grant.WriteFenced = true
	registry.mu.Unlock()
	replay := postFinding(t, handler, "candidate-1")
	if replay.Code != http.StatusOK || !strings.Contains(replay.Body.String(), `"replayed":true`) {
		t.Fatalf("fenced receipt replay = %d %s", replay.Code, replay.Body.String())
	}
	newSubmission := postFinding(t, handler, "candidate-2")
	if newSubmission.Code != http.StatusConflict ||
		!strings.Contains(newSubmission.Body.String(), "allocation_write_fenced") {
		t.Fatalf("fenced new finding = %d %s", newSubmission.Code, newSubmission.Body.String())
	}
	if findings.submits != 1 {
		t.Fatalf("finding intake submissions = %d, want 1", findings.submits)
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

func TestPrivateArtifactProductionGrantFenceWinsWhileBodyArrives(t *testing.T) {
	repository := newMemoryRepository()
	handler, registry, allocationID, principalID, binding := newProductionGrantHandler(t, repository)
	body := &gatedReader{
		data: []byte("must not commit"), started: make(chan struct{}), release: make(chan struct{}),
	}
	request := trustedRequest(
		http.MethodPut,
		"/private/v1/allocations/"+allocationID+"/artifacts/builder/race_note",
		body,
	)
	request.Header.Set("Content-Type", "text/plain")
	request.Header.Set("If-None-Match", "*")
	responseDone := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		responseDone <- response
	}()
	waitArtifactSignal(t, body.started, "request body read")
	if err := registry.SetWriteFence(allocationID); err != nil {
		t.Fatal(err)
	}
	close(body.release)
	response := waitArtifactResponse(t, responseDone)
	if response.Code != http.StatusConflict ||
		!strings.Contains(response.Body.String(), "allocation_write_fenced") {
		t.Fatalf("fence-winner response = %d %s", response.Code, response.Body.String())
	}
	if repository.writeCalls != 0 {
		t.Fatalf("fence-winner request reached Artifact mutation %d times", repository.writeCalls)
	}
	assertProductionFaultSlotsReusable(t, registry, allocationID, principalID, binding)
}

func TestPrivateArtifactProductionGrantMutationWinsBeforeFence(t *testing.T) {
	repository := newMemoryRepository()
	repository.writeStarted = make(chan struct{})
	repository.continueWrite = make(chan struct{})
	handler, registry, allocationID, principalID, binding := newProductionGrantHandler(t, repository)
	responseDone := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		responseDone <- putArtifactForAllocation(
			t, handler, allocationID, "builder", "race_note", "*", []byte("complete payload"),
		)
	}()
	waitArtifactSignal(t, repository.writeStarted, "Artifact Store write")
	fenceDone := make(chan error, 1)
	go func() { fenceDone <- registry.SetWriteFence(allocationID) }()
	select {
	case err := <-fenceDone:
		t.Fatalf("write fence overtook production private Artifact mutation: %v", err)
	case <-time.After(25 * time.Millisecond):
	}
	close(repository.continueWrite)
	response := waitArtifactResponse(t, responseDone)
	if response.Code != http.StatusCreated {
		t.Fatalf("mutation-winner response = %d %s", response.Code, response.Body.String())
	}
	select {
	case err := <-fenceDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for production write fence")
	}
	current, exists := repository.current("run-production-race", "builder", "race_note")
	if !exists || string(current.data) != "complete payload" {
		t.Fatalf("mutation-winner current = (%+v, %v)", current, exists)
	}
	later := putArtifactForAllocation(
		t, handler, allocationID, "builder", "later", "*", []byte("forbidden"),
	)
	if later.Code != http.StatusConflict || !strings.Contains(later.Body.String(), "allocation_write_fenced") {
		t.Fatalf("post-fence write = %d %s", later.Code, later.Body.String())
	}
	assertProductionFaultSlotsReusable(t, registry, allocationID, principalID, binding)
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

func TestPrivateArtifactGrantBindsPrincipalAndInstanceBeforeBodyRead(t *testing.T) {
	repository := newMemoryRepository()
	registry := &fakeRegistry{grant: testGrant("run-a")}
	handler := newTestHandler(t, registry, repository)

	body := &countingReader{data: []byte("must-not-be-read")}
	request := httptest.NewRequest(
		http.MethodPut,
		"/private/v1/allocations/allocation-1/artifacts/builder/report",
		body,
	)
	foreign := &x509.Certificate{RawSubjectPublicKeyInfo: []byte("another-valid-agent-key")}
	request.TLS = &tls.ConnectionState{
		PeerCertificates: []*x509.Certificate{foreign},
		VerifiedChains:   [][]*x509.Certificate{{foreign}},
	}
	request.Header.Set(RuntimeInstanceHeader, "runtime-1")
	request.Header.Set("Content-Type", "text/plain")
	request.Header.Set("If-None-Match", "*")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound || body.reads != 0 || repository.writeCalls != 0 {
		t.Fatalf(
			"foreign principal response = %d reads=%d writes=%d body=%s",
			response.Code, body.reads, repository.writeCalls, response.Body.String(),
		)
	}

	wrongInstance := trustedRequest(
		http.MethodGet, "/private/v1/allocations/allocation-1/artifacts", nil,
	)
	wrongInstance.Header.Set(RuntimeInstanceHeader, "runtime-2")
	wrongResponse := httptest.NewRecorder()
	handler.ServeHTTP(wrongResponse, wrongInstance)
	if wrongResponse.Code != http.StatusNotFound {
		t.Fatalf("wrong instance response = %d %s", wrongResponse.Code, wrongResponse.Body.String())
	}
}

type countingReader struct {
	data  []byte
	reads int
}

func (r *countingReader) Read(target []byte) (int, error) {
	r.reads++
	if len(r.data) == 0 {
		return 0, errors.New("unexpected second body read")
	}
	n := copy(target, r.data)
	r.data = r.data[n:]
	return n, nil
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

func trustedRequest(method, target string, body io.Reader) *http.Request {
	var request *http.Request
	if body == nil {
		request = httptest.NewRequest(method, target, nil)
	} else {
		request = httptest.NewRequest(method, target, body)
	}
	certificate := &x509.Certificate{RawSubjectPublicKeyInfo: []byte("artifact-test-key")}
	request.TLS = &tls.ConnectionState{
		PeerCertificates: []*x509.Certificate{certificate},
		VerifiedChains:   [][]*x509.Certificate{{certificate}},
	}
	request.Header.Set(RuntimeInstanceHeader, "runtime-1")
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
	return putArtifactForAllocation(t, handler, "allocation-1", namespace, name, precondition, body)
}

func putArtifactForAllocation(
	t *testing.T,
	handler http.Handler,
	allocationID string,
	namespace string,
	name string,
	precondition string,
	body []byte,
) *httptest.ResponseRecorder {
	t.Helper()
	request := trustedRequest(
		http.MethodPut,
		fmt.Sprintf("/private/v1/allocations/%s/artifacts/%s/%s", allocationID, namespace, name),
		strings.NewReader(string(body)),
	)
	request.Header.Set("Content-Type", "text/plain")
	request.Header.Set("If-None-Match", precondition)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func newProductionGrantHandler(
	t *testing.T,
	repository *memoryRepository,
) (http.Handler, *controlplane.InMemoryRegistry, string, string, controlplane.BindingRequirement) {
	t.Helper()
	sequence := 0
	registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{
		NewID: func(prefix string) (string, error) {
			sequence++
			return fmt.Sprintf("%s%d", prefix, sequence), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	principalIDs := make([]string, 2)
	for index, identity := range []struct {
		instanceID string
		spki       string
	}{
		{instanceID: "runtime-1", spki: "artifact-test-key"},
		{instanceID: "runtime-2", spki: "artifact-second-test-key"},
	} {
		principalID, err := mtls.RuntimeAgentID(&x509.Certificate{
			RawSubjectPublicKeyInfo: []byte(identity.spki),
		})
		if err != nil {
			t.Fatal(err)
		}
		principalIDs[index] = principalID
		principal := controlplane.AuthenticatedPrincipal{
			RuntimeAgentID: principalID, Labels: []string{}, LabelRevision: 1,
		}
		registration := contracts.AgentRegistration{
			APIVersion: contracts.APIVersion, InstanceID: identity.instanceID, SoftwareVersion: "0.1.0",
			StartedAt:     time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC),
			ControlURL:    "https://" + identity.instanceID + ".example:9443",
			A2AURL:        "https://" + identity.instanceID + ".example:9444",
			InitialLabels: []string{}, SupportedRuntimes: []string{"adk@1"},
			SupportedToolsets: []contracts.ToolsetCapability{{
				Ref: "run-artifacts@1", Tools: []string{"list_artifacts", "read_artifact", "write_artifact"},
			}},
			SupportedSandboxProfiles: []string{"local-workdir@1"}, SupportedRuntimeAdapters: []contracts.RuntimeAdapterRef{},
			ObservedState: contracts.AgentIdle,
		}
		if _, err := registry.RegisterAuthenticated(principal, registration); err != nil {
			t.Fatal(err)
		}
		for heartbeatIndex, echoed := range []uint64{0, 1} {
			if _, err := registry.HeartbeatAuthenticated(principalID, contracts.AgentHeartbeat{
				APIVersion: contracts.APIVersion, InstanceID: identity.instanceID,
				HeartbeatSeq: uint64(heartbeatIndex + 1), EchoedAckSeq: echoed, ObservedState: contracts.AgentIdle,
			}); err != nil {
				t.Fatal(err)
			}
		}
	}
	snapshot, err := workflowconfig.Load("../../config/testdata/valid", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	template, err := snapshot.AgentTemplate("artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	gateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	binding := controlplane.BindingRequirement{
		LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: template,
		WorkerSessionMode: contracts.WorkerSessionIsolated,
		ExecutionConfig: controlplane.AllocationExecutionConfig{
			ModelPolicy: template.ModelPolicy.Ref, LLMGateway: gateway.Ref,
		},
	}
	reservations, err := registry.ReserveAll(controlplane.ReservationRequest{
		RunID: "run-production-race", StageExecutionID: "stage-production-race",
		Bindings: []controlplane.BindingRequirement{binding},
	})
	if err != nil || len(reservations) != 1 {
		t.Fatalf("production reservation = (%+v, %v)", reservations, err)
	}
	handler, err := NewHandler(Dependencies{
		Registry: registry, Artifacts: artifacts.NewService(repository),
		NewRequestID: func() (string, error) { return "artifact-request-fixed", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	return handler, registry, reservations[0].Grant.AllocationID, principalIDs[0], binding
}

func assertProductionFaultSlotsReusable(
	t *testing.T,
	registry *controlplane.InMemoryRegistry,
	allocationID string,
	principalID string,
	binding controlplane.BindingRequirement,
) {
	t.Helper()
	if err := registry.Release(allocationID); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.GetGrant(allocationID); !errors.Is(err, controlplane.ErrAllocationNotFound) {
		t.Fatalf("released fault allocation lookup = %v", err)
	}
	if _, err := registry.HeartbeatAuthenticated(principalID, contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: "runtime-1",
		HeartbeatSeq: 3, EchoedAckSeq: 2, ObservedState: contracts.AgentIdle,
	}); err != nil {
		t.Fatal(err)
	}
	reviewer := binding
	reviewer.LogicalAgentName = "reviewer"
	reviewer.Namespace = "reviewer"
	reservations, err := registry.ReserveAll(controlplane.ReservationRequest{
		RunID: "run-after-memory-fault", StageExecutionID: "stage-after-memory-fault",
		Bindings: []controlplane.BindingRequirement{binding, reviewer},
	})
	if err != nil || len(reservations) != 2 {
		t.Fatalf("two-slot claim after Memory fault = (%+v, %v)", reservations, err)
	}
	for _, reservation := range reservations {
		if err := registry.Release(reservation.Grant.AllocationID); err != nil {
			t.Fatal(err)
		}
		if _, err := registry.GetGrant(reservation.Grant.AllocationID); !errors.Is(err, controlplane.ErrAllocationNotFound) {
			t.Fatalf("post-fault allocation remained live: %v", err)
		}
	}
}

type gatedReader struct {
	data    []byte
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (r *gatedReader) Read(target []byte) (int, error) {
	r.once.Do(func() { close(r.started) })
	<-r.release
	if len(r.data) == 0 {
		return 0, io.EOF
	}
	n := copy(target, r.data)
	r.data = r.data[n:]
	return n, nil
}

func waitArtifactSignal(t *testing.T, signal <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", description)
	}
}

func waitArtifactResponse(
	t *testing.T,
	response <-chan *httptest.ResponseRecorder,
) *httptest.ResponseRecorder {
	t.Helper()
	select {
	case value := <-response:
		return value
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for private Artifact response")
		return nil
	}
}

func testGrant(runID string) controlplane.AllocationGrant {
	principalID, _ := mtls.RuntimeAgentID(&x509.Certificate{
		RawSubjectPublicKeyInfo: []byte("artifact-test-key"),
	})
	return controlplane.AllocationGrant{
		AllocationID: "allocation-1", RuntimeAgentID: principalID,
		RuntimeInstanceID: "runtime-1", RunID: runID,
		StageExecutionID: "stage-1", LogicalAgentName: "builder", Namespace: "builder",
		ReadPolicy: controlplane.ReadCurrentRun, WritePolicy: controlplane.WriteInputsAndIntermediates,
	}
}

type fakeRegistry struct {
	mu    sync.Mutex
	grant controlplane.AllocationGrant
}

type fakeFindingIntake struct {
	receipts map[string]findingintake.Receipt
	submits  int
}

func (f *fakeFindingIntake) FindReplay(
	_ context.Context,
	_ controlplane.AllocationGrant,
	input findingintake.Submission,
) (findingintake.Receipt, bool, error) {
	value, ok := f.receipts[input.SubmissionID]
	return value, ok, nil
}

func (f *fakeFindingIntake) Submit(
	_ context.Context,
	grant controlplane.AllocationGrant,
	input findingintake.Submission,
) (findingintake.Receipt, bool, error) {
	if f.receipts == nil {
		f.receipts = make(map[string]findingintake.Receipt)
	}
	if value, ok := f.receipts[input.SubmissionID]; ok {
		return value, true, nil
	}
	f.submits++
	revision := "revision-1"
	value := findingintake.Receipt{
		ReceiptID: "receipt-1", ProposalID: "proposal-1",
		Proposal: findingintake.ExactArtifact{
			Ref: contracts.ArtifactRef{
				Namespace: "finding-proposals", Name: "proposal-1", Revision: &revision,
			},
			Digest: "sha256:" + strings.Repeat("a", 64), MediaType: "application/json", SizeBytes: 2,
		},
		Origin: findingintake.Origin{RunID: grant.RunID},
	}
	f.receipts[input.SubmissionID] = value
	return value, false, nil
}

func postFinding(t *testing.T, handler http.Handler, clientKey string) *httptest.ResponseRecorder {
	t.Helper()
	invocationID := "worker-invocation-1"
	input := findingintake.Submission{
		APIVersion: findingintake.APIVersion, InvocationID: invocationID,
		SubmissionID: findingintake.StableSubmissionID(invocationID, clientKey),
		Proposal: auditdomain.FindingProposal{
			Schema: auditdomain.FindingProposalSchema, ClientKey: clientKey,
			Title: "Candidate", Description: "Candidate description",
			Subject:       auditdomain.FindingSubject{Kind: "code", Key: "handler"},
			Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
			EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{},
			Limitations: []string{},
		},
		EvidenceRefs: []contracts.ArtifactRef{},
	}
	body, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	request := trustedRequest(
		http.MethodPost, "/private/v1/allocations/allocation-1/finding-proposals",
		bytes.NewReader(body),
	)
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
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
	revision          string
	mediaType         string
	data              []byte
	bindingCreatedAt  time.Time
	revisionCreatedAt time.Time
}

var artifactTestEpoch = time.Date(2026, 9, 1, 10, 11, 12, 123456000, time.UTC)

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
	value := storedArtifact{
		revision: revision, mediaType: "text/plain", data: append([]byte(nil), data...),
		bindingCreatedAt: artifactTestEpoch, revisionCreatedAt: artifactTestEpoch,
	}
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
	revisionCreatedAt := artifactTestEpoch.Add(time.Duration(m.next) * time.Second)
	bindingCreatedAt := revisionCreatedAt
	if exists {
		bindingCreatedAt = current.bindingCreatedAt
	}
	stored := storedArtifact{
		revision: revision, mediaType: payload.MediaType, data: append([]byte(nil), payload.Data...),
		bindingCreatedAt: bindingCreatedAt, revisionCreatedAt: revisionCreatedAt,
	}
	m.bindings[key] = stored
	m.history[historyKey(string(scope.Kind()), scope.ID(), target.Namespace, target.Name, revision)] = stored
	return artifacts.WriteResult{
		Ref:               artifacts.ArtifactRef{Namespace: target.Namespace, Name: target.Name, Revision: &revision},
		MediaType:         payload.MediaType,
		Size:              int64(len(payload.Data)),
		BindingCreatedAt:  bindingCreatedAt,
		RevisionCreatedAt: revisionCreatedAt,
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
		Ref:               artifacts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &revision},
		Payload:           artifacts.Payload{MediaType: value.mediaType, Data: append([]byte(nil), value.data...)},
		BindingCreatedAt:  value.bindingCreatedAt,
		RevisionCreatedAt: value.revisionCreatedAt,
	}, nil
}

func assertArtifactTimestampHeaders(t *testing.T, header http.Header, binding, revision time.Time) {
	t.Helper()
	if got := header.Get(bindingCreatedAtHeader); got != binding.Format(time.RFC3339Nano) {
		t.Fatalf("binding-created-at header = %q, want %q", got, binding.Format(time.RFC3339Nano))
	}
	if got := header.Get(revisionCreatedAtHeader); got != revision.Format(time.RFC3339Nano) {
		t.Fatalf("revision-created-at header = %q, want %q", got, revision.Format(time.RFC3339Nano))
	}
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

func (*memoryRepository) PinExact(context.Context, string, artifacts.Scope, artifacts.ArtifactRef, artifacts.PinKind, string) error {
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
