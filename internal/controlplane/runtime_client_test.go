package controlplane

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestRuntimeControlClientRejectsDifferentCAValidPrincipalBeforeRequest(t *testing.T) {
	root := filepath.Join(t.TempDir(), "pki")
	generator := localpki.Generator{}
	ca, err := generator.InitCA(root, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{
		DNSNames: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")},
	}
	controlPlane, err := generator.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: leaf})
	if err != nil {
		t.Fatal(err)
	}
	expectedAgent, err := generator.IssueAgent(root, "expected-agent", leaf)
	if err != nil {
		t.Fatal(err)
	}
	wrongEndpoint, err := generator.IssueAgent(root, "wrong-endpoint", leaf)
	if err != nil {
		t.Fatal(err)
	}
	serverTLS, err := mtls.RuntimeAgentServerConfig(mtls.Files{
		Certificate: wrongEndpoint.Certificate, PrivateKey: wrongEndpoint.PrivateKey, CA: ca.Certificate,
	})
	if err != nil {
		t.Fatal(err)
	}
	var requests atomic.Int32
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		requests.Add(1)
	}))
	server.TLS = serverTLS
	server.StartTLS()
	defer server.Close()

	client, err := NewMTLSRuntimeControlClient(mtls.Files{
		Certificate: controlPlane.Certificate, PrivateKey: controlPlane.PrivateKey, CA: ca.Certificate,
	}, 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	reservation := testReservation(
		"allocation_principal", "builder", server.URL, server.URL,
		testTemplate(t), time.Now().Add(time.Minute),
	)
	reservation.Grant.RuntimeAgentID = runtimePrincipalFromCertificate(t, expectedAgent.Certificate)
	err = client.Release(context.Background(), reservation)
	if err == nil || requests.Load() != 0 {
		t.Fatalf("different-SPKI release = error %v, requests %d", err, requests.Load())
	}
}

func runtimePrincipalFromCertificate(t *testing.T, path string) string {
	t.Helper()
	encoded, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode(encoded)
	if block == nil {
		t.Fatal("certificate is not PEM")
	}
	certificate, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	principal, err := mtls.RuntimeAgentID(certificate)
	if err != nil {
		t.Fatal(err)
	}
	return principal
}

func TestRuntimeControlClientPrepareSendsExactResolvedAllocation(t *testing.T) {
	template := testTemplate(t)
	template.Skills = []contracts.ArtifactRef{{Namespace: contracts.AgentSkillNamespace, Name: "review"}}
	template.ModelPolicy.ContextWindowTokens = 131_072
	cumulativeBudget := template.ModelPolicy.MaxTotalTokens - 1
	template.Summarizer = &contracts.WorkerSummarizerConfig{
		ModelPolicy: contracts.ResolvedModelPolicy{
			Ref: contracts.ModelPolicyRef{
				PolicyID: "terminal-summarizer", Version: "1",
				Digest: "sha256:" + strings.Repeat("f", 64),
			},
			Model: "worker-summarizer-model", ContextWindowTokens: 131_072,
			MaxOutputTokens: 1024, MaxModelCalls: 1,
		},
		ContextWindowRatio: 0.9, CumulativeBudget: &cumulativeBudget,
	}
	effectivePolicy := template.ModelPolicy
	effectivePolicy.Ref = contracts.ModelPolicyRef{
		PolicyID: "worker-strong", Version: "2", Digest: "sha256:" + strings.Repeat("d", 64),
	}
	effectivePolicy.Model = "worker-strong-model"
	effectivePolicy.MaxModelCalls++
	settings := testRuntimeSettings()
	lease := time.Date(2026, 8, 29, 13, 0, 0, 0, time.UTC)
	var received contracts.PrepareAllocationRequestV2
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.URL.Path != "/private/v1/allocations/allocation_1/prepare" {
			t.Errorf("unexpected request %s %s", request.Method, request.URL.Path)
		}
		if request.Header.Get(requestid.Header) != "scheduler-request-1" {
			t.Errorf("request ID = %q", request.Header.Get(requestid.Header))
		}
		decoder := json.NewDecoder(request.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&received); err != nil {
			t.Errorf("decode prepare request: %v", err)
		}
		response := contracts.PrepareAllocationResponse{
			APIVersion: contracts.APIVersion,
			WorkerHandle: contracts.WorkerHandle{
				AllocationID: "allocation_1", AgentTemplateRef: template.Ref,
				WorkerRuntimeRef: template.Runtime, LeaseExpiresAt: lease,
				AgentCard: testAgentCard(
					"builder", "allocation_1",
					serverURL(request)+"/private/v1/allocations/allocation_1/a2a",
				),
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	reservation := testReservation("allocation_1", "builder", server.URL, server.URL, template, lease)
	reservation.RunMetadataLabels = contracts.RunMetadataLabels{
		"purpose": "eval", "eval.id": "eval_01", "eval.leg": "a",
	}
	skillRevision := "run-review-1"
	reservation.ResolvedSkills = []contracts.ResolvedSkill{{
		Name: "review",
		Artifact: contracts.ArtifactRef{
			Namespace: contracts.AgentSkillNamespace, Name: "review", Revision: &skillRevision,
		},
		PackageDigest: "sha256:" + strings.Repeat("e", 64),
	}}
	workspaceRevision := "source-revision-1"
	reservation.Workspace = &contracts.AllocationWorkspaceSpecV2{
		Mode: contracts.WorkspaceModeOverlay,
		Sources: []contracts.AllocationWorkspaceSourceV2{{
			Artifact: contracts.ArtifactRef{
				Namespace: "inputs", Name: "source", Revision: &workspaceRevision,
			},
			Target: "project",
		}},
	}
	client, err := NewRuntimeControlClient(server.Client())
	if err != nil {
		t.Fatal(err)
	}

	handle, err := client.Prepare(
		requestid.With(context.Background(), "scheduler-request-1"), reservation,
		contracts.WorkerExecutionSettingsV2{
			ModelPolicy: effectivePolicy, RuntimeSettings: settings,
			ResolvedRuntimeConfigProvenance: testRuntimeProvenance(),
		},
	)
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}
	if handle.AllocationID != reservation.Grant.AllocationID ||
		handle.RuntimeAgentID != reservation.Grant.RuntimeAgentID || received.Spec.LeaseExpiresAt != lease {
		t.Fatalf("handle/request = (%+v, %+v)", handle, received.Spec)
	}
	if received.Spec.AgentTemplate.Ref != template.Ref ||
		!reflect.DeepEqual(received.Spec.RunMetadataLabels, reservation.RunMetadataLabels) ||
		received.Spec.AgentTemplate.ModelPolicy.Ref != template.ModelPolicy.Ref ||
		received.Spec.AgentTemplate.Summarizer == nil ||
		received.Spec.AgentTemplate.Summarizer.ModelPolicy.Ref != template.Summarizer.ModelPolicy.Ref ||
		received.Spec.AgentTemplate.Summarizer.CumulativeBudget == nil ||
		*received.Spec.AgentTemplate.Summarizer.CumulativeBudget != cumulativeBudget ||
		received.Spec.AgentTemplate.Summarizer.ContextWindowRatio != 0.9 ||
		len(received.Spec.ResolvedSkills) != 1 ||
		received.Spec.ResolvedSkills[0].Artifact.Revision == nil ||
		*received.Spec.ResolvedSkills[0].Artifact.Revision != skillRevision ||
		received.Spec.ModelPolicy.Ref != effectivePolicy.Ref ||
		received.Spec.ModelPolicy.Model != effectivePolicy.Model ||
		received.Spec.Workspace == nil || len(received.Spec.Workspace.Sources) != 1 ||
		received.Spec.Workspace.Sources[0].Artifact.Revision == nil ||
		*received.Spec.Workspace.Sources[0].Artifact.Revision != workspaceRevision ||
		received.Spec.RuntimeSettings.LLMGatewayToken.Reveal() != settings.LLMGatewayToken.Reveal() {
		t.Fatalf("prepare request lost resolved inputs: %+v", received.Spec)
	}
}

func TestRuntimeControlClientPrepareRejectsSecretBearingHandle(t *testing.T) {
	template := testTemplate(t)
	settings := testRuntimeSettings()
	lease := wireTime(time.Now().Add(time.Minute).UTC())
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		response := contracts.PrepareAllocationResponse{
			APIVersion: contracts.APIVersion,
			WorkerHandle: contracts.WorkerHandle{
				AllocationID: "allocation_1", AgentTemplateRef: template.Ref,
				WorkerRuntimeRef: template.Runtime, LeaseExpiresAt: lease,
				AgentCard: testAgentCard(
					settings.LLMGatewayToken.Reveal(), "allocation_1", serverURL(request)+"/a2a",
				),
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	client, _ := NewRuntimeControlClient(server.Client())
	reservation := testReservation("allocation_1", "builder", server.URL, server.URL, template, lease)

	_, err := client.Prepare(context.Background(), reservation, contracts.WorkerExecutionSettingsV2{
		ModelPolicy: template.ModelPolicy, RuntimeSettings: settings,
		ResolvedRuntimeConfigProvenance: testRuntimeProvenance(),
	})
	if err == nil || bytes.Contains([]byte(err.Error()), []byte(settings.LLMGatewayToken.Reveal())) {
		t.Fatalf("secret-bearing WorkerHandle error = %v", err)
	}
}

func TestWorkerHandleSecretScanDoesNotMatchShortCredentialAgainstJSONKeys(t *testing.T) {
	t.Parallel()

	template := testTemplate(t)
	lease := wireTime(time.Now().Add(time.Minute).UTC())
	reservation := testReservation(
		"allocation_1", "builder", "https://runtime.example", "https://runtime.example",
		template, lease,
	)
	settings := testRuntimeSettings()
	settings.HTTPProxy = &contracts.HTTPProxySettingsV2{
		Adapter: contracts.RuntimeAdapterHTTPProxy, ProxyURL: "https://proxy.example",
		BasicAuth: &contracts.HTTPProxyBasicAuthV2{
			Username: contracts.NewSecretString("worker"),
			Password: contracts.NewSecretString("short"),
		},
		Targets: []contracts.HTTPProxyTarget{contracts.ProxyTargetLLMGateway},
	}
	handle := contracts.WorkerHandle{
		AllocationID: "allocation_1", AgentTemplateRef: template.Ref,
		WorkerRuntimeRef: template.Runtime, LeaseExpiresAt: lease,
		AgentCard: testAgentCard(
			"builder", "allocation_1",
			"https://runtime.example/private/v1/allocations/allocation_1/a2a",
		),
	}
	if err := validateWorkerHandleV2(handle, reservation, settings); err != nil {
		t.Fatalf("low-entropy credential collided with a JSON key: %v", err)
	}
}

func TestRuntimeControlClientReadsWorkerStateAndRevalidatesETag(t *testing.T) {
	fixture, err := os.ReadFile(filepath.Join(
		"..", "..", "api", "testdata", "v1alpha1", "valid", "agent-state-snapshot.json",
	))
	if err != nil {
		t.Fatal(err)
	}
	const etag = `"contractor-agent-state-v1-7"`
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		requests.Add(1)
		if request.Method != http.MethodGet ||
			request.URL.Path != "/private/v1/allocations/allocation_1/agent-state" {
			t.Errorf("unexpected State request %s %s", request.Method, request.URL.Path)
		}
		if request.Header.Get("Content-Type") != "" || request.Header.Get("Accept") != "application/json" {
			t.Errorf("State request headers = %+v", request.Header)
		}
		body, _ := io.ReadAll(request.Body)
		if len(body) != 0 {
			t.Errorf("State request body = %q", body)
		}
		w.Header().Set("Cache-Control", "private, no-cache")
		w.Header().Set("ETag", etag)
		if request.Header.Get("If-None-Match") == etag {
			w.WriteHeader(http.StatusNotModified)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(fixture)
	}))
	defer server.Close()

	client, _ := NewRuntimeControlClient(server.Client())
	handle := stateWorkerHandle(server.URL, "allocation_1")
	first, err := client.ReadWorkerState(context.Background(), handle, "")
	if err != nil || first.Snapshot == nil || first.NotModified || first.ETag != etag ||
		first.Snapshot.State.StateRevision != 7 {
		t.Fatalf("first Worker State read = (%+v, %v)", first, err)
	}
	second, err := client.ReadWorkerState(context.Background(), handle, first.ETag)
	if err != nil || second.Snapshot != nil || !second.NotModified || second.ETag != etag {
		t.Fatalf("conditional Worker State read = (%+v, %v)", second, err)
	}
	if requests.Load() != 2 {
		t.Fatalf("State request count = %d", requests.Load())
	}
}

func TestRuntimeControlClientWorkerStateFailuresAreBoundedAndSafe(t *testing.T) {
	valid, err := os.ReadFile(filepath.Join(
		"..", "..", "api", "testdata", "v1alpha1", "valid", "agent-state-snapshot.json",
	))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name        string
		status      int
		etag        string
		cache       string
		body        []byte
		conditional string
		wantCode    string
	}{
		{name: "oversized", status: http.StatusOK, etag: `"contractor-agent-state-v1-7"`, cache: "private, no-cache", body: bytes.Repeat([]byte("x"), contracts.MaxAgentStateSnapshotBytes+1), wantCode: "worker_state_response_invalid"},
		{name: "wrong etag", status: http.StatusOK, etag: `"contractor-agent-state-v1-8"`, cache: "private, no-cache", body: valid, wantCode: "worker_state_response_invalid"},
		{name: "missing cache policy", status: http.StatusOK, etag: `"contractor-agent-state-v1-7"`, body: valid, wantCode: "worker_state_response_invalid"},
		{name: "false not modified", status: http.StatusNotModified, etag: `"contractor-agent-state-v1-8"`, cache: "private, no-cache", conditional: `"contractor-agent-state-v1-7"`, wantCode: "worker_state_response_invalid"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if test.cache != "" {
					w.Header().Set("Cache-Control", test.cache)
				}
				if test.etag != "" {
					w.Header().Set("ETag", test.etag)
				}
				if test.status == http.StatusOK {
					w.Header().Set("Content-Type", "application/json")
				}
				w.WriteHeader(test.status)
				_, _ = w.Write(test.body)
			}))
			defer server.Close()
			client, _ := NewRuntimeControlClient(server.Client())
			_, readErr := client.ReadWorkerState(
				context.Background(), stateWorkerHandle(server.URL, "allocation_1"), test.conditional,
			)
			var typed *WorkerStateReadError
			if !errors.As(readErr, &typed) || typed.Code != test.wantCode ||
				strings.Contains(readErr.Error(), server.URL) {
				t.Fatalf("Worker State failure = %#v", readErr)
			}
		})
	}

	client, _ := NewRuntimeControlClient(&http.Client{})
	_, err = client.ReadWorkerState(
		context.Background(), stateWorkerHandle("http://127.0.0.1:1", "allocation_1"), "",
	)
	var typed *WorkerStateReadError
	if !errors.As(err, &typed) || typed.Code != "worker_state_transport_failed" ||
		strings.Contains(err.Error(), "127.0.0.1") {
		t.Fatalf("safe transport failure = %#v", err)
	}
}

func TestRuntimeControlClientRejectsMismatchedWorkerStateEndpointBeforeRequest(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		requests.Add(1)
	}))
	defer server.Close()
	handle := stateWorkerHandle(server.URL, "allocation_1")
	interfaces := handle.AgentCard["supportedInterfaces"].([]any)
	interfaces[0].(map[string]any)["url"] = server.URL + "/unrelated/allocation_1/a2a"
	client, _ := NewRuntimeControlClient(server.Client())

	_, err := client.ReadWorkerState(context.Background(), handle, "")
	var typed *WorkerStateReadError
	if !errors.As(err, &typed) || typed.Code != "worker_state_endpoint_invalid" ||
		requests.Load() != 0 || strings.Contains(err.Error(), server.URL) {
		t.Fatalf("mismatched Worker State endpoint = (%#v, requests %d)", err, requests.Load())
	}
}

func TestValidateA2AAgentCardRequiresMutualTLS(t *testing.T) {
	registeredURL := "https://runtime.example"
	endpoint := "https://runtime.example/private/v1/allocations/allocation_1/a2a"
	card := testAgentCard("builder", "allocation_1", endpoint)
	delete(card, "securityRequirements")

	err := validateA2AAgentCard(card, "allocation_1", registeredURL)
	if err == nil || err.Error() != "Runtime Agent A2A Agent Card does not require mutual TLS" {
		t.Fatalf("validateA2AAgentCard error = %v", err)
	}
}

func TestValidateA2AAgentCardRejectsAnotherPathOnRegisteredOrigin(t *testing.T) {
	registeredURL := "https://runtime.example"
	card := testAgentCard(
		"builder", "allocation_1", "https://runtime.example/unrelated/allocation_1/a2a",
	)

	err := validateA2AAgentCard(card, "allocation_1", registeredURL)
	if err == nil || err.Error() != "Runtime Agent returned an A2A Agent Card for another endpoint" {
		t.Fatalf("validateA2AAgentCard error = %v", err)
	}
}

func TestPrepareAllCleansEveryReservationAfterPartialFailure(t *testing.T) {
	runtime := &recordingRuntime{prepareFailure: map[string]error{"allocation_2": errors.New("synthetic failure")}}
	registry := &recordingAllocationRegistry{}
	controller, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{
		Now:            func() time.Time { return time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC) },
		NewID:          func(prefix string) (string, error) { return prefix + "cleanup", nil },
		CleanupTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute)
	reservations := []Reservation{
		testReservation("allocation_1", "first", "https://first.example", "https://first.example", template, lease),
		testReservation("allocation_2", "second", "https://second.example", "https://second.example", template, lease),
	}

	handles, err := controller.PrepareAll(
		context.Background(), reservations,
		testWorkerExecutionSettings(template, testRuntimeSettings(), "first", "second"),
	)
	if err == nil || handles != nil {
		t.Fatalf("PrepareAll = (%+v, %v), want nil/error", handles, err)
	}
	for _, allocationID := range []string{"allocation_1", "allocation_2"} {
		if !slices.Contains(runtime.aborted, allocationID) || !slices.Contains(runtime.released, allocationID) ||
			!slices.Contains(registry.fenced, allocationID) || !slices.Contains(registry.released, allocationID) {
			t.Fatalf("allocation %s was not fully cleaned: runtime=%+v registry=%+v", allocationID, runtime, registry)
		}
	}
}

func TestFinalizeAllFencesBeforeRuntimeAndReleaseRetainsFailedGrant(t *testing.T) {
	runtime := &recordingRuntime{releaseFailure: map[string]error{"allocation_2": errors.New("unavailable")}}
	registry := &recordingAllocationRegistry{}
	controller, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute)
	reservations := []Reservation{
		testReservation("allocation_1", "first", "https://first.example", "https://first.example", template, lease),
		testReservation("allocation_2", "second", "https://second.example", "https://second.example", template, lease),
	}

	reports, err := controller.FinalizeAll(
		context.Background(), reservations, "finalization_1", time.Now().Add(time.Minute),
	)
	if err != nil || len(reports) != 2 {
		t.Fatalf("FinalizeAll = (%+v, %v)", reports, err)
	}
	if !slices.Equal(registry.fenced, []string{"allocation_1", "allocation_2"}) {
		t.Fatalf("fences = %v", registry.fenced)
	}
	if !slices.Equal(registry.phases, []allocationPhaseRecord{
		{"allocation_1", AllocationFinalizing}, {"allocation_2", AllocationFinalizing},
	}) || !slices.Equal(registry.reports, []string{"allocation_1", "allocation_2"}) {
		t.Fatalf("observed lifecycle = phases %v, reports %v", registry.phases, registry.reports)
	}
	if err := controller.ReleaseAll(context.Background(), reservations); err == nil {
		t.Fatal("ReleaseAll accepted a Runtime Agent release failure")
	}
	if !slices.Equal(registry.released, []string{"allocation_1"}) {
		t.Fatalf("registry releases = %v, want only confirmed Runtime release", registry.released)
	}
}

func TestRuntimeBatchTerminalCallsFanOutWithoutSiblingDeadlineStarvation(t *testing.T) {
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute)
	reservations := []Reservation{
		testReservation("allocation_1", "first", "https://first.example", "https://first.example", template, lease),
		testReservation("allocation_2", "second", "https://second.example", "https://second.example", template, lease),
	}
	reason := contracts.TerminationError{Code: "test_abort", Message: "test abort", Retryable: true}

	tests := []struct {
		name       string
		configure  func(*recordingRuntime)
		invoke     func(*testing.T, context.Context, *RuntimeBatchController, []Reservation) error
		called     func(*recordingRuntime) []string
		wantReport bool
	}{
		{
			name:      "finalize",
			configure: func(runtime *recordingRuntime) { runtime.blockFinalize = map[string]bool{"allocation_1": true} },
			invoke: func(t *testing.T, ctx context.Context, controller *RuntimeBatchController, reservations []Reservation) error {
				reports, err := controller.FinalizeAll(ctx, reservations, "finalization_1", time.Now().Add(time.Minute))
				if len(reports) != 1 || reports["second"].AllocationID != "allocation_2" {
					t.Fatalf("finalize reports = %+v", reports)
				}
				return err
			},
			called:     func(runtime *recordingRuntime) []string { return runtime.finalized },
			wantReport: true,
		},
		{
			name:      "abort",
			configure: func(runtime *recordingRuntime) { runtime.blockAbort = map[string]bool{"allocation_1": true} },
			invoke: func(t *testing.T, ctx context.Context, controller *RuntimeBatchController, reservations []Reservation) error {
				reports, err := controller.AbortAll(ctx, reservations, "abort_1", reason, time.Now().Add(time.Minute))
				if len(reports) != 1 || reports["second"].AllocationID != "allocation_2" {
					t.Fatalf("abort reports = %+v", reports)
				}
				return err
			},
			called:     func(runtime *recordingRuntime) []string { return runtime.aborted },
			wantReport: true,
		},
		{
			name:      "release",
			configure: func(runtime *recordingRuntime) { runtime.blockRelease = map[string]bool{"allocation_1": true} },
			invoke: func(_ *testing.T, ctx context.Context, controller *RuntimeBatchController, reservations []Reservation) error {
				return controller.ReleaseAll(ctx, reservations)
			},
			called: func(runtime *recordingRuntime) []string { return runtime.released },
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runtime := &recordingRuntime{}
			test.configure(runtime)
			registry := &recordingAllocationRegistry{}
			controller, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{
				CleanupTimeout: 100 * time.Millisecond,
			})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()
			if err := test.invoke(t, ctx, controller, reservations); err == nil {
				t.Fatal("batch accepted a blocked Runtime call")
			}
			if called := test.called(runtime); !slices.Contains(called, "allocation_2") {
				t.Fatalf("healthy sibling was not called: %v", called)
			}
			if test.name == "release" && !slices.Equal(registry.released, []string{"allocation_2"}) {
				t.Fatalf("registry releases = %v, want only healthy sibling", registry.released)
			}
			if test.wantReport && !slices.Equal(registry.reports, []string{"allocation_2"}) {
				t.Fatalf("recorded reports = %v", registry.reports)
			}
		})
	}
}

func TestFailedPrepareCleanupDoesNotHideHealthySibling(t *testing.T) {
	runtime := &recordingRuntime{blockAbort: map[string]bool{"allocation_1": true}}
	registry := &recordingAllocationRegistry{}
	controller, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{
		CleanupTimeout: 50 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute)
	reservations := []Reservation{
		testReservation("allocation_1", "first", "https://first.example", "https://first.example", template, lease),
		testReservation("allocation_2", "second", "https://second.example", "https://second.example", template, lease),
	}

	if err := controller.cleanupFailedPrepare(reservations); err == nil {
		t.Fatal("cleanup accepted a blocked Runtime abort")
	}
	if !slices.Contains(runtime.aborted, "allocation_2") || !slices.Contains(runtime.released, "allocation_2") {
		t.Fatalf("healthy sibling cleanup calls = aborted %v, released %v", runtime.aborted, runtime.released)
	}
	if !slices.Contains(registry.released, "allocation_2") {
		t.Fatalf("healthy sibling grant was not released: %v", registry.released)
	}
}

func TestRuntimeControlClientFinalizeAndReleaseProtocol(t *testing.T) {
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute).UTC()
	var operations []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		operations = append(operations, request.URL.Path)
		if stringsHasSuffix(request.URL.Path, "/release") {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		response := contracts.AllocationFinalResponse{
			APIVersion: contracts.APIVersion,
			Report:     testExecutionReport("allocation_1"),
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	client, _ := NewRuntimeControlClient(server.Client())
	reservation := testReservation("allocation_1", "builder", server.URL, server.URL, template, lease)
	if _, err := client.Finalize(context.Background(), reservation, "finalization_1", time.Now().Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	if err := client.Release(context.Background(), reservation); err != nil {
		t.Fatal(err)
	}
	if len(operations) != 2 {
		t.Fatalf("operations = %v", operations)
	}
}

type recordingRuntime struct {
	mu             sync.Mutex
	prepared       []string
	finalized      []string
	aborted        []string
	released       []string
	prepareFailure map[string]error
	releaseFailure map[string]error
	blockFinalize  map[string]bool
	blockAbort     map[string]bool
	blockRelease   map[string]bool
}

func (r *recordingRuntime) Prepare(
	_ context.Context, reservation Reservation, _ contracts.WorkerExecutionSettingsV2,
) (contracts.WorkerHandle, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	allocationID := reservation.Grant.AllocationID
	r.prepared = append(r.prepared, allocationID)
	if err := r.prepareFailure[allocationID]; err != nil {
		return contracts.WorkerHandle{}, err
	}
	return contracts.WorkerHandle{AllocationID: allocationID}, nil
}

func (r *recordingRuntime) Finalize(
	ctx context.Context, reservation Reservation, _ string, _ time.Time,
) (contracts.AllocationFinalReport, error) {
	if err := ctx.Err(); err != nil {
		return contracts.AllocationFinalReport{}, err
	}
	r.mu.Lock()
	allocationID := reservation.Grant.AllocationID
	r.finalized = append(r.finalized, allocationID)
	blocked := r.blockFinalize[allocationID]
	r.mu.Unlock()
	if blocked {
		<-ctx.Done()
		return contracts.AllocationFinalReport{}, ctx.Err()
	}
	return testExecutionReport(reservation.Grant.AllocationID), nil
}

func (r *recordingRuntime) Abort(
	ctx context.Context,
	reservation Reservation,
	_ string,
	_ contracts.TerminationError,
	_ time.Time,
) (contracts.AllocationFinalReport, error) {
	if err := ctx.Err(); err != nil {
		return contracts.AllocationFinalReport{}, err
	}
	r.mu.Lock()
	allocationID := reservation.Grant.AllocationID
	r.aborted = append(r.aborted, allocationID)
	blocked := r.blockAbort[allocationID]
	r.mu.Unlock()
	if blocked {
		<-ctx.Done()
		return contracts.AllocationFinalReport{}, ctx.Err()
	}
	return testExecutionReport(reservation.Grant.AllocationID), nil
}

func (r *recordingRuntime) Release(ctx context.Context, reservation Reservation) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	r.mu.Lock()
	allocationID := reservation.Grant.AllocationID
	r.released = append(r.released, allocationID)
	blocked := r.blockRelease[allocationID]
	failure := r.releaseFailure[allocationID]
	r.mu.Unlock()
	if blocked {
		<-ctx.Done()
		return ctx.Err()
	}
	return failure
}

type recordingAllocationRegistry struct {
	mu       sync.Mutex
	fenced   []string
	released []string
	phases   []allocationPhaseRecord
	reports  []string
}

type allocationPhaseRecord struct {
	allocationID string
	phase        AllocationAuthoritativePhase
}

func (r *recordingAllocationRegistry) SetWriteFence(allocationID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.fenced = append(r.fenced, allocationID)
	return nil
}

func (r *recordingAllocationRegistry) SetAllocationPhase(
	allocationID string,
	phase AllocationAuthoritativePhase,
	_ *SafeReason,
) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.phases = append(r.phases, allocationPhaseRecord{allocationID, phase})
	return nil
}

func (r *recordingAllocationRegistry) RecordAllocationReport(
	allocationID string,
	_ contracts.AllocationFinalReport,
) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.reports = append(r.reports, allocationID)
	return nil
}

func (r *recordingAllocationRegistry) Release(allocationID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.released = append(r.released, allocationID)
	return nil
}

func testReservation(
	allocationID, logicalName, controlURL, a2aURL string,
	template contracts.ResolvedAgentTemplate,
	lease time.Time,
) Reservation {
	return Reservation{
		Grant: AllocationGrant{
			AllocationID: allocationID, RuntimeAgentID: strings.Repeat("a", 64),
			RuntimeInstanceID: "runtime_" + logicalName,
			RunID:             "run_1", StageExecutionID: "stage_1", LogicalAgentName: logicalName,
			Namespace: logicalName, ReadPolicy: ReadCurrentRun, WritePolicy: WriteInputsAndIntermediates,
		},
		ControlURL: controlURL, A2AURL: a2aURL, AgentTemplate: template, LeaseExpiresAt: lease,
	}
}

func testRuntimeSettings() contracts.RuntimeSettingsV2 {
	token := contracts.NewSecretString("recognizable-runtime-secret")
	return contracts.RuntimeSettingsV2{
		LLMGatewayURL: "https://llm.example/v1", LLMGatewayToken: &token,
		ArtifactAPIURL: "https://control.example/private/v1", RequestTimeoutSeconds: 30,
	}
}

func testWorkerExecutionSettings(
	template contracts.ResolvedAgentTemplate,
	runtime contracts.RuntimeSettingsV2,
	logicalNames ...string,
) map[string]contracts.WorkerExecutionSettingsV2 {
	result := make(map[string]contracts.WorkerExecutionSettingsV2, len(logicalNames))
	for _, name := range logicalNames {
		result[name] = contracts.WorkerExecutionSettingsV2{
			ModelPolicy: template.ModelPolicy, RuntimeSettings: runtime,
			ResolvedRuntimeConfigProvenance: testRuntimeProvenance(),
		}
	}
	return result
}

func testRuntimeProvenance() contracts.ResolvedRuntimeConfigProvenanceV2 {
	gateway := contracts.LLMGatewayConfigRef{
		GatewayID: "local-litellm", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64),
	}
	return contracts.ResolvedRuntimeConfigProvenanceV2{
		Default: contracts.RuntimeLabelBindingProvenanceV2{
			Label: "default", BindingRevision: 1,
			Config: contracts.RuntimeConfigRefV2{
				Name: "contractor-empty", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
			},
		},
		RunLabels:       []contracts.RuntimeLabelBindingProvenanceV2{},
		AgentLabels:     []contracts.RuntimeLabelBindingProvenanceV2{},
		RuntimeAdapters: []contracts.RuntimeAdapterRef{}, LLMGatewayConfig: &gateway,
		RuntimeCredentialRefs: []contracts.RuntimeCredentialRefV2{},
	}
}

func testExecutionReport(allocationID string) contracts.AllocationFinalReport {
	now := time.Now().UTC()
	return contracts.AllocationFinalReport{
		ReportID: "allocation-final-" + allocationID, AllocationID: allocationID,
		StartedAt: now.Add(-time.Second), FinishedAt: now,
		Worker: contracts.ExecutionReport{
			ReportID: "worker-" + allocationID, Complete: true,
			Metrics:   contracts.ExecutionMetrics{Tools: map[string]contracts.ToolMetrics{}},
			ToolCalls: []contracts.ToolCallRecord{}, Errors: []contracts.ExecutionError{},
		},
		Runtime: contracts.RuntimeReport{Complete: true},
	}
}

func TestRuntimeAdapterMetricsAreFilteredAgainstPinnedReservation(t *testing.T) {
	t.Parallel()

	reservation := Reservation{ResolvedRuntimeConfig: &runtimeconfig.ResolvedRuntimeConfig{
		RequiredRuntimeAdapters: []contracts.RuntimeAdapterRef{contracts.RuntimeAdapterOTLPHTTP},
	}}
	report := contracts.RuntimeReport{
		Complete: true,
		Adapters: map[contracts.RuntimeAdapterRef]contracts.RuntimeAdapterMetricsV2{
			contracts.RuntimeAdapterOTLPHTTP:  {Operations: 2},
			contracts.RuntimeAdapterHTTPProxy: {Operations: 1},
		},
	}
	sanitizeRuntimeAdapterMetrics(&report, reservation)
	if report.Complete {
		t.Fatal("unexpected adapter attribution did not mark telemetry incomplete")
	}
	if len(report.Adapters) != 1 || report.Adapters[contracts.RuntimeAdapterOTLPHTTP].Operations != 2 {
		t.Fatalf("trusted adapter metrics were not retained: %+v", report.Adapters)
	}

	report = contracts.RuntimeReport{Complete: true}
	sanitizeRuntimeAdapterMetrics(&report, reservation)
	if report.Complete || len(report.Adapters) != 0 {
		t.Fatalf("missing expected adapter metrics remained complete: %+v", report)
	}
}

func serverURL(request *http.Request) string {
	return "http://" + request.Host
}

func TestRuntimeSettingSecretsIncludesProjectOriginAuthorization(t *testing.T) {
	t.Parallel()
	token := contracts.NewSecretString("project-origin-secret")
	settings := contracts.RuntimeSettingsV2{
		HTTPOriginTarget: &contracts.HTTPOriginTargetSettingsV2{
			URL: "https://app.example.test", BearerToken: &token,
		},
	}
	if !slices.Contains(runtimeSettingSecrets(settings), "project-origin-secret") {
		t.Fatal("Project origin credential is absent from WorkerHandle leak detection")
	}
}

func testAgentCard(name, allocationID, endpoint string) map[string]any {
	return map[string]any{
		"name": name,
		"supportedInterfaces": []any{map[string]any{
			"url": endpoint, "protocolBinding": "JSONRPC", "protocolVersion": "1.0",
			"tenant": allocationID,
		}},
		"defaultInputModes":  []any{stageContentMediaType},
		"defaultOutputModes": []any{workerCompletionMediaType},
		"skills": []any{map[string]any{
			"id": "contractor_stage_content", "inputModes": []any{stageContentMediaType},
			"outputModes": []any{workerCompletionMediaType},
		}},
		"securitySchemes": map[string]any{
			"mutualTLS": map[string]any{"mtlsSecurityScheme": map[string]any{}},
		},
		"securityRequirements": []any{map[string]any{
			"schemes": map[string]any{"mutualTLS": map[string]any{}},
		}},
	}
}

func stringsHasSuffix(value, suffix string) bool {
	return len(value) >= len(suffix) && value[len(value)-len(suffix):] == suffix
}

func stateWorkerHandle(baseURL, allocationID string) contracts.WorkerHandle {
	return contracts.WorkerHandle{
		AllocationID: allocationID,
		AgentCard: testAgentCard(
			"worker", allocationID,
			strings.TrimRight(baseURL, "/")+"/private/v1/allocations/"+allocationID+"/a2a",
		),
	}
}

func (r *recordingRuntime) String() string {
	return fmt.Sprintf("prepared=%v aborted=%v released=%v", r.prepared, r.aborted, r.released)
}
