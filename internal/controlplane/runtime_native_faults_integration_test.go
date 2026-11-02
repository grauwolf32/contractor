//go:build integration

package controlplane

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/planner"
	plannera2a "github.com/grauwolf32/contractor/internal/planner/a2a"
)

// These are component cross-process tests: real ADK/Gateway HTTP/mTLS/A2A,
// with local synchronization tools. No Server or business-store claim is made.
func TestCrossLanguageMTLSAllocationLifecycleNativeADKFaults(t *testing.T) {
	for _, mode := range []string{
		"native_tool_cancel_join",
		"expired_deadline_no_admission",
		"native_deadline_stops_next_dispatch",
		"native_parallel_child_cancel_keeps_completed_sibling",
		"native_cleanup_failure_fences_until_release",
		"native_lease_expiry_joins_active_adk",
		"native_terminal_cleanup_budget_expires",
		"native_failed_prepare_rollback",
		"native_successful_release_replay",
	} {
		t.Run(mode, func(t *testing.T) { testNativeADKProcessFault(t, mode) })
	}
}

type nativeSnapshot struct {
	Events          []string       `json:"events"`
	Dispatch        map[string]int `json:"dispatch"`
	Created         map[string]int `json:"created"`
	Closed          map[string]int `json:"closed"`
	Active          int            `json:"active"`
	CallbackEntries map[string]int `json:"callbackEntries"`
	Admitted        map[string]int `json:"admitted"`
	Workspace       string         `json:"workspace"`
}

type nativeProcessLog struct {
	mu     sync.Mutex
	buffer bytes.Buffer
}

func (b *nativeProcessLog) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buffer.Write(p)
}
func (b *nativeProcessLog) text() string { b.mu.Lock(); defer b.mu.Unlock(); return b.buffer.String() }

type nativeGateway struct {
	mu              sync.Mutex
	mode            string
	model           string
	main, finalizer int
	requests        []map[string]any
	problems        []string
}

func (g *nativeGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	g.mu.Lock()
	defer g.mu.Unlock()
	var request map[string]any
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 32<<20)).Decode(&request); err != nil {
		g.problems = append(g.problems, "Gateway received invalid request JSON")
		http.Error(w, "invalid fixture request", http.StatusBadRequest)
		return
	}
	g.requests = append(g.requests, request)
	if request["model"] != g.model {
		g.problems = append(g.problems, "Gateway request changed the pinned fixture model")
		http.Error(w, "wrong fixture model", http.StatusBadRequest)
		return
	}
	message := map[string]any{"role": "assistant", "content": "Completed local native work"}
	finish := "stop"
	var finalization map[string]any
	const preamble = "Contractor Worker result finalization input (JSON):\n"
	for _, raw := range request["messages"].([]any) {
		value := raw.(map[string]any)
		text, _ := value["content"].(string)
		if index := strings.Index(text, preamble); index >= 0 {
			if err := json.Unmarshal([]byte(text[index+len(preamble):]), &finalization); err != nil {
				g.problems = append(g.problems, "Invalid finalizer fixture prompt")
			}
		}
	}
	if finalization != nil {
		g.finalizer++
		value, _ := json.Marshal(map[string]any{"subtaskId": finalization["subtaskId"], "result": finalization["resultText"]})
		message["content"] = string(value)
	} else {
		g.main++
		if g.main == 1 && g.mode != "expired_deadline_no_admission" && g.mode != "native_cleanup_failure_fences_until_release" {
			message["content"] = nil
			finish = "tool_calls"
			call := func(id, name string) map[string]any {
				return map[string]any{"id": id, "type": "function", "function": map[string]any{
					"name": name, "arguments": `{"namespace":"inputs","name":"local-boundary"}`,
				}}
			}
			calls := []any{call("native-completed", "read_artifact")}
			if g.mode == "native_parallel_child_cancel_keeps_completed_sibling" {
				calls = append(calls, call("native-child-cancel", "write_artifact"))
			}
			message["tool_calls"] = calls
		}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id": "native-fixture", "object": "chat.completion", "model": request["model"],
		"choices": []any{map[string]any{"index": 0, "finish_reason": finish, "message": message}},
		"usage":   map[string]any{"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
	})
}

func (g *nativeGateway) counts() (int, int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.main, g.finalizer
}

type nativeHarness struct {
	t           *testing.T
	ctx         context.Context
	client      *RuntimeControlClient
	invoker     *plannera2a.Invoker
	reservation Reservation
	handle      contracts.WorkerHandle
	directory   string
	gateway     *nativeGateway
	settings    runtimesettings.WorkerExecutionSettings
	exited      <-chan struct{}
	output      *nativeProcessLog
	toolsets    int
}

func startNativeHarness(t *testing.T, mode string, configure ...func(*nativeHarness)) *nativeHarness {
	t.Helper()
	repository, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	temporary := t.TempDir()
	gateway := &nativeGateway{mode: mode}
	server := httptest.NewServer(gateway)
	t.Cleanup(server.Close)
	generator := localpki.Generator{}
	pki := filepath.Join(temporary, "pki")
	ca, err := generator.InitCA(pki, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{DNSNames: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	cp, err := generator.IssueControlPlane(pki, localpki.ControlPlaneOptions{LeafOptions: leaf})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := generator.IssueAgent(pki, "native-adk-fault-agent", leaf)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := os.ReadFile(agent.Certificate)
	if err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode(encoded)
	if block == nil {
		t.Fatal("missing agent certificate")
	}
	certificate, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	principal, err := mtls.RuntimeAgentID(certificate)
	if err != nil {
		t.Fatal(err)
	}
	port := unusedIntegrationPort(t)
	endpoint := fmt.Sprintf("https://127.0.0.1:%d", port)
	directory := filepath.Join(temporary, "faults")
	processCtx, stop := context.WithCancel(context.Background())
	output := &nativeProcessLog{}
	command := exec.CommandContext(processCtx, filepath.Join(repository, "runtime/.venv/bin/python"),
		"tests/lifecycle_server.py", "--port", fmt.Sprint(port), "--ca", ca.Certificate,
		"--certificate", agent.Certificate, "--private-key", agent.PrivateKey,
		"--work-root", filepath.Join(temporary, "work"), "--fault-mode", mode, "--fault-control", directory)
	command.Dir = filepath.Join(repository, "runtime")
	command.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH="+filepath.Join(repository, "runtime/src")+string(os.PathListSeparator)+filepath.Join(repository, "runtime/tests"))
	command.Stdout = output
	command.Stderr = output
	if err := command.Start(); err != nil {
		stop()
		t.Fatal(err)
	}
	done := make(chan error, 1)
	exited := make(chan struct{})
	go func() { done <- command.Wait(); close(exited) }()
	t.Cleanup(func() {
		_ = command.Process.Signal(os.Interrupt)
		select {
		case err := <-done:
			if err != nil && !t.Failed() {
				t.Errorf("Runtime process exit: %v\n%s", err, output.text())
			}
		case <-time.After(5 * time.Second):
			stop()
			<-done
		}
		stop()
	})
	files := mtls.Files{Certificate: cp.Certificate, PrivateKey: cp.PrivateKey, CA: ca.Certificate}
	client, err := NewMTLSRuntimeControlClient(files, 5*time.Second, &projectionGrantReader{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(client.Close)
	invoker, err := plannera2a.NewMTLS(files, 10*time.Second, plannera2a.Options{PollInterval: 10 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	template := testTemplate(t)
	gateway.mu.Lock()
	gateway.model = testWorkerPolicy(t).Model
	gateway.mu.Unlock()
	reservation := testReservation("allocation_native", "builder", endpoint, endpoint, template, time.Now().Add(time.Hour))
	reservation.Grant.RuntimeAgentID = principal
	client.grantReader = &projectionGrantReader{grant: reservation.Grant}
	reservation.WorkerSessionMode = contracts.WorkerSessionShared
	settings := runtimesettings.WorkerExecutionSettings{ModelPolicy: testWorkerPolicy(t), RuntimeSettings: testRuntimeSettings(), ResolvedRuntimeConfigProvenance: testRuntimeProvenance()}
	settings.RuntimeSettings.LLMGatewayURL = server.URL + "/v1"
	h := &nativeHarness{t: t, ctx: ctx, client: client, invoker: invoker, reservation: reservation, directory: directory, gateway: gateway, settings: settings, exited: exited, output: output, toolsets: 1}
	for _, option := range configure {
		option(h)
	}
	h.wait("fault_harness_ready")
	if mode == "native_failed_prepare_rollback" {
		// Observe listener readiness without consuming the one failing Prepare.
		for {
			response, readyErr := client.doRequest(ctx, http.MethodGet, endpoint+"/healthz", nil, principal, nil)
			if readyErr == nil {
				_ = response.Body.Close()
				if response.StatusCode == http.StatusOK {
					break
				}
			}
			select {
			case <-ctx.Done():
				t.Fatalf("native failed-prepare fixture readiness: %v", ctx.Err())
			case <-time.After(25 * time.Millisecond):
			}
		}
		_, err := client.Prepare(ctx, reservation, settings)
		var api *RuntimeAPIError
		if !errors.As(err, &api) || api.StatusCode != http.StatusServiceUnavailable || api.Code != "allocation_preparation_failed" || !api.Retryable {
			t.Fatalf("injected partial prepare = %v", err)
		}
		value := h.wait("prepare_failure_injected")
		if value.Active != 0 || len(value.Dispatch) != 0 || value.CallbackEntries["model"] != 0 || value.CallbackEntries["tool"] != 0 {
			t.Fatalf("failed prepare started native work: %+v", value)
		}
		assertNativeToolCounts(t, value, 1)
		if _, err := os.Stat(value.Workspace); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("failed prepare retained its workspace: %v", err)
		}
		main, finalizer := gateway.counts()
		if main != 0 || finalizer != 0 {
			t.Fatal("failed prepare reached the Gateway")
		}
		failedHandle := contracts.WorkerHandle{
			AllocationID: reservation.Grant.AllocationID, RuntimeAgentID: principal,
			AgentCard: map[string]any{"supportedInterfaces": []any{map[string]any{
				"url": endpoint + "/private/v1/allocations/" + reservation.Grant.AllocationID + "/a2a",
			}}},
		}
		_, stateErr := client.ReadWorkerState(ctx, failedHandle, "")
		var stateFailure *reporting.WorkerStateReadError
		if !errors.As(stateErr, &stateFailure) || stateFailure.StatusCode != http.StatusNotFound || stateFailure.Code != "allocation_not_found" || stateFailure.Retryable {
			t.Fatalf("failed prepare Worker State read = %v", stateErr)
		}
		reservation.Grant.AllocationID = "allocation_after_failed_prepare"
		client.grantReader = &projectionGrantReader{grant: reservation.Grant}
		h.reservation = reservation
		h.toolsets = 2
	}
	for {
		h.handle, err = client.Prepare(ctx, reservation, settings)
		if err == nil {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("native prepare: %v\n%s", err, output.text())
		case <-time.After(25 * time.Millisecond):
		}
	}
	t.Cleanup(func() {
		gateway.mu.Lock()
		defer gateway.mu.Unlock()
		if len(gateway.problems) > 0 {
			t.Errorf("Gateway fixture problems: %v", gateway.problems)
		}
	})
	return h
}

func (h *nativeHarness) snapshot() nativeSnapshot {
	h.t.Helper()
	encoded, err := os.ReadFile(filepath.Join(h.directory, "snapshot.json"))
	if err != nil {
		h.t.Fatal(err)
	}
	var value nativeSnapshot
	if err := json.Unmarshal(encoded, &value); err != nil {
		h.t.Fatal(err)
	}
	return value
}
func (h *nativeHarness) wait(event string) nativeSnapshot {
	h.t.Helper()
	for {
		encoded, err := os.ReadFile(filepath.Join(h.directory, "snapshot.json"))
		if err == nil {
			var value nativeSnapshot
			if json.Unmarshal(encoded, &value) == nil {
				for _, candidate := range value.Events {
					if candidate == event {
						return value
					}
				}
			}
		}
		select {
		case <-h.ctx.Done():
			h.t.Fatalf("native boundary did not arrive: %s\n%s", event, h.output.text())
		case <-h.exited:
			h.t.Fatalf("native Runtime exited before boundary %s\n%s", event, h.output.text())
		case <-time.After(10 * time.Millisecond):
		}
	}
}
func (h *nativeHarness) signal(name string) {
	h.t.Helper()
	file, err := os.CreateTemp(h.directory, "signal-")
	if err != nil {
		h.t.Fatal(err)
	}
	if _, err := file.Write([]byte("ready\n")); err != nil {
		h.t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		h.t.Fatal(err)
	}
	if err := os.Rename(file.Name(), filepath.Join(h.directory, name)); err != nil {
		h.t.Fatal(err)
	}
}
func nativeStage(subtask string) contracts.StageContentRequest {
	return contracts.StageContentRequest{APIVersion: contracts.APIVersion, SubtaskID: subtask, Objective: "Exercise one local native ADK boundary", Instructions: "Use the permitted local test tools.", Parameters: map[string]string{}, Artifacts: map[string]contracts.ArtifactRef{}, Deadline: time.Now().Add(time.Hour)}
}
func (h *nativeHarness) abort() (reporting.AllocationFinalReport, error) {
	return h.client.Abort(h.ctx, h.reservation, "native_abort", contracts.TerminationError{Code: "run_cancelled", Message: "Controlled native component shutdown", Retryable: false}, time.Now().Add(15*time.Second))
}
func (h *nativeHarness) finishAndRelease() {
	h.t.Helper()
	report, err := h.client.Finalize(h.ctx, h.reservation, "native_final", time.Now().Add(15*time.Second))
	if err != nil || !report.Worker.Complete || !report.Runtime.Complete {
		h.t.Fatalf("native finalize = (%+v, %v)", report, err)
	}
	if err := h.client.Release(h.ctx, h.reservation); err != nil {
		h.t.Fatal(err)
	}
	h.assertReleased()
}

func (h *nativeHarness) assertReleased() {
	h.t.Helper()
	value := h.wait("release_finished")
	if value.Active != 0 {
		h.t.Fatal("native work survived release")
	}
	assertNativeToolCounts(h.t, value, h.toolsets)
	if _, err := os.Stat(value.Workspace); !errors.Is(err, os.ErrNotExist) {
		h.t.Fatalf("released workspace still present: %v", err)
	}
}

func assertNativeToolCounts(t *testing.T, value nativeSnapshot, generations int) {
	t.Helper()
	for _, name := range []string{"list_artifacts", "read_artifact", "write_artifact"} {
		if value.Created[name] != generations || value.Closed[name] != generations {
			t.Fatalf("tool %s create/close = %d/%d, want %d/%d", name, value.Created[name], value.Closed[name], generations, generations)
		}
	}
}

func (h *nativeHarness) assertSuccessfulReleaseReplay() {
	h.t.Helper()
	before := h.snapshot()
	mainBefore, finalizerBefore := h.gateway.counts()
	if err := h.client.Release(h.ctx, h.reservation); err != nil {
		h.t.Fatalf("successful Release replay: %v", err)
	}
	after := h.snapshot()
	mainAfter, finalizerAfter := h.gateway.counts()
	if !reflect.DeepEqual(before.Created, after.Created) || !reflect.DeepEqual(before.Closed, after.Closed) ||
		!reflect.DeepEqual(before.Dispatch, after.Dispatch) || !reflect.DeepEqual(before.CallbackEntries, after.CallbackEntries) ||
		!reflect.DeepEqual(before.Admitted, after.Admitted) || before.Active != after.Active || before.Workspace != after.Workspace ||
		mainBefore != mainAfter || finalizerBefore != finalizerAfter {
		h.t.Fatal("successful Release replay repeated cleanup or semantic work")
	}
	if len(after.Events) != len(before.Events)+2 || after.Events[len(before.Events)] != "release_requested" || after.Events[len(before.Events)+1] != "release_finished" {
		h.t.Fatal("successful Release replay emitted an unexpected local side effect")
	}
	h.assertReleased()
}

func testNativeADKProcessFault(t *testing.T, mode string) {
	h := startNativeHarness(t, mode)
	request := nativeStage("0")
	switch mode {
	case "native_failed_prepare_rollback", "native_successful_release_replay":
		completion, err := h.invoker.Invoke(h.ctx, "builder", h.handle, request)
		if err != nil || completion.Result == nil {
			t.Fatalf("healthy native allocation = (%+v,%v)", completion, err)
		}
		if h.snapshot().Dispatch["read"] != 1 {
			t.Fatal("healthy allocation did not execute its local tool exactly once")
		}
		h.finishAndRelease()
		if mode == "native_successful_release_replay" {
			h.assertSuccessfulReleaseReplay()
		}
	case "expired_deadline_no_admission":
		request.Deadline = time.Now().Add(-time.Minute)
		completion, err := h.invoker.Invoke(h.ctx, "builder", h.handle, request)
		if err != nil || completion.Failure == nil || completion.Failure.Code != "worker_timeout" {
			t.Fatalf("expired deadline = (%+v,%v)", completion, err)
		}
		state, err := h.client.ReadWorkerState(h.ctx, h.handle, "")
		if err != nil {
			t.Fatal(err)
		}
		metrics := state.Snapshot.State.LastCompletedInvocation.Metrics
		if metrics.ModelCalls != 0 || metrics.ToolCalls != 0 {
			t.Fatal("expired request charged semantic work")
		}
		value := h.snapshot()
		if value.CallbackEntries["model"] != 0 || value.CallbackEntries["tool"] != 0 {
			t.Fatal("expired request entered ADK model/tool callbacks")
		}
		main, finalizer := h.gateway.counts()
		if main != 0 || finalizer != 0 {
			t.Fatal("expired request reached Gateway")
		}
		h.finishAndRelease()
	case "native_deadline_stops_next_dispatch", "native_terminal_cleanup_budget_expires":
		type result struct {
			completion contracts.WorkerCompletion
			err        error
		}
		done := make(chan result, 1)
		go func() {
			completion, err := h.invoker.Invoke(h.ctx, "builder", h.handle, request)
			done <- result{completion, err}
		}()
		value := h.wait("terminal_cleanup_started")
		if value.Active != 0 || value.Dispatch["read"] != 1 || value.Admitted["model"] != 1 || value.Admitted["tool"] != 1 {
			t.Fatalf("deadline boundary = %+v", value)
		}
		select {
		case <-done:
			t.Fatal("semantic deadline bypassed pending authorized terminal cleanup")
		default:
		}
		main, finalizer := h.gateway.counts()
		if main != 1 || finalizer != 0 {
			t.Fatal("deadline started another provider request")
		}
		if mode == "native_terminal_cleanup_budget_expires" {
			h.signal("cleanup-expire")
		} else {
			h.signal("cleanup")
		}
		out := <-done
		if out.err != nil || out.completion.Failure == nil || out.completion.Failure.Code != "worker_timeout" {
			t.Fatalf("native deadline = (%+v,%v)", out.completion, out.err)
		}
		if mode == "native_terminal_cleanup_budget_expires" {
			h.wait("terminal_cleanup_cancelled")
			rejected, err := h.invoker.Invoke(h.ctx, "builder", h.handle, nativeStage("1"))
			if err != nil || rejected.Failure == nil || rejected.Failure.Code != "worker_draining" {
				t.Fatalf("expired cleanup budget did not fence Worker: (%+v, %v)", rejected.Failure, err)
			}
		} else {
			h.wait("terminal_cleanup_done")
		}
		state, err := h.client.ReadWorkerState(h.ctx, h.handle, "")
		if err != nil {
			t.Fatal(err)
		}
		if state.Snapshot.State.LastCompletedInvocation.Phase != "failed" {
			t.Fatal("terminal State cleanup did not publish failure")
		}
		h.finishAndRelease()
	case "native_tool_cancel_join", "native_lease_expiry_joins_active_adk":
		invokeCtx, cancel := context.WithCancel(h.ctx)
		defer cancel()
		invoked := make(chan error, 1)
		go func() { _, err := h.invoker.Invoke(invokeCtx, "builder", h.handle, request); invoked <- err }()
		h.wait("read_started")
		// A real GetTask proves the production Invoker already knows its Task
		// identity, so its deferred CancelTask cannot be lost to startup order.
		h.wait("a2a_poll_seen")
		if mode == "native_tool_cancel_join" {
			cancel()
		} else {
			h.signal("lease-expire")
		}
		h.wait("native_cancel_received")
		var aborted chan error
		if mode == "native_tool_cancel_join" {
			aborted = make(chan error, 1)
			go func() { _, err := h.abort(); aborted <- err }()
			h.wait("abort_requested")
			select {
			case err := <-aborted:
				t.Fatalf("abort passed an unjoined native tool: %v", err)
			default:
			}
		}
		if h.snapshot().Active != 1 {
			t.Fatal("native cancellation did not retain its join owner")
		}
		h.signal("join")
		value := h.wait("read_joined")
		if value.Active != 0 {
			t.Fatal("native tool join did not finish")
		}
		if aborted != nil {
			if err := <-aborted; err != nil {
				t.Fatal(err)
			}
		} else {
			h.wait("lease_expiry_finished")
		}
		cancel()
		if err := <-invoked; err == nil {
			t.Fatal("cancelled invocation returned success")
		}
		if err := h.client.Release(h.ctx, h.reservation); err != nil {
			t.Fatal(err)
		}
		h.assertReleased()
	case "native_parallel_child_cancel_keeps_completed_sibling":
		first, err := h.invoker.Invoke(h.ctx, "builder", h.handle, request)
		if err == nil || planner.FailureFrom(err).Code != "worker_task_cancelled" {
			state, _ := h.client.ReadWorkerState(h.ctx, h.handle, "")
			t.Fatalf("child-originated cancellation = (%+v, %v); state %+v; snapshot %+v", first.Failure, err, state.Snapshot.State.LastCompletedInvocation, h.snapshot())
		}
		value := h.wait("completed_sibling_settled")
		if value.Dispatch["read"] != 1 {
			t.Fatal("completed sibling was not dispatched once")
		}
		state, err := h.client.ReadWorkerState(h.ctx, h.handle, "")
		if err != nil {
			t.Fatal(err)
		}
		if state.Snapshot.State.LastCompletedInvocation.Phase != "cancelled" {
			t.Fatal("child cancellation lost terminal authority")
		}
		metrics := state.Snapshot.State.LastCompletedInvocation.Metrics
		if metrics.ModelCalls != 1 || metrics.ModelErrors != 0 || metrics.InputTokens != 7 || metrics.OutputTokens != 3 || metrics.TotalTokens != 10 || metrics.ToolCalls != 2 || metrics.ToolErrors != 1 {
			t.Fatalf("child cancellation changed charged work: %+v", metrics)
		}
		cancelledCalls := 0
		for _, call := range state.Snapshot.State.Metrics.ToolCalls {
			if call.Tool == "write_artifact" {
				cancelledCalls++
				if call.Error == nil || call.Error.Code != "tool_call_cancelled" || call.Error.Retryable == nil || !*call.Error.Retryable {
					t.Fatalf("child cancellation metric lost its classification: %+v", call)
				}
			}
		}
		if cancelledCalls != 1 {
			t.Fatalf("cancelled child metric count = %d", cancelledCalls)
		}
		main, finalizer := h.gateway.counts()
		settled := h.snapshot()
		if main != 1 || finalizer != 0 || settled.CallbackEntries["model"] != 1 || settled.CallbackEntries["tool"] != 2 || settled.Admitted["model"] != 1 || settled.Admitted["tool"] != 2 {
			t.Fatal("child cancellation started another semantic attempt")
		}
		completion, err := h.invoker.Invoke(h.ctx, "builder", h.handle, nativeStage("1"))
		if err != nil || completion.Result == nil {
			t.Fatalf("shared next invocation = (%+v,%v)", completion, err)
		}
		h.gateway.mu.Lock()
		carried := h.gateway.requests[1]
		h.gateway.mu.Unlock()
		assertNativeCarriedPairs(t, carried, state.Snapshot.State.LastCompletedInvocation.InvocationID)
		if h.snapshot().Dispatch["read"] != 1 || h.snapshot().Dispatch["write"] != 1 {
			t.Fatal("shared recovery replayed a native sibling")
		}
		h.finishAndRelease()
	case "native_cleanup_failure_fences_until_release":
		completion, err := h.invoker.Invoke(h.ctx, "builder", h.handle, request)
		if err != nil || completion.Result == nil {
			t.Fatalf("cleanup prerequisite = (%+v,%v)", completion, err)
		}
		if _, err := h.client.Finalize(h.ctx, h.reservation, "cleanup_final", time.Now().Add(15*time.Second)); err != nil {
			t.Fatal(err)
		}
		err = h.client.Release(h.ctx, h.reservation)
		var api *RuntimeAPIError
		if !errors.As(err, &api) || api.Code != "allocation_cleanup_failed" || !api.Retryable {
			t.Fatalf("cleanup failure = %v", err)
		}
		h.wait("release_failed")
		other := h.reservation
		other.Grant.AllocationID = "allocation_too_early"
		// Authorize this exact competing request in the Server fixture so the
		// assertion below reaches the Runtime's independent cleanup fence.
		h.client.grantReader = &projectionGrantReader{grant: other.Grant}
		_, err = h.client.Prepare(h.ctx, other, h.settings)
		h.client.grantReader = &projectionGrantReader{grant: h.reservation.Grant}
		api = nil
		if !errors.As(err, &api) || api.StatusCode != http.StatusConflict || api.Code != "allocation_conflict" || api.Retryable {
			t.Fatalf("failed cleanup did not reject another allocation deterministically: %v", err)
		}
		if _, err := h.client.ReadWorkerState(h.ctx, h.handle, ""); err == nil {
			t.Fatal("failed cleanup exposed destroyed Worker")
		}
		if err := h.client.Release(h.ctx, h.reservation); err != nil {
			t.Fatal(err)
		}
		h.assertReleased()
	}
}

func assertNativeCarriedPairs(t *testing.T, request map[string]any, logicalID string) {
	t.Helper()
	pending := map[string]bool{}
	seen := map[string]int{}
	for _, raw := range request["messages"].([]any) {
		message := raw.(map[string]any)
		if message["role"] != "tool" && len(pending) != 0 {
			t.Fatal("shared history inserted non-tool content before the native batch was paired")
		}
		if calls, ok := message["tool_calls"].([]any); ok {
			for _, rawCall := range calls {
				id := rawCall.(map[string]any)["id"].(string)
				pending[id] = true
				seen[id]++
			}
		}
		if message["role"] == "tool" {
			id := message["tool_call_id"].(string)
			if !pending[id] {
				t.Fatalf("unpaired native response %s", id)
			}
			delete(pending, id)
			if id == "native-completed" && !strings.Contains(message["content"].(string), "sibling-once") {
				t.Fatal("completed sibling result was replaced")
			}
			var content map[string]any
			if err := json.Unmarshal([]byte(message["content"].(string)), &content); err != nil {
				t.Fatal(err)
			}
			if id == "native-completed" && content["logicalInvocation"] != logicalID {
				t.Fatal("carried sibling result changed its original logical owner")
			}
			if id == "native-child-cancel" {
				errorValue, ok := content["error"].(map[string]any)
				if !ok || errorValue["code"] != "tool_call_cancelled" || errorValue["retryable"] != true {
					t.Fatal("dispatched child cancellation was rewritten as proven nonexecution")
				}
			}
		}
	}
	if len(pending) != 0 || seen["native-completed"] != 1 || seen["native-child-cancel"] != 1 {
		t.Fatalf("native carried IDs = %v; pending %v", seen, pending)
	}
}
