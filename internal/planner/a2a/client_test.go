package a2a

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	sdk "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/requestid"
)

func TestInvokerHandlesImmediateMessageAndTerminalTaskEqually(t *testing.T) {
	want := successResult()
	tests := []struct {
		name     string
		response sdk.SendMessageResult
	}{
		{name: "message", response: resultMessage(want)},
		{
			name: "completed task",
			response: &sdk.Task{
				ID: "task-1", ContextID: "context-1",
				Status: sdk.TaskStatus{State: sdk.TaskStateCompleted, Message: resultMessage(want)},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := &fakeClient{response: test.response}
			invoker := fakeInvoker(client)

			got, err := invoker.Invoke(
				context.Background(), "builder", workerHandle("https://runtime.example/a2a"),
				stageRequest(),
			)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("Invoke = (%+v, %v), want %+v", got, err, want)
			}
			if client.sendCalls != 1 || client.getCalls != 0 ||
				client.request.Tenant != "allocation-1" || client.request.Message.Role != sdk.MessageRoleUser ||
				len(client.request.Message.Parts) != 1 || client.request.Config == nil ||
				!client.request.Config.ReturnImmediately {
				t.Fatalf("SDK request/calls = (%+v, %d, %d)", client.request, client.sendCalls, client.getCalls)
			}
			part := client.request.Message.Parts[0]
			if _, ok := part.Content.(sdk.Data); !ok || part.MediaType != stageContentMediaType {
				t.Fatalf("request part = %+v", part)
			}
		})
	}
}

func TestInvokerMapsInteractionStatesAndDeadlineToTypedFailures(t *testing.T) {
	tests := []struct {
		name  string
		state sdk.TaskState
		code  string
	}{
		{name: "input", state: sdk.TaskStateInputRequired, code: "worker_input_required"},
		{name: "auth", state: sdk.TaskStateAuthRequired, code: "worker_auth_required"},
		{name: "cancelled", state: sdk.TaskStateCanceled, code: "worker_task_cancelled"},
		{name: "rejected", state: sdk.TaskStateRejected, code: "worker_task_rejected"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := &fakeClient{response: &sdk.Task{
				ID: "task-1", ContextID: "context-1", Status: sdk.TaskStatus{State: test.state},
			}}
			_, err := fakeInvoker(client).Invoke(
				context.Background(), "builder", workerHandle("https://runtime.example/a2a"),
				stageRequest(),
			)
			assertPlannerCode(t, err, test.code)
			if client.getCalls != 0 {
				t.Fatalf("interaction state polled %d times", client.getCalls)
			}
		})
	}

	working := &sdk.Task{
		ID: "task-1", ContextID: "context-1", Status: sdk.TaskStatus{State: sdk.TaskStateWorking},
	}
	client := &fakeClient{response: working, tasks: []*sdk.Task{working}}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Millisecond)
	defer cancel()
	started := time.Now()
	_, err := fakeInvoker(client).Invoke(
		ctx, "builder", workerHandle("https://runtime.example/a2a"), stageRequest(),
	)
	assertPlannerCode(t, err, "worker_deadline_exceeded")
	if time.Since(started) > time.Second || client.getCalls == 0 {
		t.Fatalf("bounded polling elapsed=%s getCalls=%d", time.Since(started), client.getCalls)
	}
}

func TestInvokerRejectsInvalidEnvelopeAndFailedTaskSuccessMismatch(t *testing.T) {
	invalid := sdk.NewMessage(sdk.MessageRoleAgent, sdk.NewTextPart("not data"))
	_, err := fakeInvoker(&fakeClient{response: invalid}).Invoke(
		context.Background(), "builder", workerHandle("https://runtime.example/a2a"),
		stageRequest(),
	)
	assertPlannerCode(t, err, "invalid_a2a_response")

	mismatched := &sdk.Task{
		ID: "task-1", ContextID: "context-1",
		Status: sdk.TaskStatus{State: sdk.TaskStateFailed, Message: resultMessage(successResult())},
	}
	_, err = fakeInvoker(&fakeClient{response: mismatched}).Invoke(
		context.Background(), "builder", workerHandle("https://runtime.example/a2a"),
		stageRequest(),
	)
	assertPlannerCode(t, err, "invalid_a2a_response")

	unknown := sdk.NewMessage(sdk.MessageRoleAgent, sdk.NewDataPart(map[string]any{
		"apiVersion": contracts.APIVersion,
		"outcome":    "succeeded", "summary": "done", "artifacts": map[string]any{},
		"invented": true,
	}))
	_, err = decodeResultMessage(unknown)
	assertPlannerCode(t, err, "invalid_worker_result")
}

func TestInvokerReturnsValidatedFailedCandidateFromFailedTask(t *testing.T) {
	want := contracts.StageContentResult{
		APIVersion: contracts.APIVersion,
		Outcome:    contracts.StageFailed,
		Summary:    "Worker could not build the report",
		Artifacts:  map[string]contracts.ArtifactRef{},
		Error: &contracts.TerminationError{
			Code: "source_invalid", Message: "The source is invalid", Retryable: false,
		},
	}
	response := &sdk.Task{
		ID: "task-1", ContextID: "context-1",
		Status: sdk.TaskStatus{State: sdk.TaskStateFailed, Message: resultMessage(want)},
	}
	got, err := fakeInvoker(&fakeClient{response: response}).Invoke(
		context.Background(), "builder", workerHandle("https://runtime.example/a2a"),
		stageRequest(),
	)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("failed Task = (%+v, %v), want %+v", got, err, want)
	}
}

func TestOfficialGoSDKJSONRPCRoundTrip(t *testing.T) {
	executor := a2asrv.AgentExecutorFunc(func(
		_ context.Context, execution *a2asrv.ExecutorContext,
	) iter.Seq2[sdk.Event, error] {
		return func(yield func(sdk.Event, error) bool) {
			if execution.Tenant != "allocation-1" || execution.Message == nil ||
				len(execution.Message.Parts) != 1 {
				yield(nil, errors.New("invalid Contractor request"))
				return
			}
			data, ok := execution.Message.Parts[0].Content.(sdk.Data)
			if !ok {
				yield(nil, errors.New("request is not DataPart"))
				return
			}
			encoded, _ := json.Marshal(data.Value)
			if _, err := contracts.DecodeStrict[contracts.StageContentRequest](encoded); err != nil {
				yield(nil, err)
				return
			}
			yield(resultMessage(successResult()), nil)
		}
	})
	a2aHandler := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor))
	receivedRequestID := ""
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedRequestID = r.Header.Get(requestid.Header)
		a2aHandler.ServeHTTP(w, r)
	}))
	defer server.Close()
	invoker, err := New(server.Client(), Options{PollInterval: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}

	result, err := invoker.Invoke(
		requestid.With(context.Background(), "planner-request-1"),
		"builder", workerHandle(server.URL), stageRequest(),
	)
	if err != nil || !reflect.DeepEqual(result, successResult()) {
		t.Fatalf("official SDK round trip = (%+v, %v)", result, err)
	}
	if receivedRequestID != "planner-request-1" {
		t.Fatalf("A2A request ID = %q", receivedRequestID)
	}
}

func fakeInvoker(client protocolClient) *Invoker {
	return &Invoker{
		build:        func(context.Context, *sdk.AgentCard) (protocolClient, error) { return client, nil },
		pollInterval: time.Millisecond,
	}
}

type fakeClient struct {
	response  sdk.SendMessageResult
	tasks     []*sdk.Task
	request   *sdk.SendMessageRequest
	sendCalls int
	getCalls  int
}

func (c *fakeClient) SendMessage(
	_ context.Context, request *sdk.SendMessageRequest,
) (sdk.SendMessageResult, error) {
	c.sendCalls++
	c.request = request
	return c.response, nil
}

func (c *fakeClient) GetTask(
	_ context.Context, _ *sdk.GetTaskRequest,
) (*sdk.Task, error) {
	c.getCalls++
	if len(c.tasks) == 0 {
		return nil, errors.New("no scripted Task")
	}
	if len(c.tasks) == 1 {
		return c.tasks[0], nil
	}
	result := c.tasks[0]
	c.tasks = c.tasks[1:]
	return result, nil
}

func (*fakeClient) Destroy() error { return nil }

func workerHandle(endpoint string) contracts.WorkerHandle {
	card := sdk.AgentCard{
		Name: "Contractor Worker builder",
		SupportedInterfaces: []*sdk.AgentInterface{{
			URL: endpoint, ProtocolBinding: sdk.TransportProtocolJSONRPC,
			ProtocolVersion: sdk.Version, Tenant: "allocation-1",
		}},
	}
	encoded, _ := json.Marshal(card)
	var mapped map[string]any
	_ = json.Unmarshal(encoded, &mapped)
	return contracts.WorkerHandle{AllocationID: "allocation-1", AgentCard: mapped}
}

func TestDecodeCardAcceptsPythonProtoJSONEmptySecurityScopes(t *testing.T) {
	handle := workerHandle("https://runtime.test/private/v1/allocations/allocation-1/a2a")
	handle.AgentCard["securitySchemes"] = map[string]any{
		"mutualTLS": map[string]any{
			"mtlsSecurityScheme": map[string]any{"description": "deployment mTLS"},
		},
	}
	handle.AgentCard["securityRequirements"] = []any{
		map[string]any{"schemes": map[string]any{"mutualTLS": map[string]any{}}},
	}

	card, err := decodeCard(handle, true)
	if err != nil {
		t.Fatalf("decode Python protobuf JSON card: %v", err)
	}
	if len(card.SecurityRequirements) != 1 || len(card.SecurityRequirements[0]) != 1 {
		t.Fatalf("security requirements = %+v", card.SecurityRequirements)
	}
}

func stageRequest() contracts.StageContentRequest {
	return contracts.StageContentRequest{
		APIVersion: contracts.APIVersion, SubtaskID: "0", Objective: "Build a report",
		Instructions: "Use the source.", Parameters: map[string]string{"mode": "strict"},
		Artifacts: map[string]contracts.ArtifactRef{},
	}
}

func successResult() contracts.StageContentResult {
	revision := "result-r1"
	return contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: "done",
		Artifacts: map[string]contracts.ArtifactRef{
			"report": {Namespace: "builder", Name: "report", Revision: &revision},
		},
	}
}

func resultMessage(result contracts.StageContentResult) *sdk.Message {
	part := sdk.NewDataPart(result)
	part.MediaType = stageContentMediaType
	return sdk.NewMessage(sdk.MessageRoleAgent, part)
}

func assertPlannerCode(t *testing.T, err error, code string) {
	t.Helper()
	var plannerError *planner.Error
	if !errors.As(err, &plannerError) || plannerError.Code != code {
		t.Fatalf("error = %v, want Planner code %q", err, code)
	}
}
