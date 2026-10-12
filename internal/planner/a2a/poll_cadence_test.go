package a2a

import (
	"context"
	"reflect"
	"testing"
	"time"

	sdk "github.com/a2aproject/a2a-go/v2/a2a"
)

// TestInvokerBacksOffPollingOfARunningTask polls a Worker Task that runs for
// a virtual minute with the default poll interval: the delay between GetTask
// requests starts at 100 milliseconds and doubles up to two seconds
// (LIM-PLN-7), so the minute costs at most 60 requests, a tenth of the 600 a
// fixed 100-millisecond cadence made, and the result is noticed within one
// maximal delay after the Task completes.
func TestInvokerBacksOffPollingOfARunningTask(t *testing.T) {
	want := successResult()
	working := &sdk.Task{
		ID: "task-1", ContextID: "context-1", Status: sdk.TaskStatus{State: sdk.TaskStateWorking},
	}
	completed := &sdk.Task{
		ID: "task-1", ContextID: "context-1",
		Status: sdk.TaskStatus{State: sdk.TaskStateCompleted, Message: resultMessage(want)},
	}
	client := &fakeClient{response: working, tasks: []*sdk.Task{working}}
	var elapsed time.Duration
	var delays []time.Duration
	invoker := &Invoker{
		build:        func(context.Context, *sdk.AgentCard) (protocolClient, error) { return client, nil },
		pollInterval: defaultPollInterval,
		wait: func(ctx context.Context, delay time.Duration) error {
			delays = append(delays, delay)
			elapsed += delay
			if elapsed >= time.Minute {
				// The Worker finishes after a minute of virtual time.
				client.tasks = []*sdk.Task{completed}
			}
			return ctx.Err()
		},
	}

	got, err := invoker.Invoke(t.Context(), "builder", workerHandle("https://runtime.example/a2a"), stageRequest())
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("Invoke = (%+v, %v), want %+v", got, err, want)
	}
	if client.getCalls > 60 || client.getCalls != len(delays) {
		t.Fatalf("a one-minute Task took %d GetTask requests after %d waits, want at most 60", client.getCalls, len(delays))
	}
	wantStart := []time.Duration{
		100 * time.Millisecond, 200 * time.Millisecond, 400 * time.Millisecond,
		800 * time.Millisecond, 1600 * time.Millisecond, maxPollInterval,
	}
	if len(delays) < len(wantStart) || !reflect.DeepEqual(delays[:len(wantStart)], wantStart) {
		t.Fatalf("poll delays start %v, want %v", delays[:min(len(delays), len(wantStart))], wantStart)
	}
	for _, delay := range delays[len(wantStart):] {
		if delay != maxPollInterval {
			t.Fatalf("poll delay %s after the backoff reached %s", delay, maxPollInterval)
		}
	}
	if elapsed > time.Minute+maxPollInterval {
		t.Fatalf("the completed Task was noticed after %s, more than one poll delay after a minute", elapsed)
	}
}

// TestInvokerKeepsALongerConfiguredPollInterval configures a poll interval
// above the backoff bound: every poll waits exactly the configured interval,
// never shorter and never longer.
func TestInvokerKeepsALongerConfiguredPollInterval(t *testing.T) {
	working := &sdk.Task{
		ID: "task-1", ContextID: "context-1", Status: sdk.TaskStatus{State: sdk.TaskStateWorking},
	}
	client := &fakeClient{response: working, tasks: []*sdk.Task{working}}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	var delays []time.Duration
	invoker := &Invoker{
		build:        func(context.Context, *sdk.AgentCard) (protocolClient, error) { return client, nil },
		pollInterval: 5 * time.Second,
		wait: func(ctx context.Context, delay time.Duration) error {
			delays = append(delays, delay)
			if len(delays) == 4 {
				cancel()
			}
			return ctx.Err()
		},
	}
	if _, err := invoker.Invoke(ctx, "builder", workerHandle("https://runtime.example/a2a"), stageRequest()); err == nil {
		t.Fatal("cancelled invocation returned a result")
	}
	if want := []time.Duration{5 * time.Second, 5 * time.Second, 5 * time.Second, 5 * time.Second}; !reflect.DeepEqual(delays, want) {
		t.Fatalf("poll delays = %v, want %v", delays, want)
	}
}
