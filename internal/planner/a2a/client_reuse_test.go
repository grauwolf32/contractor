package a2a

import (
	"context"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	sdk "github.com/a2aproject/a2a-go/v2/a2a"
)

// TestNewMTLSClosesConnectionsOfACancelledInvocation cancels a production
// invocation during a poll: the cancelled poll's connection is closed, the
// best-effort CancelTask reaches the Worker once on a verified connection, no
// request is repeated, and Invoke returns only after closing every connection
// it owned (CTL-R164, CTL-R166).
func TestNewMTLSClosesConnectionsOfACancelledInvocation(t *testing.T) {
	polling := make(chan struct{}, 1)
	var sends, polls, cancels atomic.Int64
	f := newConnectionExperiment(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rpc, ok := lifecycleExperimentRPC(t, w, r)
		if !ok {
			return
		}
		switch rpc.Method {
		case "SendMessage":
			sends.Add(1)
			experimentResponse(w, rpc.ID, sdk.StreamResponse{Event: experimentWorkingTask()})
		case "GetTask":
			if polls.Add(1) < 3 {
				experimentResponse(w, rpc.ID, experimentWorkingTask())
				return
			}
			polling <- struct{}{}
			<-r.Context().Done()
		case "CancelTask":
			cancels.Add(1)
			experimentResponse(w, rpc.ID, experimentWorkingTask())
		default:
			http.Error(w, "unexpected RPC", http.StatusBadRequest)
		}
	}))
	invoker, err := NewMTLS(f.files, 2*time.Second, Options{PollInterval: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	finished := make(chan error, 1)
	go func() {
		_, invokeErr := invoker.Invoke(ctx, "builder", f.handle(), stageRequest())
		finished <- invokeErr
	}()
	lifecycleExperimentWait(t, polling, "third poll")
	cancel()
	select {
	case err := <-finished:
		assertPlannerCode(t, err, "planner_cancelled")
	case <-time.After(5 * time.Second):
		t.Fatal("cancelled invocation did not return")
	}
	experimentAwaitClosed(t, f)
	if sends.Load() != 1 || polls.Load() != 3 || cancels.Load() != 1 {
		t.Fatalf("SendMessage=%d GetTask=%d CancelTask=%d, want 1, 3 and 1", sends.Load(), polls.Load(), cancels.Load())
	}
	// SendMessage and the first two polls share one connection; the
	// cancelled poll closed it, so CancelTask verified a second one.
	if f.accepts.Load() != 2 || f.handshakes.Load() != 2 {
		t.Fatalf("connections=%d handshakes=%d, want 2", f.accepts.Load(), f.handshakes.Load())
	}
}
