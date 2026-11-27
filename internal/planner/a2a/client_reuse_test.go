package a2a

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	a2apb "github.com/a2aproject/a2a-go/v2/a2apb/v1"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/grpc"
	"google.golang.org/grpc/stats"
)

type connectionStats struct {
	opened atomic.Int64
	ended  chan struct{}
}

func (*connectionStats) TagRPC(ctx context.Context, _ *stats.RPCTagInfo) context.Context { return ctx }
func (*connectionStats) HandleRPC(context.Context, stats.RPCStats)                       {}
func (*connectionStats) TagConn(ctx context.Context, _ *stats.ConnTagInfo) context.Context {
	return ctx
}
func (s *connectionStats) HandleConn(_ context.Context, event stats.ConnStats) {
	switch event.(type) {
	case *stats.ConnBegin:
		s.opened.Add(1)
	case *stats.ConnEnd:
		s.ended <- struct{}{}
	}
}

func TestNewMTLSClosesConnectionsOfACancelledInvocation(t *testing.T) {
	polling := make(chan struct{}, 1)
	var sends, polls, cancels atomic.Int64
	statistics := &connectionStats{ended: make(chan struct{}, 16)}
	working := func() *a2apb.Task {
		return &a2apb.Task{Id: "task-1", ContextId: "context-1", Status: &a2apb.TaskStatus{State: a2apb.TaskState_TASK_STATE_WORKING}}
	}
	interceptor := func(ctx context.Context, request any, info *grpc.UnaryServerInfo, next grpc.UnaryHandler) (any, error) {
		switch info.FullMethod {
		case a2apb.A2AService_SendMessage_FullMethodName:
			sends.Add(1)
			return &a2apb.SendMessageResponse{Payload: &a2apb.SendMessageResponse_Task{Task: working()}}, nil
		case a2apb.A2AService_GetTask_FullMethodName:
			if polls.Add(1) < 3 {
				return working(), nil
			}
			polling <- struct{}{}
			<-ctx.Done()
			return nil, ctx.Err()
		case a2apb.A2AService_CancelTask_FullMethodName:
			cancels.Add(1)
			return working(), nil
		default:
			return next(ctx, request)
		}
	}
	var executor a2asrv.AgentExecutor
	invoker, handle, _ := nativeSDK(t, executor, grpc.ChainUnaryInterceptor(interceptor), grpc.StatsHandler(statistics))
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	finished := make(chan error, 1)
	go func() { _, err := invoker.Invoke(ctx, "builder", handle, stageRequest()); finished <- err }()
	select {
	case <-polling:
	case <-time.After(5 * time.Second):
		t.Fatal("third poll not reached")
	}
	cancel()
	select {
	case err := <-finished:
		assertPlannerCode(t, err, "planner_cancelled")
	case <-time.After(5 * time.Second):
		t.Fatal("invocation did not return")
	}
	select {
	case <-statistics.ended:
	case <-time.After(5 * time.Second):
		t.Fatal("invocation left its channel open")
	}
	if sends.Load() != 1 || polls.Load() != 3 || cancels.Load() != 1 {
		t.Fatalf("calls send=%d poll=%d cancel=%d", sends.Load(), polls.Load(), cancels.Load())
	}
	// Cancelling one HTTP/2 RPC preserves the verified channel for CancelTask.
	if statistics.opened.Load() != 1 {
		t.Fatalf("opened %d channels", statistics.opened.Load())
	}
}
