package a2a

import (
	"context"
	"iter"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	sdk "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
)

func TestOfficialSDKAcceptsCompactUTF8WorkerResult(t *testing.T) {
	want := successResult()
	want.Result.Result = strings.Repeat("<&>", 21000)
	executor := a2asrv.AgentExecutorFunc(func(
		_ context.Context, _ *a2asrv.ExecutorContext,
	) iter.Seq2[sdk.Event, error] {
		return func(yield func(sdk.Event, error) bool) { yield(resultMessage(want), nil) }
	})
	server := httptest.NewServer(a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor)))
	defer server.Close()
	invoker, err := New(server.Client(), Options{PollInterval: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	got, err := invoker.Invoke(t.Context(), "builder", workerHandle(server.URL), stageRequest())
	if err != nil {
		t.Fatal(err)
	}
	if got.Result == nil || got.Result.Result != want.Result.Result || got.InvocationID != want.InvocationID {
		t.Fatal("SDK round trip changed Worker text or identity")
	}
}
