package privateartifacts

import (
	"github.com/grauwolf32/contractor/internal/artifacts"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"strings"
	"testing"
)

// The allocation principal is checked before protobuf decoding. Once authority
// is established, malformed names reject the typed header before payload/store.
func TestPrivateArtifactNamesRejectBeforeStore(t *testing.T) {
	repository := newMemoryRepository()
	fixture := newAllocationNativeHarness(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(repository)}})
	for _, name := range []string{"review notes", "отчет", "_report", strings.Repeat("a", 129)} {
		for _, namespace := range []bool{false, true} {
			frame := uploadHeader(1)
			if namespace {
				frame.GetHeader().Artifact.Namespace = proto.String(name)
			} else {
				frame.GetHeader().Artifact.Name = proto.String(name)
			}
			stream, err := fixture.Artifact.Write(fixture.Context)
			if err != nil {
				t.Fatal(err)
			}
			_ = stream.Send(frame)
			_, err = stream.CloseAndRecv()
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("malformed name=%q: %v", name, err)
			}
		}
	}
	ctx := metadata.NewOutgoingContext(fixture.Context, metadata.Pairs("x-contractor-runtime-instance-id", "runtime-1", "x-contractor-allocation-id", "allocation-missing"))
	stream, err := fixture.Artifact.Write(ctx)
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Send(uploadHeader(1))
	_, err = stream.CloseAndRecv()
	if status.Code(err) != codes.NotFound {
		t.Fatalf("unknown allocation=%v", err)
	}
	repository.mu.Lock()
	defer repository.mu.Unlock()
	if repository.writeCalls != 0 || repository.readCalls != 0 {
		t.Fatal("invalid request reached storage")
	}
}
