package privateartifacts

import (
	"bytes"
	"context"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

type observedTransferAdmission struct {
	artifacttransfer.Admission
	entered, released chan struct{}
}

func (a observedTransferAdmission) AcquireTransfer(ctx context.Context) (context.Context, func(), error) {
	ctx, release, err := a.Admission.AcquireTransfer(ctx)
	if err != nil {
		return ctx, nil, err
	}
	a.entered <- struct{}{}
	return ctx, func() { release(); a.released <- struct{}{} }, nil
}
func observedTransfers() observedTransferAdmission {
	return observedTransferAdmission{artifactpgtest.NewRuntime(), make(chan struct{}, 32), make(chan struct{}, 32)}
}
func awaitTransferSignals(t *testing.T, ch <-chan struct{}, count int) {
	t.Helper()
	for range count {
		select {
		case <-ch:
		case <-t.Context().Done():
			t.Fatal(t.Context().Err())
		case <-time.After(10 * time.Second):
			t.Fatal("native transfer owner did not join")
		}
	}
}
func uploadHeader(size uint64) *pb.ArtifactWriteFrame {
	return &pb.ArtifactWriteFrame{Frame: &pb.ArtifactWriteFrame_Header{Header: &pb.ArtifactWriteHeader{Artifact: &pb.ArtifactRef{Namespace: proto.String("builder"), Name: proto.String("payload")}, MediaType: proto.String("application/octet-stream"), Size: proto.Uint64(size), Precondition: &pb.ArtifactWriteHeader_CreateOnly{CreateOnly: true}}}}
}
func payloadReadTarget() *pb.ArtifactReadTarget {
	return &pb.ArtifactReadTarget{Artifact: &pb.ArtifactRef{Namespace: proto.String("inputs"), Name: proto.String("large")}}
}

func TestPrivateArtifactStalledUploadsReleaseTransferSlots(t *testing.T) {
	admission := observedTransfers()
	repository := newMemoryRepository()
	fixture := newAllocationNativeHarness(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(repository), Transfers: admission}})
	cancels := []context.CancelFunc{}
	for range 4 {
		ctx, cancel := context.WithCancel(fixture.Context)
		cancels = append(cancels, cancel)
		defer cancel()
		stream, err := fixture.Artifact.Write(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if err = stream.Send(uploadHeader(1)); err != nil {
			t.Fatal(err)
		}
	}
	awaitTransferSignals(t, admission.entered, 4)
	stream, err := fixture.Artifact.Write(fixture.Context)
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Send(uploadHeader(0))
	_, err = stream.CloseAndRecv()
	if status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("fifth native upload: %v", err)
	}
	for _, cancel := range cancels {
		cancel()
	}
	awaitTransferSignals(t, admission.released, 4)
	stream, err = fixture.Artifact.Write(fixture.Context)
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Send(uploadHeader(0))
	reply, err := stream.CloseAndRecv()
	if err != nil || reply.GetSize() != 0 {
		t.Fatalf("upload after cancellation: %v %v", reply, err)
	}
	awaitTransferSignals(t, admission.released, 1)
	repository.mu.Lock()
	defer repository.mu.Unlock()
	if repository.writeCalls != 1 {
		t.Fatalf("partial uploads reached storage: %d", repository.writeCalls)
	}
}

func TestPrivateArtifactStalledDownloadsReleaseTransferSlots(t *testing.T) {
	admission := observedTransfers()
	repository := newMemoryRepository()
	repository.seed("run-a", "inputs", "large", "revision-a", bytes.Repeat([]byte("x"), 8<<20))
	fixture := newAllocationNativeHarness(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(repository), Transfers: admission}})
	cancels := []context.CancelFunc{}
	for range 4 {
		ctx, cancel := context.WithCancel(fixture.Context)
		cancels = append(cancels, cancel)
		defer cancel()
		if _, err := fixture.Artifact.Read(ctx, payloadReadTarget()); err != nil {
			t.Fatal(err)
		}
	}
	awaitTransferSignals(t, admission.entered, 4)
	fifth, err := fixture.Artifact.Read(fixture.Context, payloadReadTarget())
	if err != nil {
		t.Fatal(err)
	}
	_, err = fifth.Recv()
	if status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("fifth native download: %v", err)
	}
	for _, cancel := range cancels {
		cancel()
	}
	awaitTransferSignals(t, admission.released, 4)
	stream, err := fixture.Artifact.Read(fixture.Context, payloadReadTarget())
	if err != nil {
		t.Fatal(err)
	}
	if _, err = stream.Recv(); err != nil {
		t.Fatalf("download after cancellation: %v", err)
	}
	// The fixture owns and joins the final download through its context cleanup.
}

type nativeDeadlineWriter struct {
	http.ResponseWriter
	readArmed, writeArmed *atomic.Bool
}

func (w *nativeDeadlineWriter) Flush()                      { w.ResponseWriter.(http.Flusher).Flush() }
func (w *nativeDeadlineWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
func (w *nativeDeadlineWriter) SetReadDeadline(d time.Time) error {
	w.readArmed.Store(!d.IsZero())
	return http.NewResponseController(w.ResponseWriter).SetReadDeadline(d)
}
func (w *nativeDeadlineWriter) SetWriteDeadline(d time.Time) error {
	w.writeArmed.Store(!d.IsZero())
	return http.NewResponseController(w.ResponseWriter).SetWriteDeadline(d)
}

func TestPrivateArtifactSlowStorageOutlivesTheClientDeadline(t *testing.T) {
	repository := newMemoryRepository()
	repository.writeStarted = make(chan struct{})
	repository.continueWrite = make(chan struct{})
	var readArmed, writeArmed atomic.Bool
	fixture := newAllocationNativeHarness(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(repository)}}, func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			next.ServeHTTP(&nativeDeadlineWriter{w, &readArmed, &writeArmed}, r)
		})
	})
	stream, err := fixture.Artifact.Write(fixture.Context)
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Send(uploadHeader(0))
	done := make(chan error, 1)
	go func() { _, err := stream.CloseAndRecv(); done <- err }()
	waitArtifactSignal(t, repository.writeStarted, "native Artifact storage")
	if readArmed.Load() || writeArmed.Load() {
		t.Error("payload socket deadline leaked into storage")
	}
	close(repository.continueWrite)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
