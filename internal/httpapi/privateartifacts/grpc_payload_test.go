package privateartifacts

import (
	"bytes"
	"io"
	"strconv"
	"testing"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

func TestArtifactNativeStreamRejectsMalformedUploadBeforePublication(t *testing.T) {
	for _, test := range []struct {
		name   string
		frames func() []*pb.ArtifactWriteFrame
		want   codes.Code
	}{
		{"short-eof", func() []*pb.ArtifactWriteFrame { return []*pb.ArtifactWriteFrame{uploadHeader(1)} }, codes.InvalidArgument},
		{"extra", func() []*pb.ArtifactWriteFrame {
			return []*pb.ArtifactWriteFrame{uploadHeader(0), {Frame: &pb.ArtifactWriteFrame_Chunk{Chunk: []byte("x")}}}
		}, codes.InvalidArgument},
		{"repeated-header", func() []*pb.ArtifactWriteFrame { return []*pb.ArtifactWriteFrame{uploadHeader(1), uploadHeader(1)} }, codes.InvalidArgument},
		{"wrong-size", func() []*pb.ArtifactWriteFrame {
			return []*pb.ArtifactWriteFrame{uploadHeader(1), {Frame: &pb.ArtifactWriteFrame_Chunk{Chunk: []byte("xy")}}}
		}, codes.InvalidArgument},
		{"aggregate", func() []*pb.ArtifactWriteFrame {
			return []*pb.ArtifactWriteFrame{uploadHeader(artifacts.MaxPayloadSize + 1)}
		}, codes.ResourceExhausted},
		{"absent-precondition", func() []*pb.ArtifactWriteFrame {
			h := uploadHeader(0)
			h.GetHeader().Precondition = nil
			return []*pb.ArtifactWriteFrame{h}
		}, codes.InvalidArgument},
		{"false-create", func() []*pb.ArtifactWriteFrame {
			h := uploadHeader(0)
			h.GetHeader().Precondition = &pb.ArtifactWriteHeader_CreateOnly{CreateOnly: false}
			return []*pb.ArtifactWriteFrame{h}
		}, codes.InvalidArgument},
		{"blank-revision", func() []*pb.ArtifactWriteFrame {
			h := uploadHeader(0)
			h.GetHeader().Precondition = &pb.ArtifactWriteHeader_ExpectedRevision{ExpectedRevision: " "}
			return []*pb.ArtifactWriteFrame{h}
		}, codes.InvalidArgument},
		{"unknown-header", func() []*pb.ArtifactWriteFrame {
			h := uploadHeader(0)
			fixtureUnknown(h.GetHeader())
			return []*pb.ArtifactWriteFrame{h}
		}, codes.InvalidArgument},
	} {
		t.Run(test.name, func(t *testing.T) {
			repository := newMemoryRepository()
			fixture := newAllocationNativeHarness(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(repository)}})
			stream, err := fixture.Artifact.Write(fixture.Context)
			if err != nil {
				t.Fatal(err)
			}
			for _, frame := range test.frames() {
				if err = stream.Send(frame); err != nil && err != io.EOF {
					t.Fatal(err)
				}
			}
			_, err = stream.CloseAndRecv()
			if status.Code(err) != test.want {
				t.Fatalf("malformed native upload=%v", err)
			}
			repository.mu.Lock()
			defer repository.mu.Unlock()
			if repository.writeCalls != 0 {
				t.Fatal("malformed stream published a revision")
			}
		})
	}
}

func TestArtifactNativeStreamByteExactRoundTripAndOpaqueCAS(t *testing.T) {
	for _, size := range []int{0, 1, runtimewire.BulkChunkBytes, 2*runtimewire.BulkChunkBytes + 7} {
		t.Run(strconv.Itoa(size), func(t *testing.T) {
			repository := newMemoryRepository()
			opaque := "revision,with opaque punctuation"
			repository.seed("run-a", "builder", "payload", opaque, []byte("old"))
			fixture := newAllocationNativeHarness(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(repository)}})
			expected := uploadHeader(uint64(size))
			expected.GetHeader().Precondition = &pb.ArtifactWriteHeader_ExpectedRevision{ExpectedRevision: opaque}
			stream, err := fixture.Artifact.Write(fixture.Context)
			if err != nil {
				t.Fatal(err)
			}
			_ = stream.Send(expected)
			payload := bytes.Repeat([]byte{0, 255, 1, 2}, (size+3)/4)[:size]
			for offset := 0; offset < size; {
				length := runtimewire.BulkChunkSize(size - offset)
				if err = stream.Send(&pb.ArtifactWriteFrame{Frame: &pb.ArtifactWriteFrame_Chunk{Chunk: payload[offset : offset+length]}}); err != nil {
					t.Fatal(err)
				}
				offset += length
			}
			receipt, err := stream.CloseAndRecv()
			if err != nil {
				t.Fatal(err)
			}
			read, err := fixture.Artifact.Read(fixture.Context, &pb.ArtifactReadTarget{Artifact: receipt.Artifact})
			if err != nil {
				t.Fatal(err)
			}
			first, err := read.Recv()
			if err != nil || !proto.Equal(first.GetHeader(), receipt) {
				t.Fatalf("metadata=%v %v", first, err)
			}
			var received []byte
			for {
				frame, err := read.Recv()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatal(err)
				}
				if len(frame.GetChunk()) != runtimewire.BulkChunkSize(size-len(received)) {
					t.Fatal("noncanonical output chunk")
				}
				received = append(received, frame.GetChunk()...)
			}
			if !bytes.Equal(received, payload) {
				t.Fatal("native payload bytes changed")
			}
		})
	}
}
