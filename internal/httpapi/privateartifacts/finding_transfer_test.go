package privateartifacts

import (
	"context"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

type findingBudgetIntake struct {
	FindingIntake
	check func(context.Context)
}

func (i *findingBudgetIntake) FindReplay(ctx context.Context, g controlplane.AllocationGrant, s findingintake.Submission) (findingintake.Receipt, bool, error) {
	i.check(ctx)
	return i.FindingIntake.FindReplay(ctx, g, s)
}

func TestFindingTransferBoundsSocketAndStorageAndReleasesBeforeResponse(t *testing.T) {
	admission := observedTransfers()
	var readArmed, writeArmed atomic.Bool
	calls := 0
	intake := &findingBudgetIntake{FindingIntake: &fakeFindingIntake{}, check: func(ctx context.Context) {
		calls++
		if _, ok := ctx.Deadline(); !ok {
			t.Error("finding storage has no independent deadline")
		}
		if readArmed.Load() || writeArmed.Load() {
			t.Error("finding storage inherited payload socket deadline")
		}
	}}
	owner := &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(newMemoryRepository()), Transfers: admission, Findings: intake}}
	fixture := newAllocationNativeHarness(t, owner, func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			next.ServeHTTP(&nativeDeadlineWriter{w, &readArmed, &writeArmed}, r)
		})
	})
	adapter := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reader, err := payloadFixtureFrames(r, strings.Split(strings.TrimPrefix(r.URL.Path, "/private/v1/allocations/"), "/"), pb.Artifact_SubmitFinding_FullMethodName)
		if err != nil {
			t.Fatal(err)
		}
		raw, err := io.ReadAll(reader)
		if err != nil {
			t.Fatal(err)
		}
		frames, err := fixtureMessages(raw)
		if err != nil {
			t.Fatal(err)
		}
		stream, err := fixture.Artifact.SubmitFinding(fixture.Context)
		if err != nil {
			t.Fatal(err)
		}
		for _, raw := range frames {
			frame := new(pb.FindingSubmissionFrame)
			if err = proto.Unmarshal(raw, frame); err != nil {
				t.Fatal(err)
			}
			if err = stream.Send(frame); err != nil {
				t.Fatal(err)
			}
		}
		reply, err := stream.CloseAndRecv()
		if err != nil {
			t.Fatal(err)
		}
		var value map[string]any
		if err = runtimewire.DecodeDomain(reply, &value, runtimewire.BulkHeaderBytes, runtimewire.BulkFrameBytes); err != nil {
			t.Fatal(err)
		}
		httpStatus := 201
		if reply.GetReplayed() {
			httpStatus = 200
		}
		writeJSON(w, httpStatus, value)
	})
	for _, want := range []int{201, 200} {
		if response := postFinding(t, adapter, "deadline"); response.Code != want {
			t.Fatalf("native finding=%d %s", response.Code, response.Body.String())
		}
	}

	if calls != 2 {
		t.Fatalf("finding storage calls=%d", calls)
	}
	awaitTransferSignals(t, admission.released, 2)
}

func TestFindingTransferTimeoutReleasesAllSlots(t *testing.T) {
	admission := observedTransfers()
	intake := &fakeFindingIntake{}
	fixture := newAllocationNativeHarness(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(newMemoryRepository()), Transfers: admission, Findings: intake}})
	for range 4 {
		ctx, cancel := context.WithCancel(fixture.Context)
		stream, err := fixture.Artifact.SubmitFinding(ctx)
		if err != nil {
			t.Fatal(err)
		}
		err = stream.Send(&pb.FindingSubmissionFrame{Frame: &pb.FindingSubmissionFrame_Header{Header: &pb.FindingSubmissionHeader{ApiVersion: proto.String("contractor/v1alpha1"), InvocationId: proto.String("invocation-1"), SubmissionId: proto.String("submission-1"), EvidenceRefs: &pb.ListOfArtifactRef{}, ProposalSize: proto.Uint64(2)}}})
		if err != nil {
			t.Fatal(err)
		}
		awaitTransferSignals(t, admission.entered, 1)
		cancel()
		_, err = stream.CloseAndRecv()
		if status.Code(err) != codes.Canceled {
			t.Fatalf("cancelled finding stream=%v", err)
		}
		awaitTransferSignals(t, admission.released, 1)
	}
	if intake.submits != 0 {
		t.Fatal("partial finding reached intake")
	}
}
