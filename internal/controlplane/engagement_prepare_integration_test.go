//go:build integration

package controlplane

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc/codes"
)

// The actual Runtime gRPC reader must admit only metadata, preserve replay, and
// create no target/model work during prepare. The native harness uses real mTLS
// and a separate Python Runtime process; its local tool factory records work.
func TestCrossLanguageMTLSEngagementMetadataPrepare(t *testing.T) {
	revision := "engagement-r1"
	metadata := &engagementcontract.AllocationProjection{
		Schema:           "contractor.engagement.allocation.v1",
		Engagement:       contracts.ArtifactRef{Namespace: "engagements", Name: "native", Revision: &revision},
		EngagementDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		AuthorizedBy:     "owner-native", AuthorizedAt: time.Date(2026, 10, 28, 12, 0, 0, 0, time.UTC),
	}
	h := startNativeHarness(t, "native_successful_release_replay", func(h *nativeHarness) {
		h.client.preparation = allocationPreparationFunc(func(_ context.Context, _ Reservation, spec control.AllocationSpec, dispatch func(control.AllocationSpec) (contracts.WorkerHandle, error)) (contracts.WorkerHandle, error) {
			spec.Engagement = metadata
			return dispatch(spec)
		})
	})
	first := h.handle
	before := h.snapshot()
	invalid := control.AllocationSpec{
		APIVersion: contracts.APIVersion, AllocationID: h.reservation.Grant.AllocationID,
		RunID: h.reservation.Grant.RunID, StageExecutionID: h.reservation.Grant.StageExecutionID,
		LogicalAgentName: h.reservation.Grant.LogicalAgentName, Namespace: h.reservation.Grant.Namespace,
		WorkerSessionMode: h.reservation.WorkerSessionMode, ContextManagement: h.reservation.ContextManagement.Clone(),
		RunMetadataLabels: h.reservation.RunMetadataLabels.Clone(),
		LeaseExpiresAt:    wireTime(h.reservation.LeaseExpiresAt), AgentTemplate: h.reservation.AgentTemplate.Clone(),
		ResolvedSkills: []contracts.ResolvedSkill{}, ModelPolicy: h.settings.ModelPolicy.Clone(),
		RuntimeSettings: h.settings.RuntimeSettings, ResolvedRuntimeConfigProvenance: h.settings.ResolvedRuntimeConfigProvenance,
		Engagement: metadata,
	}
	if err := invalid.Validate(); err != nil {
		t.Fatalf("invalid process test baseline: %v", err)
	}
	request := &pb.PrepareAllocationRequest{}
	if err := runtimewire.Encode(control.PrepareAllocationRequest{APIVersion: contracts.APIVersion, Spec: invalid}, request); err != nil {
		t.Fatal(err)
	}
	defer runtimewire.Clear(request)
	request.Spec.Engagement.Http = &pb.HTTPAssignment{}
	channel, release, err := h.client.channels.Borrow(h.ctx, h.reservation.ControlURL, h.reservation.Grant.RuntimeAgentID)
	if err != nil {
		t.Fatal(err)
	}
	_, rpcErr := pb.NewRuntimeLifecycleClient(channel).Prepare(h.ctx, request)
	release()
	err = decodeLifecycleError(rpcErr)
	var apiError *RuntimeAPIError
	if !errors.As(err, &apiError) || apiError.RPCStatus != codes.InvalidArgument || apiError.Code != "invalid_request" || apiError.Retryable {
		t.Fatalf("Runtime accepted target assignment or wrong failure: %v", err)
	}
	replay, err := h.client.Prepare(h.ctx, h.reservation, h.settings)
	if err != nil || !reflect.DeepEqual(first, replay) {
		t.Fatalf("prepare replay changed handle: %v", err)
	}
	after := h.snapshot()
	if !reflect.DeepEqual(before.Created, after.Created) || len(after.Dispatch) != 0 {
		t.Fatal("metadata prepare replay repeated creation or dispatched target work")
	}
	if main, finalizer := h.gateway.counts(); main != 0 || finalizer != 0 {
		t.Fatal("metadata prepare reached model Gateway")
	}
	h.finishAndRelease()
	h.assertSuccessfulReleaseReplay()
}
