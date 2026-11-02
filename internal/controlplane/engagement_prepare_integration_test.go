//go:build integration

package controlplane

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
)

// The actual Runtime HTTP reader must admit only metadata, preserve replay, and
// create no target/model work during prepare. The native harness uses real mTLS
// and a separate Python Runtime process; its local tool factory records work.
func TestCrossLanguageMTLSEngagementMetadataPrepare(t *testing.T) {
	policy := engagementcontract.Policy{Version: "1", ActiveChecks: "automatic", RequiredCapabilities: []string{engagementcontract.Capability}, OptionalCapabilities: []string{}}
	digest, err := policy.Digest()
	if err != nil {
		t.Fatal(err)
	}
	revision := "engagement-r1"
	metadata := &engagementcontract.AllocationProjection{
		Schema: "contractor.engagement.allocation.v1", AuditID: "audit-native",
		Engagement:       contracts.ArtifactRef{Namespace: "engagements", Name: "native", Revision: &revision},
		EngagementDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", PolicyDigest: digest,
		AuthorizedBy: "owner-native", AuthorizedAt: time.Date(2026, 10, 28, 12, 0, 0, 0, time.UTC), AccessMode: "blackbox",
		Policy: policy,
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
	projection := *invalid.Engagement
	projection.HTTP = &engagementcontract.HTTPAssignment{}
	invalid.Engagement = &projection
	var rejected control.PrepareAllocationResponse
	err = h.client.postJSON(h.ctx, h.reservation.ControlURL, invalid.AllocationID, h.reservation.Grant.RuntimeAgentID, "prepare",
		control.PrepareAllocationRequest{APIVersion: contracts.APIVersion, Spec: invalid}, &rejected, true)
	var apiError *RuntimeAPIError
	if !errors.As(err, &apiError) || apiError.StatusCode != http.StatusUnprocessableEntity || apiError.Code != "invalid_request" || apiError.Retryable {
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
