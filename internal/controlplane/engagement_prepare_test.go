package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

type projectionGrantReader struct{ grant AllocationGrant }

func (r *projectionGrantReader) GetGrant(string) (AllocationGrant, error) { return r.grant, nil }

func TestSharedEngagementPreparePinsMetadataAndRechecksGrantAfterRead(t *testing.T) {
	fixture, err := contracts.DecodePrivateStrict[control.AllocationSpec](contractstest.ReadFixture(t, "valid", "allocation-spec-engagement-metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, outcome := range []string{"success", "fenced", "released", "storage-failure", "cancelled"} {
		t.Run(outcome, func(t *testing.T) {
			template := testTemplate(t)
			lease := time.Now().Add(time.Minute).UTC().Truncate(time.Microsecond)
			var received *control.PrepareAllocationRequest
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				received = &control.PrepareAllocationRequest{}
				if err := json.NewDecoder(r.Body).Decode(received); err != nil {
					t.Error(err)
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(control.PrepareAllocationResponse{APIVersion: contracts.APIVersion, WorkerHandle: contracts.WorkerHandle{
					AllocationID: "allocation", AgentTemplateRef: template.Ref, WorkerRuntimeRef: template.Runtime, LeaseExpiresAt: lease,
					AgentCard: testAgentCard("worker", "allocation", serverURL(r)+"/private/v1/allocations/allocation/a2a"),
				}})
			}))
			defer server.Close()
			reservation := testReservation("allocation", "worker", server.URL, server.URL, template, lease)
			grants := &projectionGrantReader{grant: reservation.Grant}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			calls := 0
			client, err := NewRuntimeControlClient(server.Client())
			if err != nil {
				t.Fatal(err)
			}
			client.WithEngagementResolver(func(ctx context.Context, selected Reservation, _ runtimesettings.WorkerExecutionSettings) (*engagementcontract.AllocationProjection, error) {
				calls++
				if selected.Grant.RunID != reservation.Grant.RunID || selected.Grant.StageExecutionID != reservation.Grant.StageExecutionID {
					t.Fatal("untrusted Run/Stage selected")
				}
				switch outcome {
				case "fenced":
					grants.grant.WriteFenced = true
				case "released":
					grants.grant.RuntimeInstanceID = "replacement"
				case "storage-failure":
					return nil, errors.New("storage unavailable")
				case "cancelled":
					cancel()
				}
				return fixture.Engagement, nil
			}, grants)
			_, err = client.Prepare(ctx, reservation, runtimesettings.WorkerExecutionSettings{ModelPolicy: template.ModelPolicy, RuntimeSettings: testRuntimeSettings(), ResolvedRuntimeConfigProvenance: testRuntimeProvenance()})
			if calls != 1 {
				t.Fatalf("projection reads=%d", calls)
			}
			if outcome != "success" {
				if err == nil || received != nil {
					t.Fatalf("outcome %s dispatched despite invalid authority: %v", outcome, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if received == nil || !reflect.DeepEqual(received.Spec.Engagement, fixture.Engagement) {
				t.Fatal("pinned metadata changed")
			}
		})
	}
}

func TestSharedControlledPrepareMakesOnlyOrdinaryPrepareRPC(t *testing.T) {
	fixture, err := contracts.DecodePrivateStrict[control.AllocationSpec](contractstest.ReadFixture(t, "valid", "engagement-allocation-preflight.json"))
	if err != nil {
		t.Fatal(err)
	}
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute).UTC().Truncate(time.Microsecond)
	var calls []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls = append(calls, r.URL.Path)
		if !strings.HasSuffix(r.URL.Path, "/prepare") {
			t.Error("unexpected target control RPC", r.URL.Path)
			w.WriteHeader(500)
			return
		}
		var request control.PrepareAllocationRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		if !reflect.DeepEqual(request.Spec.Engagement, fixture.Engagement) {
			t.Error("HTTP authority changed")
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(control.PrepareAllocationResponse{APIVersion: contracts.APIVersion, WorkerHandle: contracts.WorkerHandle{AllocationID: "allocation", AgentTemplateRef: template.Ref, WorkerRuntimeRef: template.Runtime, LeaseExpiresAt: lease, AgentCard: testAgentCard("worker", "allocation", serverURL(r)+"/private/v1/allocations/allocation/a2a")}})
	}))
	defer server.Close()
	reservation := testReservation("allocation", "worker", server.URL, server.URL, template, lease)
	reservation.Grant.RuntimeInstanceID = "instance"
	client, err := NewRuntimeControlClient(server.Client())
	if err != nil {
		t.Fatal(err)
	}
	client.WithEngagementResolver(func(context.Context, Reservation, runtimesettings.WorkerExecutionSettings) (*engagementcontract.AllocationProjection, error) {
		return fixture.Engagement, nil
	}, &projectionGrantReader{grant: reservation.Grant})
	handle, err := client.Prepare(t.Context(), reservation, runtimesettings.WorkerExecutionSettings{ModelPolicy: template.ModelPolicy, RuntimeSettings: testRuntimeSettings(), ResolvedRuntimeConfigProvenance: testRuntimeProvenance()})
	if err != nil || handle.AllocationID != "allocation" || len(calls) != 1 || !strings.HasSuffix(calls[0], "/prepare") {
		t.Fatal(handle, calls, err)
	}
}
