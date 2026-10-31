package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

func TestSharedEngagementPrivatePrepareRechecksAuthorityRedactsAndClearsMaterial(t *testing.T) {
	for _, mode := range []string{"fenced", "provider-failure", "cancelled", "agent-card-secret"} {
		t.Run(mode, func(t *testing.T) {
			fixture, err := contracts.DecodePrivateStrict[control.AllocationSpec](contractstest.ReadFixture(t, "valid", "engagement-allocation-preflight.json"))
			if err != nil {
				t.Fatal(err)
			}
			ref := runtimesettings.RuntimeCredentialRef{CredentialID: "held-target", Kind: runtimesettings.RuntimeCredentialTargetBearer}
			fixture.Engagement.HTTP.Identities = append(fixture.Engagement.HTTP.Identities, engagementcontract.Identity{ID: "account-a", Kind: "bearer", Origins: []string{"https://target.example"}, Credential: &ref})
			material := map[string]runtimesettings.HTTPIdentityMaterial{"account-a": {Credential: ref, Fields: map[string]contracts.SecretString{"token": contracts.NewSecretString("private-target-canary")}}}
			template := testTemplate(t)
			lease := time.Now().Add(time.Minute).UTC().Truncate(time.Microsecond)
			received := false
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				received = true
				var request control.PrepareAllocationRequest
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Fatal(err)
				}
				if request.Spec.RuntimeSettings.HTTPIdentities["account-a"].Fields["token"].Reveal() != "private-target-canary" {
					t.Error("private prepare did not reveal assigned material")
				}
				public, _ := json.Marshal(request.Spec.Engagement)
				if strings.Contains(string(public), "private-target-canary") {
					t.Error("private material entered public projection")
				}
				card := testAgentCard("worker", "allocation", serverURL(r)+"/private/v1/allocations/allocation/a2a")
				card["description"] = "private-target-canary"
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(control.PrepareAllocationResponse{APIVersion: contracts.APIVersion, WorkerHandle: contracts.WorkerHandle{AllocationID: "allocation", AgentTemplateRef: template.Ref, WorkerRuntimeRef: template.Runtime, LeaseExpiresAt: lease, AgentCard: card}})
			}))
			defer server.Close()
			reservation := testReservation("allocation", "worker", server.URL, server.URL, template, lease)
			grants := &projectionGrantReader{grant: reservation.Grant}
			client, err := NewRuntimeControlClient(server.Client())
			if err != nil {
				t.Fatal(err)
			}
			defer client.Close()
			client.WithEngagementResolver(func(context.Context, Reservation, runtimesettings.WorkerExecutionSettings) (*engagementcontract.AllocationProjection, error) {
				return fixture.Engagement, nil
			}, grants)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			client.WithEngagementMaterials(func(ctx context.Context, projection *engagementcontract.AllocationProjection) (map[string]runtimesettings.HTTPIdentityMaterial, error) {
				if projection != fixture.Engagement {
					t.Fatal("untrusted material assignment")
				}
				if _, ok := ctx.Deadline(); !ok {
					t.Fatal("private material lookup is unbounded")
				}
				switch mode {
				case "fenced":
					grants.grant.WriteFenced = true
				case "provider-failure":
					return material, errors.New("provider unavailable")
				case "cancelled":
					cancel()
				}
				return material, nil
			})
			_, err = client.Prepare(ctx, reservation, runtimesettings.WorkerExecutionSettings{ModelPolicy: testWorkerPolicy(t), RuntimeSettings: testRuntimeSettings(), ResolvedRuntimeConfigProvenance: testRuntimeProvenance()})
			if err == nil || strings.Contains(err.Error(), "private-target-canary") {
				t.Fatalf("unsafe material outcome: %v", err)
			}
			if len(material) != 0 {
				t.Fatal("private material retained after prepare failure")
			}
			if received != (mode == "agent-card-secret") {
				t.Fatalf("private prepare received=%v mode=%s err=%v", received, mode, err)
			}
		})
	}
}
