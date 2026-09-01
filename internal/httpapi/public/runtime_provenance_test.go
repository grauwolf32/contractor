package public

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestStageRuntimeConfigurationReadModelIsSafeAndHistorical(t *testing.T) {
	releasedAt := time.Date(2026, time.August, 31, 12, 0, 0, 0, time.UTC)
	digest := "sha256:" + strings.Repeat("1", 64)
	configuration := &runstore.AllocationRuntimeConfiguration{
		Origins: runtimeconfig.ResolvedRuntimeConfigOrigins{
			WorkerTelemetry: &runtimeconfig.RuntimeFieldOrigin{
				Layer:   runtimeconfig.LayerAgentLabels,
				Configs: []runtimeconfig.Ref{{Name: "site-debug", Version: "2", Digest: digest}},
			},
		},
		Provenance: contracts.ResolvedRuntimeConfigProvenanceV2{
			AgentLabels: []contracts.RuntimeLabelBindingProvenanceV2{{
				Label: "debug", BindingRevision: 7,
				Config: contracts.RuntimeConfigRefV2{Name: "site-debug", Version: "2", Digest: digest},
			}},
			RuntimeAdapters: []contracts.RuntimeAdapterRef{contracts.RuntimeAdapterOTLPHTTP},
		},
	}
	projection := stageRuntimeConfigurationReadModel([]runstore.StageAllocation{
		{
			AllocationID: "allocation-secret-physical-identity", LogicalAgentName: "reviewer",
			RuntimeAgentID: "principal-secret-physical-identity", RuntimeAgentInstanceID: "runtime-secret-physical-identity",
			RuntimeConfiguration: configuration, ReleaseCompletedAt: &releasedAt,
		},
	})
	if projection == nil || len(projection.Allocations) != 1 {
		t.Fatalf("projection = %+v", projection)
	}
	allocation := projection.Allocations[0]
	if allocation.LogicalAgent != "reviewer" || allocation.Status != "released" ||
		len(allocation.AgentLabels) != 1 || allocation.AgentLabels[0].BindingRevision != "7" ||
		len(allocation.RuntimeAdapters) != 1 {
		t.Fatalf("safe allocation = %+v", allocation)
	}
	encoded, err := json.Marshal(projection)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"allocation-secret", "principal-secret", "runtime-secret", "endpoint", "token", "password"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("projection leaked %q: %s", forbidden, encoded)
		}
	}
}

func TestStageRuntimeConfigurationReadModelWaitsForCommittedProvenance(t *testing.T) {
	if projection := stageRuntimeConfigurationReadModel([]runstore.StageAllocation{{
		LogicalAgentName: "reviewer", RuntimeConfiguration: nil,
	}}); projection != nil {
		t.Fatalf("uncommitted provenance became visible: %+v", projection)
	}
}
