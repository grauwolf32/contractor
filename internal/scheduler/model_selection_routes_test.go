package scheduler

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestReservationRoutesUseEffectiveSummaryModel(t *testing.T) {
	reservation := controlplane.Reservation{
		AgentTemplate: contracts.ResolvedAgentTemplate{Summarizer: &contracts.WorkerSummarizerConfig{ContextWindowRatio: 0.9}},
		ResolvedRuntimeConfig: &runtimeconfig.ResolvedRuntimeConfig{
			ModelPolicy:           contracts.ResolvedModelPolicy{Model: "selected-worker"},
			SummarizerModelPolicy: &contracts.ResolvedModelPolicy{Model: "selected-summary"},
			LLMGateway:            llmgateway.ResolvedLLMGatewayConfig{Ref: llmgateway.LLMGatewayConfigRef{Digest: "gateway"}},
		},
	}
	routes := reservationModelRoutes("owner", reservation)
	if len(routes) != 2 || routes[0].Model != "selected-worker" || routes[1].Model != "selected-summary" {
		t.Fatalf("effective recovery routes = %+v", routes)
	}
	reservation.ResolvedRuntimeConfig.SummarizerModelPolicy.Model = "selected-worker"
	if len(reservationModelRoutes("owner", reservation)) != 1 {
		t.Fatal("identical main/summary route was registered twice")
	}
}

func TestReservationRoutesIncludeTheCompactorModel(t *testing.T) {
	reservation := controlplane.Reservation{
		ResolvedRuntimeConfig: &runtimeconfig.ResolvedRuntimeConfig{
			ModelPolicy:          contracts.ResolvedModelPolicy{Model: "worker"},
			CompactorModelPolicy: &contracts.ResolvedModelPolicy{Model: "compactor"},
			LLMGateway:           llmgateway.ResolvedLLMGatewayConfig{Ref: llmgateway.LLMGatewayConfigRef{Digest: "gateway"}},
		},
	}
	routes := reservationModelRoutes("owner", reservation)
	if len(routes) != 2 || routes[0].Model != "worker" || routes[1].Model != "compactor" ||
		routes[1].GatewayDigest != "gateway" {
		t.Fatalf("compact@1 recovery routes = %+v", routes)
	}
	reservation.ResolvedRuntimeConfig.CompactorModelPolicy.Model = "worker"
	if len(reservationModelRoutes("owner", reservation)) != 1 {
		t.Fatal("identical main/compactor route was registered twice")
	}
}
