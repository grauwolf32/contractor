package projectsitemap

import (
	"context"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// CollectionIngress is allocation-private transient transport metadata.
// Headers and bodies are absent. The ordinary mutation projector normalizes
// metadata before identities, receipt digests or Artifact bytes are constructed.
type CollectionIngress struct {
	Observations []HTTPObservation `json:"observations" strictjson:"required"`
}

type HTTPObservation struct {
	Observation
	SessionGeneration string `json:"sessionGeneration" strictjson:"required"`
}

// CollectHTTPAllocation is the only collector RPC. Go owns projection and
// coalescing in the existing reducer; Runtime only batches private input.
func (s *Service) CollectHTTPAllocation(ctx context.Context, access AllocationAccess, idempotencyKey string, input CollectionIngress) (Result, error) {
	if !access.Policy.CollectObservations || input.Observations == nil || len(input.Observations) == 0 || len(input.Observations) > MaxBatch || len(jsonBytes(input)) > MaxBatchBytes {
		return Result{}, ErrInvalid
	}
	deployment := ""
	if access.Policy.Deployment != nil {
		deployment = *access.Policy.Deployment
	}
	authority := Authority{OwnerID: access.OwnerID, ProjectID: access.ProjectID, RunID: access.RunID,
		SourceID: key("http", deployment), Kind: "http",
		Deployment: deployment, CanObserve: true}
	command := Mutation{Operation: "observe", Observations: make([]Observation, len(input.Observations)), sessionGenerations: make([]string, len(input.Observations))}
	for index, observation := range input.Observations {
		if !contracts.ValidResourceID(observation.SessionGeneration) {
			return Result{}, ErrInvalid
		}
		command.Observations[index] = observation.Observation
		command.sessionGenerations[index] = observation.SessionGeneration
	}
	return s.Mutate(ctx, authority, idempotencyKey, command)
}
