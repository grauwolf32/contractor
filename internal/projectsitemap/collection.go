package projectsitemap

import (
	"context"
	"slices"
	"time"
)

// CollectionIngress is allocation-private transient transport metadata.
// Headers and bodies are absent. The ordinary mutation projector normalizes
// metadata before identities, receipt digests or Artifact bytes are constructed.
type CollectionIngress struct {
	Kind         string        `json:"kind" strictjson:"required"`
	Observations []Observation `json:"observations" strictjson:"required"`
}

// CollectAllocation accepts selected HTTP and Caido metadata. Go owns source
// identity and reduction; a provider row that cannot be represented is a bounded
// omission and must not discard its otherwise valid siblings.
func (s *Service) CollectAllocation(ctx context.Context, access AllocationAccess, idempotencyKey string, input CollectionIngress) (Result, error) {
	if !access.Policy.CollectObservations || !slices.Contains([]string{"http", "caido"}, input.Kind) || input.Observations == nil || len(input.Observations) == 0 || len(input.Observations) > MaxBatch || len(jsonBytes(input)) > MaxBatchBytes {
		return Result{}, ErrInvalid
	}
	authority := observationAuthority(access, input.Kind)
	command := Mutation{Operation: "observe", Observations: input.Observations}
	if input.Kind == "caido" {
		command.Observations = make([]Observation, 0, len(input.Observations))
		var omitted uint64
		for _, observation := range input.Observations {
			projected, valid := caidoObservation(observation)
			if !valid {
				omitted++
				continue
			}
			command.Observations = append(command.Observations, projected)
		}
		if omitted != 0 {
			command.Gaps = []Gap{{SourceID: authority.SourceID, Reason: "projection", Count: omitted}}
		}
	}
	return s.Mutate(ctx, authority, idempotencyKey, command)
}

func observationAuthority(access AllocationAccess, kind string) Authority {
	deployment := ""
	if access.Policy.Deployment != nil {
		deployment = *access.Policy.Deployment
	}
	return Authority{OwnerID: access.OwnerID, ProjectID: access.ProjectID, RunID: access.RunID,
		SourceID: key(kind, deployment), Kind: kind, Deployment: deployment, CanObserve: true}
}

func caidoObservation(observation Observation) (Observation, bool) {
	at, err := time.Parse(time.RFC3339Nano, observation.At)
	if err != nil || !methodPattern.MatchString(observation.Method) || !validOutcome(observation.Outcome, observation.Status) || len(observation.Parameters) > MaxBatch {
		return Observation{}, false
	}
	_, _, parameters, err := projectURL(observation.URL)
	if err != nil {
		return Observation{}, false
	}
	for _, name := range observation.Parameters {
		if !validLabel(name) {
			return Observation{}, false
		}
		parameters = append(parameters, name)
	}
	slices.Sort(parameters)
	if len(slices.Compact(parameters)) > MaxBatch {
		return Observation{}, false
	}
	observation.At = at.UTC().Format(time.RFC3339Nano)
	return observation, validTime(observation.At)
}
