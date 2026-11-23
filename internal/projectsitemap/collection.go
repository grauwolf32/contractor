package projectsitemap

import (
	"context"
	"slices"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// CollectionIngress is the allocation-private normalized observation boundary.
// Adapters own source formats and omissions. The Server owns generic validation,
// source identity, grouping, exact evidence metadata and publication.
type CollectionIngress struct {
	Kind         string                 `json:"kind" strictjson:"required"`
	Observations []Observation          `json:"observations" strictjson:"required"`
	Gaps         []CollectionGap        `json:"gaps,omitempty"`
	Artifact     *contracts.ArtifactRef `json:"artifact,omitempty"`
}

// CollectionGap carries a factual aggregate without caller-supplied provenance.
type CollectionGap struct {
	Reason string `json:"reason" strictjson:"required"`
	Count  uint64 `json:"count" strictjson:"required"`
}

func (input CollectionIngress) Validate() error {
	if !observedKind(input.Kind) || input.Observations == nil || len(input.Observations) > MaxBatch || len(input.Gaps) > 16 || len(jsonBytes(input)) > MaxBatchBytes {
		return ErrInvalid
	}
	if input.Artifact != nil && input.Artifact.ValidateExact() != nil {
		return ErrInvalid
	}
	seen := map[string]bool{}
	for _, gap := range input.Gaps {
		if !validGapReason(gap.Reason) || gap.Count == 0 || gap.Count > MaxOmissionCount || seen[gap.Reason] {
			return ErrInvalid
		}
		seen[gap.Reason] = true
	}
	for _, observation := range input.Observations {
		if err := validateCollectedObservation(observation); err != nil {
			return err
		}
	}
	return nil
}

// CollectAllocation receives a fully prepared batch. Its authenticated adapter
// checks automatic collection or explicit tool grants before this boundary.
// Every source has identical validation; invalid rows reject the whole batch.
func (s *Service) CollectAllocation(ctx context.Context, access AllocationAccess, idempotencyKey string, input CollectionIngress) (Result, error) {
	if input.Validate() != nil || access.Policy.Validate() != nil {
		return Result{}, ErrInvalid
	}
	authority := observationAuthority(access, input.Kind)
	command := Mutation{Operation: "observe", Observations: input.Observations, Artifact: input.Artifact}
	for _, gap := range input.Gaps {
		command.Gaps = append(command.Gaps, Gap{SourceID: authority.SourceID, Reason: gap.Reason, Count: gap.Count})
	}
	slices.SortFunc(command.Gaps, func(a, b Gap) int { return strings.Compare(a.Reason, b.Reason) })
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

func validateCollectedObservation(observation Observation) error {
	if _, err := projectAnnotation(observation.Note, observation.Tags, Authority{}); err != nil {
		return err
	}
	if !methodPattern.MatchString(observation.Method) || !validOutcome(observation.Outcome, observation.Status) || !validTime(observation.At) || len(observation.Parameters) > MaxBatch {
		return ErrInvalid
	}
	_, _, parameters, err := projectURL(observation.URL)
	if err != nil {
		return err
	}
	for _, name := range observation.Parameters {
		if !validLabel(name) {
			return ErrInvalid
		}
		parameters = append(parameters, name)
	}
	slices.Sort(parameters)
	if len(slices.Compact(parameters)) > MaxBatch {
		return ErrInvalid
	}
	return nil
}
