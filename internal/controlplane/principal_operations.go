package controlplane

import (
	"context"
	"errors"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type RuntimeAgentPrincipalAvailability string

const (
	PrincipalAvailable                 RuntimeAgentPrincipalAvailability = "available"
	PrincipalOffline                   RuntimeAgentPrincipalAvailability = "offline"
	PrincipalBusy                      RuntimeAgentPrincipalAvailability = "busy"
	PrincipalSlotUnavailable           RuntimeAgentPrincipalAvailability = "slot_unavailable"
	PrincipalAdapterCapabilityMismatch RuntimeAgentPrincipalAvailability = "adapter_capability_mismatch"
)

type RuntimeAgentPrincipalProjection struct {
	Principal               runtimeconfig.RuntimeAgentPrincipal
	RequiredRuntimeAdapters []string
	MissingRuntimeAdapters  []string
	Availability            RuntimeAgentPrincipalAvailability
	Live                    *RuntimeAgentObservation
}

func (p RuntimeAgentPrincipalProjection) Validate() error {
	if len(p.RequiredRuntimeAdapters) > maximumRuntimeCapabilityRefs ||
		len(p.MissingRuntimeAdapters) > len(p.RequiredRuntimeAdapters) ||
		!validUniqueCapabilityRefs(p.RequiredRuntimeAdapters) ||
		!validUniqueCapabilityRefs(p.MissingRuntimeAdapters) {
		return errors.New("Runtime Agent principal adapter projection is invalid")
	}
	required := make(map[string]struct{}, len(p.RequiredRuntimeAdapters))
	for _, adapter := range p.RequiredRuntimeAdapters {
		required[adapter] = struct{}{}
	}
	for _, adapter := range p.MissingRuntimeAdapters {
		if _, ok := required[adapter]; !ok {
			return errors.New("Runtime Agent principal missing adapter projection is invalid")
		}
	}
	if p.Availability == PrincipalOffline {
		if p.Live != nil || len(p.MissingRuntimeAdapters) != 0 {
			return errors.New("offline Runtime Agent principal contains live facts")
		}
		return nil
	}
	if p.Live == nil || p.Live.Validate() != nil {
		return errors.New("live Runtime Agent principal observation is invalid")
	}
	switch p.Availability {
	case PrincipalAdapterCapabilityMismatch:
		if len(p.MissingRuntimeAdapters) == 0 {
			return errors.New("Runtime Agent adapter mismatch has no missing adapter")
		}
	case PrincipalAvailable:
		if len(p.MissingRuntimeAdapters) != 0 || p.Live.SlotState != SlotIdle {
			return errors.New("available Runtime Agent principal has incompatible live facts")
		}
	case PrincipalBusy:
		if len(p.MissingRuntimeAdapters) != 0 ||
			(p.Live.SlotState != SlotReserved && p.Live.SlotState != SlotBusy && p.Live.SlotState != SlotDraining) {
			return errors.New("busy Runtime Agent principal has incompatible live facts")
		}
	case PrincipalSlotUnavailable:
		if len(p.MissingRuntimeAdapters) != 0 || p.Live.SlotState != SlotFenced {
			return errors.New("unavailable Runtime Agent principal has incompatible live facts")
		}
	default:
		return errors.New("Runtime Agent principal availability is invalid")
	}
	return nil
}

type PrincipalCatalog interface {
	Get(context.Context, string) (runtimeconfig.RuntimeAgentPrincipal, error)
	List(context.Context, string, int) ([]runtimeconfig.RuntimeAgentPrincipal, error)
	RequiredRuntimeAdapters(context.Context, []string) ([]string, error)
	ReplaceLabelsIdempotent(context.Context, string, uint64, []string, string, string, time.Time) (runtimeconfig.PrincipalMutationResult, error)
	DeleteIdempotent(context.Context, string, uint64, string, string, time.Time) (runtimeconfig.PrincipalMutationResult, error)
}

type PrincipalRegistry interface {
	PrincipalRuntimeObservation(string) (*RuntimeAgentObservation, bool)
	ApplyPrincipalLabels(AuthenticatedPrincipal) error
}

type PrincipalOperations struct {
	catalog  PrincipalCatalog
	registry PrincipalRegistry
}

func NewPrincipalOperations(catalog PrincipalCatalog, registry PrincipalRegistry) (*PrincipalOperations, error) {
	if catalog == nil || registry == nil {
		return nil, errors.New("Runtime Agent principal Operations dependencies are incomplete")
	}
	return &PrincipalOperations{catalog: catalog, registry: registry}, nil
}

func (s *PrincipalOperations) List(
	ctx context.Context, afterRuntimeAgentID string, limit int,
) ([]RuntimeAgentPrincipalProjection, error) {
	principals, err := s.catalog.List(ctx, afterRuntimeAgentID, limit)
	if err != nil {
		return nil, err
	}
	result := make([]RuntimeAgentPrincipalProjection, 0, len(principals))
	for _, principal := range principals {
		projection, err := s.project(ctx, principal)
		if err != nil {
			return nil, err
		}
		result = append(result, projection)
	}
	return result, nil
}

func (s *PrincipalOperations) Get(
	ctx context.Context, runtimeAgentID string,
) (RuntimeAgentPrincipalProjection, error) {
	principal, err := s.catalog.Get(ctx, runtimeAgentID)
	if err != nil {
		return RuntimeAgentPrincipalProjection{}, err
	}
	return s.project(ctx, principal)
}

func (s *PrincipalOperations) ReplaceLabels(
	ctx context.Context,
	runtimeAgentID string,
	expectedRevision uint64,
	labels []string,
	idempotencyKey string,
	actor string,
	at time.Time,
) (RuntimeAgentPrincipalProjection, bool, error) {
	result, err := s.catalog.ReplaceLabelsIdempotent(
		ctx, runtimeAgentID, expectedRevision, labels, idempotencyKey, actor, at,
	)
	if err != nil {
		return RuntimeAgentPrincipalProjection{}, false, err
	}
	if result.Principal == nil {
		return RuntimeAgentPrincipalProjection{}, false, errors.New("Runtime Agent label mutation returned no principal")
	}
	current, err := s.catalog.Get(ctx, runtimeAgentID)
	if err != nil {
		return RuntimeAgentPrincipalProjection{}, false, err
	}
	if err := s.registry.ApplyPrincipalLabels(AuthenticatedPrincipal{
		RuntimeAgentID: current.RuntimeAgentID,
		Labels:         append([]string{}, current.Labels...),
		LabelRevision:  current.LabelRevision,
	}); err != nil {
		return RuntimeAgentPrincipalProjection{}, false, err
	}
	projection, err := s.project(ctx, *result.Principal)
	return projection, result.Replayed, err
}

func (s *PrincipalOperations) Delete(
	ctx context.Context,
	runtimeAgentID string,
	expectedRevision uint64,
	idempotencyKey string,
	actor string,
	at time.Time,
) (bool, error) {
	result, err := s.catalog.DeleteIdempotent(
		ctx, runtimeAgentID, expectedRevision, idempotencyKey, actor, at,
	)
	if err != nil {
		if errors.Is(err, ErrRegistrationConflict) {
			return false, runtimeconfig.ErrPrincipalInUse
		}
		return false, err
	}
	if !result.Deleted {
		return false, errors.New("Runtime Agent principal mutation did not delete the principal")
	}
	return result.Replayed, nil
}

func (s *PrincipalOperations) project(
	ctx context.Context,
	principal runtimeconfig.RuntimeAgentPrincipal,
) (RuntimeAgentPrincipalProjection, error) {
	required, err := s.catalog.RequiredRuntimeAdapters(ctx, principal.Labels)
	if err != nil {
		return RuntimeAgentPrincipalProjection{}, err
	}
	result := RuntimeAgentPrincipalProjection{
		Principal: principal, RequiredRuntimeAdapters: append([]string{}, required...),
		Availability: PrincipalOffline,
	}
	live, ok := s.registry.PrincipalRuntimeObservation(principal.RuntimeAgentID)
	if !ok {
		return result, result.Validate()
	}
	result.Live = live
	result.MissingRuntimeAdapters = missingStrings(required, live.SupportedRuntimeAdapters)
	if len(result.MissingRuntimeAdapters) != 0 {
		result.Availability = PrincipalAdapterCapabilityMismatch
	} else {
		switch live.SlotState {
		case SlotIdle:
			result.Availability = PrincipalAvailable
		case SlotReserved, SlotBusy, SlotDraining:
			result.Availability = PrincipalBusy
		default:
			result.Availability = PrincipalSlotUnavailable
		}
	}
	return result, result.Validate()
}

func missingStrings(required, available []string) []string {
	present := make(map[string]struct{}, len(available))
	for _, value := range available {
		present[value] = struct{}{}
	}
	result := make([]string, 0)
	for _, value := range required {
		if _, ok := present[value]; !ok {
			result = append(result, value)
		}
	}
	sort.Strings(result)
	return result
}
