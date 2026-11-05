package controlplane

import (
	"context"
	"errors"
	"fmt"
	"slices"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type PlacementGatewayLookup interface {
	LLMGateway(string) (llmgateway.ResolvedLLMGatewayConfig, error)
}

type PlacementCredentialGuard interface {
	WithAllocationReferences(context.Context, func() error) error
}

type PlacementRuntimeCredentialLookup interface {
	Get(context.Context, string) (credentials.RuntimeCredentialMetadata, error)
}

type PlacementAllocatorOptions struct {
	Storage                       *artifactpostgres.Storage
	AdditionalAdapterRequirements func(context.Context, string, string, []BindingRequirement) ([]runtimesettings.RuntimeAdapterRef, error)
	Pool                          *pgxpool.Pool
	Registry                      *InMemoryRegistry
	Gateways                      PlacementGatewayLookup
	LLMCredentials                resolved.CredentialLookup
	TransactionLLMCredentials     runtimeconfig.TransactionLLMCredentialLookupFactory
	RuntimeCredentials            PlacementRuntimeCredentialLookup
	CredentialGuard               PlacementCredentialGuard
	PerformanceMetrics            bool
}

// PlacementAllocator composes immutable SQL catalogs with the process-local
// liveness Registry. Every potentially blocking lookup happens outside the
// Registry mutex; only compact compatibility edges cross that boundary.
type PlacementAllocator struct {
	storage                       *artifactpostgres.Storage
	additionalAdapterRequirements func(context.Context, string, string, []BindingRequirement) ([]runtimesettings.RuntimeAdapterRef, error)
	pool                          *pgxpool.Pool
	registry                      *InMemoryRegistry
	gateways                      PlacementGatewayLookup
	llmCredentials                resolved.CredentialLookup
	transactionLLMCredentials     runtimeconfig.TransactionLLMCredentialLookupFactory
	runtimeCredentials            PlacementRuntimeCredentialLookup
	credentialGuard               PlacementCredentialGuard
	performanceMetrics            bool
}

func NewPlacementAllocator(options PlacementAllocatorOptions) (*PlacementAllocator, error) {
	if options.Storage == nil || options.Pool == nil || options.Registry == nil || options.Gateways == nil ||
		options.LLMCredentials == nil || options.TransactionLLMCredentials == nil ||
		options.RuntimeCredentials == nil || options.CredentialGuard == nil {
		return nil, errors.New("candidate placement dependencies are incomplete")
	}
	return &PlacementAllocator{
		storage:                       options.Storage,
		additionalAdapterRequirements: options.AdditionalAdapterRequirements,
		pool:                          options.Pool, registry: options.Registry, gateways: options.Gateways,
		llmCredentials: options.LLMCredentials, runtimeCredentials: options.RuntimeCredentials,
		transactionLLMCredentials: options.TransactionLLMCredentials,
		credentialGuard:           options.CredentialGuard, performanceMetrics: options.PerformanceMetrics,
	}, nil
}

func (a *PlacementAllocator) ReserveAllContext(
	ctx context.Context,
	request ReservationRequest,
) ([]Reservation, error) {
	if err := contracts.ValidatePlacementLabels(request.RequiredPlacementLabels); err != nil {
		return nil, fmt.Errorf("%w: invalid required placement labels: %v", ErrInvalidRequest, err)
	}
	// This read is outside Registry locks and pinning transactions. The owning
	// baseline is immutable, so the same requirement survives placement retries.
	if a.additionalAdapterRequirements != nil {
		required, err := a.additionalAdapterRequirements(ctx, request.RunID, request.StageExecutionID, request.Bindings)
		if err != nil {
			return nil, err
		}
		request.additionalAdapters = slices.Clone(required)
	}
	attempt := newPlacementAttemptCache()
	reservations, err := a.reserveAllContext(ctx, request, attempt)
	if !errors.Is(err, ErrInsufficientCapacity) || ctx.Err() != nil {
		return reservations, err
	}
	// A failed attempt may have selected a stale principal, or stale labels
	// may have prevented any compatible edge from being built. Refresh only on
	// failure, after provisional reservations have been discarded, and retry
	// once with the current durable principals.
	changed, refreshErr := a.refreshCandidatePrincipals(ctx, matchingLocationCandidates(request.RequiredPlacementLabels, a.registry.PlacementCandidates()))
	if refreshErr != nil {
		return nil, refreshErr
	}
	if changed {
		attempt.discardMutable()
		reservations, err = a.reserveAllContext(ctx, request, attempt)
	}
	if errors.Is(err, ErrInsufficientCapacity) && !request.ModelFree && ctx.Err() == nil {
		a.holdForWaitingModelStage(ctx, request, attempt)
	}
	return reservations, err
}

func (a *PlacementAllocator) reserveAllContext(
	ctx context.Context,
	request ReservationRequest,
	attempt *placementAttemptCache,
) ([]Reservation, error) {
	if existing, err := a.registry.reservationsForRequest(request); err == nil {
		return existing, nil
	} else if !errors.Is(err, ErrAllocationNotFound) {
		return nil, err
	}
	if request.RuntimeConfig == nil {
		return nil, fmt.Errorf("%w: production placement requires a Run RuntimeConfig snapshot", ErrInvalidRequest)
	}
	candidates := a.registry.PlacementCandidates()
	if len(candidates) == 0 {
		return nil, ErrInsufficientCapacity
	}
	edges, optimistic, err := a.candidateEdges(ctx, request, candidates, attempt)
	if err != nil {
		return nil, err
	}
	if len(edges) == 0 {
		return nil, ErrInsufficientCapacity
	}
	reservations, err := a.registry.ReserveCandidateEdges(request, edges)
	if err != nil {
		return nil, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = a.registry.DiscardCandidateReservations(request.StageExecutionID)
		}
	}()

	if request.Admit != nil {
		candidates := make([]Reservation, len(reservations))
		for index, reservation := range reservations {
			runtimeConfig, ok := optimistic[placementResolutionKey(
				reservation.Grant.LogicalAgentName, reservation.Grant.RuntimeInstanceID,
			)]
			if !ok {
				return nil, errors.New("candidate placement omitted a selected resolution")
			}
			// The attempt cache keeps the resolution for later passes, so
			// admission receives a detached copy.
			detached := runtimeConfig.Clone()
			reservation.ResolvedRuntimeConfig = &detached
			candidates[index] = reservation
		}
		if err := request.Admit(ctx, candidates); err != nil {
			return nil, err
		}
	}

	selectedCandidates := make(map[string]AgentSnapshot, len(reservations))
	for _, candidate := range candidates {
		selectedCandidates[candidate.Registration.InstanceID] = candidate
	}
	resolvedByAllocation := make(map[string]runtimeconfig.ResolvedRuntimeConfig, len(reservations))
	policyByAllocation := make(map[string]reporting.PerformanceCollectionPolicy, len(reservations))
	err = a.credentialGuard.WithAllocationReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, a.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			return a.pinReservations(
				ctx, tx, request, reservations, selectedCandidates, optimistic,
				resolvedByAllocation, policyByAllocation,
			)
		})
	})
	if err != nil {
		if errors.Is(err, errPlacementRevisionChanged) {
			attempt.discardMutable()
			return nil, ErrInsufficientCapacity
		}
		return nil, err
	}
	pinned := make(map[string]PinnedReservationConfig, len(reservations))
	for _, reservation := range reservations {
		runtimeConfig, ok := resolvedByAllocation[reservation.Grant.AllocationID]
		if !ok {
			return nil, errors.New("durable placement omitted a selected allocation")
		}
		pinned[reservation.Grant.AllocationID] = PinnedReservationConfig{
			RuntimeAgentLabelRevision:   reservation.RuntimeAgentLabelRevision,
			Resolved:                    runtimeConfig,
			PerformanceCollectionPolicy: policyByAllocation[reservation.Grant.AllocationID],
			PerformanceMetrics:          policyByAllocation[reservation.Grant.AllocationID].Request(),
		}
	}
	result, err := a.registry.CommitCandidateReservations(request.StageExecutionID, pinned)
	if err != nil {
		return nil, persistencepostgres.WrapError("commit durable placement to live Registry", err)
	}
	committed = true
	return result, nil
}

func (a *PlacementAllocator) GetGrant(allocationID string) (AllocationGrant, error) {
	return a.registry.GetGrant(allocationID)
}

func (a *PlacementAllocator) GetReservation(allocationID string) (Reservation, error) {
	return a.registry.GetReservation(allocationID)
}

func (a *PlacementAllocator) SetWriteFence(allocationID string) error {
	return a.registry.SetWriteFence(allocationID)
}

// DiscardUnprepared drops a committed Stage reservation that was never
// prepared, so the same StageExecution can be placed again.
func (a *PlacementAllocator) DiscardUnprepared(stageExecutionID string) error {
	return a.registry.DiscardUnpreparedReservations(stageExecutionID)
}

func (a *PlacementAllocator) PollAllocationLosses() []AllocationLoss {
	return a.registry.PollAllocationLosses()
}
