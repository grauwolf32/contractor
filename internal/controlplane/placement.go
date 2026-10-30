package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

var errPlacementRevisionChanged = errors.New("placement revisions changed")

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
	AuditAdapterRequirements  func(context.Context, string, string) ([]runtimesettings.RuntimeAdapterRef, error)
	Pool                      *pgxpool.Pool
	Registry                  *InMemoryRegistry
	Gateways                  PlacementGatewayLookup
	LLMCredentials            resolved.CredentialLookup
	TransactionLLMCredentials runtimeconfig.TransactionLLMCredentialLookupFactory
	RuntimeCredentials        PlacementRuntimeCredentialLookup
	CredentialGuard           PlacementCredentialGuard
	PerformanceMetrics        bool
}

// PlacementAllocator composes immutable SQL catalogs with the process-local
// liveness Registry. Every potentially blocking lookup happens outside the
// Registry mutex; only compact compatibility edges cross that boundary.
type PlacementAllocator struct {
	auditAdapterRequirements  func(context.Context, string, string) ([]runtimesettings.RuntimeAdapterRef, error)
	pool                      *pgxpool.Pool
	registry                  *InMemoryRegistry
	gateways                  PlacementGatewayLookup
	llmCredentials            resolved.CredentialLookup
	transactionLLMCredentials runtimeconfig.TransactionLLMCredentialLookupFactory
	runtimeCredentials        PlacementRuntimeCredentialLookup
	credentialGuard           PlacementCredentialGuard
	performanceMetrics        bool
}

type placementCachedValue[T any] struct {
	value T
	err   error
}

// placementAttemptCache shares optimistic observations among the passes of
// one placement attempt, that is one ReserveAllContext call: the first pass,
// the retry after a principal refresh and the model Stage hold (CTL-R162).
// The pinning transaction never reads it: selected edges are re-read and
// drift-checked under that transaction (CTL-R88). It dies with the attempt,
// so the next Scheduler poll reads every mutable dependency again.
//
// It holds at most one entry per distinct Agent label, credential ID and
// (binding, ordered Agent-label set) among the location-matching live
// candidates of the attempt, so its size follows the fleet, not the number of
// passes or candidates sharing a configuration.
type placementAttemptCache struct {
	// The pinned default and Run-label versions are immutable references of
	// the Run snapshot, so discardMutable keeps them.
	requestLoaded bool
	defaultConfig runtimeconfig.PinnedRuntimeConfig
	runConfigs    []runtimeconfig.PinnedRuntimeConfig
	requestErr    error

	agentLabels        map[string]placementCachedValue[runtimeconfig.PinnedRuntimeConfig]
	llmCredentials     map[string]placementCachedValue[runtimeconfig.LLMCredentialAuthorization]
	runtimeCredentials map[string]placementCachedValue[runtimesettings.RuntimeCredentialKind]
	resolutions        map[placementResolutionInput]placementCachedValue[runtimeconfig.ResolvedRuntimeConfig]
}

// placementResolutionInput names everything a candidate contributes to
// resolveCandidate: the binding and the principal's exact ordered Agent-label
// set. Labels never contain NUL (runtimeconfig.ValidateLabel).
type placementResolutionInput struct {
	logicalAgentName string
	labels           string
}

func newPlacementAttemptCache() *placementAttemptCache {
	return &placementAttemptCache{
		agentLabels:        make(map[string]placementCachedValue[runtimeconfig.PinnedRuntimeConfig]),
		llmCredentials:     make(map[string]placementCachedValue[runtimeconfig.LLMCredentialAuthorization]),
		runtimeCredentials: make(map[string]placementCachedValue[runtimesettings.RuntimeCredentialKind]),
		resolutions:        make(map[placementResolutionInput]placementCachedValue[runtimeconfig.ResolvedRuntimeConfig]),
	}
}

// discardMutable drops every observation of mutable durable state: Agent-label
// bindings, credential metadata and the resolutions derived from them. A
// changed principal or a drift found by the pinning transaction proves that a
// mutation committed after these reads, and the attempt cannot tell which
// binding or credential mutations accompanied it (CTL-R162).
func (cache *placementAttemptCache) discardMutable() {
	clear(cache.agentLabels)
	clear(cache.llmCredentials)
	clear(cache.runtimeCredentials)
	clear(cache.resolutions)
}

func NewPlacementAllocator(options PlacementAllocatorOptions) (*PlacementAllocator, error) {
	if options.Pool == nil || options.Registry == nil || options.Gateways == nil ||
		options.LLMCredentials == nil || options.TransactionLLMCredentials == nil ||
		options.RuntimeCredentials == nil || options.CredentialGuard == nil {
		return nil, errors.New("candidate placement dependencies are incomplete")
	}
	return &PlacementAllocator{
		auditAdapterRequirements: options.AuditAdapterRequirements,
		pool:                     options.Pool, registry: options.Registry, gateways: options.Gateways,
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
	// This read is outside Registry locks and pinning transactions. The Audit
	// baseline is immutable, so the same requirement survives placement retries.
	if a.auditAdapterRequirements != nil {
		required, err := a.auditAdapterRequirements(ctx, request.RunID, request.StageExecutionID)
		if err != nil {
			return nil, err
		}
		request.auditAdapters = slices.Clone(required)
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

// holdForWaitingModelStage holds the live Agents that a model Stage waiting for
// allocated capacity could use (CTL-R155). The hold is advisory: an error
// while resolving it holds nothing and never changes the Stage's own
// placement outcome, and the Stage's next failed attempt computes it again.
// Live candidates that the failed passes already resolved reuse the attempt's
// resolutions; only allocated or newly live candidates are resolved here.
func (a *PlacementAllocator) holdForWaitingModelStage(
	ctx context.Context,
	request ReservationRequest,
	attempt *placementAttemptCache,
) {
	if request.RuntimeConfig == nil {
		return
	}
	edges, _, err := a.candidateEdges(ctx, request, a.registry.liveCandidates(), attempt)
	if err != nil {
		return
	}
	_ = a.registry.holdForModelStage(request, edges)
}

// candidateEdges resolves every compatible binding and candidate pair outside
// the Registry lock, sharing the attempt's observations, and returns the edges
// with the Runtime configuration resolved for each. The returned resolutions
// may share memory with the attempt cache and must not be mutated.
func (a *PlacementAllocator) candidateEdges(
	ctx context.Context,
	request ReservationRequest,
	candidates []AgentSnapshot,
	attempt *placementAttemptCache,
) ([]CandidateEdge, map[string]runtimeconfig.ResolvedRuntimeConfig, error) {
	candidates = matchingLocationCandidates(request.RequiredPlacementLabels, candidates)
	optimistic := make(map[string]runtimeconfig.ResolvedRuntimeConfig)
	lookups := placementCredentialLookups{
		llm: a.llmCredentials, runtime: a.runtimeCredentials,
		llmCache: attempt.llmCredentials, runtimeCache: attempt.runtimeCredentials,
	}
	edges := make([]CandidateEdge, 0, len(request.Bindings)*len(candidates))
	for _, binding := range request.Bindings {
		toolAdapters, err := resolved.RequiredRuntimeAdaptersForTemplate(binding.AgentTemplate)
		if err != nil {
			return nil, nil, fmt.Errorf("%w: AgentTemplate infrastructure requirements are invalid", ErrInvalidRequest)
		}
		for _, candidate := range candidates {
			if !isBindingCompatible(candidate.Registration, binding) {
				continue
			}
			runtimeConfig, err := a.resolveAttemptCandidate(ctx, request, binding, candidate.Principal, lookups, attempt)
			if err != nil {
				if contextError := ctx.Err(); contextError != nil {
					return nil, nil, contextError
				}
				if candidateResolutionUnavailable(err) {
					continue
				}
				return nil, nil, err
			}
			if !containsRuntimeAdapters(candidate.Registration.SupportedRuntimeAdapters, runtimeConfig.RequiredRuntimeAdapters) || !containsRuntimeAdapters(candidate.Registration.SupportedRuntimeAdapters, request.auditAdapters) {
				continue
			}
			if !containsRuntimeAdapters(runtimeConfig.RequiredRuntimeAdapters, toolAdapters) {
				continue
			}
			key := placementResolutionKey(binding.LogicalAgentName, candidate.Registration.InstanceID)
			optimistic[key] = runtimeConfig
			edges = append(edges, CandidateEdge{
				LogicalAgentName: binding.LogicalAgentName, RuntimeAgentID: candidate.Principal.RuntimeAgentID,
				RuntimeAgentInstanceID:    candidate.Registration.InstanceID,
				RuntimeAgentLabelRevision: candidate.Principal.LabelRevision,
				RequiredRuntimeAdapters:   append(slices.Clone(runtimeConfig.RequiredRuntimeAdapters), request.auditAdapters...),
			})
		}
	}
	return edges, optimistic, nil
}

// matchingLocationCandidates filters immutable declarations before candidate
// configuration reads. A wrong-location principal cannot become eligible through
// a mutable RuntimeConfig-label refresh, so capacity retries never query it.
func matchingLocationCandidates(required []string, candidates []AgentSnapshot) []AgentSnapshot {
	if len(required) == 0 {
		return candidates
	}
	matching := make([]AgentSnapshot, 0, len(candidates))
	for _, candidate := range candidates {
		if contracts.PlacementLabelsMatch(required, candidate.Registration.PlacementLabels) {
			matching = append(matching, candidate)
		}
	}
	return matching
}

func (a *PlacementAllocator) refreshCandidatePrincipals(ctx context.Context, candidates []AgentSnapshot) (bool, error) {
	repository := agentprincipal.NewRepository(a.pool)
	seen := make(map[string]struct{}, len(candidates))
	changed := false
	for _, candidate := range candidates {
		id := candidate.Principal.RuntimeAgentID
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		current, err := repository.Get(ctx, id)
		if errors.Is(err, runtimeconfig.ErrNotFound) {
			changed = a.registry.MarkPrincipalMissing(id) || changed
			continue
		}
		if err != nil {
			return false, err
		}
		principal := AuthenticatedPrincipal{
			RuntimeAgentID: id, Labels: current.Labels, LabelRevision: current.LabelRevision,
		}
		if equalAuthenticatedPrincipal(candidate.Principal, principal) {
			continue
		}
		if err := a.registry.ApplyPrincipalLabels(principal); err != nil {
			return false, err
		}
		changed = true
	}
	return changed, nil
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

func (a *PlacementAllocator) pinReservations(
	ctx context.Context,
	tx pgx.Tx,
	request ReservationRequest,
	reservations []Reservation,
	candidates map[string]AgentSnapshot,
	optimistic map[string]runtimeconfig.ResolvedRuntimeConfig,
	result map[string]runtimeconfig.ResolvedRuntimeConfig,
	policies map[string]reporting.PerformanceCollectionPolicy,
) error {
	labels := make(map[string]struct{})
	principalIDs := make([]string, 0, len(reservations))
	for _, reservation := range reservations {
		candidate, ok := candidates[reservation.Grant.RuntimeInstanceID]
		if !ok || candidate.Principal.RuntimeAgentID != reservation.Grant.RuntimeAgentID ||
			candidate.Principal.LabelRevision != reservation.RuntimeAgentLabelRevision {
			return errPlacementRevisionChanged
		}
		principalIDs = append(principalIDs, candidate.Principal.RuntimeAgentID)
		for _, label := range candidate.Principal.Labels {
			labels[label] = struct{}{}
		}
	}
	orderedLabels := sortedStringSet(labels)
	repository := runtimeconfig.NewRepository(tx)
	if len(orderedLabels) != 0 {
		if _, err := repository.LockBindings(ctx, orderedLabels); err != nil {
			return placementCatalogError(err)
		}
	}
	principals, err := agentprincipal.NewRepository(tx).LockMany(ctx, principalIDs)
	if err != nil {
		return placementCatalogError(err)
	}
	principalsByID := make(map[string]agentprincipal.Principal, len(principals))
	for _, principal := range principals {
		principalsByID[principal.RuntimeAgentID] = principal
	}

	bindings := make(map[string]BindingRequirement, len(request.Bindings))
	for _, binding := range request.Bindings {
		bindings[binding.LogicalAgentName] = binding
	}
	llmLookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, a.transactionLLMCredentials)
	if err != nil {
		return err
	}
	lookups := placementCredentialLookups{llm: llmLookup, runtime: credentials.NewRuntimeCredentialRepository(tx)}
	for _, reservation := range reservations {
		candidate := candidates[reservation.Grant.RuntimeInstanceID]
		principal, ok := principalsByID[reservation.Grant.RuntimeAgentID]
		if !ok || principal.LabelRevision != candidate.Principal.LabelRevision ||
			!reflect.DeepEqual(principal.Labels, candidate.Principal.Labels) {
			return errPlacementRevisionChanged
		}
		binding, ok := bindings[reservation.Grant.LogicalAgentName]
		if !ok {
			return errors.New("selected placement references an unknown logical Agent")
		}
		runtimeConfig, err := a.resolveCandidate(ctx, tx, request, binding, candidate.Principal, lookups)
		if err != nil {
			return placementCatalogError(err)
		}
		observed, ok := optimistic[placementResolutionKey(
			reservation.Grant.LogicalAgentName, reservation.Grant.RuntimeInstanceID,
		)]
		if !ok || !sameResolvedRuntimeConfig(observed, runtimeConfig) ||
			!contracts.PlacementLabelsMatch(request.RequiredPlacementLabels, candidate.Registration.PlacementLabels) ||
			!isBindingCompatible(candidate.Registration, binding) ||
			!containsRuntimeAdapters(candidate.Registration.SupportedRuntimeAdapters, runtimeConfig.RequiredRuntimeAdapters) || !containsRuntimeAdapters(candidate.Registration.SupportedRuntimeAdapters, request.auditAdapters) {
			return errPlacementRevisionChanged
		}
		result[reservation.Grant.AllocationID] = runtimeConfig
		policies[reservation.Grant.AllocationID] = a.collectionPolicy(candidate.Registration)
	}

	store := runstore.NewPostgresTxStore(tx)
	if _, err := store.LockStageState(
		ctx, request.RunID, request.StageExecutionID, runstore.StagePreparing,
	); err != nil {
		switch {
		case errors.Is(err, runstore.ErrNotFound):
			return runstore.ErrNotFound
		case errors.Is(err, runstore.ErrConflict):
			return runstore.ErrConflict
		default:
			return persistencepostgres.WrapError("lock StageExecution for allocation placement", err)
		}
	}
	storedRun, err := store.GetRun(ctx, request.RunID)
	if err != nil {
		return err
	}
	if !slices.Equal(storedRun.RequiredPlacementLabels, request.RequiredPlacementLabels) {
		return fmt.Errorf("%w: requirements differ from the immutable Run", ErrInvalidRequest)
	}
	for _, reservation := range reservations {
		runtimeConfig := result[reservation.Grant.AllocationID]
		configuration := &runstore.AllocationRuntimeConfiguration{
			ModelPolicy: runtimeConfig.ModelPolicy.Ref,
			Origins:     runtimeConfig.Origins,
			Provenance:  runtimeConfig.Provenance,
		}
		if err := store.RecordStageAllocation(ctx, runstore.StageAllocation{
			RuntimePlacementLabels: append([]string{}, reservation.PlacementLabels...),
			CompletionContract:     contracts.CloneWorkerCompletionContract(reservation.CompletionContract),
			AllocationID:           reservation.Grant.AllocationID, StageExecutionID: request.StageExecutionID,
			LogicalAgentName: reservation.Grant.LogicalAgentName, Namespace: reservation.Grant.Namespace,
			AgentTemplateRef: reservation.AgentTemplate.Ref, WorkerRuntimeRef: reservation.AgentTemplate.Runtime,
			RuntimeAgentID:                    reservation.Grant.RuntimeAgentID,
			RuntimeAgentInstanceID:            reservation.Grant.RuntimeInstanceID,
			RuntimeAgentLabelRevision:         reservation.RuntimeAgentLabelRevision,
			RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion,
			RuntimeConfiguration:              configuration,
			PerformanceCollectionPolicy:       policies[reservation.Grant.AllocationID],
		}); err != nil {
			return err
		}
	}
	return nil
}

func (a *PlacementAllocator) collectionPolicy(registration control.AgentRegistration) reporting.PerformanceCollectionPolicy {
	if !a.performanceMetrics {
		return reporting.PerformanceCollectionDisabled
	}
	for _, version := range registration.SupportedPerformanceMetricsVersions {
		if version == reporting.PerformanceMetricsVersion {
			return reporting.PerformanceCollectionRequested
		}
	}
	return reporting.PerformanceCollectionUnsupported
}

func (a *PlacementAllocator) resolveCandidate(
	ctx context.Context,
	db persistencepostgres.DBTX,
	request ReservationRequest,
	binding BindingRequirement,
	principal AuthenticatedPrincipal,
	lookups placementCredentialLookups,
) (runtimeconfig.ResolvedRuntimeConfig, error) {
	return a.resolveCandidateWithCache(ctx, db, request, binding, principal, lookups, nil)
}

// resolveAttemptCandidate resolves binding for principal's exact ordered
// Agent-label set at most once per attempt (CTL-R162). Another candidate with
// the same set, in this or a later pass, reuses the result, including a
// candidate-local failure.
func (a *PlacementAllocator) resolveAttemptCandidate(
	ctx context.Context,
	request ReservationRequest,
	binding BindingRequirement,
	principal AuthenticatedPrincipal,
	lookups placementCredentialLookups,
	attempt *placementAttemptCache,
) (runtimeconfig.ResolvedRuntimeConfig, error) {
	input := placementResolutionInput{
		logicalAgentName: binding.LogicalAgentName, labels: strings.Join(principal.Labels, "\x00"),
	}
	if cached, ok := attempt.resolutions[input]; ok {
		return cached.value, cached.err
	}
	runtimeConfig, err := a.resolveCandidateWithCache(ctx, a.pool, request, binding, principal, lookups, attempt)
	attempt.resolutions[input] = placementCachedValue[runtimeconfig.ResolvedRuntimeConfig]{value: runtimeConfig, err: err}
	return runtimeConfig, err
}

func (a *PlacementAllocator) resolveCandidateWithCache(
	ctx context.Context,
	db persistencepostgres.DBTX,
	request ReservationRequest,
	binding BindingRequirement,
	principal AuthenticatedPrincipal,
	lookups placementCredentialLookups,
	cache *placementAttemptCache,
) (runtimeconfig.ResolvedRuntimeConfig, error) {
	if request.RuntimeConfig == nil || binding.RuntimeSelection == nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, runtimeconfig.ErrInvalid
	}
	repository := runtimeconfig.NewRepository(db)
	var defaultConfig runtimeconfig.PinnedRuntimeConfig
	var runConfigs []runtimeconfig.PinnedRuntimeConfig
	var err error
	if cache == nil {
		defaultConfig, runConfigs, err = loadRequestPinnedRuntimeConfigs(ctx, repository, request.RuntimeConfig)
	} else {
		defaultConfig, runConfigs, err = cache.requestConfigs(ctx, repository, request.RuntimeConfig)
	}
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	agentConfigs := make([]runtimeconfig.PinnedRuntimeConfig, 0, len(principal.Labels))
	for _, label := range principal.Labels {
		var loaded runtimeconfig.PinnedRuntimeConfig
		if cache == nil {
			loaded, err = loadAgentPinnedRuntimeConfig(ctx, repository, label)
		} else {
			loaded, err = cache.agentConfig(ctx, repository, label)
		}
		if err != nil {
			return runtimeconfig.ResolvedRuntimeConfig{}, err
		}
		agentConfigs = append(agentConfigs, loaded)
	}
	workflowPatch, runPatch, escalationPatch, err := runtimeRoutePatches(*binding.RuntimeSelection)
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	if binding.AgentTemplate.IsToolWorker() {
		defaultConfig = runtimeconfig.WithoutWorkerModel(defaultConfig)
		for i := range runConfigs {
			runConfigs[i] = runtimeconfig.WithoutWorkerModel(runConfigs[i])
		}
		for i := range agentConfigs {
			agentConfigs[i] = runtimeconfig.WithoutWorkerModel(agentConfigs[i])
		}
	}
	allConfigs := append([]runtimeconfig.PinnedRuntimeConfig{defaultConfig}, runConfigs...)
	allConfigs = append(allConfigs, agentConfigs...)
	gateways, err := a.gatewayCatalog(*binding.RuntimeSelection, allConfigs)
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	llmCredentials, runtimeCredentials, err := lookups.catalogs(ctx, *binding.RuntimeSelection, allConfigs)
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	var summarizerModelPolicy *contracts.ResolvedModelPolicy
	// Only the terminal branch has a summarizer consumer (WRK-R184).
	if binding.ContextManagement.Strategy == contracts.ContextStrategyTerminal {
		policy := binding.AgentTemplate.Summarizer.ModelPolicy
		summarizerModelPolicy = &policy
	}
	if binding.RuntimeSelection.SummarizerModelPolicy != nil {
		policy := binding.RuntimeSelection.SummarizerModelPolicy.Clone()
		summarizerModelPolicy = &policy
	}
	return runtimeconfig.ResolveRuntimeConfig(runtimeconfig.ResolveRuntimeConfigInput{
		ModelFree:   binding.AgentTemplate.IsToolWorker(),
		ModelPolicy: binding.RuntimeSelection.ModelPolicy, SummarizerModelPolicy: summarizerModelPolicy,
		Default:  defaultConfig,
		Workflow: workflowPatch, RunLabels: runConfigs, RunOverride: runPatch,
		Escalation: escalationPatch, AgentLabels: agentConfigs,
		Gateways: gateways, LLMCredentials: llmCredentials, RuntimeCredentials: runtimeCredentials,
	})
}

func loadRequestPinnedRuntimeConfigs(
	ctx context.Context,
	repository *runtimeconfig.Repository,
	snapshot *runtimeconfig.RunSnapshot,
) (runtimeconfig.PinnedRuntimeConfig, []runtimeconfig.PinnedRuntimeConfig, error) {
	defaultConfig, err := loadPinnedRuntimeConfig(ctx, repository, snapshot.Default)
	if err != nil {
		return runtimeconfig.PinnedRuntimeConfig{}, nil, err
	}
	runConfigs := make([]runtimeconfig.PinnedRuntimeConfig, 0, len(snapshot.Labels))
	for _, pin := range snapshot.Labels {
		loaded, err := loadPinnedRuntimeConfig(ctx, repository, pin)
		if err != nil {
			return runtimeconfig.PinnedRuntimeConfig{}, nil, err
		}
		runConfigs = append(runConfigs, loaded)
	}
	return defaultConfig, runConfigs, nil
}

func (cache *placementAttemptCache) requestConfigs(
	ctx context.Context,
	repository *runtimeconfig.Repository,
	snapshot *runtimeconfig.RunSnapshot,
) (runtimeconfig.PinnedRuntimeConfig, []runtimeconfig.PinnedRuntimeConfig, error) {
	if !cache.requestLoaded {
		cache.requestLoaded = true
		cache.defaultConfig, cache.runConfigs, cache.requestErr = loadRequestPinnedRuntimeConfigs(ctx, repository, snapshot)
	}
	// Model-free projection changes slice entries, so each candidate owns its
	// own shallow slice while the cached pinned specs remain read-only.
	return cache.defaultConfig, append([]runtimeconfig.PinnedRuntimeConfig(nil), cache.runConfigs...), cache.requestErr
}

func loadAgentPinnedRuntimeConfig(
	ctx context.Context,
	repository *runtimeconfig.Repository,
	label string,
) (runtimeconfig.PinnedRuntimeConfig, error) {
	binding, err := repository.GetBinding(ctx, label)
	if err != nil {
		return runtimeconfig.PinnedRuntimeConfig{}, err
	}
	version, err := repository.GetVersionByRef(ctx, binding.Ref)
	if err != nil {
		return runtimeconfig.PinnedRuntimeConfig{}, err
	}
	return runtimeconfig.PinnedRuntimeConfig{
		Label: label, BindingRevision: binding.Revision, Config: binding.Ref, Spec: version.Spec,
	}, nil
}

func (cache *placementAttemptCache) agentConfig(
	ctx context.Context,
	repository *runtimeconfig.Repository,
	label string,
) (runtimeconfig.PinnedRuntimeConfig, error) {
	if cached, ok := cache.agentLabels[label]; ok {
		return cached.value, cached.err
	}
	loaded, err := loadAgentPinnedRuntimeConfig(ctx, repository, label)
	cache.agentLabels[label] = placementCachedValue[runtimeconfig.PinnedRuntimeConfig]{value: loaded, err: err}
	return loaded, err
}

func loadPinnedRuntimeConfig(
	ctx context.Context,
	repository *runtimeconfig.Repository,
	pin runtimeconfig.PinnedLabel,
) (runtimeconfig.PinnedRuntimeConfig, error) {
	version, err := repository.GetVersionByRef(ctx, pin.Config)
	if err != nil {
		return runtimeconfig.PinnedRuntimeConfig{}, err
	}
	return runtimeconfig.PinnedRuntimeConfig{
		Label: pin.Label, BindingRevision: pin.BindingRevision, Config: pin.Config, Spec: version.Spec,
	}, nil
}

func (a *PlacementAllocator) gatewayCatalog(
	selection resolved.ResolvedConsumerExecutionConfig,
	configs []runtimeconfig.PinnedRuntimeConfig,
) (map[llmgateway.LLMGatewayConfigRef]llmgateway.ResolvedLLMGatewayConfig, error) {
	refs := make(map[llmgateway.LLMGatewayConfigRef]struct{})
	if selection.LLMGateway != nil {
		refs[selection.LLMGateway.Ref] = struct{}{}
	}
	for _, pinned := range configs {
		if field := pinned.Spec.Worker.LLMGateway.Gateway; field.Present && !field.Clear {
			refs[field.Value] = struct{}{}
		}
	}
	result := make(map[llmgateway.LLMGatewayConfigRef]llmgateway.ResolvedLLMGatewayConfig, len(refs))
	for ref := range refs {
		gateway, err := a.gateways.LLMGateway(ref.GatewayID + "@" + ref.Version)
		if err != nil || gateway.Ref != ref {
			return nil, runtimeconfig.ErrNotFound
		}
		result[ref] = gateway
	}
	return result, nil
}

// Keep SQL-backed lookup ownership explicit. Optimistic resolution uses pool
// readers; the recheck under the lifecycle barrier uses transaction readers.
type placementCredentialLookups struct {
	llm          resolved.CredentialLookup
	runtime      PlacementRuntimeCredentialLookup
	llmCache     map[string]placementCachedValue[runtimeconfig.LLMCredentialAuthorization]
	runtimeCache map[string]placementCachedValue[runtimesettings.RuntimeCredentialKind]
}

func (lookups placementCredentialLookups) catalogs(
	ctx context.Context,
	selection resolved.ResolvedConsumerExecutionConfig,
	configs []runtimeconfig.PinnedRuntimeConfig,
) (
	map[string]runtimeconfig.LLMCredentialAuthorization,
	map[string]runtimesettings.RuntimeCredentialKind,
	error,
) {
	llmIDs := make(map[string]struct{})
	runtimeIDs := make(map[string]struct{})
	if selection.Credential != nil {
		llmIDs[selection.Credential.CredentialID] = struct{}{}
	}
	for _, pinned := range configs {
		collectSpecCredentialIDs(pinned.Spec, llmIDs, runtimeIDs)
	}
	llmResult := make(map[string]runtimeconfig.LLMCredentialAuthorization, len(llmIDs))
	for id := range llmIDs {
		authorization, err := lookups.llmAuthorization(ctx, id)
		if err != nil {
			return nil, nil, err
		}
		llmResult[id] = authorization
	}
	runtimeResult := make(map[string]runtimesettings.RuntimeCredentialKind, len(runtimeIDs))
	for id := range runtimeIDs {
		kind, err := lookups.runtimeKind(ctx, id)
		if err != nil {
			return nil, nil, err
		}
		runtimeResult[id] = kind
	}
	return llmResult, runtimeResult, nil
}

func (lookups placementCredentialLookups) llmAuthorization(
	ctx context.Context,
	id string,
) (runtimeconfig.LLMCredentialAuthorization, error) {
	if cached, ok := lookups.llmCache[id]; ok {
		return cached.value, cached.err
	}
	metadata, err := lookups.llm.LookupLLMCredential(ctx, id)
	var authorization runtimeconfig.LLMCredentialAuthorization
	if err == nil {
		authorization = runtimeconfig.LLMCredentialAuthorization{
			Ref: metadata.Ref, LLMGateway: metadata.LLMGateway,
			ModelPolicies: append([]contracts.ModelPolicyRef{}, metadata.ModelPolicies...),
			Models:        append([]string{}, metadata.Models...), Unrestricted: metadata.Unrestricted,
		}
	}
	if lookups.llmCache != nil {
		lookups.llmCache[id] = placementCachedValue[runtimeconfig.LLMCredentialAuthorization]{value: authorization, err: err}
	}
	return authorization, err
}

func (lookups placementCredentialLookups) runtimeKind(
	ctx context.Context,
	id string,
) (runtimesettings.RuntimeCredentialKind, error) {
	if cached, ok := lookups.runtimeCache[id]; ok {
		return cached.value, cached.err
	}
	metadata, err := lookups.runtime.Get(ctx, id)
	kind := runtimesettings.RuntimeCredentialKind(metadata.Kind)
	if lookups.runtimeCache != nil {
		lookups.runtimeCache[id] = placementCachedValue[runtimesettings.RuntimeCredentialKind]{value: kind, err: err}
	}
	return kind, err
}

func collectSpecCredentialIDs(
	spec runtimeconfig.Spec,
	llmIDs map[string]struct{},
	runtimeIDs map[string]struct{},
) {
	if field := spec.Worker.LLMGateway.Credential; field.Present && !field.Clear && field.Value != "" {
		llmIDs[field.Value] = struct{}{}
	}
	if field := spec.Worker.Telemetry; field.Present && !field.Clear && field.Value.Credential != "" {
		runtimeIDs[field.Value.Credential] = struct{}{}
	}
	if field := spec.Worker.HTTPProxy; field.Present && !field.Clear && field.Value.Credential != "" {
		runtimeIDs[field.Value.Credential] = struct{}{}
	}
	if field := spec.Worker.Caido; field.Present && !field.Clear && field.Value.Credential != "" {
		runtimeIDs[field.Value.Credential] = struct{}{}
	}
	if field := spec.Worker.Interactsh; field.Present && !field.Clear && field.Value.Credential != "" {
		runtimeIDs[field.Value.Credential] = struct{}{}
	}
	if field := spec.Planner.Telemetry; field.Present && !field.Clear && field.Value.Credential != "" {
		runtimeIDs[field.Value.Credential] = struct{}{}
	}
}

func runtimeRoutePatches(
	selection resolved.ResolvedConsumerExecutionConfig,
) (runtimeconfig.WorkerRoutePatch, runtimeconfig.WorkerRoutePatch, runtimeconfig.WorkerRoutePatch, error) {
	var workflow, run, escalation runtimeconfig.WorkerRoutePatch
	patchFor := func(origin string) (*runtimeconfig.WorkerRoutePatch, error) {
		switch {
		case strings.HasPrefix(origin, "run.executionConfig"):
			return &run, nil
		case strings.HasPrefix(origin, "executionConfig."), strings.Contains(origin, ".escalate.executionConfig"):
			return &escalation, nil
		case strings.HasPrefix(origin, "workflow.executionConfig"):
			return &workflow, nil
		default:
			return nil, runtimeconfig.ErrInvalid
		}
	}
	if selection.Origins.LLMGateway != "" {
		if selection.LLMGateway == nil {
			return workflow, run, escalation, runtimeconfig.ErrInvalid
		}
		patch, err := patchFor(selection.Origins.LLMGateway)
		if err != nil {
			return workflow, run, escalation, err
		}
		patch.Gateway = runtimeconfig.Field[llmgateway.LLMGatewayConfigRef]{
			Present: true, Value: selection.LLMGateway.Ref,
		}
	}
	if selection.Origins.Credential != "" {
		patch, err := patchFor(selection.Origins.Credential)
		if err != nil {
			return workflow, run, escalation, err
		}
		patch.Credential.Present = true
		if selection.Credential == nil {
			patch.Credential.Clear = true
		} else {
			patch.Credential.Value = selection.Credential.CredentialID
		}
	}
	return workflow, run, escalation, nil
}

func placementCatalogError(err error) error {
	if errors.Is(err, runtimeconfig.ErrPrecondition) || errors.Is(err, runtimeconfig.ErrConflict) ||
		candidateResolutionUnavailable(err) {
		return errPlacementRevisionChanged
	}
	return err
}

// Candidate-local policy, reference and kind failures remove only that edge.
// Opaque repository/provider errors are deliberately not collapsed into
// capacity: a database or credential backend outage must remain observable to
// Scheduler instead of looking like an idle-capacity shortage forever.
func candidateResolutionUnavailable(err error) bool {
	var resolution *runtimeconfig.ResolutionError
	return errors.As(err, &resolution) ||
		errors.Is(err, runtimeconfig.ErrInvalid) ||
		errors.Is(err, runtimeconfig.ErrNotFound) ||
		errors.Is(err, credentials.ErrNotFound) ||
		errors.Is(err, credentials.ErrRuntimeCredentialNotFound)
}

func placementResolutionKey(logicalAgentName, instanceID string) string {
	return logicalAgentName + "\x00" + instanceID
}

func sameResolvedRuntimeConfig(left, right runtimeconfig.ResolvedRuntimeConfig) bool {
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && string(leftJSON) == string(rightJSON)
}

func sortedStringSet(values map[string]struct{}) []string {
	result := make([]string, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
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
