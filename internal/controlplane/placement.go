package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

var errPlacementRevisionChanged = errors.New("placement revisions changed")

type PlacementGatewayLookup interface {
	LLMGateway(string) (contracts.ResolvedLLMGatewayConfig, error)
}

type PlacementCredentialGuard interface {
	WithAllocationReferences(context.Context, func() error) error
}

type PlacementRuntimeCredentialLookup interface {
	Get(context.Context, string) (credentials.RuntimeCredentialMetadata, error)
}

type PlacementAllocatorOptions struct {
	Pool               *pgxpool.Pool
	Registry           *InMemoryRegistry
	Gateways           PlacementGatewayLookup
	LLMCredentials     workflowconfig.CredentialLookup
	RuntimeCredentials PlacementRuntimeCredentialLookup
	CredentialGuard    PlacementCredentialGuard
}

// PlacementAllocator composes immutable SQL catalogs with the process-local
// liveness Registry. Every potentially blocking lookup happens outside the
// Registry mutex; only compact compatibility edges cross that boundary.
type PlacementAllocator struct {
	pool               *pgxpool.Pool
	registry           *InMemoryRegistry
	gateways           PlacementGatewayLookup
	llmCredentials     workflowconfig.CredentialLookup
	runtimeCredentials PlacementRuntimeCredentialLookup
	credentialGuard    PlacementCredentialGuard
}

func NewPlacementAllocator(options PlacementAllocatorOptions) (*PlacementAllocator, error) {
	if options.Pool == nil || options.Registry == nil || options.Gateways == nil ||
		options.LLMCredentials == nil || options.RuntimeCredentials == nil || options.CredentialGuard == nil {
		return nil, errors.New("candidate placement dependencies are incomplete")
	}
	return &PlacementAllocator{
		pool: options.Pool, registry: options.Registry, gateways: options.Gateways,
		llmCredentials: options.LLMCredentials, runtimeCredentials: options.RuntimeCredentials,
		credentialGuard: options.CredentialGuard,
	}, nil
}

// ReserveAll keeps the legacy Allocator shape for embedding callers. Scheduler
// uses ReserveAllContext so its preparation deadline also bounds SQL work.
func (a *PlacementAllocator) ReserveAll(request ReservationRequest) ([]Reservation, error) {
	return a.ReserveAllContext(context.Background(), request)
}

func (a *PlacementAllocator) ReserveAllContext(
	ctx context.Context,
	request ReservationRequest,
) ([]Reservation, error) {
	if existing, err := a.registry.GetStageReservations(request.StageExecutionID); err == nil {
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

	optimistic := make(map[string]runtimeconfig.ResolvedRuntimeConfig)
	edges := make([]CandidateEdge, 0, len(request.Bindings)*len(candidates))
	for _, binding := range request.Bindings {
		for _, candidate := range candidates {
			if !isCompatible(candidate.Registration, binding.AgentTemplate, binding.Workspace) {
				continue
			}
			resolved, err := a.resolveCandidate(ctx, a.pool, request, binding, candidate.Principal)
			if err != nil {
				if contextError := ctx.Err(); contextError != nil {
					return nil, contextError
				}
				if candidateResolutionUnavailable(err) {
					continue
				}
				return nil, err
			}
			if !containsRuntimeAdapters(candidate.Registration.SupportedRuntimeAdapters, resolved.RequiredRuntimeAdapters) {
				continue
			}
			key := placementResolutionKey(binding.LogicalAgentName, candidate.Registration.InstanceID)
			optimistic[key] = resolved
			edges = append(edges, CandidateEdge{
				LogicalAgentName: binding.LogicalAgentName, RuntimeAgentID: candidate.Principal.RuntimeAgentID,
				RuntimeAgentInstanceID:    candidate.Registration.InstanceID,
				RuntimeAgentLabelRevision: candidate.Principal.LabelRevision,
				RequiredRuntimeAdapters:   append([]contracts.RuntimeAdapterRef{}, resolved.RequiredRuntimeAdapters...),
			})
		}
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

	selectedCandidates := make(map[string]AgentSnapshot, len(reservations))
	for _, candidate := range candidates {
		selectedCandidates[candidate.Registration.InstanceID] = candidate
	}
	resolvedByAllocation := make(map[string]runtimeconfig.ResolvedRuntimeConfig, len(reservations))
	err = a.credentialGuard.WithAllocationReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, a.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			return a.pinReservations(
				ctx, tx, request, reservations, selectedCandidates, optimistic, resolvedByAllocation,
			)
		})
	})
	if err != nil {
		if errors.Is(err, errPlacementRevisionChanged) {
			return nil, ErrInsufficientCapacity
		}
		return nil, err
	}
	pinned := make(map[string]PinnedReservationConfig, len(reservations))
	for _, reservation := range reservations {
		resolved, ok := resolvedByAllocation[reservation.Grant.AllocationID]
		if !ok {
			return nil, errors.New("durable placement omitted a selected allocation")
		}
		pinned[reservation.Grant.AllocationID] = PinnedReservationConfig{
			RuntimeAgentLabelRevision: reservation.RuntimeAgentLabelRevision,
			Resolved:                  resolved,
		}
	}
	result, err := a.registry.CommitCandidateReservations(request.StageExecutionID, pinned)
	if err != nil {
		return nil, errors.New("commit durable placement to live Registry")
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
	principals, err := runtimeconfig.NewPrincipalRepository(tx).LockMany(ctx, principalIDs)
	if err != nil {
		return placementCatalogError(err)
	}
	principalsByID := make(map[string]runtimeconfig.RuntimeAgentPrincipal, len(principals))
	for _, principal := range principals {
		principalsByID[principal.RuntimeAgentID] = principal
	}

	bindings := make(map[string]BindingRequirement, len(request.Bindings))
	for _, binding := range request.Bindings {
		bindings[binding.LogicalAgentName] = binding
	}
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
		resolved, err := a.resolveCandidate(ctx, tx, request, binding, candidate.Principal)
		if err != nil {
			return placementCatalogError(err)
		}
		observed, ok := optimistic[placementResolutionKey(
			reservation.Grant.LogicalAgentName, reservation.Grant.RuntimeInstanceID,
		)]
		if !ok || !sameResolvedRuntimeConfig(observed, resolved) ||
			!isCompatible(candidate.Registration, binding.AgentTemplate, binding.Workspace) ||
			!containsRuntimeAdapters(candidate.Registration.SupportedRuntimeAdapters, resolved.RequiredRuntimeAdapters) {
			return errPlacementRevisionChanged
		}
		result[reservation.Grant.AllocationID] = resolved
	}

	var state, runID string
	if err := tx.QueryRow(ctx, `
SELECT state, run_id
FROM stage_executions
WHERE stage_execution_id = $1
FOR UPDATE`, request.StageExecutionID).Scan(&state, &runID); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return runstore.ErrNotFound
		}
		return errors.New("lock StageExecution for allocation placement")
	}
	if state != string(runstore.StagePreparing) || runID != request.RunID {
		return runstore.ErrConflict
	}
	store := runstore.NewPostgresStore(tx)
	for _, reservation := range reservations {
		resolved := result[reservation.Grant.AllocationID]
		configuration := &runstore.AllocationRuntimeConfiguration{
			ModelPolicy: resolved.ModelPolicy.Ref,
			Origins:     resolved.Origins,
			Provenance:  resolved.Provenance,
		}
		if err := store.RecordStageAllocation(ctx, runstore.StageAllocation{
			AllocationID: reservation.Grant.AllocationID, StageExecutionID: request.StageExecutionID,
			LogicalAgentName: reservation.Grant.LogicalAgentName, Namespace: reservation.Grant.Namespace,
			AgentTemplateRef: reservation.AgentTemplate.Ref, WorkerRuntimeRef: reservation.AgentTemplate.Runtime,
			RuntimeAgentID:                    reservation.Grant.RuntimeAgentID,
			RuntimeAgentInstanceID:            reservation.Grant.RuntimeInstanceID,
			RuntimeAgentLabelRevision:         reservation.RuntimeAgentLabelRevision,
			RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion,
			RuntimeConfiguration:              configuration,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (a *PlacementAllocator) resolveCandidate(
	ctx context.Context,
	db persistencepostgres.DBTX,
	request ReservationRequest,
	binding BindingRequirement,
	principal AuthenticatedPrincipal,
) (runtimeconfig.ResolvedRuntimeConfig, error) {
	if request.RuntimeConfig == nil || binding.RuntimeSelection == nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, runtimeconfig.ErrInvalid
	}
	repository := runtimeconfig.NewRepository(db)
	defaultConfig, err := loadPinnedRuntimeConfig(ctx, repository, request.RuntimeConfig.Default)
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	runConfigs := make([]runtimeconfig.PinnedRuntimeConfig, 0, len(request.RuntimeConfig.Labels))
	for _, pin := range request.RuntimeConfig.Labels {
		loaded, err := loadPinnedRuntimeConfig(ctx, repository, pin)
		if err != nil {
			return runtimeconfig.ResolvedRuntimeConfig{}, err
		}
		runConfigs = append(runConfigs, loaded)
	}
	agentConfigs := make([]runtimeconfig.PinnedRuntimeConfig, 0, len(principal.Labels))
	for _, label := range principal.Labels {
		binding, err := repository.GetBinding(ctx, label)
		if err != nil {
			return runtimeconfig.ResolvedRuntimeConfig{}, err
		}
		version, err := repository.GetVersionByRef(ctx, binding.Ref)
		if err != nil {
			return runtimeconfig.ResolvedRuntimeConfig{}, err
		}
		agentConfigs = append(agentConfigs, runtimeconfig.PinnedRuntimeConfig{
			Label: label, BindingRevision: binding.Revision, Config: binding.Ref, Spec: version.Spec,
		})
	}
	workflowPatch, runPatch, escalationPatch, err := runtimeRoutePatches(*binding.RuntimeSelection)
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	allConfigs := append([]runtimeconfig.PinnedRuntimeConfig{defaultConfig}, runConfigs...)
	allConfigs = append(allConfigs, agentConfigs...)
	gateways, err := a.gatewayCatalog(*binding.RuntimeSelection, allConfigs)
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	llmCredentials, runtimeCredentials, err := a.credentialCatalogs(ctx, *binding.RuntimeSelection, allConfigs)
	if err != nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, err
	}
	return runtimeconfig.ResolveRuntimeConfig(runtimeconfig.ResolveRuntimeConfigInput{
		ModelPolicy: binding.RuntimeSelection.ModelPolicy, Default: defaultConfig,
		Workflow: workflowPatch, RunLabels: runConfigs, RunOverride: runPatch,
		Escalation: escalationPatch, AgentLabels: agentConfigs,
		Gateways: gateways, LLMCredentials: llmCredentials, RuntimeCredentials: runtimeCredentials,
	})
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
	selection workflowconfig.ResolvedConsumerExecutionConfig,
	configs []runtimeconfig.PinnedRuntimeConfig,
) (map[contracts.LLMGatewayConfigRef]contracts.ResolvedLLMGatewayConfig, error) {
	refs := make(map[contracts.LLMGatewayConfigRef]struct{})
	if selection.LLMGateway != nil {
		refs[selection.LLMGateway.Ref] = struct{}{}
	}
	for _, pinned := range configs {
		if field := pinned.Spec.Worker.LLMGateway.Gateway; field.Present && !field.Clear {
			refs[field.Value] = struct{}{}
		}
	}
	result := make(map[contracts.LLMGatewayConfigRef]contracts.ResolvedLLMGatewayConfig, len(refs))
	for ref := range refs {
		gateway, err := a.gateways.LLMGateway(ref.GatewayID + "@" + ref.Version)
		if err != nil || gateway.Ref != ref {
			return nil, runtimeconfig.ErrNotFound
		}
		result[ref] = gateway
	}
	return result, nil
}

func (a *PlacementAllocator) credentialCatalogs(
	ctx context.Context,
	selection workflowconfig.ResolvedConsumerExecutionConfig,
	configs []runtimeconfig.PinnedRuntimeConfig,
) (
	map[string]runtimeconfig.LLMCredentialAuthorization,
	map[string]contracts.RuntimeCredentialKind,
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
		metadata, err := a.llmCredentials.LookupLLMCredential(ctx, id)
		if err != nil {
			return nil, nil, err
		}
		llmResult[id] = runtimeconfig.LLMCredentialAuthorization{
			Ref: metadata.Ref, LLMGateway: metadata.LLMGateway,
			ModelPolicies: append([]contracts.ModelPolicyRef{}, metadata.ModelPolicies...),
			Models:        append([]string{}, metadata.Models...), Unrestricted: metadata.Unrestricted,
		}
	}
	runtimeResult := make(map[string]contracts.RuntimeCredentialKind, len(runtimeIDs))
	for id := range runtimeIDs {
		metadata, err := a.runtimeCredentials.Get(ctx, id)
		if err != nil {
			return nil, nil, err
		}
		runtimeResult[id] = contracts.RuntimeCredentialKind(metadata.Kind)
	}
	return llmResult, runtimeResult, nil
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
	if field := spec.Planner.Telemetry; field.Present && !field.Clear && field.Value.Credential != "" {
		runtimeIDs[field.Value.Credential] = struct{}{}
	}
}

func runtimeRoutePatches(
	selection workflowconfig.ResolvedConsumerExecutionConfig,
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
		patch.Gateway = runtimeconfig.Field[contracts.LLMGatewayConfigRef]{
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

func (a *PlacementAllocator) Release(allocationID string) error {
	return a.registry.Release(allocationID)
}

func (a *PlacementAllocator) PollAllocationLosses() []AllocationLoss {
	return a.registry.PollAllocationLosses()
}
