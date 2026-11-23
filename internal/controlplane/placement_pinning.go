package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

var errPlacementRevisionChanged = errors.New("placement revisions changed")

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
			!containsRuntimeAdapters(candidate.Registration.SupportedRuntimeAdapters, runtimeConfig.RequiredRuntimeAdapters) || !containsRuntimeAdapters(candidate.Registration.SupportedRuntimeAdapters, request.additionalAdapters) {
			return errPlacementRevisionChanged
		}
		result[reservation.Grant.AllocationID] = runtimeConfig
		policies[reservation.Grant.AllocationID] = a.collectionPolicy(candidate.Registration)
	}

	store := runstore.NewPostgresTxStore(tx, a.storage)
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
		if err := a.allocations.RecordStageAllocation(ctx, tx, runstore.StageAllocation{
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

func placementCatalogError(err error) error {
	if errors.Is(err, runtimeconfig.ErrPrecondition) || errors.Is(err, runtimeconfig.ErrConflict) ||
		candidateResolutionUnavailable(err) {
		return errPlacementRevisionChanged
	}
	return err
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
