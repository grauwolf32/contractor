package controlplane

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type RuntimeLifecycle interface {
	Prepare(context.Context, Reservation, contracts.WorkerExecutionSettings) (contracts.WorkerHandle, error)
	Finalize(context.Context, Reservation, string, time.Time) (contracts.AllocationFinalReport, error)
	Abort(context.Context, Reservation, string, contracts.TerminationError, time.Time) (contracts.AllocationFinalReport, error)
	Release(context.Context, Reservation) error
}

type AllocationRegistry interface {
	SetWriteFence(string) error
	SetAllocationPhase(string, AllocationAuthoritativePhase, *SafeReason) error
	RecordAllocationReport(string, contracts.AllocationFinalReport) error
	Release(string) error
}

// RuntimeBatchController coordinates allocation lifecycle calls and Registry
// observations. Registry grants remain owned until Runtime release succeeds.
type RuntimeBatchController struct {
	runtime        RuntimeLifecycle
	registry       AllocationRegistry
	now            func() time.Time
	nowMu          sync.Mutex
	newID          func(string) (string, error)
	idMu           sync.Mutex
	cleanupTimeout time.Duration
}

type RuntimeBatchOptions struct {
	Now            func() time.Time
	NewID          func(string) (string, error)
	CleanupTimeout time.Duration
}

func NewRuntimeBatchController(
	runtime RuntimeLifecycle,
	registry AllocationRegistry,
	options RuntimeBatchOptions,
) (*RuntimeBatchController, error) {
	if runtime == nil || registry == nil {
		return nil, errors.New("Runtime lifecycle and allocation registry are required")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	if options.NewID == nil {
		options.NewID = lifecycleID
	}
	if options.CleanupTimeout == 0 {
		options.CleanupTimeout = 10 * time.Second
	}
	if options.CleanupTimeout <= 0 {
		return nil, errors.New("cleanup timeout must be positive")
	}
	return &RuntimeBatchController{
		runtime: runtime, registry: registry, now: options.Now,
		newID: options.NewID, cleanupTimeout: options.CleanupTimeout,
	}, nil
}

func (c *RuntimeBatchController) PrepareAll(
	ctx context.Context,
	reservations []Reservation,
	settings map[string]contracts.WorkerExecutionSettings,
) (map[string]contracts.WorkerHandle, error) {
	if err := validateReservationBatch(reservations); err != nil {
		return nil, err
	}
	// Validate the whole settings map before any Runtime call; a mismatch
	// found mid-batch would leave earlier Agents prepared and active.
	for _, reservation := range reservations {
		if _, ok := settings[reservation.Grant.LogicalAgentName]; !ok {
			return nil, fmt.Errorf("missing execution settings for logical Agent %q", reservation.Grant.LogicalAgentName)
		}
	}
	if len(settings) != len(reservations) {
		return nil, errors.New("Worker execution settings contain an unknown logical Agent")
	}
	handles := make(map[string]contracts.WorkerHandle, len(reservations))
	for _, reservation := range reservations {
		logicalName := reservation.Grant.LogicalAgentName
		handle, err := c.runtime.Prepare(ctx, reservation, settings[logicalName])
		if err != nil {
			prepareErr := fmt.Errorf("prepare logical Agent %q: %w", reservation.Grant.LogicalAgentName, err)
			cleanupErr := c.cleanupFailedPrepare(reservations)
			return nil, errors.Join(prepareErr, cleanupErr)
		}
		if err := c.registry.SetAllocationPhase(
			reservation.Grant.AllocationID, AllocationActive, nil,
		); err != nil {
			prepareErr := fmt.Errorf("observe prepared logical Agent %q: %w", logicalName, err)
			cleanupErr := c.cleanupFailedPrepare(reservations)
			return nil, errors.Join(prepareErr, cleanupErr)
		}
		handles[reservation.Grant.LogicalAgentName] = handle
	}
	return handles, nil
}

func (c *RuntimeBatchController) FinalizeAll(
	ctx context.Context,
	reservations []Reservation,
	finalizationID string,
	deadline time.Time,
) (map[string]contracts.AllocationFinalReport, error) {
	ctx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	if err := validateReservationBatch(reservations); err != nil {
		return nil, err
	}
	var failures []error
	for _, reservation := range reservations {
		if err := c.registry.SetWriteFence(reservation.Grant.AllocationID); err != nil {
			failures = append(failures, fmt.Errorf("fence allocation %q: %w", reservation.Grant.AllocationID, err))
		}
		if err := c.registry.SetAllocationPhase(
			reservation.Grant.AllocationID, AllocationFinalizing, nil,
		); err != nil {
			failures = append(failures, fmt.Errorf("observe finalizing allocation %q: %w", reservation.Grant.AllocationID, err))
		}
	}
	results := fanOutRuntimeCalls(ctx, reservations, func(
		callContext context.Context, _ int, reservation Reservation,
	) (contracts.AllocationFinalReport, error) {
		return c.runtime.Finalize(callContext, reservation, finalizationID, deadline)
	})
	reports := make(map[string]contracts.AllocationFinalReport, len(reservations))
	for index, reservation := range reservations {
		result := results[index]
		if result.err != nil {
			failures = append(failures, fmt.Errorf(
				"finalize allocation %q: %w", reservation.Grant.AllocationID, result.err,
			))
			continue
		}
		if err := c.registry.RecordAllocationReport(reservation.Grant.AllocationID, result.value); err != nil {
			failures = append(failures, fmt.Errorf("observe final report for allocation %q: %w", reservation.Grant.AllocationID, err))
		}
		reports[reservation.Grant.LogicalAgentName] = result.value
	}
	return reports, errors.Join(failures...)
}

func (c *RuntimeBatchController) AbortAll(
	ctx context.Context,
	reservations []Reservation,
	abortID string,
	reason contracts.TerminationError,
	deadline time.Time,
) (map[string]contracts.AllocationFinalReport, error) {
	ctx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	if err := validateReservationBatch(reservations); err != nil {
		return nil, err
	}
	var failures []error
	reports := make(map[string]contracts.AllocationFinalReport, len(reservations))
	safeReason := safeTerminationReason(reason)
	for _, reservation := range reservations {
		if err := c.registry.SetWriteFence(reservation.Grant.AllocationID); err != nil {
			failures = append(failures, fmt.Errorf("fence allocation %q: %w", reservation.Grant.AllocationID, err))
		}
		if err := c.registry.SetAllocationPhase(
			reservation.Grant.AllocationID, AllocationAborting, &safeReason,
		); err != nil {
			failures = append(failures, fmt.Errorf("observe aborting allocation %q: %w", reservation.Grant.AllocationID, err))
		}
	}
	results := fanOutRuntimeCalls(ctx, reservations, func(
		callContext context.Context, _ int, reservation Reservation,
	) (contracts.AllocationFinalReport, error) {
		return c.runtime.Abort(callContext, reservation, abortID, reason, deadline)
	})
	for index, reservation := range reservations {
		result := results[index]
		if result.err != nil {
			failures = append(failures, fmt.Errorf(
				"abort allocation %q: %w", reservation.Grant.AllocationID, result.err,
			))
			continue
		}
		if err := c.registry.RecordAllocationReport(reservation.Grant.AllocationID, result.value); err != nil {
			failures = append(failures, fmt.Errorf("observe abort report for allocation %q: %w", reservation.Grant.AllocationID, err))
		}
		reports[reservation.Grant.LogicalAgentName] = result.value
	}
	return reports, errors.Join(failures...)
}

func (c *RuntimeBatchController) ReleaseAll(ctx context.Context, reservations []Reservation) error {
	ctx, cancel := context.WithTimeout(ctx, c.cleanupTimeout)
	defer cancel()
	if err := validateReservationBatch(reservations); err != nil {
		return err
	}
	var failures []error
	for _, reservation := range reservations {
		if err := c.registry.SetAllocationPhase(
			reservation.Grant.AllocationID, AllocationReleasing, nil,
		); err != nil {
			failures = append(failures, fmt.Errorf("observe releasing allocation %q: %w", reservation.Grant.AllocationID, err))
		}
	}
	results := fanOutRuntimeCalls(ctx, reservations, func(
		callContext context.Context, _ int, reservation Reservation,
	) (struct{}, error) {
		return struct{}{}, c.runtime.Release(callContext, reservation)
	})
	for index, reservation := range reservations {
		if results[index].err != nil {
			failures = append(failures, fmt.Errorf(
				"release Runtime Agent allocation %q: %w",
				reservation.Grant.AllocationID,
				results[index].err,
			))
			continue
		}
		if err := c.registry.Release(reservation.Grant.AllocationID); err != nil {
			failures = append(failures, fmt.Errorf("release registry allocation %q: %w", reservation.Grant.AllocationID, err))
		}
	}
	return errors.Join(failures...)
}

func (c *RuntimeBatchController) cleanupFailedPrepare(reservations []Reservation) error {
	deadline := c.currentTime().Add(c.cleanupTimeout)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()
	reason := contracts.TerminationError{
		Code: "allocation_batch_preparation_failed", Message: "Stage allocation batch preparation failed", Retryable: true,
	}
	var failures []error
	safeReason := safeTerminationReason(reason)
	for _, reservation := range reservations {
		allocationID := reservation.Grant.AllocationID
		if err := c.registry.SetWriteFence(allocationID); err != nil {
			failures = append(failures, fmt.Errorf("fence failed prepare allocation %q: %w", allocationID, err))
		}
		if err := c.registry.SetAllocationPhase(allocationID, AllocationAborting, &safeReason); err != nil {
			failures = append(failures, fmt.Errorf("observe failed prepare allocation %q: %w", allocationID, err))
		}
	}
	abortIDs := make([]string, len(reservations))
	abortIDErrors := make([]error, len(reservations))
	for index := range reservations {
		abortIDs[index], abortIDErrors[index] = c.nextID("abort_")
	}
	type cleanupResult struct {
		report              contracts.AllocationFinalReport
		reportAvailable     bool
		abortErr            error
		releasingPhaseError error
		releaseErr          error
	}
	results := fanOutRuntimeCalls(ctx, reservations, func(
		callContext context.Context, index int, reservation Reservation,
	) (cleanupResult, error) {
		result := cleanupResult{}
		if abortIDErrors[index] != nil {
			result.abortErr = fmt.Errorf("create cleanup abort ID: %w", abortIDErrors[index])
		} else {
			result.report, result.abortErr = c.runtime.Abort(
				callContext, reservation, abortIDs[index], reason, deadline,
			)
			result.reportAvailable = result.abortErr == nil
		}
		result.releasingPhaseError = c.registry.SetAllocationPhase(
			reservation.Grant.AllocationID, AllocationReleasing, nil,
		)
		result.releaseErr = c.runtime.Release(callContext, reservation)
		return result, nil
	})
	for index, reservation := range reservations {
		allocationID := reservation.Grant.AllocationID
		if results[index].err != nil {
			failures = append(failures, fmt.Errorf("cleanup failed prepare allocation %q: %w", allocationID, results[index].err))
			continue
		}
		result := results[index].value
		if result.abortErr != nil {
			failures = append(failures, fmt.Errorf("abort failed prepare allocation %q: %w", allocationID, result.abortErr))
		}
		if result.reportAvailable {
			if err := c.registry.RecordAllocationReport(allocationID, result.report); err != nil {
				failures = append(failures, fmt.Errorf("observe failed prepare report %q: %w", allocationID, err))
			}
		}
		if result.releasingPhaseError != nil {
			failures = append(failures, fmt.Errorf("observe failed prepare release %q: %w", allocationID, result.releasingPhaseError))
		}
		if result.releaseErr != nil {
			failures = append(failures, fmt.Errorf("release failed prepare Runtime Agent allocation %q: %w", allocationID, result.releaseErr))
			continue
		}
		if err := c.registry.Release(allocationID); err != nil {
			failures = append(failures, fmt.Errorf("release failed prepare registry allocation %q: %w", allocationID, err))
		}
	}
	return errors.Join(failures...)
}

func (c *RuntimeBatchController) nextID(prefix string) (string, error) {
	c.idMu.Lock()
	defer c.idMu.Unlock()
	return c.newID(prefix)
}

func (c *RuntimeBatchController) currentTime() time.Time {
	c.nowMu.Lock()
	defer c.nowMu.Unlock()
	return c.now()
}

type runtimeCallResult[T any] struct {
	value T
	err   error
}

func fanOutRuntimeCalls[T any](
	ctx context.Context,
	reservations []Reservation,
	call func(context.Context, int, Reservation) (T, error),
) []runtimeCallResult[T] {
	results := make([]runtimeCallResult[T], len(reservations))
	var group sync.WaitGroup
	group.Add(len(reservations))
	for index := range reservations {
		go func(index int) {
			defer group.Done()
			results[index].value, results[index].err = call(ctx, index, reservations[index])
		}(index)
	}
	group.Wait()
	return results
}

func safeTerminationReason(source contracts.TerminationError) SafeReason {
	result := SafeReason{Code: source.Code, Retryable: source.Retryable}
	if !validSafeReason(result) {
		result.Code = "allocation_aborted"
	}
	return result
}

func validateReservationBatch(reservations []Reservation) error {
	if len(reservations) == 0 {
		return errors.New("allocation reservation batch must not be empty")
	}
	allocationIDs := make(map[string]struct{}, len(reservations))
	logicalNames := make(map[string]struct{}, len(reservations))
	for _, reservation := range reservations {
		if strings.TrimSpace(reservation.Grant.AllocationID) == "" || strings.TrimSpace(reservation.Grant.LogicalAgentName) == "" {
			return errors.New("allocation reservation is incomplete")
		}
		if _, duplicate := allocationIDs[reservation.Grant.AllocationID]; duplicate {
			return errors.New("allocation reservation batch contains duplicate allocation IDs")
		}
		if _, duplicate := logicalNames[reservation.Grant.LogicalAgentName]; duplicate {
			return errors.New("allocation reservation batch contains duplicate logical Agent names")
		}
		allocationIDs[reservation.Grant.AllocationID] = struct{}{}
		logicalNames[reservation.Grant.LogicalAgentName] = struct{}{}
	}
	return nil
}

func lifecycleID(prefix string) (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(buffer), nil
}
