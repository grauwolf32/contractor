package controlplane

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

func (c *RuntimeControlClient) Finalize(
	ctx context.Context,
	reservation Reservation,
	finalizationID string,
	deadline time.Time,
) (reporting.AllocationFinalReport, error) {
	request := control.FinalizeAllocationRequest{
		APIVersion: contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
		FinalizationID: finalizationID, Deadline: deadline,
	}
	if err := request.Validate(); err != nil {
		return reporting.AllocationFinalReport{}, fmt.Errorf("build finalize request: %w", err)
	}
	var response reporting.AllocationFinalResponse
	if err := c.callLifecycle(
		ctx, reservation.ControlURL, reservation.Grant.AllocationID,
		reservation.Grant.RuntimeAgentID, "finalize", request, &response, false,
	); err != nil {
		return reporting.AllocationFinalReport{}, err
	}
	sanitizeRuntimeAdapterMetrics(&response.Report.Runtime, reservation)
	response.Report.Runtime.ApplyResourcePolicy(reservation.PerformanceCollectionPolicy)
	if err := response.Validate(); err != nil {
		return reporting.AllocationFinalReport{}, errors.New("Runtime Agent returned a lifecycle response that violates the contract")
	}
	if response.Report.AllocationID != reservation.Grant.AllocationID {
		return reporting.AllocationFinalReport{}, errors.New("Runtime Agent final report identifies another allocation")
	}
	return response.Report, nil
}

func (c *RuntimeControlClient) Abort(
	ctx context.Context,
	reservation Reservation,
	abortID string,
	reason contracts.TerminationError,
	deadline time.Time,
) (reporting.AllocationFinalReport, error) {
	request := control.AbortAllocationRequest{
		APIVersion: contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
		AbortID: abortID, Reason: reason, Deadline: deadline,
	}
	if err := request.Validate(); err != nil {
		return reporting.AllocationFinalReport{}, fmt.Errorf("build abort request: %w", err)
	}
	var response reporting.AllocationFinalResponse
	if err := c.callLifecycle(
		ctx, reservation.ControlURL, reservation.Grant.AllocationID,
		reservation.Grant.RuntimeAgentID, "abort", request, &response, false,
	); err != nil {
		return reporting.AllocationFinalReport{}, err
	}
	sanitizeRuntimeAdapterMetrics(&response.Report.Runtime, reservation)
	response.Report.Runtime.ApplyResourcePolicy(reservation.PerformanceCollectionPolicy)
	if err := response.Validate(); err != nil {
		return reporting.AllocationFinalReport{}, errors.New("Runtime Agent returned a lifecycle response that violates the contract")
	}
	if response.Report.AllocationID != reservation.Grant.AllocationID {
		return reporting.AllocationFinalReport{}, errors.New("Runtime Agent abort report identifies another allocation")
	}
	return response.Report, nil
}

func (c *RuntimeControlClient) Release(ctx context.Context, reservation Reservation) error {
	if c.transports != nil {
		// Retire this principal's cached connections at release (CTL-R164).
		// Other slots' exchanges finish; later calls verify a new connection.
		defer c.transports.Evict(reservation.Grant.RuntimeAgentID)
	}
	request := control.ReleaseAllocationRequest{
		APIVersion: contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
	}
	if err := request.Validate(); err != nil {
		return fmt.Errorf("build release request: %w", err)
	}
	if c.channels != nil {
		defer c.channels.Evict(reservation.Grant.RuntimeAgentID)
	}
	return c.callLifecycle(ctx, reservation.ControlURL, reservation.Grant.AllocationID,
		reservation.Grant.RuntimeAgentID, "release", request, nil, false)
}

func sanitizeRuntimeAdapterMetrics(report *reporting.RuntimeReport, reservation Reservation) {
	expected := map[runtimesettings.RuntimeAdapterRef]struct{}{}
	if reservation.ResolvedRuntimeConfig != nil {
		for _, ref := range reservation.ResolvedRuntimeConfig.RequiredRuntimeAdapters {
			expected[ref] = struct{}{}
		}
	}
	if report.Adapters == nil {
		report.Adapters = map[runtimesettings.RuntimeAdapterRef]reporting.RuntimeAdapterMetrics{}
	}
	for ref, metrics := range report.Adapters {
		_, selected := expected[ref]
		if !selected || metrics.Validate() != nil {
			delete(report.Adapters, ref)
			report.Complete = false
		}
	}
	for ref := range expected {
		if _, reported := report.Adapters[ref]; !reported {
			report.Complete = false
		}
	}
}
