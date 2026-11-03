package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
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
	if err := c.postJSON(
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
	if err := c.postJSON(
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
		// One slot per Runtime Agent: after a release attempt no lifecycle
		// call of this allocation follows, and a retry or the next allocation
		// opens and verifies a new connection (CTL-R164).
		defer c.transports.Evict(reservation.Grant.RuntimeAgentID)
	}
	request := control.ReleaseAllocationRequest{
		APIVersion: contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
	}
	if err := request.Validate(); err != nil {
		return fmt.Errorf("build release request: %w", err)
	}
	target, err := c.endpoint(reservation.ControlURL, reservation.Grant.AllocationID, "release")
	if err != nil {
		return err
	}
	body, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("encode release request: %w", err)
	}
	response, err := c.do(ctx, target, body, reservation.Grant.RuntimeAgentID)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusNoContent {
		return decodeRuntimeError(response)
	}
	limited := io.LimitReader(response.Body, 1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return persistencepostgres.WrapError("read Runtime Agent release response", err)
	}
	if len(data) != 0 {
		return errors.New("Runtime Agent release response must be empty")
	}
	return nil
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
