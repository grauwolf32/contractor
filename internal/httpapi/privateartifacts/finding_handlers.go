package privateartifacts

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
)

func (h *handler) postFindingProposal(w http.ResponseWriter, r *http.Request) {
	if h.dependencies.Findings == nil {
		h.handleError(w, errors.New("finding intake is not configured"))
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, fmt.Errorf("%w: finding proposal requires application/json", errInvalidRequest))
		return
	}
	grant, identity, err := h.allocationGrant(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	r, transfer, err := artifacttransfer.Acquire(w, r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	defer transfer.Close()
	input, err := readFindingSubmission(r, transfer)
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		h.handleError(w, err)
		return
	}
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	r = r.WithContext(ctx)
	var receipt findingintake.Receipt
	var found bool
	var replayed bool
	err = h.dependencies.Registry.WithWriteGrant(
		grant.AllocationID,
		func(current controlplane.AllocationGrant) error {
			if _, storeErr := h.runStoreForGrant(grant.AllocationID, current, identity, false); storeErr != nil {
				return storeErr
			}
			var replayErr error
			receipt, found, replayErr = h.dependencies.Findings.FindReplay(r.Context(), current, input)
			if replayErr != nil || found {
				replayed = found
				return replayErr
			}
			if _, storeErr := h.runStoreForGrant(grant.AllocationID, current, identity, true); storeErr != nil {
				return storeErr
			}
			var submitErr error
			receipt, replayed, submitErr = h.dependencies.Findings.Submit(r.Context(), current, input)
			return submitErr
		},
	)
	transfer.ReleaseBeforeWrite(0)
	if err != nil {
		h.handleError(w, err)
		return
	}
	status := http.StatusCreated
	if replayed {
		status = http.StatusOK
	}
	writeJSON(w, status, findingSubmissionResponse(receipt, replayed))
}

func readFindingSubmission(r *http.Request, transfer *artifacttransfer.Transfer) (findingintake.Submission, error) {
	var result findingintake.Submission
	if r.ContentLength < -1 {
		return result, findingintake.ErrInvalid
	}
	data, err := transfer.ReadBody(r, findingintake.MaxRequestBytes)
	if err != nil || decodeRequestJSON(data, &result) != nil {
		return result, findingintake.ErrInvalid
	}
	return result, nil
}

func findingSubmissionResponse(
	receipt findingintake.Receipt,
	replayed bool,
) findingintake.SubmissionResponse {
	evidence := receipt.Evidence
	if evidence == nil {
		evidence = []findingintake.ExactArtifact{}
	}
	return findingintake.SubmissionResponse{
		FindingID:        receipt.FindingID,
		PublicationError: receipt.PublicationError,
		APIVersion:       findingintake.APIVersion,
		ProposalID:       receipt.ProposalID,
		ReceiptID:        receipt.ReceiptID,
		Proposal:         receipt.Proposal,
		Evidence:         evidence,
		Replayed:         replayed,
	}
}
