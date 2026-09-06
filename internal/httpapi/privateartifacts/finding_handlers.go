package privateartifacts

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
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
	input, err := readFindingSubmission(w, r)
	if err != nil {
		h.handleError(w, err)
		return
	}
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

func readFindingSubmission(w http.ResponseWriter, r *http.Request) (findingintake.Submission, error) {
	var result findingintake.Submission
	if r.ContentLength < -1 || r.ContentLength > findingintake.MaxRequestBytes {
		return result, findingintake.ErrInvalid
	}
	body := http.MaxBytesReader(w, r.Body, findingintake.MaxRequestBytes)
	data, err := io.ReadAll(body)
	if err != nil {
		return result, findingintake.ErrInvalid
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&result); err != nil {
		return result, findingintake.ErrInvalid
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return result, findingintake.ErrInvalid
	}
	return result, nil
}

func findingSubmissionResponse(
	receipt findingintake.Receipt,
	replayed bool,
) findingintake.SubmissionResponse {
	return findingintake.SubmissionResponse{
		APIVersion: findingintake.APIVersion,
		ProposalID: receipt.ProposalID,
		ReceiptID:  receipt.ReceiptID,
		Proposal:   receipt.Proposal,
		Replayed:   replayed,
	}
}
