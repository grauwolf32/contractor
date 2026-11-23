package privateartifacts

import (
	"errors"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/runstore"
	"net/http"
)

func (h *handler) postAuditJudgments(w http.ResponseWriter, r *http.Request) {
	if h.dependencies.AuditJudgments == nil {
		h.handleError(w, errors.New("Audit result service unavailable"))
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	media, err := requestMediaType(r)
	if err != nil || media != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	grant, identity, err := h.allocationGrant(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	r, transfer, err := artifacttransfer.Acquire(w, r, h.dependencies.Transfers)
	if err != nil {
		h.handleError(w, err)
		return
	}
	defer transfer.Close()
	raw, err := transfer.ReadBody(r, auditcontract.MaximumDocumentBytes)
	var input auditcontract.JudgmentSubmission
	if err != nil || decodeRequestJSON(raw, &input) != nil || input.Validate() != nil {
		transfer.ReleaseBeforeWrite(0)
		h.handleError(w, errInvalidRequest)
		return
	}
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	var receipt auditcontract.JudgmentReceipt
	err = h.dependencies.Registry.WithWriteGrant(grant.AllocationID, func(current controlplane.AllocationGrant) error {
		if _, err := h.runStoreForGrant(grant.AllocationID, current, identity, false); err != nil {
			return err
		}
		var err error
		receipt, err = h.dependencies.AuditJudgments.RecordJudgments(ctx, current, input)
		return err
	})
	transfer.ReleaseBeforeWrite(0)
	var stateConflict *runstore.StateConflictError
	var stale *auditservice.JudgmentStaleError
	switch {
	case errors.As(err, &stale):
		writeJSON(w, http.StatusConflict, struct {
			errorResponse
			APIVersion         string                             `json:"apiVersion"`
			CurrentGenerations []auditcontract.JudgmentGeneration `json:"currentGenerations"`
		}{errorResponse: errorResponse{Code: "audit_result_stale", Message: "result changed; a later model response may submit a new judgment", Retryable: false, RequestID: requestid.FromResponse(w)}, APIVersion: contracts.APIVersion, CurrentGenerations: stale.Generations})
		return
	case errors.Is(err, auditstore.ErrEvidenceBudgetExhausted):
		h.writeError(w, http.StatusConflict, "audit_result_limit_exceeded", "selected results exceed the Audit collection limit", false)
		return
	case errors.Is(err, auditstore.ErrConflict):
		h.writeError(w, http.StatusConflict, "audit_result_conflict", "result submission identity conflicts", false)
		return
	case errors.Is(err, auditstore.ErrPrecondition), errors.Is(err, runstore.ErrNotFound), errors.Is(err, runstore.ErrPrecondition), errors.Is(err, runstore.ErrProjectDeleting), errors.Is(err, auditstore.ErrProjectDeleting), errors.As(err, &stateConflict):
		h.writeError(w, http.StatusConflict, "audit_result_closed", "Audit attempt no longer accepts results", false)
		return
	case errors.Is(err, auditstore.ErrInvalid), errors.Is(err, auditservice.ErrInvalid), errors.Is(err, auditstore.ErrNotFound):
		h.handleError(w, errInvalidRequest)
		return
	case err != nil:
		h.handleError(w, err)
		return
	}
	status := http.StatusCreated
	if receipt.Replayed {
		status = http.StatusOK
	}
	writeJSON(w, status, receipt)
}
