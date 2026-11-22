package privateartifacts

import (
	"errors"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/runstore"
	"net/http"
)

func (h *handler) postAuditEvidence(w http.ResponseWriter, r *http.Request) {
	if h.dependencies.AuditEvidence == nil {
		h.handleError(w, errors.New("Audit evidence service unavailable"))
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
	var input auditcontract.EvidenceSubmission
	if err != nil || decodeRequestJSON(raw, &input) != nil || input.Validate() != nil {
		transfer.ReleaseBeforeWrite(0)
		h.handleError(w, errInvalidRequest)
		return
	}
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	var receipt auditcontract.EvidenceReceipt
	err = h.dependencies.Registry.WithWriteGrant(grant.AllocationID, func(current controlplane.AllocationGrant) error {
		if _, err := h.runStoreForGrant(grant.AllocationID, current, identity, false); err != nil {
			return err
		}
		var err error
		receipt, err = h.dependencies.AuditEvidence.RecordEvidence(ctx, current, input)
		return err
	})
	transfer.ReleaseBeforeWrite(0)
	var stateConflict *runstore.StateConflictError
	switch {
	case errors.Is(err, auditstore.ErrEvidenceBudgetExhausted):
		h.writeError(w, http.StatusConflict, "audit_evidence_limit_exceeded", "selected evidence exceeds the Audit collection limit", false)
		return
	case errors.Is(err, auditstore.ErrConflict):
		h.writeError(w, http.StatusConflict, "audit_evidence_conflict", "evidence submission identity conflicts", false)
		return
	case errors.Is(err, auditstore.ErrPrecondition), errors.Is(err, runstore.ErrNotFound), errors.Is(err, runstore.ErrPrecondition), errors.Is(err, runstore.ErrProjectDeleting), errors.Is(err, auditstore.ErrProjectDeleting), errors.As(err, &stateConflict):
		h.writeError(w, http.StatusConflict, "audit_evidence_closed", "Audit attempt no longer accepts evidence", false)
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
