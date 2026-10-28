package public

import (
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/auditservice"
)

// preflightAudit validates one owner-selected draft revision. Its POST is a
// bounded read: it does not create a start receipt, hold or target action.
func (h *auditHandlers) preflightAudit(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	revision, err := requiredRevision(r, "Audit preflight")
	if err != nil {
		handleError(w, err)
		return
	}
	if err := requireEmptyBody(w, r); err != nil {
		handleError(w, err)
		return
	}
	credentialUser := runtimeCredentialUser(r.Context())
	result, err := h.Audits.Preflight(r.Context(), auditservice.PreflightParams{
		OwnerID: credentialUser.UserID, AuditID: r.PathValue("auditId"), ExpectedRevision: revision,
		OperationsPrincipal: credentialUser.Operations,
	})
	if err != nil {
		handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(result.Revision, 10)))
	writeJSON(w, http.StatusOK, result)
}
