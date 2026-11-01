package privateartifacts

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

var (
	errInvalidRequest        = errors.New("invalid private Artifact API request")
	errArtifactAccessDenied  = errors.New("allocation artifact access denied")
	errAllocationWriteFenced = errors.New("allocation artifact writes are fenced")
)

type errorResponse struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
	RequestID string `json:"requestId"`
}

func (h *handler) handleError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, errSitemapAccessDenied):
		h.writeError(w, http.StatusForbidden, "sitemap_access_denied", "allocation does not permit this sitemap operation", false)
	case errors.Is(err, errSitemapUnavailable):
		h.writeError(w, http.StatusServiceUnavailable, "sitemap_unavailable", "sitemap service is unavailable", true)
	case errors.Is(err, projectsitemap.ErrConflict):
		h.writeError(w, http.StatusConflict, "sitemap_conflict", "sitemap edit or source precondition did not match", false)
	case errors.Is(err, contracts.ErrIdempotencyConflict):
		h.writeError(w, http.StatusConflict, "idempotency_conflict", "the mutation key is already bound to another intent", false)
	case errors.Is(err, projectstore.ErrDeleting), errors.Is(err, artifacts.ErrScopeDeleting):
		h.writeError(w, http.StatusConflict, "project_deleting", "the Project is deleting", false)
	case errors.Is(err, projectstore.ErrNotFound):
		h.writeError(w, http.StatusNotFound, "artifact_not_found", "artifact was not found", false)
	case errors.Is(err, controlplane.ErrAllocationNotFound):
		h.writeError(w, http.StatusNotFound, "allocation_not_found", "active allocation was not found", false)
	case errors.Is(err, errAllocationWriteFenced):
		h.writeError(w, http.StatusConflict, "allocation_write_fenced", "allocation no longer permits artifact writes", false)
	case errors.Is(err, errArtifactAccessDenied), errors.Is(err, artifacts.ErrReservedNamespace):
		h.writeError(w, http.StatusForbidden, "artifact_access_denied", "allocation does not permit this artifact operation", false)
	case errors.Is(err, artifacts.ErrPayloadTooLarge):
		h.writeError(w, http.StatusRequestEntityTooLarge, "artifact_too_large", "artifact exceeds the configured payload limit", false)
	case errors.Is(err, artifacts.ErrTransferCapacity):
		h.writeError(w, http.StatusServiceUnavailable, "artifact_transfer_capacity", "artifact transfer capacity is exhausted", true)
	case errors.Is(err, artifacts.ErrArtifactIntegrity):
		h.writeError(w, http.StatusServiceUnavailable, "artifact_content_corrupt", "artifact content failed integrity verification", false)
	case errors.Is(err, artifacts.ErrBlobMissing):
		h.writeError(w, http.StatusServiceUnavailable, "artifact_content_missing", "artifact content is unavailable", false)
	case errors.Is(err, artifacts.ErrArtifactFrozen):
		// A frozen binding never advances, so no re-read can make a retry succeed.
		h.writeError(w, http.StatusConflict, "artifact_frozen", "artifact binding is frozen and cannot change", false)
	case errors.Is(err, artifacts.ErrArtifactConflict):
		h.writeError(w, http.StatusConflict, "artifact_conflict", "artifact revision precondition did not match", true)
	case errors.Is(err, artifacts.ErrHistoryProtected):
		// Only the release of a protecting reference changes the outcome
		// (ART-R126).
		h.writeError(w, http.StatusConflict, "artifact_history_protected",
			"artifact history is protected and cannot fit its retention limit", false)
	case errors.Is(err, artifacts.ErrHistoryMaintenancePending):
		w.Header().Set("Retry-After", strconv.Itoa(int(artifacts.HistoryRetryAfter/time.Second)))
		h.writeError(w, http.StatusServiceUnavailable, "artifact_history_maintenance_pending",
			"artifact history maintenance is pending; retry shortly", true)
	case errors.Is(err, artifacts.ErrRunRevisionLimit):
		// The Run never regains revisions, so no retry can succeed (ART-R118).
		h.writeError(w, http.StatusConflict, "artifact_limit_exceeded", "the Run has reached its artifact revision limit", false)
	case errors.Is(err, findingintake.ErrConflict):
		h.writeError(w, http.StatusConflict, "finding_submission_conflict", "finding submission identity already has different content", false)
	case errors.Is(err, findingintake.ErrAccessDenied), errors.Is(err, findingintake.ErrToolNotSelected):
		h.writeError(w, http.StatusForbidden, "finding_access_denied", "allocation does not permit finding submission", false)
	case errors.Is(err, findingintake.ErrNotFound):
		h.writeError(w, http.StatusNotFound, "finding_receipt_not_found", "finding receipt was not found", false)
	case errors.Is(err, artifacts.ErrArtifactNotFound):
		h.writeError(w, http.StatusNotFound, "artifact_not_found", "artifact was not found", false)
	case errors.Is(err, errInvalidRequest), errors.Is(err, projectsitemap.ErrInvalid), errors.Is(err, artifacts.ErrInvalidScope),
		errors.Is(err, artifacts.ErrInvalidName), errors.Is(err, artifacts.ErrInvalidMediaType),
		errors.Is(err, artifacts.ErrVersionedWriteTarget), errors.Is(err, contracts.ErrValidation):
		h.writeError(w, http.StatusBadRequest, "invalid_request", "request does not satisfy the private Artifact API contract", false)
	case errors.Is(err, findingintake.ErrInvalid):
		h.writeError(w, http.StatusBadRequest, "invalid_finding_request", "request does not satisfy the finding intake contract", false)
	case persistencepostgres.IsTransactionConflict(err):
		h.writeError(w, http.StatusServiceUnavailable, "storage_transaction_conflict", "database transaction conflicted; retry the request", true)
	case persistencepostgres.Classify(err) == persistencepostgres.ClassInvalid:
		h.writeError(w, http.StatusBadRequest, "invalid_request", "request does not satisfy the private Artifact API contract", false)
	default:
		h.writeError(w, http.StatusInternalServerError, "internal_error", "artifact operation could not be processed", true)
	}
}

func (h *handler) writeError(w http.ResponseWriter, status int, code, message string, retryable bool) {
	writeJSON(w, status, errorResponse{
		Code: code, Message: message, Retryable: retryable,
		RequestID: requestid.FromResponse(w),
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	body, err := json.Marshal(value)
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	body = append(body, '\n')
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
