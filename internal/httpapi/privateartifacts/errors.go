package privateartifacts

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
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

type privateFailure struct {
	status        int
	code, message string
	retryable     bool
}

func (f privateFailure) Status() int     { return f.status }
func (f privateFailure) Code() string    { return f.code }
func (f privateFailure) Retryable() bool { return f.retryable }
func newPrivateFailure(status int, code, message string, retryable bool) privateFailure {
	return privateFailure{status, code, message, retryable}
}
func (h *handler) handleError(w http.ResponseWriter, err error) {
	failure := classifyPrivateError(err)
	if failure.code == "artifact_history_maintenance_pending" {
		w.Header().Set("Retry-After", strconv.Itoa(int(artifactpostgres.HistoryRetryAfter/time.Second)))
	}
	h.writeError(w, failure.Status(), failure.Code(), failure.message, failure.Retryable())
}

func classifyPrivateError(err error) privateFailure {
	var validation *auditdomain.ValidationError
	switch {
	case errors.As(err, &validation):
		kind := validation.ErrorKind()
		return newPrivateFailure(kind.Status(), kind.Code(), "finding selection does not satisfy the collection contract", kind.Retryable())
	case errors.Is(err, errSitemapAccessDenied):
		return newPrivateFailure(http.StatusForbidden, "sitemap_access_denied", "allocation does not permit this sitemap operation", false)
	case errors.Is(err, errSitemapUnavailable):
		return newPrivateFailure(http.StatusServiceUnavailable, "sitemap_unavailable", "sitemap service is unavailable", true)
	case errors.Is(err, projectsitemap.ErrConflict):
		return newPrivateFailure(http.StatusConflict, "sitemap_conflict", "sitemap edit or source precondition did not match", false)
	case errors.Is(err, contracts.ErrIdempotencyConflict):
		return newPrivateFailure(http.StatusConflict, "idempotency_conflict", "the mutation key is already bound to another intent", false)
	case errors.Is(err, projectstore.ErrDeleting), errors.Is(err, artifacts.ErrScopeDeleting):
		return newPrivateFailure(http.StatusConflict, "project_deleting", "the Project is deleting", false)
	case errors.Is(err, projectstore.ErrNotFound):
		return newPrivateFailure(http.StatusNotFound, "artifact_not_found", "artifact was not found", false)
	case errors.Is(err, controlplane.ErrAllocationNotFound):
		return newPrivateFailure(http.StatusNotFound, "allocation_not_found", "active allocation was not found", false)
	case errors.Is(err, errAllocationWriteFenced):
		return newPrivateFailure(http.StatusConflict, "allocation_write_fenced", "allocation no longer permits artifact writes", false)
	case errors.Is(err, errArtifactAccessDenied), errors.Is(err, artifacts.ErrReservedNamespace):
		return newPrivateFailure(http.StatusForbidden, "artifact_access_denied", "allocation does not permit this artifact operation", false)
	case errors.Is(err, blob.ErrTooLarge):
		return newPrivateFailure(http.StatusRequestEntityTooLarge, "artifact_too_large", "artifact exceeds the configured payload limit", false)
	case errors.Is(err, blob.ErrTransferCapacity):
		return newPrivateFailure(http.StatusServiceUnavailable, "artifact_transfer_capacity", "artifact transfer capacity is exhausted", true)
	case errors.Is(err, blob.ErrIntegrity):
		return newPrivateFailure(http.StatusServiceUnavailable, "artifact_content_corrupt", "artifact content failed integrity verification", false)
	case errors.Is(err, blob.ErrMissing):
		return newPrivateFailure(http.StatusServiceUnavailable, "artifact_content_missing", "artifact content is unavailable", false)
	case errors.Is(err, artifacts.ErrArtifactFrozen):
		// A frozen binding never advances, so no re-read can make a retry succeed.
		return newPrivateFailure(http.StatusConflict, "artifact_frozen", "artifact binding is frozen and cannot change", false)
	case errors.Is(err, artifacts.ErrArtifactConflict):
		return newPrivateFailure(http.StatusConflict, "artifact_conflict", "artifact revision precondition did not match", true)
	case errors.Is(err, artifacts.ErrHistoryProtected):
		// Only the release of a protecting reference changes the outcome
		// (ART-R126).
		return newPrivateFailure(http.StatusConflict, "artifact_history_protected",
			"artifact history is protected and cannot fit its retention limit", false)
	case errors.Is(err, artifacts.ErrHistoryMaintenancePending):
		return newPrivateFailure(http.StatusServiceUnavailable, "artifact_history_maintenance_pending",
			"artifact history maintenance is pending; retry shortly", true)
	case errors.Is(err, artifacts.ErrRunRevisionLimit):
		// The Run never regains revisions, so no retry can succeed (ART-R118).
		return newPrivateFailure(http.StatusConflict, "artifact_limit_exceeded", "the Run has reached its artifact revision limit", false)
	case errors.Is(err, findingintake.ErrCollectionKeyConflict):
		return newPrivateFailure(http.StatusConflict, "idempotency_conflict", "selection key is bound to another selection", false)
	case errors.Is(err, findingintake.ErrConflict):
		return newPrivateFailure(http.StatusConflict, "finding_submission_conflict", "finding identity or revision conflicts", false)
	case errors.Is(err, findingintake.ErrAccessDenied), errors.Is(err, findingintake.ErrToolNotSelected):
		return newPrivateFailure(http.StatusForbidden, "finding_access_denied", "allocation does not permit this finding operation", false)
	case errors.Is(err, findingintake.ErrNotFound):
		return newPrivateFailure(http.StatusNotFound, "finding_receipt_not_found", "finding receipt was not found", false)
	case errors.Is(err, artifacts.ErrArtifactNotFound):
		return newPrivateFailure(http.StatusNotFound, "artifact_not_found", "artifact was not found", false)
	case errors.Is(err, errInvalidRequest), errors.Is(err, projectsitemap.ErrInvalid), errors.Is(err, artifacts.ErrInvalidScope),
		errors.Is(err, artifacts.ErrInvalidName), errors.Is(err, artifacts.ErrInvalidMediaType),
		errors.Is(err, artifacts.ErrVersionedWriteTarget), errors.Is(err, contracts.ErrValidation):
		return newPrivateFailure(http.StatusBadRequest, "invalid_request", "request does not satisfy the private Artifact API contract", false)
	case errors.Is(err, findingintake.ErrProjectRequired):
		return newPrivateFailure(http.StatusBadRequest, "invalid_finding_request", "finding requires an active Project", false)
	case errors.Is(err, findingintake.ErrEvidenceRequired):
		return newPrivateFailure(http.StatusBadRequest, "invalid_finding_request", "finding requires selected exact evidence", false)
	case errors.Is(err, findingintake.ErrInvalid):
		return newPrivateFailure(http.StatusBadRequest, "invalid_finding_request", "request does not satisfy the finding intake contract", false)
	case persistencepostgres.IsTransactionConflict(err):
		return newPrivateFailure(http.StatusServiceUnavailable, "storage_transaction_conflict", "database transaction conflicted; retry the request", true)
	case persistencepostgres.Classify(err) == persistencepostgres.ClassInvalid:
		return newPrivateFailure(http.StatusBadRequest, "invalid_request", "request does not satisfy the private Artifact API contract", false)
	default:
		return newPrivateFailure(http.StatusInternalServerError, "internal_error", "artifact operation could not be processed", true)
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
