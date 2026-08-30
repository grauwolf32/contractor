package public

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (h *handler) handleError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, config.ErrPublicationConflict):
		h.writeError(w, http.StatusConflict, "configuration_conflict", "immutable configuration identity or idempotency key conflicts", false)
	case errors.Is(err, artifacts.ErrPayloadTooLarge):
		h.writeError(w, http.StatusRequestEntityTooLarge, "artifact_too_large", "artifact exceeds the 16 MiB limit", false)
	case errors.Is(err, artifacts.ErrArtifactConflict), errors.Is(err, artifacts.ErrArtifactFrozen),
		errors.Is(err, runstore.ErrConflict):
		h.writeError(w, http.StatusConflict, "conflict", "resource state changed; retry with the current revision", true)
	case errors.Is(err, artifacts.ErrArtifactNotFound), errors.Is(err, runstore.ErrNotFound),
		errors.Is(err, config.ErrConfigurationNotFound):
		h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
	case errors.Is(err, errInvalidRequest), errors.Is(err, artifacts.ErrInvalidScope),
		errors.Is(err, artifacts.ErrInvalidName), errors.Is(err, artifacts.ErrInvalidMediaType),
		errors.Is(err, artifacts.ErrVersionedWriteTarget), errors.Is(err, artifacts.ErrExactRevisionRequired),
		errors.Is(err, artifacts.ErrReservedNamespace), errors.Is(err, contracts.ErrValidation),
		errors.Is(err, runstore.ErrInvalid), errors.Is(err, config.ErrInvalidConfigurationKind),
		errors.Is(err, config.ErrInvalidPublication):
		h.writeError(w, http.StatusBadRequest, "invalid_request", "request does not satisfy the API contract", false)
	default:
		h.writeError(w, http.StatusInternalServerError, "internal_error", "request could not be processed", true)
	}
}

func (h *handler) writeError(
	w http.ResponseWriter,
	status int,
	code string,
	message string,
	retryable bool,
) {
	writeJSON(w, status, errorResponse{
		Code: code, Message: message, Retryable: retryable,
		RequestID: requestid.FromResponse(w),
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
