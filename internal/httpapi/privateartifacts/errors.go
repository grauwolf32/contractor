package privateartifacts

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/requestid"
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
	case errors.Is(err, controlplane.ErrAllocationNotFound):
		h.writeError(w, http.StatusNotFound, "allocation_not_found", "active allocation was not found", false)
	case errors.Is(err, errAllocationWriteFenced):
		h.writeError(w, http.StatusConflict, "allocation_write_fenced", "allocation no longer permits artifact writes", false)
	case errors.Is(err, errArtifactAccessDenied), errors.Is(err, artifacts.ErrReservedNamespace):
		h.writeError(w, http.StatusForbidden, "artifact_access_denied", "allocation does not permit this artifact operation", false)
	case errors.Is(err, artifacts.ErrPayloadTooLarge):
		h.writeError(w, http.StatusRequestEntityTooLarge, "artifact_too_large", "artifact exceeds the 16 MiB limit", false)
	case errors.Is(err, artifacts.ErrArtifactConflict), errors.Is(err, artifacts.ErrArtifactFrozen):
		h.writeError(w, http.StatusConflict, "artifact_conflict", "artifact revision precondition did not match", true)
	case errors.Is(err, artifacts.ErrArtifactNotFound):
		h.writeError(w, http.StatusNotFound, "artifact_not_found", "artifact was not found", false)
	case errors.Is(err, errInvalidRequest), errors.Is(err, artifacts.ErrInvalidScope),
		errors.Is(err, artifacts.ErrInvalidName), errors.Is(err, artifacts.ErrInvalidMediaType),
		errors.Is(err, artifacts.ErrVersionedWriteTarget), errors.Is(err, contracts.ErrValidation):
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
