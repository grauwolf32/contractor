package public

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func (h *handler) handleError(w http.ResponseWriter, err error) {
	var credentialInUse *credentials.CredentialInUseError
	var runtimeCredentialInUse *credentials.RuntimeCredentialInUseError
	var runtimeLabelInUse *runtimeconfig.LabelInUseError
	switch {
	case errors.Is(err, runtimeconfig.ErrPrecondition):
		h.writeError(w, http.StatusPreconditionFailed, "precondition_failed", "resource revision precondition failed", false)
	case errors.Is(err, runtimeconfig.ErrAgentLabelNotApplicable):
		h.writeError(w, http.StatusBadRequest, "runtime_agent_label_not_applicable", "Runtime label has no Worker-applicable setting", false)
	case errors.Is(err, runtimeconfig.ErrPrincipalInUse):
		h.writeError(w, http.StatusConflict, "runtime_agent_in_use", "Runtime Agent principal is live, labeled, or allocation-referenced", false)
	case errors.As(err, &runtimeLabelInUse):
		writeJSON(w, http.StatusConflict, errorResponse{
			Code: "runtime_label_in_use", Message: "Runtime label is assigned to a Runtime Agent",
			Retryable: false, RequestID: requestid.FromResponse(w),
			Details: &runtimeLabelInUseDetailsResponse{
				Kind: "runtime_label_in_use", RuntimeAgentIDs: append([]string(nil), runtimeLabelInUse.RuntimeAgentIDs...),
			},
		})
	case errors.Is(err, runtimeconfig.ErrReserved):
		h.writeError(w, http.StatusConflict, "runtime_label_in_use", "reserved Runtime label cannot be removed", false)
	case errors.Is(err, runtimeconfig.ErrNotFound):
		h.writeError(w, http.StatusBadRequest, "runtime_label_unknown", "a selected Runtime label is unavailable", false)
	case errors.Is(err, runtimeconfig.ErrConflict):
		h.writeError(w, http.StatusConflict, "runtime_config_conflict", "selected Runtime labels conflict", false)
	case errors.Is(err, runtimeconfig.ErrInvalid):
		h.writeError(w, http.StatusBadRequest, "runtime_config_invalid", "Runtime label configuration is invalid", false)
	case errors.As(err, &runtimeCredentialInUse):
		writeJSON(w, http.StatusConflict, errorResponse{
			Code: "runtime_credential_in_use", Message: "Runtime credential is referenced by active configuration",
			Retryable: false, RequestID: requestid.FromResponse(w),
			Details: &runtimeCredentialInUseDetailsResponse{
				Kind:          "runtime_credential_in_use",
				BindingLabels: append([]string(nil), runtimeCredentialInUse.Usage.BindingLabels...),
				RunIDs:        append([]string(nil), runtimeCredentialInUse.Usage.RunIDs...),
				AllocationIDs: append([]string(nil), runtimeCredentialInUse.Usage.AllocationIDs...),
			},
		})
	case errors.Is(err, credentials.ErrRuntimeCredentialNotFound):
		h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
	case errors.Is(err, credentials.ErrRuntimeCredentialConflict):
		h.writeError(w, http.StatusConflict, "runtime_config_conflict", "Runtime credential identity or idempotency key conflicts", false)
	case errors.Is(err, credentials.ErrRuntimeCredentialInvalid):
		h.writeError(w, http.StatusBadRequest, "runtime_config_invalid", "Runtime credential request is invalid", false)
	case errors.As(err, &credentialInUse):
		writeJSON(w, http.StatusConflict, errorResponse{
			Code: "credential_in_use", Message: "credential is pinned by a non-terminal Run",
			Retryable: false, RequestID: requestid.FromResponse(w),
			Details: &credentialInUseDetailsResponse{
				Kind: "credential_in_use", RunIDs: append([]string(nil), credentialInUse.RunIDs...),
			},
		})
	case errors.Is(err, credentials.ErrGatewayUnavailable), errors.Is(err, credentials.ErrManagerUnavailable):
		h.writeError(w, http.StatusBadGateway, "gateway_unavailable", "managed Gateway operation is unavailable", true)
	case errors.Is(err, credentials.ErrConflict):
		h.writeError(w, http.StatusConflict, "credential_conflict", "credential identity or idempotency key conflicts", false)
	case errors.Is(err, credentials.ErrNotFound):
		h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
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
		errors.Is(err, config.ErrInvalidPublication), errors.Is(err, credentials.ErrInvalid):
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
