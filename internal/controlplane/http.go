package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

const maxPrivateJSONBody = 1 << 20

type privateErrorResponse struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
	RequestID string `json:"requestId"`
}

type HTTPOptions struct {
	NewRequestID func() (string, error)
	Logger       *slog.Logger
	Principals   PrincipalRegistrar
}

type PrincipalRegistrar interface {
	Register(context.Context, string, []string) (runtimeconfig.RuntimeAgentPrincipal, error)
}

func NewHTTPHandler(registry Registry, supplied ...HTTPOptions) (http.Handler, error) {
	if registry == nil {
		return nil, errors.New("Control Plane registry is required")
	}
	if len(supplied) > 1 {
		return nil, errors.New("at most one Control Plane HTTP options value is allowed")
	}
	options := HTTPOptions{}
	if len(supplied) == 1 {
		options = supplied[0]
	}
	if options.Principals == nil {
		return nil, errors.New("Runtime Agent principal registrar is required")
	}
	handler := &privateHTTPHandler{registry: registry, principals: options.Principals}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /private/v1/agents/register", handler.register)
	mux.HandleFunc("POST /private/v1/agents/{instanceID}/heartbeat", handler.heartbeat)
	mux.HandleFunc("/private/v1/agents/{instanceID}/heartbeat", handler.methodNotAllowed)
	mux.HandleFunc("/private/v1/agents/register", handler.methodNotAllowed)
	mux.HandleFunc("/", handler.notFound)
	return requestid.Middleware(handler.requireMTLS(mux), requestid.Options{
		Generator: options.NewRequestID, Logger: options.Logger,
		Boundary: "control-plane-private-api", TrustIncoming: true,
	}), nil
}

type privateHTTPHandler struct {
	registry   Registry
	principals PrincipalRegistrar
}

type runtimeAgentPrincipalContextKey struct{}

func (h *privateHTTPHandler) requireMTLS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		runtimeAgentID, err := mtls.RuntimeAgentIDFromConnection(r.TLS)
		if err != nil {
			writePrivateError(w, http.StatusUnauthorized, "mtls_required", "a verified client certificate is required", false)
			return
		}
		ctx := context.WithValue(r.Context(), runtimeAgentPrincipalContextKey{}, runtimeAgentID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (h *privateHTTPHandler) register(w http.ResponseWriter, r *http.Request) {
	if r.URL.RawQuery != "" {
		h.handleError(w, fmt.Errorf("%w: query parameters are not supported", ErrInvalidRequest))
		return
	}
	registration, err := decodePrivateV2JSON[contracts.AgentRegistrationV2](w, r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	runtimeAgentID, ok := r.Context().Value(runtimeAgentPrincipalContextKey{}).(string)
	if !ok {
		h.handleError(w, errors.New("authenticated Runtime Agent principal is unavailable"))
		return
	}
	principal, err := h.principals.Register(r.Context(), runtimeAgentID, registration.InitialLabels)
	if err != nil {
		h.handleError(w, err)
		return
	}
	authenticated := AuthenticatedPrincipal{
		RuntimeAgentID: principal.RuntimeAgentID,
		Labels:         principal.Labels, LabelRevision: principal.LabelRevision,
	}
	if _, err := h.registry.RegisterAuthenticated(authenticated, registration); err != nil {
		h.handleError(w, err)
		return
	}
	writePrivateJSON(w, http.StatusOK, h.registry.RegistrationResponse(authenticated))
}

func (h *privateHTTPHandler) heartbeat(w http.ResponseWriter, r *http.Request) {
	if r.URL.RawQuery != "" {
		h.handleError(w, fmt.Errorf("%w: query parameters are not supported", ErrInvalidRequest))
		return
	}
	heartbeat, err := decodePrivateJSON[contracts.AgentHeartbeat](w, r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if heartbeat.InstanceID != r.PathValue("instanceID") {
		h.handleError(w, fmt.Errorf("%w: path and body instance IDs differ", ErrInvalidRequest))
		return
	}
	runtimeAgentID, ok := r.Context().Value(runtimeAgentPrincipalContextKey{}).(string)
	if !ok {
		h.handleError(w, errors.New("authenticated Runtime Agent principal is unavailable"))
		return
	}
	response, err := h.registry.HeartbeatAuthenticated(runtimeAgentID, heartbeat)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writePrivateJSON(w, http.StatusOK, response)
}

func (h *privateHTTPHandler) handleError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, ErrInvalidRequest), errors.Is(err, contracts.ErrValidation):
		writePrivateError(w, http.StatusBadRequest, "invalid_request", "request does not satisfy the private API contract", false)
	case errors.Is(err, runtimeconfig.ErrInvalid), errors.Is(err, runtimeconfig.ErrNotFound),
		errors.Is(err, runtimeconfig.ErrConflict), errors.Is(err, runtimeconfig.ErrPrecondition):
		writePrivateError(w, http.StatusBadRequest, "invalid_runtime_labels", "Runtime Agent labels cannot be registered", false)
	case errors.Is(err, ErrRegistrationConflict), errors.Is(err, ErrHeartbeatOutOfOrder):
		writePrivateError(w, http.StatusConflict, "conflict", "request conflicts with current Runtime Agent state", true)
	default:
		writePrivateError(w, http.StatusInternalServerError, "internal_error", "request could not be processed", true)
	}
}

func (*privateHTTPHandler) methodNotAllowed(w http.ResponseWriter, _ *http.Request) {
	writePrivateError(w, http.StatusMethodNotAllowed, "method_not_allowed", "method is not allowed", false)
}

func (*privateHTTPHandler) notFound(w http.ResponseWriter, _ *http.Request) {
	writePrivateError(w, http.StatusNotFound, "not_found", "resource was not found", false)
}

func decodePrivateJSON[T contracts.Validatable](w http.ResponseWriter, r *http.Request) (T, error) {
	var zero T
	values := r.Header.Values("Content-Type")
	if len(values) != 1 {
		return zero, fmt.Errorf("%w: exactly one Content-Type is required", ErrInvalidRequest)
	}
	mediaType, parameters, err := mime.ParseMediaType(values[0])
	if err != nil || mediaType != "application/json" || len(parameters) != 0 {
		return zero, fmt.Errorf("%w: Content-Type must be application/json without parameters", ErrInvalidRequest)
	}
	if r.ContentLength > maxPrivateJSONBody {
		return zero, fmt.Errorf("%w: request body is too large", ErrInvalidRequest)
	}
	body := http.MaxBytesReader(w, r.Body, maxPrivateJSONBody)
	data, err := io.ReadAll(body)
	if err != nil {
		return zero, fmt.Errorf("%w: read request body", ErrInvalidRequest)
	}
	value, err := contracts.DecodeStrict[T](data)
	if err != nil {
		return zero, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	return value, nil
}

func decodePrivateV2JSON[T contracts.Validatable](w http.ResponseWriter, r *http.Request) (T, error) {
	var zero T
	data, err := readPrivateJSON(w, r)
	if err != nil {
		return zero, err
	}
	value, err := contracts.DecodePrivateV2Strict[T](data)
	if err != nil {
		return zero, fmt.Errorf("%w: private protocol v2 input is invalid", ErrInvalidRequest)
	}
	return value, nil
}

func readPrivateJSON(w http.ResponseWriter, r *http.Request) ([]byte, error) {
	values := r.Header.Values("Content-Type")
	if len(values) != 1 {
		return nil, fmt.Errorf("%w: exactly one Content-Type is required", ErrInvalidRequest)
	}
	mediaType, parameters, err := mime.ParseMediaType(values[0])
	if err != nil || mediaType != "application/json" || len(parameters) != 0 {
		return nil, fmt.Errorf("%w: Content-Type must be application/json without parameters", ErrInvalidRequest)
	}
	if r.ContentLength > maxPrivateJSONBody {
		return nil, fmt.Errorf("%w: request body is too large", ErrInvalidRequest)
	}
	body := http.MaxBytesReader(w, r.Body, maxPrivateJSONBody)
	data, err := io.ReadAll(body)
	if err != nil {
		return nil, fmt.Errorf("%w: read request body", ErrInvalidRequest)
	}
	return data, nil
}

func writePrivateError(w http.ResponseWriter, status int, code, message string, retryable bool) {
	writePrivateJSON(w, status, privateErrorResponse{
		Code: code, Message: message, Retryable: retryable,
		RequestID: requestid.FromResponse(w),
	})
}

func writePrivateJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	encoder := json.NewEncoder(w)
	_ = encoder.Encode(value)
}
