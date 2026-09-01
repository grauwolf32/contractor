// Package privateartifacts exposes allocation-bound RunScope artifact access
// to mTLS-authenticated Runtime Agents.
package privateartifacts

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/requestid"
)

const RuntimeInstanceHeader = "X-Contractor-Runtime-Instance-ID"

type AllocationRegistry interface {
	GetGrant(string) (controlplane.AllocationGrant, error)
	WithWriteGrant(string, func(controlplane.AllocationGrant) error) error
}

type Dependencies struct {
	Registry     AllocationRegistry
	Artifacts    *artifacts.Service
	NewRequestID func() (string, error)
	Logger       *slog.Logger
}

type handler struct{ dependencies Dependencies }

type authenticatedRuntime struct {
	principalID string
	instanceID  string
}

type authenticatedRuntimeContextKey struct{}

func NewHandler(dependencies Dependencies) (http.Handler, error) {
	if dependencies.Registry == nil || dependencies.Artifacts == nil {
		return nil, errors.New("private Artifact API dependencies are incomplete")
	}
	current := &handler{dependencies: dependencies}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /private/v1/allocations/{allocationID}/artifacts", current.listArtifacts)
	mux.HandleFunc("GET /private/v1/allocations/{allocationID}/artifacts/{namespace}/{name}", current.getArtifact)
	mux.HandleFunc("PUT /private/v1/allocations/{allocationID}/artifacts/{namespace}/{name}", current.putArtifact)
	mux.HandleFunc("/private/v1/allocations/{allocationID}/artifacts/{namespace}/{name}", current.methodNotAllowed)
	mux.HandleFunc("/private/v1/allocations/{allocationID}/artifacts", current.methodNotAllowed)
	mux.HandleFunc("/", current.notFound)
	return requestid.Middleware(current.requireMTLS(mux), requestid.Options{
		Generator: dependencies.NewRequestID, Logger: dependencies.Logger,
		Boundary: "artifact-private-api", TrustIncoming: true,
	}), nil
}

func (h *handler) requireMTLS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		principalID, err := mtls.RuntimeAgentIDFromConnection(r.TLS)
		if err != nil {
			h.writeError(w, http.StatusUnauthorized, "mtls_required", "a verified Runtime Agent certificate is required", false)
			return
		}
		values := r.Header.Values(RuntimeInstanceHeader)
		if len(values) != 1 || len(values[0]) > 256 || strings.TrimSpace(values[0]) == "" || values[0] != strings.TrimSpace(values[0]) {
			h.writeError(w, http.StatusBadRequest, "invalid_runtime_instance", "one bounded Runtime Agent instance header is required", false)
			return
		}
		identity := authenticatedRuntime{principalID: principalID, instanceID: values[0]}
		ctx := context.WithValue(r.Context(), authenticatedRuntimeContextKey{}, identity)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (h *handler) runStore(r *http.Request, write bool) (artifacts.ScopedStore, error) {
	allocationID := r.PathValue("allocationID")
	if strings.TrimSpace(allocationID) == "" {
		return artifacts.ScopedStore{}, errInvalidRequest
	}
	grant, err := h.dependencies.Registry.GetGrant(allocationID)
	if err != nil {
		return artifacts.ScopedStore{}, err
	}
	identity, ok := r.Context().Value(authenticatedRuntimeContextKey{}).(authenticatedRuntime)
	if !ok {
		return artifacts.ScopedStore{}, errArtifactAccessDenied
	}
	return h.runStoreForGrant(allocationID, grant, identity, write)
}

func (h *handler) runStoreForGrant(
	allocationID string,
	grant controlplane.AllocationGrant,
	identity authenticatedRuntime,
	write bool,
) (artifacts.ScopedStore, error) {
	if grant.AllocationID != allocationID || strings.TrimSpace(grant.RunID) == "" ||
		grant.RuntimeAgentID != identity.principalID || grant.RuntimeInstanceID != identity.instanceID {
		return artifacts.ScopedStore{}, controlplane.ErrAllocationNotFound
	}
	if grant.ReadPolicy != controlplane.ReadCurrentRun {
		return artifacts.ScopedStore{}, errArtifactAccessDenied
	}
	if write {
		if grant.WriteFenced {
			return artifacts.ScopedStore{}, errAllocationWriteFenced
		}
		if grant.WritePolicy != controlplane.WriteInputsAndIntermediates {
			return artifacts.ScopedStore{}, errArtifactAccessDenied
		}
	}
	store, err := h.dependencies.Artifacts.Run(grant.RunID)
	if err != nil {
		return artifacts.ScopedStore{}, fmt.Errorf("bind allocation RunScope: %w", err)
	}
	return store, nil
}

func (h *handler) methodNotAllowed(w http.ResponseWriter, _ *http.Request) {
	h.writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "method is not allowed", false)
}

func (h *handler) notFound(w http.ResponseWriter, _ *http.Request) {
	h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
}
