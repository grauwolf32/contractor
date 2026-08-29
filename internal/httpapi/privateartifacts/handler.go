// Package privateartifacts exposes allocation-bound RunScope artifact access
// to mTLS-authenticated Runtime Agents.
package privateartifacts

import (
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/controlplane"
)

type AllocationRegistry interface {
	GetGrant(string) (controlplane.AllocationGrant, error)
}

type Dependencies struct {
	Registry  AllocationRegistry
	Artifacts *artifacts.Service
}

type handler struct{ dependencies Dependencies }

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
	return current.requireMTLS(mux), nil
}

func (h *handler) requireMTLS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.TLS == nil || len(r.TLS.VerifiedChains) == 0 || len(r.TLS.PeerCertificates) == 0 {
			h.writeError(w, http.StatusUnauthorized, "mtls_required", "a verified Runtime Agent certificate is required", false)
			return
		}
		next.ServeHTTP(w, r)
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
	if grant.AllocationID != allocationID || strings.TrimSpace(grant.RunID) == "" {
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
