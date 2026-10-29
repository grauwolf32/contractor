// Package privateartifacts exposes allocation-bound RunScope access and
// explicitly granted Project Artifact reads to mTLS-authenticated Runtime Agents.
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
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const RuntimeInstanceHeader = "X-Contractor-Runtime-Instance-ID"

type AllocationRegistry interface {
	GetGrant(string) (controlplane.AllocationGrant, error)
	WithWriteGrant(string, func(controlplane.AllocationGrant) error) error
}

type Dependencies struct {
	PentestActions  PentestActions
	GatewayRecovery *gatewayrecovery.Service
	Registry        AllocationRegistry
	Artifacts       *artifacts.Service
	Findings        FindingIntake
	NewRequestID    func() (string, error)
	Logger          *slog.Logger
	Sitemap         *projectsitemap.Service
	RunProjects     RunProjectReader
	Projects        ProjectReader
}

type RunProjectReader interface {
	ProjectContext(context.Context, string) (runstore.ProjectContext, error)
}

type ProjectReader interface {
	Get(context.Context, string, string) (projectstore.Project, error)
}

type FindingIntake interface {
	FindReplay(context.Context, controlplane.AllocationGrant, findingintake.Submission) (findingintake.Receipt, bool, error)
	Submit(context.Context, controlplane.AllocationGrant, findingintake.Submission) (findingintake.Receipt, bool, error)
}

type handler struct {
	dependencies Dependencies
	recovery     gatewayRecoveryUpdater
}

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
	if dependencies.GatewayRecovery != nil {
		current.recovery = dependencies.GatewayRecovery
	}
	mux := http.NewServeMux()
	httpx.ServeRoutes(mux, current.routes(), current.methodNotAllowed, current.notFound)
	return requestid.Middleware(current.requireMTLS(mux), requestid.Options{
		Generator: dependencies.NewRequestID, Logger: dependencies.Logger,
		Boundary: "artifact-private-api", TrustIncoming: mtls.HasVerifiedRuntimeAgent,
	}), nil
}

// routes lists every route of the private allocation API. ServeRoutes
// answers HEAD and any other undeclared method of their paths with 405 and
// any other path with 404 (ART-R120).
func (h *handler) routes() []httpx.Route {
	const allocation = "/private/v1/allocations/{allocationID}"
	return []httpx.Route{
		{Method: http.MethodPost, Path: allocation + "/gateway-recovery", Handler: h.gatewayRecovery},
		{Method: http.MethodPost, Path: allocation + "/pentest-actions", Handler: h.pentestAction},
		{Method: http.MethodGet, Path: allocation + "/artifacts", Handler: h.listArtifacts},
		{Method: http.MethodGet, Path: allocation + "/artifacts/{namespace}/{name}", Handler: h.getArtifact},
		{Method: http.MethodGet, Path: allocation + "/artifacts/{namespace}/{name}/metadata", Handler: h.getArtifactMetadata},
		{Method: http.MethodGet, Path: allocation + "/artifacts/{namespace}/{name}/access", Handler: h.getArtifactAccess},
		{Method: http.MethodPut, Path: allocation + "/artifacts/{namespace}/{name}", Handler: h.putArtifact},
		{Method: http.MethodPost, Path: allocation + "/finding-proposals", Handler: h.postFindingProposal},
		{Method: http.MethodPost, Path: allocation + "/sitemap/mutations", Handler: h.mutateSitemap},
		{Method: http.MethodPost, Path: allocation + "/sitemap/snapshots", Handler: h.snapshotSitemap},
		{Method: http.MethodPost, Path: allocation + "/sitemap/observations", Handler: h.collectSitemapObservations},
		{Method: http.MethodPost, Path: allocation + "/sitemap/imports", Handler: h.importSitemapSource},
	}
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
	grant, identity, err := h.allocationGrant(r)
	if err != nil {
		return artifacts.ScopedStore{}, err
	}
	return h.runStoreForGrant(r.PathValue("allocationID"), grant, identity, write)
}

func (h *handler) allocationGrant(
	r *http.Request,
) (controlplane.AllocationGrant, authenticatedRuntime, error) {
	allocationID := r.PathValue("allocationID")
	if strings.TrimSpace(allocationID) == "" {
		return controlplane.AllocationGrant{}, authenticatedRuntime{}, errInvalidRequest
	}
	grant, err := h.dependencies.Registry.GetGrant(allocationID)
	if err != nil {
		return controlplane.AllocationGrant{}, authenticatedRuntime{}, err
	}
	identity, ok := r.Context().Value(authenticatedRuntimeContextKey{}).(authenticatedRuntime)
	if !ok {
		return controlplane.AllocationGrant{}, authenticatedRuntime{}, errArtifactAccessDenied
	}
	if grant.AllocationID != allocationID || grant.RuntimeAgentID != identity.principalID ||
		grant.RuntimeInstanceID != identity.instanceID {
		return controlplane.AllocationGrant{}, authenticatedRuntime{}, controlplane.ErrAllocationNotFound
	}
	return grant, identity, nil
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
