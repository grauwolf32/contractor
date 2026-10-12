// Package public implements the authenticated single-user HTTP API. Scope IDs
// are derived from authentication and route-owned Run records, never accepted
// as arbitrary request fields.
//
// Handlers are grouped by context: session, catalog, Projects, Runs, Run
// repeat drafts, the Run Queue, Artifacts, Audits, findings, Evals,
// Operations, LLM credentials, Runtime configuration and the owner Git key.
// Each context is a struct that holds only the services its routes use, and
// every one of them is required; TestContextFilesReachOnlyTheirOwnDependencies
// keeps a context's files on its own struct.
package public

import (
	"context"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// ProjectReader reads one owner's Project; contexts that act inside a
// Project use it to keep foreign Projects indistinguishable from missing ones.
type ProjectReader interface {
	Get(context.Context, string, string) (projectstore.Project, error)
}

// RunGetter reads one Run; ownedRun keeps another owner's Run hidden.
type RunGetter interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
}

// Waker nudges a background controller to look for new work now.
type Waker interface {
	Wake()
}

type OperationsInvalidator interface {
	InvalidateOperations(controlplane.OperationsResource, string) error
}

var (
	errInvalidRequest = apierror.New(apierror.InvalidRequest, "invalid public API request")
	// errRequestTooLarge reports a JSON request body beyond its declared
	// bound. It is 413 like an oversized artifact, not a malformed request.
	errRequestTooLarge = apierror.New(apierror.RequestTooLarge, "public API request body is too large")
)
