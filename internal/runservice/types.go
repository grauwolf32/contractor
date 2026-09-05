// Package runservice owns WorkflowRun creation orchestration shared by the
// public API and the trusted Audit Controller. It is the only layer that may
// request an Audit-managed publication mode.
package runservice

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

var (
	ErrInvalid       = errors.New("Run creation request is invalid")
	ErrNotConfigured = errors.New("Run creation capability is not configured")
)

type RunReader interface {
	LookupRunIdempotency(context.Context, string, string, string) (runstore.WorkflowRun, bool, error)
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
}

type PublicRunWriter interface {
	PinRuntimeLabels(context.Context, []string) (runtimeconfig.RunSnapshot, error)
	CreateRunIdempotent(context.Context, runstore.CreateRunIdempotentParams) (runstore.WorkflowRun, bool, error)
	SetRunSkillSelections(context.Context, string, []contracts.RunSkillSnapshot) error
	TransitionRun(context.Context, string, runstore.WorkflowRunState, runstore.WorkflowRunState, runstore.Reason) (runstore.WorkflowRun, error)
}

type AuditRunWriter interface {
	CreateAuditRun(context.Context, runstore.CreateAuditRunParams) (runstore.WorkflowRun, error)
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	SetRunSkillSelections(context.Context, string, []contracts.RunSkillSnapshot) error
	TransitionRun(context.Context, string, runstore.WorkflowRunState, runstore.WorkflowRunState, runstore.Reason) (runstore.WorkflowRun, error)
}

type AuditExecutionWriter interface {
	GetRunCreationIntent(context.Context, auditstore.ControllerClaim, string) (auditstore.RunCreationIntent, error)
	BindRun(context.Context, auditstore.BindRunParams) (auditstore.Execution, error)
}

type PublicTransaction func(context.Context, func(PublicRunWriter, *artifacts.Service) error) error
type AuditTransaction func(context.Context, func(AuditRunWriter, *artifacts.Service, AuditExecutionWriter) error) error

type WorkflowResolver interface {
	ResolveRunWorkflow(context.Context, string, config.ExecutionConfigPatch, config.CredentialLookup) (config.ResolvedWorkflow, error)
}

type CredentialGuard interface {
	// WithRunCreation must hold the shared lifecycle barrier used by both LLM
	// and Runtime credential deletion. RuntimeCredentialValidator deliberately
	// performs only the lookup while this single guard is held; acquiring the
	// same read barrier recursively can deadlock behind a waiting deletion.
	WithRunCreation(context.Context, func() error) error
}

type RuntimeCredentialValidator interface {
	ValidateRuntimeCredential(context.Context, string, ...string) error
}

type ProjectReader interface {
	Get(context.Context, string, string) (projectstore.Project, error)
}

type Options struct {
	Runs                         RunReader
	Workflows                    WorkflowResolver
	LLMCredentials               config.CredentialLookup
	CredentialGuard              CredentialGuard
	RuntimeCredentials           RuntimeCredentialValidator
	Projects                     ProjectReader
	PublicTransaction            PublicTransaction
	AuditTransaction             AuditTransaction
	SkillInitializationAvailable bool
}

type Service struct {
	runs                         RunReader
	workflows                    WorkflowResolver
	llmCredentials               config.CredentialLookup
	credentialGuard              CredentialGuard
	runtimeCredentials           RuntimeCredentialValidator
	projects                     ProjectReader
	publicTransaction            PublicTransaction
	auditTransaction             AuditTransaction
	skillInitializationAvailable bool
}

type PublicCreateParams struct {
	OwnerID         string
	ProjectID       *string
	Workflow        string
	ExecutionConfig config.ExecutionConfigPatch
	RuntimeLabels   []string
	MetadataLabels  runstore.RunMetadataLabels
	Parameters      map[string]string
	Inputs          map[string]contracts.ArtifactRef
	IdempotencyKey  string
	RequestDigest   string
	NewRunID        func() (string, error)
}

// AuditCreateParams contains only already-pinned server-side values. It has no
// publication-mode field and accepts credential identities only through the
// immutable Workflow/Runtime/Project snapshots; raw secret bytes cannot enter
// this boundary.
type AuditCreateParams struct {
	Claim             auditstore.ControllerClaim
	ExecutionID       string
	Workflow          config.ResolvedWorkflow
	RuntimeConfig     runtimeconfig.RunSnapshot
	Skills            []contracts.RunSkillSnapshot
	ProjectHTTPTarget *contracts.HTTPOriginTargetRef
	Parameters        map[string]string
	Inputs            map[string]auditstore.ExactArtifact
	ExecutionManifest auditstore.ExactArtifact
	RequestDigest     string
	NewRunID          func() (string, error)
}

type CreateResult struct {
	Run      runstore.WorkflowRun
	Created  bool
	Replayed bool
}
