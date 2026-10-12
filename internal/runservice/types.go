// Package runservice owns WorkflowRun creation orchestration shared by the
// public API and the trusted Audit Controller, and the public Run detail read
// from one database snapshot. It is the only layer that may request an
// Audit-managed publication mode.
package runservice

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
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
	PinRuntimeLabels(context.Context, []string, ...bool) (runtimeconfig.RunSnapshot, error)
	CreateRunIdempotent(context.Context, runstore.CreateRunIdempotentParams) (runstore.WorkflowRun, bool, error)
	SetRunSkillSelections(context.Context, string, []agentskills.RunSkillSnapshot) error
	TransitionRun(context.Context, string, runstore.WorkflowRunState, runstore.WorkflowRunState, runstore.Reason) (runstore.WorkflowRun, error)
}

type AuditRunWriter interface {
	SetAuditCompletion(context.Context, string, runstore.AuditCompletionSnapshot) error
	CreateAuditRun(context.Context, runstore.CreateAuditRunParams) (runstore.WorkflowRun, error)
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	SetRunSkillSelections(context.Context, string, []agentskills.RunSkillSnapshot) error
	TransitionRun(context.Context, string, runstore.WorkflowRunState, runstore.WorkflowRunState, runstore.Reason) (runstore.WorkflowRun, error)
}

type AuditExecutionWriter interface {
	GetRunCreationIntent(context.Context, auditstore.ControllerClaim, string) (auditstore.RunCreationIntent, error)
	BindRun(context.Context, auditstore.BindRunParams) (auditstore.Execution, error)
}

type PublicTransaction func(context.Context, func(PublicRunWriter, *artifacts.Service) error) error
type AuditTransaction func(context.Context, func(AuditRunWriter, *artifacts.Service, AuditExecutionWriter) error) error

type WorkflowResolver interface {
	ResolveRunWorkflow(context.Context, string, resolved.ExecutionConfigPatch, resolved.CredentialLookup) (resolved.ResolvedWorkflow, error)
}

type CredentialGuard interface {
	// WithRunCreation must hold the shared lifecycle barrier used by both LLM
	// and Runtime credential deletion. RuntimeCredentialValidator deliberately
	// performs only the lookup while this single guard is held; acquiring the
	// same read barrier recursively can deadlock behind a waiting deletion.
	WithRunCreation(context.Context, func() error) error
}

// RuntimeCredentialValidator checks that the requesting principal may pin an
// active Runtime credential of an allowed kind. A credential the principal may
// not use is reported as missing.
type RuntimeCredentialValidator interface {
	ValidateRuntimeCredentialUse(context.Context, credentials.RuntimeCredentialUser, string, ...string) error
}

type ProjectReader interface {
	Get(context.Context, string, string) (projectstore.Project, error)
}

type Options struct {
	Runs                         RunReader
	Workflows                    WorkflowResolver
	LLMCredentials               resolved.CredentialLookup
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
	llmCredentials               resolved.CredentialLookup
	credentialGuard              CredentialGuard
	runtimeCredentials           RuntimeCredentialValidator
	projects                     ProjectReader
	publicTransaction            PublicTransaction
	auditTransaction             AuditTransaction
	skillInitializationAvailable bool
}

type PublicCreateParams struct {
	// Trusted preparation expectations are not public request fields.
	ExpectedWorkflowSHA256 string
	ExpectedRuntimeSHA256  string
	ExpectedSkillsSHA256   string
	OwnerID                string
	ProjectID              *string
	Workflow               string
	ExecutionConfig        resolved.ExecutionConfigPatch
	RuntimeLabels          []string
	MetadataLabels         runstore.RunMetadataLabels
	Parameters             map[string]string
	Inputs                 map[string]contracts.ArtifactRef
	IdempotencyKey         string
	RequestDigest          string
	NewRunID               func() (string, error)

	// OperationsPrincipal reports that the requesting principal holds the
	// Operations capability and may pin any Runtime credential. Otherwise a
	// Project HTTP target may pin only a credential OwnerID created.
	OperationsPrincipal bool
}

// AuditCreateParams contains only already-pinned server-side values. It has no
// publication-mode field and accepts credential identities only through the
// immutable Workflow/Runtime/Project snapshots; raw secret bytes cannot enter
// this boundary.
type AuditCreateParams struct {
	Claim             auditstore.ControllerClaim
	ExecutionID       string
	Workflow          resolved.ResolvedWorkflow
	RuntimeConfig     runtimeconfig.RunSnapshot
	Skills            []agentskills.RunSkillSnapshot
	ProjectHTTPTarget *runtimesettings.HTTPOriginTargetRef
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
