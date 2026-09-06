package auditservice

import (
	"context"
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	DraftSelectionSchema = "contractor.audit.draft-selection.v1"
	BaselineSchema       = "contractor.audit.baseline.v1"
	maximumScopeFields   = 3
	maximumScopeValue    = 64 << 10
)

type ProfileCatalog interface {
	AuditProfile(string) (config.ResolvedAuditProfile, error)
	AuditProfiles() []config.ResolvedAuditProfile
}

// CredentialReferenceGuard must fence both managed LLM and Runtime credential
// deletion while start validates identities and commits durable Audit holds.
// The production services share one lifecycle barrier, so the managed guard
// supplies this combined critical section without nested RWMutex acquisition.
type CredentialReferenceGuard interface {
	WithRunCreation(context.Context, func() error) error
}

type Options struct {
	Pool                      *pgxpool.Pool
	Profiles                  ProfileCatalog
	TransactionLLMCredentials runtimeconfig.TransactionLLMCredentialLookupFactory
	CredentialGuard           CredentialReferenceGuard
	Now                       func() time.Time
}

type Service struct {
	pool                      *pgxpool.Pool
	profiles                  ProfileCatalog
	transactionLLMCredentials runtimeconfig.TransactionLLMCredentialLookupFactory
	credentialGuard           CredentialReferenceGuard
	findings                  *findingintake.Service
	standards                 *auditstandards.Catalog
	now                       func() time.Time
}

type ProfileSelector struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type Scope struct {
	Objective          string `json:"objective,omitempty"`
	Target             string `json:"target,omitempty"`
	AuthorizationScope string `json:"authorizationScope,omitempty"`
}

func (s Scope) Values() map[string]string {
	result := make(map[string]string, maximumScopeFields)
	if s.Objective != "" {
		result["objective"] = s.Objective
	}
	if s.Target != "" {
		result["target"] = s.Target
	}
	if s.AuthorizationScope != "" {
		result["authorizationScope"] = s.AuthorizationScope
	}
	return result
}

type DraftSelection struct {
	Schema        string                              `json:"schema"`
	Inputs        map[string]auditstore.ExactArtifact `json:"inputs"`
	RuntimeLabels []string                            `json:"runtimeLabels"`
	Scope         Scope                               `json:"scope"`
}

type BaselineInventory struct {
	SourceContentDigest      string                         `json:"sourceContentDigest"`
	CanonicalInventoryDigest string                         `json:"canonicalInventoryDigest"`
	StandardSelection        *config.AuditStandardSelection `json:"standardSelection,omitempty"`
	Gaps                     []string                       `json:"gaps"`
	Worklist                 auditstore.ExactArtifact       `json:"worklist"`
	ExecutionManifest        auditdomain.ExecutionManifest  `json:"executionManifest"`
}

type BaselineSnapshot struct {
	Schema               string                              `json:"schema"`
	Inputs               map[string]auditstore.ExactArtifact `json:"inputs"`
	Scope                Scope                               `json:"scope"`
	RuntimeLabels        []string                            `json:"runtimeLabels"`
	RuntimeConfig        runtimeconfig.RunSnapshot           `json:"runtimeConfig"`
	Skills               []contracts.RunSkillSnapshot        `json:"skills"`
	LLMCredentialIDs     []string                            `json:"llmCredentialIds"`
	RuntimeCredentialIDs []string                            `json:"runtimeCredentialIds"`
	ProjectHTTPTarget    *contracts.HTTPOriginTargetRef      `json:"projectHttpTarget,omitempty"`
	Standards            []auditstandards.PinnedPackage      `json:"standards"`
	Inventory            BaselineInventory                   `json:"inventory"`
}

type Compatibility struct {
	ServerCompatible        bool                  `json:"serverCompatible"`
	RequiresInputValidation bool                  `json:"requiresInputValidation"`
	Reasons                 []CompatibilityReason `json:"reasons"`
}

type ProfileProjection struct {
	Profile       config.ResolvedAuditProfile
	Compatibility Compatibility
}

type CreateDraftParams struct {
	AuditID        string
	OwnerID        string
	ProjectID      string
	Profile        ProfileSelector
	Inputs         map[string]contracts.ArtifactRef
	RuntimeLabels  []string
	Scope          Scope
	IdempotencyKey string
	RequestDigest  string
}

type StartParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	IdempotencyKey   string
	RequestDigest    string
}

type MutationParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	IdempotencyKey   string
	RequestDigest    string
}

type MutationResult struct {
	Audit    auditstore.Audit
	Replayed bool
}

type StartedAudit struct {
	Audit    auditstore.Audit
	Round    auditstore.Round
	Items    []auditstore.Item
	Replayed bool
}

type AuditPageParams = auditstore.ListParams
type ItemPageParams = auditstore.ListItemsParams

type ReportStatus string

const (
	ReportPending     ReportStatus = "pending"
	ReportProposed    ReportStatus = "proposed"
	ReportReady       ReportStatus = "ready"
	ReportUnavailable ReportStatus = "unavailable"
)

type ReportProjection struct {
	Status          ReportStatus              `json:"status"`
	MachineArtifact *auditstore.ExactArtifact `json:"machineArtifact,omitempty"`
	SummaryArtifact *auditstore.ExactArtifact `json:"summaryArtifact,omitempty"`
	Machine         json.RawMessage           `json:"machine,omitempty"`
	Summary         string                    `json:"summary,omitempty"`
}
