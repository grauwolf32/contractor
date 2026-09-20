package auditservice

import (
	"context"
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/auditbaseline"
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
	BaselineSchema       = auditbaseline.Schema
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

type Scope = auditbaseline.Scope

type DraftSelection struct {
	Schema        string                              `json:"schema"`
	Inputs        map[string]auditstore.ExactArtifact `json:"inputs"`
	RuntimeLabels []string                            `json:"runtimeLabels"`
	Scope         Scope                               `json:"scope"`
}

type BaselineInventory = auditbaseline.BaselineInventory
type BaselineSnapshot = auditbaseline.BaselineSnapshot

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
	ExpectedProfileSHA256 string
	AuditID               string
	OwnerID               string
	ProjectID             string
	Profile               ProfileSelector
	Inputs                map[string]contracts.ArtifactRef
	RuntimeLabels         []string
	Scope                 Scope
	IdempotencyKey        string
	RequestDigest         string
}

type StartParams struct {
	ExpectedRuntimeSHA256   string
	ExpectedSkillsSHA256    string
	ExpectedStandardsSHA256 string
	OwnerID                 string
	AuditID                 string
	ExpectedRevision        uint64
	IdempotencyKey          string
	RequestDigest           string
	DeadlineSeconds         *int
}

type MutationParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	IdempotencyKey   string
	RequestDigest    string
	DeadlineSeconds  *int
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
	Review          *ReviewRequest            `json:"review,omitempty"`
	Status          ReportStatus              `json:"status"`
	MachineArtifact *auditstore.ExactArtifact `json:"machineArtifact,omitempty"`
	SummaryArtifact *auditstore.ExactArtifact `json:"summaryArtifact,omitempty"`
	Machine         json.RawMessage           `json:"machine,omitempty"`
	Summary         string                    `json:"summary,omitempty"`
}
