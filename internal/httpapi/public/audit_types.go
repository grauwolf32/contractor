package public

// Audits: the management and finding proposal ports. The Audit request
// and response bodies live with their handlers in audit_handlers.go.

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

type AuditManagement interface {
	GetWorkspace(context.Context, string, string) (auditservice.WorkspaceSummary, error)
	ListFindingsPage(context.Context, auditservice.FindingListParams) (auditservice.FindingPage, error)
	ListReviewsPage(context.Context, auditservice.ReviewListParams) (auditservice.ReviewPage, error)
	Profiles() []auditservice.ProfileProjection
	Profile(auditservice.ProfileSelector) (auditservice.ProfileProjection, error)
	Standards(context.Context, string) ([]auditstandards.PackageProjection, error)
	Standard(context.Context, string, auditstandards.Reference) (auditstandards.PackageProjection, error)
	CreateDraft(context.Context, auditservice.CreateDraftParams) (auditstore.Audit, bool, error)
	Start(context.Context, auditservice.StartParams) (auditservice.StartedAudit, error)
	Pause(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Resume(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Cancel(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Delete(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Get(context.Context, string, string) (auditstore.Audit, error)
	List(context.Context, auditstore.ListParams) ([]auditstore.Audit, error)
	ListItems(context.Context, auditstore.ListItemsParams) ([]auditstore.Item, error)
	ListItemAttempts(context.Context, string, string, []string) (map[string][]auditstore.ItemAttempt, error)
	GetRound(context.Context, string, string, string) (auditstore.Round, error)
	ListCoverage(context.Context, string, string, string, int, int) ([]auditstore.CoverageRow, error)
	GetReport(context.Context, string, string) (auditservice.ReportProjection, error)
	ListFindings(context.Context, auditservice.FindingListParams) ([]auditservice.Finding, error)
	GetFinding(context.Context, string, string, string) (auditservice.Finding, error)
	CreateFindingReview(context.Context, auditservice.CreateFindingReviewParams) (auditservice.FindingReviewResult, error)
	DecideFinding(context.Context, auditservice.DecideFindingParams) (auditservice.FindingDecisionResult, error)
	DecideActionReview(context.Context, auditservice.DecideActionReviewParams) (auditservice.ActionReviewDecisionResult, error)
	GetReview(context.Context, string, string, string) (auditservice.ReviewRequest, error)
	ListReviews(context.Context, auditservice.ReviewListParams) ([]auditservice.ReviewRequest, error)
	ListFindingProvenance(context.Context, auditservice.ProvenanceListParams) ([]auditservice.FindingProvenance, error)
}

type FindingProposalManagement interface {
	ListRun(context.Context, string, string, findingintake.ListQuery) ([]findingintake.Receipt, error)
	ListAuditInbox(context.Context, string, string, findingintake.ListQuery) ([]findingintake.Receipt, error)
	ImportIntoAudit(context.Context, findingintake.ImportRequest) (findingintake.AuditHold, bool, error)
}
