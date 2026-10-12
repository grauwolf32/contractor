// Package auditimport validates frozen Audit child-Run outputs and commits
// exact retained evidence, settlement receipts, coverage, and reports.
package auditimport

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
)

var ErrPermanent = errors.New("Audit import cannot be completed")

// Store is what the importer commits under the Controller's claim: collection
// receipts, report pages and the report itself. It is the claim seam of the
// Audit store.
type Store interface {
	Collect(context.Context, auditstore.CollectParams) (auditstore.CollectionReceipt, bool, error)
	RecordReportPage(context.Context, auditstore.RecordReportPageParams) error
	CommitReport(context.Context, auditstore.CommitReportParams) (auditstore.Audit, error)
	ProposeReport(context.Context, auditstore.ProposeReportParams) (auditstore.Audit, bool, error)
}

// Reader is what the importer reads of an Audit while it collects and
// assembles a report. It is the read seam of the Audit store.
type Reader interface {
	GetArtifactLink(context.Context, string, string) (auditstore.ArtifactLink, error)
	ListExecutionItems(context.Context, string) ([]auditstore.ExecutionItem, error)
	ListItemsByIDs(context.Context, string, []string) ([]auditstore.Item, error)
	ListRounds(context.Context, string) ([]auditstore.Round, error)
	ListReportItems(context.Context, string, string, int, int) ([]auditstore.ReportItem, error)
	ListReportFindings(context.Context, auditstore.ReportFindingPageParams) ([]auditstore.ReportFinding, bool, error)
	ListReportPages(context.Context, string, uint64) ([]auditstore.ReportPage, error)
	ReadReportSections(context.Context, string, uint64) (map[auditstore.ReportSection]auditstore.ReportSectionContent, error)
	CollectionDispositionCounts(context.Context, string) (auditstore.CollectionDispositionCounts, error)
	GetReportCandidate(context.Context, string) (auditstore.ReportCandidate, error)
}

type RunReader interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
}

// FindingRetention is the purpose-specific bridge used only while collecting
// an Audit child Run. It transfers exact proposal/evidence revisions into the
// owning Audit inbox; it does not create findings, items, or review decisions.
type FindingRetention interface {
	ListAuditCollection(context.Context, string, string, string, findingintake.ListQuery) ([]findingintake.CollectionReceipt, error)
	GetAuditReceipt(context.Context, string, string, string) (findingintake.Receipt, error)
	RetainAuditCollectionBatch(context.Context, []findingintake.ImportRequest) error
	RejectAuditCollection(context.Context, findingintake.ImportRequest, string) error
	ResolveAuditProposals(
		context.Context, string, string, string, string, []findingintake.ProposalKey,
	) ([]findingintake.ResolvedProposal, error)
}

type Importer struct {
	store     Store
	reads     Reader
	runs      RunReader
	artifacts ArtifactAccess
	findings  FindingRetention
	budget    reportBudget
}

func New(
	store Store,
	reads Reader,
	runs RunReader,
	artifactAccess ArtifactAccess,
	findings ...FindingRetention,
) (*Importer, error) {
	if store == nil || reads == nil || runs == nil || artifactAccess == nil {
		return nil, errors.New("Audit importer dependencies are incomplete")
	}
	if len(findings) > 1 {
		return nil, errors.New("Audit importer accepts at most one finding intake")
	}
	result := &Importer{store: store, reads: reads, runs: runs, artifacts: artifactAccess, budget: defaultReportBudget}
	if len(findings) == 1 {
		if findings[0] == nil {
			return nil, errors.New("Audit importer finding intake is nil")
		}
		result.findings = findings[0]
	}
	return result, nil
}

type sourceProvenance struct {
	Schema                string                   `json:"schema"`
	AuditID               string                   `json:"auditId"`
	ExecutionID           string                   `json:"executionId"`
	ExecutionItemID       string                   `json:"executionItemId"`
	ItemID                string                   `json:"itemId"`
	ItemKey               string                   `json:"itemKey"`
	ItemAttempt           int                      `json:"itemAttempt"`
	RunID                 string                   `json:"runId"`
	WorkflowName          string                   `json:"workflowName"`
	WorkflowVersion       string                   `json:"workflowVersion"`
	WorkflowSchemaVersion string                   `json:"workflowSchemaVersion"`
	WorkflowClosureDigest string                   `json:"workflowClosureDigest"`
	ExecutionManifest     auditstore.ExactArtifact `json:"executionManifest"`
	Task                  auditstore.ExactArtifact `json:"task"`
	TaskSource            json.RawMessage          `json:"taskSource"`
	SourceOutput          auditstore.ExactArtifact `json:"sourceOutput"`
}
