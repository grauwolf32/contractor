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

type Store interface {
	ListExecutionItems(context.Context, string) ([]auditstore.ExecutionItem, error)
	ListItems(context.Context, string) ([]auditstore.Item, error)
	ListCoverage(context.Context, string, string, int, int) ([]auditstore.CoverageRow, error)
	ListReportFindings(context.Context, string) ([]auditstore.ReportFinding, error)
	CollectionDispositionCounts(context.Context, string) (auditstore.CollectionDispositionCounts, error)
	Collect(context.Context, auditstore.CollectParams) (auditstore.CollectionReceipt, bool, error)
	CommitReport(context.Context, auditstore.CommitReportParams) (auditstore.Audit, error)
}

type RunReader interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
}

// FindingRetention is the purpose-specific bridge used only while collecting
// an Audit child Run. It transfers exact proposal/evidence revisions into the
// owning Audit inbox; it does not create findings, items, or review decisions.
type FindingRetention interface {
	ListRun(context.Context, string, string, findingintake.ListQuery) ([]findingintake.Receipt, error)
	GetAuditReceipt(context.Context, string, string, string) (findingintake.Receipt, error)
	ImportIntoAudit(context.Context, findingintake.ImportRequest) (findingintake.AuditHold, bool, error)
	ResolveAuditProposals(
		context.Context, string, string, string, string, []findingintake.ProposalKey,
	) ([]findingintake.ResolvedProposal, error)
}

type Importer struct {
	store     Store
	runs      RunReader
	artifacts ArtifactAccess
	findings  FindingRetention
}

func New(
	store Store,
	runs RunReader,
	artifactAccess ArtifactAccess,
	findings ...FindingRetention,
) (*Importer, error) {
	if store == nil || runs == nil || artifactAccess == nil {
		return nil, errors.New("Audit importer dependencies are incomplete")
	}
	if len(findings) > 1 {
		return nil, errors.New("Audit importer accepts at most one finding intake")
	}
	result := &Importer{store: store, runs: runs, artifacts: artifactAccess}
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
