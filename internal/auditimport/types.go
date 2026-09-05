// Package auditimport validates frozen Audit child-Run outputs and commits
// exact retained evidence, settlement receipts, coverage, and reports.
package auditimport

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

var ErrPermanent = errors.New("Audit import cannot be completed")

type Store interface {
	ListExecutionItems(context.Context, string) ([]auditstore.ExecutionItem, error)
	ListItems(context.Context, string) ([]auditstore.Item, error)
	ListCoverage(context.Context, string, string, int, int) ([]auditstore.CoverageRow, error)
	CollectionDispositionCounts(context.Context, string) (auditstore.CollectionDispositionCounts, error)
	Collect(context.Context, auditstore.CollectParams) (auditstore.CollectionReceipt, bool, error)
	CommitReport(context.Context, auditstore.CommitReportParams) (auditstore.Audit, error)
}

type RunReader interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
}

type Importer struct {
	store     Store
	runs      RunReader
	artifacts ArtifactAccess
}

func New(store Store, runs RunReader, artifactAccess ArtifactAccess) (*Importer, error) {
	if store == nil || runs == nil || artifactAccess == nil {
		return nil, errors.New("Audit importer dependencies are incomplete")
	}
	return &Importer{store: store, runs: runs, artifacts: artifactAccess}, nil
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
