// Package auditimport validates frozen Audit child-Run outputs and commits
// exact retained evidence, settlement receipts, coverage, and reports.
package auditimport

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
)

var ErrPermanent = errors.New("Audit import cannot be completed")

// Store is what the importer commits under the Controller's claim: collection
// receipts, report pages and the report itself. It is the claim seam of the
// Audit store.
type Store interface {
	Collect(context.Context, auditstore.CollectParams) (auditstore.CollectionReceipt, bool, error)
}

// Reader returns only the bounded selected Task membership and collection receipt.
type Reader interface {
	ListExecutionTasks(context.Context, string, string) ([]auditstore.Task, error)
	CollectionReceipt(context.Context, string, string) (auditstore.CollectionReceipt, error)
}

type RunReader interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
}

// FindingRetention is the purpose-specific bridge used only while collecting
// an Audit child Run. It transfers exact proposal/evidence revisions into the
// owning Audit inbox; it does not create findings, items, or review decisions.
type FindingRetention interface {
	ListAuditCollection(context.Context, string, string, string, findingintake.ListQuery) ([]findingintake.CollectionReceipt, error)

	RetainAuditCollectionBatch(context.Context, []findingintake.ImportRequest) error
	RejectAuditCollection(context.Context, findingintake.ImportRequest, string) error
	AuditExecutionReceiptReferences(context.Context, string, string, string, string) (map[string]findingintake.ReceiptReferences, error)
}

type Importer struct {
	intake    TaskWorkIntake
	store     Store
	reads     Reader
	runs      RunReader
	artifacts ArtifactAccess
	findings  FindingRetention
}

type TaskWorkIntake interface {
	AcceptPreparedWork(context.Context, auditstore.ControllerClaim, auditstore.ReconcileSnapshot, auditstore.Execution, auditcontract.Material, []byte) (auditstore.TaskIntakeResult, error)
}

func (i *Importer) WithTaskIntake(intake TaskWorkIntake) *Importer {
	bound := *i
	bound.intake = intake
	return &bound
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
	result := &Importer{store: store, reads: reads, runs: runs, artifacts: artifactAccess}
	if len(findings) == 1 {
		if findings[0] == nil {
			return nil, errors.New("Audit importer finding intake is nil")
		}
		result.findings = findings[0]
	}
	return result, nil
}
