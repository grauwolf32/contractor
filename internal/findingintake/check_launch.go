package findingintake

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

// PrepareCheck captures one current selection and its launch provenance in the
// Run creation transaction. Its ordinary Artifact acknowledgement survives Run purge.
func PrepareCheck(ctx context.Context, tx pgx.Tx, storage *artifactpostgres.Storage, runID string, check contracts.FindingCheck) (Receipt, error) {
	run, err := runstore.NewPostgresStore(tx, storage).GetRun(ctx, runID)
	if err != nil {
		return Receipt{}, err
	}
	if check.Validate() != nil || run.PublicationMode != runstore.PublicationOrdinary || run.ProjectID == nil || run.ProjectHTTPTarget == nil || run.State != runstore.RunInitializing {
		return Receipt{}, ErrInvalid
	}
	projectID := *run.ProjectID
	events := hookstore.NewFindingEventsTx(tx, storage)
	if err = events.LockFindingCorrelationFence(ctx, projectID); err != nil {
		return Receipt{}, err
	}
	if err = CheckSource(ctx, tx, storage, run.OwnerID, projectID, check); err != nil {
		return Receipt{}, err
	}
	finding, err := querystore.New(tx, storage).GetProjectFinding(ctx, run.OwnerID, projectID, check.FindingID)
	if err != nil {
		return Receipt{}, err
	}
	if finding.Revision != check.Revision {
		return Receipt{}, ErrConflict
	}
	receipt, err := scanReceiptRow(tx.QueryRow(ctx, getProjectReceiptSQL, run.OwnerID, projectID, check.ReceiptID))
	if err != nil {
		return Receipt{}, err
	}
	if receipt.ProjectProposal == nil || !sameCollectionContent(*receipt.ProjectProposal, receipt.Proposal) || len(receipt.ProjectEvidence) != len(receipt.Evidence) || len(receipt.ProjectEvidence) > MaxEvidenceRefs {
		return Receipt{}, blob.ErrIntegrity
	}
	bytes := receipt.ProjectProposal.SizeBytes
	for i, source := range receipt.ProjectEvidence {
		if source.Validate() != nil || !sameCollectionContent(source, receipt.Evidence[i]) {
			return Receipt{}, blob.ErrIntegrity
		}
		bytes += source.SizeBytes
	}
	if bytes > auditdomain.MaximumExpandedBytes {
		return Receipt{}, ErrInvalid
	}
	data, err := strictjson.Canonical(map[string]any{"schema": auditdomain.FindingSourceHistorySchema, "kind": "validation", "receiptId": check.ReceiptID, "runId": runID, "finding": check, "workflow": json.RawMessage(run.WorkflowSnapshot), "target": run.ProjectHTTPTarget, "proposal": receipt.ProjectProposal, "evidence": receipt.ProjectEvidence})
	if err != nil {
		return Receipt{}, err
	}
	if len(data) > auditdomain.MaximumCollectionJSONBytes {
		return Receipt{}, ErrInvalid
	}
	scope, _ := artifacts.ProjectScope(projectID)
	_, err = artifacts.NewService(storage.Repository(tx)).CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: artifactpolicy.FindingHistoryNamespace, Name: auditdomain.FindingSourceHistoryName(check.ReceiptID, "check-launch", runID)}, artifacts.Payload{MediaType: "application/json", Data: data})
	if err != nil {
		return Receipt{}, err
	}
	if err = events.AdvanceFindingHistory(ctx, projectID); err != nil {
		return Receipt{}, err
	}
	return receipt, nil
}
