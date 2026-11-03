package auditservice

import (
	"context"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/jackc/pgx/v5"
)

func (s *Service) FindingExportScope(ctx context.Context, tx pgx.Tx, ownerID, auditID string, lock bool) (string, error) {
	id, err := querystore.New(tx, s.storage).FindingExportScope(ctx, ownerID, auditID, lock)
	return id, collectionError(err)
}

// SelectExportFindings freezes only this Audit's finding revisions, full
// contribution membership, current decisions and current assessment metadata.
func (s *Service) SelectExportFindings(ctx context.Context, tx pgx.Tx, ownerID, auditID string, selections []findingexport.Selection) ([]findingexport.Result, error) {
	ids := make([]string, len(selections))
	for i, v := range selections {
		ids[i] = v.FindingID
	}
	reads := querystore.New(tx, s.storage)
	records, err := reads.ExportFindings(ctx, ownerID, auditID, ids)
	if err != nil {
		return nil, collectionError(err)
	}
	for _, selection := range selections {
		row, found := records[selection.FindingID]
		if !found {
			return nil, findingintake.ErrNotFound
		}
		if row.Revision != selection.Revision {
			return nil, findingintake.ErrConflict
		}
	}
	contributions, err := reads.ExportContributions(ctx, auditID, ids)
	if err != nil {
		return nil, collectionError(err)
	}
	results := make([]findingexport.Result, 0, len(selections))
	all := map[string]bool{}
	for _, selection := range selections {
		row := records[selection.FindingID]
		members := map[string]bool{row.FirstReceiptID: true}
		for _, id := range contributions[row.FindingID] {
			members[id] = true
		}
		presentation := row.FirstReceiptID
		if selection.PresentationReceiptID != "" {
			presentation = selection.PresentationReceiptID
		}
		if !members[presentation] {
			return nil, findingintake.ErrNotFound
		}
		result := findingexport.Result{FindingID: row.FindingID, Revision: row.Revision, PresentationReceiptID: presentation, ReceiptIDs: []string{}, Review: findingexport.Review{}}
		for id := range members {
			result.ReceiptIDs = append(result.ReceiptIDs, id)
			all[id] = true
		}
		if len(all) > auditdomain.MaximumCollectionEntries {
			return nil, findingintake.ErrInvalid
		}
		sort.Strings(result.ReceiptIDs)
		if row.DuplicateTargetID != nil {
			result.Review.DuplicateTargetID = *row.DuplicateTargetID
		}
		if row.State != nil {
			value := string(*row.State)
			result.Review.State = &value
		}
		if row.AnalystVerdict != nil {
			result.Review.Verdict = string(*row.AnalystVerdict)
		}
		if row.AnalystSeverity != nil {
			result.Review.Severity = string(*row.AnalystSeverity)
		}
		if row.Rationale != nil {
			result.Review.Rationale = *row.Rationale
		}
		if row.ActorID != nil {
			result.Review.ActorID = *row.ActorID
		}
		if row.AssessedAt != nil {
			result.Review.DecisionAt = row.AssessedAt.UTC().Format(time.RFC3339Nano)
		}
		if row.CorrelationGroupKey != "" {
			result.PrimaryKey = &findingexport.PrimaryKey{ProjectID: row.ProjectID, Algorithm: row.CorrelationAlgorithm, Value: row.CorrelationValue, GroupKey: row.CorrelationGroupKey, Policy: auditdomain.FindingCorrelationPolicy}
		}
		results = append(results, result)
	}
	return results, nil
}
