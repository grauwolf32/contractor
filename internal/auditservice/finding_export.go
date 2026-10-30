package auditservice

import (
	"context"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/jackc/pgx/v5"
)

func (s *Service) FindingExportScope(ctx context.Context, tx pgx.Tx, ownerID, auditID string, lock bool) (string, error) {
	id, err := querystore.New(tx).FindingExportScope(ctx, ownerID, auditID, lock)
	return id, collectionError(err)
}

// SelectExportFindings freezes only this Audit's finding revisions, full
// contribution membership, current decisions and current assessment metadata.
func (s *Service) SelectExportFindings(ctx context.Context, tx pgx.Tx, ownerID, auditID string, selections []findingexport.Selection) ([]findingexport.Result, error) {
	ids := make([]string, len(selections))
	for i, v := range selections {
		ids[i] = v.FindingID
	}
	reads := querystore.New(tx)
	records, err := reads.ExportFindings(ctx, ownerID, auditID, ids)
	if err != nil {
		return nil, collectionError(err)
	}
	var decisionAudits, decisionIDs, assessmentAudits, assessmentIDs []string
	for _, v := range selections {
		row, ok := records[v.FindingID]
		if !ok {
			return nil, findingintake.ErrNotFound
		}
		if row.Revision != v.Revision {
			return nil, findingintake.ErrConflict
		}
		if row.CurrentDecisionID != nil {
			decisionAudits = append(decisionAudits, auditID)
			decisionIDs = append(decisionIDs, *row.CurrentDecisionID)
		}
		if row.CurrentAssessmentID != nil {
			assessmentAudits = append(assessmentAudits, auditID)
			assessmentIDs = append(assessmentIDs, *row.CurrentAssessmentID)
		}
	}
	contributions, err := reads.ExportContributions(ctx, auditID, ids)
	if err != nil {
		return nil, collectionError(err)
	}
	decisions, err := reads.ReadFindingDecisions(ctx, ownerID, decisionAudits, decisionIDs)
	if err != nil {
		return nil, collectionError(err)
	}
	assessments, err := reads.ReadFindingAssessments(ctx, ownerID, assessmentAudits, assessmentIDs)
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
		result := findingexport.Result{FindingID: row.FindingID, Revision: row.Revision, PresentationReceiptID: presentation, ReceiptIDs: []string{}, Review: findingexport.Review{State: string(row.State)}}
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
		if row.CurrentAssessmentID != nil {
			result.Review.AssessmentID = *row.CurrentAssessmentID
			a, ok := assessments[*row.CurrentAssessmentID]
			if !ok {
				return nil, artifacts.ErrArtifactIntegrity
			}
			result.Assessment = &findingexport.Assessment{ID: a.AssessmentID, SemanticAssessment: a.SemanticAssessment, ReceiptID: a.ReceiptID, Result: a.Result, DirectVerification: a.DirectVerification, Contract: a.Contract, AcceptedAt: a.AcceptedAt.UTC().Format(time.RFC3339Nano)}
			if a.ItemID != nil {
				result.Assessment.ItemID = *a.ItemID
			}
			if a.ExecutionItemID != nil {
				result.Assessment.ExecutionItemID = *a.ExecutionItemID
			}
			if a.CollectionReceiptID != nil {
				result.Assessment.CollectionReceiptID = *a.CollectionReceiptID
			}
		}
		if row.CurrentDecisionID != nil {
			d, ok := decisions[*row.CurrentDecisionID]
			if !ok || d.AuditID != auditID || d.FindingID != row.FindingID {
				return nil, artifacts.ErrArtifactIntegrity
			}
			result.Review.DecisionID = d.DecisionID
			result.Review.Verdict = string(d.Verdict)
			result.Review.Rationale = d.Rationale
			result.Review.ActorID = d.ActorID
			result.Review.DecisionAt = d.CreatedAt.UTC().Format(time.RFC3339Nano)
			if d.Severity != nil {
				result.Review.Severity = string(*d.Severity)
			}
		}
		if row.CorrelationGroupKey != "" {
			result.PrimaryKey = &findingexport.PrimaryKey{ProjectID: row.CorrelationProjectID, Algorithm: row.CorrelationAlgorithm, Value: row.CorrelationValue, GroupKey: row.CorrelationGroupKey, Policy: auditdomain.FindingCorrelationPolicy}
		}
		results = append(results, result)
	}
	return results, nil
}
