package evalservice

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

// Apply only the explicitly selected records. Re-observation may invalidate
// their completeness, but cannot silently substitute a newer result or review.
func applySelectedRecords(ctx context.Context, st *evalstore.Store, e evalstore.Experiment, member evalstore.Member, checks []evaldomain.Check, observed *evaldomain.ResultInput, view *evaldomain.MemberView) (bool, error) {
	selected, err := st.Selected(ctx, e.OwnerID, e.ID, member.MemberID)
	if err != nil || selected == nil {
		return false, err
	}
	record, err := st.Record(ctx, e.OwnerID, e.ID, member.MemberID, evaldomain.RecordKindResult, selected.ResultSHA256)
	if err != nil {
		return false, err
	}
	var result evaldomain.ResultInput
	if err = json.Unmarshal(record.Document.Bytes(), &result); err != nil {
		return false, err
	}
	view.ResultSHA256 = &selected.ResultSHA256
	view.AssessmentSHA256 = selected.AssessmentSHA256
	complete := selectedCollectionComplete(result, observed, member.Recipe.Case.Outputs)
	if !sameExecution(result.Execution, *view.Execution) {
		complete = false
		view.Conflicting = true
	}
	available, err := selectedEvidenceAvailable(ctx, st, result)
	if err != nil {
		return false, err
	}
	complete = complete && available
	// A changed token scope does not erase unchanged parent duration.
	usage := reconcileSelectedUsage(result.Usage, observed)
	view.Usage = &usage
	if selected.AssessmentSHA256 == nil {
		return complete, nil
	}
	record, err = st.Record(ctx, e.OwnerID, e.ID, member.MemberID, evaldomain.RecordKindAssessment, *selected.AssessmentSHA256)
	if err != nil {
		return false, err
	}
	var assessment evaldomain.AssessmentInput
	if err = json.Unmarshal(record.Document.Bytes(), &assessment); err != nil {
		return false, err
	}
	view.Assessment = evaldomain.AssessmentDecision(checks, &assessment, complete)
	return complete, nil
}

func selectedEvidenceAvailable(ctx context.Context, st *evalstore.Store, result evaldomain.ResultInput) (bool, error) {
	refs := make(map[evaldomain.Artifact]struct{}, len(result.Outputs)+len(result.Evidence))
	for _, ref := range result.Outputs {
		refs[ref] = struct{}{}
	}
	for _, evidence := range result.Evidence {
		refs[evidence.Artifact] = struct{}{}
	}
	available := true
	for ref := range refs {
		if err := st.VerifyEvidence(ctx, ref); err != nil {
			if !evaldomain.IsCode(err, "eval_evidence_unavailable") {
				return false, err
			}
			available = false
		}
	}
	return available, nil
}
