package pgstore

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func (s *AdmissionSession) validatePreparedSource(ctx context.Context, item auditdomain.PreparedRoundItem) error {
	plan := item.Execution.AcceptedPlan
	if plan == nil || auditdomain.ValidateAcceptedPlan(*plan, item.Execution.Dependencies) != nil || plan.AuditID != s.State.AuditID || plan.OwnerID != s.reader.owner || plan.ProjectID != s.reader.project || plan.BaselineDigest != s.State.BaselineDigest || plan.ProfileDigest != s.State.ProfileDigest {
		return auditstore.ErrInvalid
	}
	source, err := s.Reads.FindingSource(ctx, plan.Source.ReceiptID)
	if err != nil {
		return err
	}
	ordinal := *plan.Source.ProposedCheckOrdinal
	if ordinal < 0 || ordinal >= len(source.Projection.Checks) || source.Original.Verify(plan.Source.Proposal) != nil || source.Retained.Verify(plan.Source.RetainedProposal) != nil || source.OriginRunID != plan.Source.RunID {
		return auditstore.ErrInvalid
	}
	header := source.Projection.Checks[ordinal]
	if header.Status != "registered" || header.PlanDigest != plan.PlanDigest || header.SourceMethod != plan.Plan.Method || header.Phase != plan.Plan.Phase || header.PlanSchema != plan.Plan.Schema {
		return auditstore.ErrInvalid
	}
	if s.validateRoute != nil {
		return s.validateRoute(*plan, item.Execution.Dependencies, item.WorkflowRole)
	}
	audit, err := s.reader.Get(ctx, s.reader.owner, s.State.AuditID)
	if err != nil {
		return err
	}
	profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil {
		return err
	}
	var baseline auditbaseline.BaselineSnapshot
	if strictjson.Decode(audit.BaselineSnapshot, &baseline) != nil {
		return auditstore.ErrInvalid
	}
	routes, err := auditbaseline.NewCheckRoutes(audit, profile, baseline)
	if err != nil {
		return err
	}
	s.validateRoute = routes.Validate
	return routes.Validate(*plan, item.Execution.Dependencies, item.WorkflowRole)
}

func (s *AdmissionSession) stageItems(ctx context.Context, page auditdomain.PreparedRoundPage) error {
	state := s.State
	if state.Root == nil {
		return auditstore.ErrPrecondition
	}
	ref, _ := json.Marshal(state.Root.Ref)
	created, err := s.tx.Exec(ctx, admissionCreateProposedSQL, state.AuditID, state.RoundID, state.RoundOrdinal, ref, state.Root.Digest, state.SelectedCount, state.Generation)
	if err != nil {
		return err
	}
	if created.RowsAffected() != 1 {
		return auditstore.ErrConflict
	}
	encoded, err := preparedPageStageRows(page)
	if err != nil {
		return err
	}
	if err := s.reader.work.add(0, 0, len(encoded), 0); err != nil {
		return err
	}
	tag, err := s.tx.Exec(ctx, admissionStageItemsSQL, state.AuditID, state.RoundID, encoded)
	if err != nil {
		return writeError("stage proposed Audit items", err)
	}
	if tag.RowsAffected() != int64(len(page.Items)) {
		return auditstore.ErrConflict
	}
	_, err = s.tx.Exec(ctx, admissionStageReviewsSQL, state.AuditID, state.RoundID, page.FirstItemOrdinal, page.FirstItemOrdinal+len(page.Items))
	if err != nil {
		return err
	}
	reviews := 0
	for _, item := range page.Items {
		if item.ApprovalKind != "none" {
			reviews++
		}
	}
	tag, err = s.tx.Exec(ctx, `UPDATE audit_rounds SET prepared_staged_count=prepared_staged_count+$3,prepared_review_count=prepared_review_count+$4,prepared_generation=$5 WHERE audit_id=$1 AND round_id=$2 AND state='proposed' AND prepared_staged_count=$6`, state.AuditID, state.RoundID, len(page.Items), reviews, state.Generation, state.StagedCount)
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return auditstore.ErrConflict
	}
	return nil
}

func preparedPageStageRows(page auditdomain.PreparedRoundPage) ([]byte, error) {
	input := make([]map[string]any, 0, len(page.Items))
	for _, item := range page.Items {
		if err := auditdomain.ValidateAcceptedPlan(*item.Execution.AcceptedPlan, item.Execution.Dependencies); err != nil {
			return nil, err
		}
		execution, err := strictjson.Canonical(item.Execution)
		if err != nil {
			return nil, err
		}
		dependencies := ""
		if item.Execution.Dependencies != nil {
			data, err := auditdomain.EncodeRetainedDependencies(*item.Execution.Dependencies)
			if err != nil {
				return nil, err
			}
			dependencies = string(data)
		}
		originRef := item.Origin.SourceRef
		origin := auditstore.ItemOrigin{Schema: item.Origin.Schema, SourceRef: &originRef, SourceContentDigest: item.Origin.SourceContentDigest,
			SourceMediaType: item.Origin.SourceMediaType, CanonicalInventoryDigest: item.Origin.CanonicalInventoryDigest, EntryKey: item.Origin.EntryKey}
		source := item.ProposalSources[0]
		one := page
		one.Items = []auditdomain.PreparedRoundItem{item}
		input = append(input, map[string]any{"item_id": item.ItemID, "item_key": item.ItemKey, "ordinal": item.Ordinal, "kind": item.Kind,
			"subject_key": item.SubjectKey, "task_ref": item.Task.Ref, "task_digest": item.Task.Digest, "origin": origin, "workflow_role": item.WorkflowRole,
			"initial_state": item.InitialState, "approval_kind": item.ApprovalKind, "approval_digest": item.ApprovalDigest, "status": item.Coverage.Status,
			"requested": item.Coverage.Requested, "completed": item.Coverage.Completed, "gaps": item.Coverage.Gaps, "rationale": item.Coverage.Rationale,
			"receipt_id": source.ReceiptID, "proposed_check_ordinal": source.ProposedCheckOrdinal, "proposal_ref": source.Proposal.Ref, "proposal_digest": source.Proposal.Digest,
			"retained_dependencies": dependencies, "prepared_execution": string(execution), "prepared_page_ordinal": page.PageOrdinal,
			"prepared_related_count": auditdomain.PreparedRoundPageDescriptors(one)})
	}
	return json.Marshal(input)
}

// PreparedPageStageBytes measures exactly the row envelope the staged writer stores.
func PreparedPageStageBytes(page auditdomain.PreparedRoundPage) (int, error) {
	data, err := preparedPageStageRows(page)
	return len(data), err
}
