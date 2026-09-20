package evalservice

import (
	"context"
	"encoding/json"
	"slices"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

type MemberIdentity = evaldomain.MemberIdentity
type ExecutionView = evaldomain.ExecutionView
type MemberView = evaldomain.MemberView
type MemberPageParams struct {
	OwnerID, ExperimentID, Snapshot, Filter, VariantID, SuiteID, MeasurementScope string
	Bin                                                                           *evalstore.BinFilter
	AfterOrdinal, Limit                                                           int
}

type MemberPage struct {
	Snapshot      string
	Freshness     string
	Revision      int64
	Summary       json.RawMessage
	FilteredCount int
	Items         []MemberView
	HasMore       bool
	LastOrdinal   int
}

func executionView(r evalstore.ExecutionObservation) ExecutionView {
	out := ExecutionView{State: "not_submitted", StartedAt: utcTime(r.StartedAt), FinishedAt: utcTime(r.FinishedAt)}
	if r.ExecutionID != nil {
		out.Ref = &evaldomain.ExecutionRef{Kind: r.Kind, ID: *r.ExecutionID}
		out.State = "unknown"
	}
	if r.SubmissionState != nil && *r.SubmissionState == "intent" {
		out.State = "unknown"
		reason := "Submission outcome is being reconciled."
		out.Reason = &reason
	}
	if r.SubmissionState != nil && *r.SubmissionState == "rejected" {
		reason := "The member was not started."
		out.Reason = &reason
	}
	if r.OrdinaryState != nil {
		switch *r.OrdinaryState {
		case "succeeded", "completed":
			out.State = "succeeded"
		case "failed":
			out.State = "failed"
		case "cancelled":
			out.State = "cancelled"
		case "initializing", "draft":
			out.State = "accepted"
		case "running", "active", "paused", "waiting_review", "finalizing", "cancelling":
			out.State = "running"
		default:
			out.State = "unknown"
		}
	}
	if r.Deleted {
		out.State = "unknown"
		out.FinishedAt = nil
		reason := "Execution was deleted; retained identity prevents another submission."
		out.Reason = &reason
	}
	if !isTerminal(out.State) {
		out.FinishedAt = nil
	}
	return out
}

func isTerminal(state string) bool {
	return state == "succeeded" || state == "failed" || state == "cancelled"
}

func executionMember(r evalstore.ExecutionObservation) MemberView {
	ex := executionView(r)
	return MemberView{
		Member: MemberIdentity{
			ID: r.MemberID, SuiteID: r.SuiteID, CaseID: r.CaseID, Sample: r.Sample, VariantID: r.VariantID,
			CaseSHA256: r.CaseSHA256, BindingSHA256: r.BindingSHA256, Eligibility: r.Eligibility, Reason: r.Reason,
		},
		Execution:  &ex,
		Assessment: "unscored",
	}
}

func (s *Service) Members(ctx context.Context, p MemberPageParams) (MemberPage, error) {
	result := MemberPage{Items: []MemberView{}, LastOrdinal: -1}
	if !validPage(p.AfterOrdinal, p.Limit) || !slices.Contains([]string{"", "all", "unresolved", "failed", "unscored", "unsupported", "blocked", "conflicting"}, p.Filter) {
		return result, evaldomain.Failure("eval_invalid")
	}
	err := s.withSelectedView(ctx, p, func(st *evalstore.Store, e evalstore.Experiment, v evalstore.View, setup preparedSetup) error {
		if p.VariantID != "" {
			if _, ok := v.Summary.Counts[p.VariantID]; !ok {
				return evaldomain.Failure("eval_invalid")
			}
		}
		page, err := st.SelectedMemberPage(ctx, selectedPageParams(p, v))
		if err != nil {
			return err
		}
		result.Snapshot, result.Freshness, result.Revision = v.Snapshot, v.Freshness, e.Revision
		result.Summary, err = json.Marshal(v.Summary)
		result.Items, result.FilteredCount, result.HasMore, result.LastOrdinal = page.Items, page.FilteredCount, page.HasMore, page.LastOrdinal
		return err
	})
	return result, err
}
