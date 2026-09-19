package evalservice

import (
	"context"
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"time"
)

type MemberIdentity struct {
	ID            string  `json:"memberId"`
	SuiteID       string  `json:"suiteId"`
	CaseID        string  `json:"caseId"`
	Sample        int     `json:"sample"`
	VariantID     string  `json:"variantId"`
	CaseSHA256    string  `json:"caseSha256"`
	BindingSHA256 string  `json:"bindingSha256"`
	Eligibility   string  `json:"eligibility"`
	Reason        *string `json:"reason"`
}
type ExecutionView struct {
	Ref        *evaldomain.ExecutionRef `json:"ref"`
	State      string                   `json:"state"`
	StartedAt  *time.Time               `json:"startedAt"`
	FinishedAt *time.Time               `json:"finishedAt"`
	Reason     *string                  `json:"reason"`
}
type MemberView struct {
	Member           MemberIdentity    `json:"member"`
	Execution        *ExecutionView    `json:"execution"`
	ResultSHA256     *string           `json:"resultSha256"`
	AssessmentSHA256 *string           `json:"assessmentSha256"`
	Conflicting      bool              `json:"conflicting"`
	Assessment       string            `json:"assessment"`
	Usage            *evaldomain.Usage `json:"usage"`
}
type MemberPageParams struct {
	OwnerID, ExperimentID, Snapshot, Filter, VariantID string
	AfterOrdinal, Limit                                int
}
type MemberPage struct {
	Snapshot      string
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
	return MemberView{Member: MemberIdentity{r.MemberID, r.SuiteID, r.CaseID, r.Sample, r.VariantID, r.CaseSHA256, r.BindingSHA256, r.Eligibility, r.Reason}, Execution: &ex, Assessment: "unscored"}
}
func ratio(n, d int) map[string]any {
	var value *float64
	if d > 0 {
		v := float64(n) / float64(d)
		value = &v
	}
	return map[string]any{"numerator": n, "denominator": d, "value": value}
}

// Execution summaries report only observed execution facts. Result/assessment
// records and their selected view are incorporated by the collection reducer.
func executionSummary(rows []evalstore.ExecutionObservation) (json.RawMessage, error) {
	counts := map[string]map[string]int{}
	pairs := map[string]int{}
	for _, r := range rows {
		c := counts[r.VariantID]
		if c == nil {
			c = map[string]int{}
			for _, name := range []string{"expected", "eligible", "unsupported", "blocked", "submitted", "terminal", "missing", "conflicting", "collectionComplete", "scored", "qualityPassed", "executionSucceeded", "endToEndPassed"} {
				c[name] = 0
			}
			counts[r.VariantID] = c
		}
		c["expected"]++
		c[r.Eligibility]++
		if r.ExecutionID != nil {
			c["submitted"]++
		} else {
			c["missing"]++
		}
		ex := executionView(r)
		if r.ExecutionID != nil && (isTerminal(ex.State) || r.SubmissionState != nil && *r.SubmissionState == "terminal") {
			c["terminal"]++
			pairs[r.PairID]++
		}
		if ex.State == "succeeded" {
			c["executionSucceeded"]++
		}
	}
	terminalPairs := 0
	for _, n := range pairs {
		if n == 2 {
			terminalPairs++
		}
	}
	quality := map[string]any{}
	for arm, c := range counts {
		quality[arm] = map[string]any{"executionSuccess": ratio(c["executionSucceeded"], c["expected"]), "endToEndPass": ratio(0, c["expected"]), "conditionalQuality": ratio(0, 0)}
	}
	out, err := jsonBytes(map[string]any{"counts": counts, "quality": quality, "terminalPairs": terminalPairs, "completeQualityPairs": 0, "completeTokenPairs": 0, "conclusion": "inconclusive"})
	if err != nil {
		return nil, err
	}
	return out, evaldomain.Validate("Summary", out)
}
func memberMatches(v MemberView, filter, variant string) bool {
	if variant != "" && v.Member.VariantID != variant {
		return false
	}
	switch filter {
	case "", "all":
		return true
	case "unresolved":
		return !isTerminal(v.Execution.State) || v.Assessment == "unscored"
	case "failed":
		return v.Execution.State == "failed"
	case "unscored":
		return v.Assessment == "unscored"
	case "unsupported", "blocked":
		return v.Member.Eligibility == filter
	case "conflicting":
		return v.Conflicting
	}
	return false
}
func (s *Service) Members(ctx context.Context, p MemberPageParams) (MemberPage, error) {
	result := MemberPage{Items: []MemberView{}, LastOrdinal: -1}
	if p.Limit < 1 || p.Limit > evaldomain.MaxPageSize || p.AfterOrdinal < -1 || !contains([]string{"", "all", "unresolved", "failed", "unscored", "unsupported", "blocked", "conflicting"}, p.Filter) {
		return result, evaldomain.Failure("eval_invalid")
	}
	err := pg.InTx(ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		st := evalstore.NewTxStore(tx)
		e, err := st.Get(ctx, p.OwnerID, p.ExperimentID)
		if err != nil {
			return err
		}
		if e.Expected == 0 {
			return evaldomain.Failure("eval_not_ready")
		}
		rows, err := st.ExecutionObservations(ctx, p.OwnerID, p.ExperimentID)
		if err != nil {
			return err
		}
		if len(rows) != e.Expected {
			return evaldomain.Failure("eval_member_conflict")
		}
		if p.VariantID != "" {
			found := false
			for _, r := range rows {
				found = found || r.VariantID == p.VariantID
			}
			if !found {
				return evaldomain.Failure("eval_invalid")
			}
		}
		digest, err := hashJSON(struct {
			Revision int64
			Rows     []evalstore.ExecutionObservation
		}{e.Revision, rows})
		if err != nil {
			return err
		}
		result.Snapshot = "view-" + digest[7:]
		result.Revision = e.Revision
		if p.Snapshot != "" && p.Snapshot != result.Snapshot {
			return evaldomain.Failure("eval_view_changed")
		}
		result.Summary, err = executionSummary(rows)
		if err != nil {
			return err
		}
		for _, r := range rows {
			v := executionMember(r)
			if !memberMatches(v, p.Filter, p.VariantID) {
				continue
			}
			result.FilteredCount++
			if r.Ordinal <= p.AfterOrdinal {
				continue
			}
			if len(result.Items) >= p.Limit {
				result.HasMore = true
				continue
			}
			result.Items = append(result.Items, v)
			result.LastOrdinal = r.Ordinal
		}
		return nil
	})
	return result, err
}
