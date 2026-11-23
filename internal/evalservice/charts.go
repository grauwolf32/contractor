package evalservice

import (
	"context"
	"slices"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

type ChartParams struct {
	PairPageParams
	Chart string
}

type ChartView struct {
	ViewMetadata
	Chart              string                             `json:"chart"`
	SuiteID            *string                            `json:"suiteId"`
	MeasurementScope   *string                            `json:"measurementScope"`
	Unit               string                             `json:"unit"`
	Coverage           evaldomain.Coverage                `json:"coverage"`
	Quality            map[string]evaldomain.Quality      `json:"quality,omitempty"`
	AssessmentCoverage map[string]evaldomain.Ratio        `json:"assessmentCoverage,omitempty"`
	Bins               *[]evaldomain.Bin                  `json:"bins,omitempty"`
	Distributions      map[string]evaldomain.Distribution `json:"distributions,omitempty"`
	Differences        *[]evaldomain.Delta                `json:"differences,omitempty"`
	Points             *[]evaldomain.ProgressPoint        `json:"points,omitempty"`
	BucketMS           *int64                             `json:"bucketMs,omitempty"`
	HasMore            bool                               `json:"-"`
	LastOrdinal        int                                `json:"-"`
	LastDifference     *float64                           `json:"-"`
}

func (s *Service) Chart(ctx context.Context, p ChartParams) (ChartView, error) {
	out := ChartView{Chart: p.Chart}
	if !slices.Contains([]string{"quality", "tokens", "duration", "pair-deltas", "progress"}, p.Chart) || p.Filter != "" || p.VariantID != "" || p.Bin != nil {
		return out, evaldomain.Failure("eval_invalid")
	}
	metric := p.Chart
	if p.Chart == "pair-deltas" {
		metric = p.Metric
		if !validPage(p.AfterOrdinal, p.Limit) || !slices.Contains([]string{"tokens", "duration"}, metric) {
			return out, evaldomain.Failure("eval_invalid")
		}
	} else if p.Metric != "" || p.Absolute || p.AfterOrdinal >= 0 {
		return out, evaldomain.Failure("eval_invalid")
	}
	if p.Chart == "quality" || p.Chart == "progress" {
		if p.MeasurementScope != "" {
			return out, evaldomain.Failure("eval_invalid")
		}
	}
	err := s.withSelectedView(ctx, p.MemberPageParams, func(st *evalstore.Store, e evalstore.Experiment, v evalstore.View, setup preparedSetup) error {
		out.ViewMetadata = ViewMetadata{v.Snapshot, v.Freshness, v.Summary}
		summary := v.Summary
		if p.SuiteID != "" {
			out.SuiteID = &p.SuiteID
			summary = v.Suites[p.SuiteID]
		}
		expected := summary.Counts[setup.Comparison.Baseline].Expected
		out.Coverage = evaldomain.Coverage{Expected: expected, Reasons: map[string]int{}}
		switch p.Chart {
		case "quality":
			out.Unit = "fraction"
			out.Quality = summary.Quality
			out.AssessmentCoverage = map[string]evaldomain.Ratio{}
			for arm, counts := range summary.Counts {
				out.AssessmentCoverage[arm] = evaldomain.Fraction(counts.Scored, counts.Expected)
			}
			out.Coverage.Included = summary.CompleteQualityPairs
			out.Coverage.Excluded = expected - out.Coverage.Included
			if out.Coverage.Excluded > 0 {
				out.Coverage.Reasons["quality-incomplete"] = out.Coverage.Excluded
			}
			return nil
		case "progress":
			out.Unit = "members"
			points, width, err := st.Progress(ctx, e, v, p.SuiteID, setup.Comparison.Baseline, setup.Comparison.Candidate)
			if err != nil {
				return err
			}
			out.Points, out.BucketMS = &points, &width
			out.Coverage.Included = summary.TerminalPairs
			out.Coverage.Excluded = expected - summary.TerminalPairs
			if out.Coverage.Excluded > 0 {
				out.Coverage.Reasons["nonterminal"] = out.Coverage.Excluded
			}
			return nil
		}
		scope := setup.Variants[0].Kind
		out.MeasurementScope = &scope
		out.Unit = "tokens"
		if metric == "duration" {
			out.Unit = "milliseconds"
		}
		cohort, err := st.ChartCohort(ctx, e.OwnerID, e.ID, v.Generation, p.SuiteID, metric)
		if err != nil {
			return err
		}
		out.Coverage = cohort.Coverage
		if p.Chart != "pair-deltas" {
			out.Bins = &cohort.Bins
			out.Distributions = cohort.Distributions
			return nil
		}
		params := selectedPageParams(p.MemberPageParams, v)
		params.Metric, params.Absolute, params.AfterDifference = metric, p.Absolute, p.AfterDifference
		page, err := st.PairPage(ctx, params)
		if err != nil {
			return err
		}
		deltas := make([]evaldomain.Delta, 0, len(page.Items))
		for _, pair := range page.Items {
			a, b, gap := evaldomain.ComparableMeasure(pair.A, pair.B, metric, v.PinsVerified)
			if gap != "" {
				return evaldomain.Failure("eval_member_conflict")
			}
			deltas = append(deltas, evaldomain.Delta{PairID: pair.ID, SuiteID: pair.SuiteID, CaseID: pair.CaseID, Sample: pair.Sample, A: *a.Value, B: *b.Value, Difference: *b.Value - *a.Value, Regression: pair.Regression})
		}
		out.Differences = &deltas
		out.HasMore, out.LastOrdinal, out.LastDifference = page.HasMore, page.LastOrdinal, page.LastDifference
		return nil
	})
	return out, err
}
